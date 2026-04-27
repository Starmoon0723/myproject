import argparse
import ast
import inspect
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


def _safe_json_loads(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except Exception:
            return value
    return value


def _read_jsonl(path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _parse_layer_ids(layer_text: str) -> list[int]:
    out: list[int] = []
    for x in layer_text.split(","):
        x = x.strip()
        if not x:
            continue
        out.append(int(x))
    if not out:
        raise ValueError("layers不能为空")
    return out


def _unpack_video_features(video_outputs: Any) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    # HF不同版本在can_return_tuple下返回结构略有差异，这里做兼容解析。
    if hasattr(video_outputs, "pooler_output"):
        pooler = video_outputs.pooler_output
        deepstack = getattr(video_outputs, "deepstack_features", None)
    elif isinstance(video_outputs, (list, tuple)):
        if len(video_outputs) == 0:
            raise ValueError("get_video_features返回空tuple/list")
        pooler = video_outputs[0]
        deepstack = video_outputs[1] if len(video_outputs) > 1 else None
    else:
        raise ValueError(f"无法解析get_video_features返回类型: {type(video_outputs)}")

    if isinstance(pooler, torch.Tensor):
        pooler_list = [pooler]
    else:
        pooler_list = list(pooler)

    if deepstack is None:
        deepstack_list: list[torch.Tensor] = []
    elif isinstance(deepstack, torch.Tensor):
        deepstack_list = [deepstack]
    else:
        deepstack_list = list(deepstack)
    return pooler_list, deepstack_list


def _normalize_keep_indices(
    keep_indices: Any,
    *,
    num_frames: int,
    tokens_per_frame: int,
) -> list[list[int]]:
    keep_indices = _safe_json_loads(keep_indices)
    if keep_indices is None:
        raise ValueError("缺少 video_token_keep_indices，无法对齐裁切token")
    if isinstance(keep_indices, torch.Tensor):
        keep_indices = keep_indices.detach().cpu().tolist()
    if not isinstance(keep_indices, (list, tuple)):
        raise ValueError(f"video_token_keep_indices类型错误: {type(keep_indices)}")

    out: list[list[int]] = []
    for frame_idx in range(num_frames):
        src = keep_indices[frame_idx] if frame_idx < len(keep_indices) else []
        if isinstance(src, torch.Tensor):
            src = src.detach().cpu().tolist()
        if not isinstance(src, (list, tuple)):
            src = []
        vals: set[int] = set()
        for v in src:
            try:
                iv = int(v)
            except Exception:
                continue
            if 0 <= iv < tokens_per_frame:
                vals.add(iv)
        out.append(sorted(vals))
    return out


def _build_retention_mask(
    keep_indices: list[list[int]],
    *,
    num_frames: int,
    tokens_per_frame: int,
    device: torch.device,
) -> torch.Tensor:
    mask = torch.zeros(num_frames * tokens_per_frame, dtype=torch.bool, device=device)
    for frame_idx in range(num_frames):
        local = keep_indices[frame_idx] if frame_idx < len(keep_indices) else []
        if not local:
            continue
        offset = frame_idx * tokens_per_frame
        mask[offset + torch.tensor(local, device=device, dtype=torch.long)] = True
    return mask


def _find_frame_video_positions(
    prompt_token_ids: list[int],
    *,
    num_frames: int,
    video_token_id: int,
    vision_start_token_id: int,
    vision_end_token_id: int,
    llm_grid_h: int,
    llm_grid_w: int,
) -> list[dict[str, Any]]:
    frames: list[dict[str, Any]] = []
    cursor = 0
    seq_len = len(prompt_token_ids)

    for frame_idx in range(num_frames):
        try:
            start = prompt_token_ids.index(vision_start_token_id, cursor)
        except ValueError as e:
            raise ValueError(f"在prompt中未找到第{frame_idx}帧的vision_start") from e
        try:
            end = prompt_token_ids.index(vision_end_token_id, start + 1)
        except ValueError as e:
            raise ValueError(f"在prompt中未找到第{frame_idx}帧的vision_end") from e

        frame_video_positions = [
            pos
            for pos in range(start + 1, end)
            if prompt_token_ids[pos] == video_token_id
        ]
        frames.append(
            {
                "frame_idx": int(frame_idx),
                "timestamp_token_positions": list(range(cursor, start)),
                "vision_start_position": int(start),
                "vision_end_position": int(end),
                "video_token_count_in_prompt": int(len(frame_video_positions)),
                "video_token_positions": frame_video_positions,
            }
        )
        cursor = end + 1

    if cursor > seq_len:
        raise ValueError("帧位置解析超出prompt长度")
    base_tokens_per_frame = int(llm_grid_h * llm_grid_w)
    full_layout = all(
        int(frame["video_token_count_in_prompt"]) == base_tokens_per_frame for frame in frames
    )
    for frame in frames:
        frame["prompt_layout_full_tokens"] = bool(full_layout)
        frame["base_tokens_per_frame"] = base_tokens_per_frame
    return frames


def _select_kept_prompt_positions(
    frame_positions: list[dict[str, Any]],
    keep_indices: list[list[int]],
    *,
    llm_grid_w: int,
) -> tuple[list[int], list[dict[str, Any]]]:
    selected_positions: list[int] = []
    frame_attn_meta: list[dict[str, Any]] = []

    for frame in frame_positions:
        frame_idx = int(frame["frame_idx"])
        frame_video_positions = list(frame["video_token_positions"])
        frame_keep_indices = keep_indices[frame_idx] if frame_idx < len(keep_indices) else []
        full_layout = bool(frame["prompt_layout_full_tokens"])

        token_items = []
        if full_layout:
            for keep_idx in frame_keep_indices:
                if 0 <= keep_idx < len(frame_video_positions):
                    pos = int(frame_video_positions[keep_idx])
                    selected_positions.append(pos)
                    token_items.append(
                        {
                            "keep_index": int(keep_idx),
                            "llm_row": int(keep_idx // llm_grid_w),
                            "llm_col": int(keep_idx % llm_grid_w),
                            "prompt_position": pos,
                        }
                    )
        else:
            pair_count = min(len(frame_video_positions), len(frame_keep_indices))
            for i in range(pair_count):
                keep_idx = int(frame_keep_indices[i])
                pos = int(frame_video_positions[i])
                selected_positions.append(pos)
                token_items.append(
                    {
                        "keep_index": keep_idx,
                        "llm_row": int(keep_idx // llm_grid_w),
                        "llm_col": int(keep_idx % llm_grid_w),
                        "prompt_position": pos,
                    }
                )

        frame_attn_meta.append(
            {
                "frame_idx": frame_idx,
                "timestamp_token_positions": frame["timestamp_token_positions"],
                "vision_start_position": frame["vision_start_position"],
                "vision_end_position": frame["vision_end_position"],
                "video_token_count_in_prompt": frame["video_token_count_in_prompt"],
                "keep_index_count": len(frame_keep_indices),
                "prompt_layout_full_tokens": full_layout,
                "tokens": token_items,
            }
        )

    return selected_positions, frame_attn_meta


def _build_videomme_messages(row: dict[str, Any], data_root: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    video_path = os.path.join(data_root, row["video_path"])
    frame_indices = _safe_json_loads(row.get("frame_indices", None))

    content_for_text: list[dict[str, Any]] = []
    content_for_vision: list[dict[str, Any]] = []

    video_item_text = {"type": "video", "video": video_path}
    video_item_vision = {"type": "video", "video": video_path}
    if frame_indices is not None:
        video_item_text["frame_indices"] = frame_indices
        video_item_vision["frame_indices"] = frame_indices

    for k in ("min_pixels", "max_pixels", "total_pixels", "min_frames", "max_frames", "fps", "nframes"):
        if k in row and row[k] is not None:
            video_item_text[k] = row[k]
            video_item_vision[k] = row[k]

    content_for_text.append(video_item_text)
    content_for_vision.append(video_item_vision)

    candidates = row.get("candidates", "[]")
    if isinstance(candidates, str):
        candidates = ast.literal_eval(candidates)
    question = "\n" + row["question"] + "\n" + "\n".join(candidates)
    text_item = {
        "type": "text",
        "text": f"\nThese are the frames of a video. {question}\nThe best answer is: ",
    }
    content_for_text.append(text_item)
    content_for_vision.append(text_item)

    messages_for_text = [{"role": "user", "content": content_for_text}]
    messages_for_vision = [{"role": "user", "content": content_for_vision}]
    return messages_for_text, messages_for_vision


def _extract_one_sample(
    row: dict[str, Any],
    *,
    processor,
    model,
    process_vision_info,
    data_root: str,
    layer_ids_1based: list[int],
    append_generated_token: bool,
) -> list[dict[str, Any]]:
    prompt_token_ids = _safe_json_loads(row.get("prompt_token_ids", None))
    if prompt_token_ids is None:
        raise ValueError("结果中缺少 prompt_token_ids，请先用更新后的evaluator重新推理")
    prompt_token_ids = [int(x) for x in prompt_token_ids]

    generated_token_ids = _safe_json_loads(row.get("generated_token_ids", None))
    if generated_token_ids is None:
        generated_token_ids = []
    generated_token_ids = [int(x) for x in generated_token_ids]

    messages_for_text, messages_for_vision = _build_videomme_messages(row, data_root=data_root)
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages_for_vision,
        image_patch_size=processor.image_processor.patch_size,
        return_video_kwargs=True,
        return_video_metadata=True,
    )
    if video_inputs is None or len(video_inputs) == 0:
        raise ValueError("当前样本未解析到video_inputs")
    if video_kwargs is None:
        video_kwargs = {}
    # 预采样视频时显式提供fps，避免Qwen3VL时间戳fallback到24fps。
    if "fps" not in video_kwargs:
        default_fps = float(row.get("fps", 2.0) if row.get("fps", None) is not None else 2.0)
        video_kwargs["fps"] = [default_fps]

    # overwrite_vision_process(return_video_metadata=True) 返回的元素可能是
    # (video_tensor, video_metadata)。HF processor 只接受 videos=list[video]，
    # metadata 需单独通过 video_metadata 传入。
    normalized_videos: list[Any] = []
    normalized_video_metadata: list[Any] = []
    for item in video_inputs:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            normalized_videos.append(item[0])
            normalized_video_metadata.append(item[1])
        else:
            normalized_videos.append(item)

    processor_kwargs = dict(video_kwargs)
    if len(normalized_video_metadata) > 0:
        processor_kwargs["video_metadata"] = normalized_video_metadata

    video_proc = processor(
        text=["<|vision_start|><|video_pad|><|vision_end|>"],
        images=image_inputs,
        videos=normalized_videos,
        return_tensors="pt",
        **processor_kwargs,
    )
    pixel_values_videos = video_proc["pixel_values_videos"].to(model.device)
    video_grid_thw = video_proc["video_grid_thw"].to(model.device)
    if video_grid_thw.ndim != 2 or video_grid_thw.shape[0] != 1:
        raise ValueError(f"当前脚本仅支持单视频样本，video_grid_thw={video_grid_thw.shape}")

    t, h, w = [int(x) for x in video_grid_thw[0].tolist()]
    merge_size = int(model.model.visual.spatial_merge_size)
    llm_grid_h = int(h // merge_size)
    llm_grid_w = int(w // merge_size)
    tokens_per_frame = int(llm_grid_h * llm_grid_w)

    keep_indices = _normalize_keep_indices(
        row.get("video_token_keep_indices", None),
        num_frames=t,
        tokens_per_frame=tokens_per_frame,
    )
    retention_mask = _build_retention_mask(
        keep_indices,
        num_frames=t,
        tokens_per_frame=tokens_per_frame,
        device=model.device,
    )

    frame_positions = _find_frame_video_positions(
        prompt_token_ids,
        num_frames=t,
        video_token_id=int(model.config.video_token_id),
        vision_start_token_id=int(model.config.vision_start_token_id),
        vision_end_token_id=int(model.config.vision_end_token_id),
        llm_grid_h=llm_grid_h,
        llm_grid_w=llm_grid_w,
    )
    kept_prompt_positions, frame_attn_meta = _select_kept_prompt_positions(
        frame_positions,
        keep_indices,
        llm_grid_w=llm_grid_w,
    )

    with torch.no_grad():
        video_outputs = model.model.get_video_features(
            pixel_values_videos=pixel_values_videos,
            video_grid_thw=video_grid_thw,
        )
        video_embed_list, deepstack_video_embeds = _unpack_video_features(video_outputs)
        video_embeds = torch.cat(video_embed_list, dim=0)

        video_embeds = video_embeds[retention_mask]
        deepstack_video_embeds = [x[retention_mask] for x in deepstack_video_embeds]

        full_input_ids = list(prompt_token_ids)
        query_position = len(prompt_token_ids) - 1
        if append_generated_token and len(generated_token_ids) > 0:
            full_input_ids.append(int(generated_token_ids[0]))
            query_position = len(prompt_token_ids)

        input_ids = torch.tensor([full_input_ids], device=model.device, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=model.device)
        inputs_embeds = model.model.get_input_embeddings()(input_ids)

        video_positions = torch.tensor(
            kept_prompt_positions,
            device=model.device,
            dtype=torch.long,
        )
        if int(video_positions.numel()) != int(video_embeds.shape[0]):
            raise ValueError(
                "按keep_indices映射后的prompt视频token数量与裁切后video embedding数量不一致: "
                f"{int(video_positions.numel())} vs {int(video_embeds.shape[0])}. "
                "请检查frame_indices/keep_indices与prompt_token_ids是否来自同一次运行。"
            )
        inputs_embeds[0, video_positions, :] = video_embeds.to(inputs_embeds.dtype)

        visual_pos_masks = torch.zeros(
            (1, input_ids.shape[1]),
            dtype=torch.bool,
            device=model.device,
        )
        visual_pos_masks[0, video_positions] = True

        video_token_id = int(model.config.video_token_id)
        mm_token_type_ids = torch.zeros_like(input_ids, dtype=torch.int32, device=model.device)
        mm_token_type_ids[input_ids == video_token_id] = 2

        if hasattr(model.model, "compute_3d_position_ids"):
            position_ids = model.model.compute_3d_position_ids(
                input_ids=input_ids,
                video_grid_thw=video_grid_thw,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                past_key_values=None,
                mm_token_type_ids=mm_token_type_ids,
            )
        elif hasattr(model.model, "get_rope_index"):
            rope_fn = model.model.get_rope_index
            sig = inspect.signature(rope_fn)
            rope_kwargs: dict[str, Any] = {
                "input_ids": input_ids,
                "image_grid_thw": None,
                "video_grid_thw": video_grid_thw,
                "attention_mask": attention_mask,
            }
            if "mm_token_type_ids" in sig.parameters:
                rope_kwargs["mm_token_type_ids"] = mm_token_type_ids
            rope_out = rope_fn(**rope_kwargs)
            if isinstance(rope_out, tuple):
                position_ids = rope_out[0]
            else:
                position_ids = rope_out
        else:
            raise AttributeError("Qwen3VLModel既不包含compute_3d_position_ids也不包含get_rope_index")

        seq_len = int(input_ids.shape[1])
        if query_position <= 0 or query_position >= seq_len:
            raise ValueError(f"非法query_position={query_position}, seq_len={seq_len}")

        # Stage-1: prefill到query之前，构建KV cache（不取attention，避免LxL显存爆炸）
        prefill_len = int(query_position)
        prefill_outputs = model.model.language_model(
            input_ids=None,
            position_ids=position_ids[:, :, :prefill_len],
            attention_mask=attention_mask[:, :prefill_len],
            past_key_values=None,
            inputs_embeds=inputs_embeds[:, :prefill_len, :],
            visual_pos_masks=visual_pos_masks[:, :prefill_len],
            deepstack_visual_embeds=deepstack_video_embeds,
            output_attentions=False,
            use_cache=True,
            return_dict=True,
        )
        past_key_values = prefill_outputs.past_key_values

        # Stage-2: 只跑query token，取注意力（形状约为 [heads, 1, L]）
        query_outputs = model.model.language_model(
            input_ids=None,
            position_ids=position_ids[:, :, query_position : query_position + 1],
            attention_mask=attention_mask[:, : query_position + 1],
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds[:, query_position : query_position + 1, :],
            visual_pos_masks=visual_pos_masks[:, query_position : query_position + 1],
            deepstack_visual_embeds=None,
            output_attentions=True,
            use_cache=False,
            return_dict=True,
        )

    results: list[dict[str, Any]] = []
    num_layers = len(query_outputs.attentions)
    for layer_id in layer_ids_1based:
        layer_idx = int(layer_id) - 1
        if layer_idx < 0 or layer_idx >= num_layers:
            raise ValueError(f"层号{layer_id}超出范围，模型共{num_layers}层")

        attn = query_outputs.attentions[layer_idx][0]  # [heads, 1, k]
        q_to_all = attn[:, 0, :]  # [heads, seq]
        q_to_all_mean = q_to_all.mean(dim=0)

        frame_items = []
        for frame in frame_attn_meta:
            token_items = []
            for token_info in frame["tokens"]:
                pos = int(token_info["prompt_position"])
                token_items.append(
                    {
                        **token_info,
                        "attention": float(q_to_all_mean[pos].item()),
                    }
                )
            frame_items.append(
                {
                    "frame_idx": frame["frame_idx"],
                    "timestamp_token_positions": frame["timestamp_token_positions"],
                    "vision_start_position": frame["vision_start_position"],
                    "vision_end_position": frame["vision_end_position"],
                    "video_token_count_in_prompt": frame["video_token_count_in_prompt"],
                    "keep_index_count": frame["keep_index_count"],
                    "prompt_layout_full_tokens": frame["prompt_layout_full_tokens"],
                    "tokens": token_items,
                }
            )

        results.append(
            {
                "index": row.get("index", None),
                "video_path": row.get("video_path", None),
                "prediction": row.get("prediction", None),
                "layer_id_1based": int(layer_id),
                "query_position": int(query_position),
                "query_token_id": int(full_input_ids[query_position]),
                "query_token_is_generated": bool(query_position >= len(prompt_token_ids)),
                "prompt_length": int(len(prompt_token_ids)),
                "sequence_length_for_attention": int(len(full_input_ids)),
                "video_grid_thw": [int(t), int(h), int(w)],
                "llm_grid_hw": [int(llm_grid_h), int(llm_grid_w)],
                "frame_attentions": frame_items,
            }
        )
    return results


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="提取Qwen3VL裁切token在指定层的attention score。")
    p.add_argument("--result_jsonl", type=str, required=True, help="evaluator输出jsonl路径")
    p.add_argument("--output_jsonl", type=str, required=True, help="attention输出jsonl路径")
    p.add_argument("--model_path", type=str, required=True, help="HF模型路径，例如Qwen/Qwen3-VL-8B-Instruct")
    p.add_argument("--data_root", type=str, required=True, help="Video-MME数据根目录")
    p.add_argument("--layers", type=str, default="2,17,32", help="1-based层号，逗号分隔")
    p.add_argument("--indices", type=str, default="", help="只处理指定样本index，逗号分隔")
    p.add_argument("--max_samples", type=int, default=0, help="最多处理样本数，0表示不限制")
    p.add_argument("--device", type=str, default="cuda", help="cuda/cpu")
    p.add_argument("--append_generated_token", action="store_true", help="将首个生成token拼到序列末尾作为query")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    custom_code_dir = Path(__file__).resolve().parents[1] / "models" / "ours" / "code"
    if str(custom_code_dir) not in sys.path:
        sys.path.append(str(custom_code_dir))
    from overwrite_vision_process import process_vision_info  # noqa: WPS433

    rows = _read_jsonl(args.result_jsonl)
    wanted_indices = set()
    if args.indices.strip():
        wanted_indices = {int(x.strip()) for x in args.indices.split(",") if x.strip()}

    selected_rows: list[dict[str, Any]] = []
    for row in rows:
        if wanted_indices and int(row.get("index", -1)) not in wanted_indices:
            continue
        selected_rows.append(row)
        if args.max_samples > 0 and len(selected_rows) >= args.max_samples:
            break
    if not selected_rows:
        raise ValueError("未筛选到任何样本")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    processor = AutoProcessor.from_pretrained(args.model_path)
    try:
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            args.model_path,
            dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
            attn_implementation="eager",
        )
    except Exception:
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            args.model_path,
            dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        )
    model.to(device)
    model.eval()

    layer_ids_1based = _parse_layer_ids(args.layers)

    output_rows: list[dict[str, Any]] = []
    for row in selected_rows:
        output_rows.extend(
            _extract_one_sample(
                row=row,
                processor=processor,
                model=model,
                process_vision_info=process_vision_info,
                data_root=args.data_root,
                layer_ids_1based=layer_ids_1based,
                append_generated_token=bool(args.append_generated_token),
            )
        )

    out_path = Path(args.output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in output_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"[DONE] wrote {len(output_rows)} rows to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

