import torch
from PIL import Image
from lavis.models import load_model_and_preprocess
from lavis.processors import load_processor
from transformers import CLIPProcessor, CLIPModel, BlipProcessor, BlipForImageTextRetrieval

import pandas as pd
import json
from decord import VideoReader
from decord import cpu, gpu
import numpy as np
import os
import cv2

import numpy as np
import pickle

import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description='Extract Video Feature')

    parser.add_argument('--dataset_name', type=str, default='longvideobench', help='support longvideobench and videomme')
    parser.add_argument('--label_path', type=str, default='./datasets/longvideobench/lvb_val.json',help='your path of the label')
    parser.add_argument('--video_path', type=str, default='./datasets/longvideobench/videos',help='your path of the video')
    parser.add_argument('--extract_feature_model', type=str,default='blip', help='blip/clip/sevila')
    parser.add_argument('--output_file', type=str,default='./outscores',help='path of output scores and frames')
    parser.add_argument('--device', type=str,default='cuda')
    parser.add_argument('--blip_model_path', type=str, default='', help='local path for BLIP (e.g., ./models/blip-itm-large-coco)')
    parser.add_argument('--clip_model_path', type=str, default='', help='local path for CLIP (e.g., ./models/clip-vit-base-patch32)')

    return parser.parse_args()


def main(args):
    # --- Path Setup ---
    # # videomme, mvbench, lvbench, mlvu, mmvu, videommmu
    if args.dataset_name =="longvideobench":
        label_path = os.path.join(args.label_path,'lvb_val.json')
        video_path = os.path.join(args.video_path,'videos')
    elif args.dataset_name =="videomme":
        label_path = os.path.join(args.label_path,'videomme.json')
        video_path = os.path.join(args.video_path,'data')
    elif args.dataset_name =="mlvu":
        label_path = os.path.join(args.label_path,'mlvu_all.json')
        video_path = args.video_path
    elif args.dataset_name == "mmvu":
        label_path = os.path.join(args.label_path, 'validation_process.json')
        video_root = args.video_path
    elif args.dataset_name == "videommmu":
        label_path = os.path.join(args.label_path, 'VideoMMMU.tsv')  # 注意：这里是 .tsv
        video_root = args.video_path 
    elif args.dataset_name == "lvbench":
        # label_path = os.path.join(args.label_path, 'video_info.meta.json')
        label_path = os.path.join(args.label_path, 'video_info.meta_wo_options.json')
        video_root = args.video_path
    elif args.dataset_name == "mvbench":
        label_path = os.path.join(args.label_path, 'mvbench_all.json')
        video_root = args.video_path
    elif args.dataset_name == "fcmbench":
        label_path = os.path.join(args.label_path, 'fcmbench_longvideo_v1.0_20260205_absolute.jsonl')
        video_root = args.video_path
    else:
        raise ValueError("dataset_name: longvideobench or videomme or mlvu or mmvu or videommmu or lvbench or mvbench or fcmbench")

    if os.path.exists(label_path):
        if label_path.endswith('.tsv'):
            # Load VideoMMMU TSV
            df = pd.read_csv(label_path, sep='\t')
            # Convert to list of dicts (like JSON)
            datas = df.to_dict(orient='records')
        elif label_path.endswith('.jsonl'):
            with open(label_path, 'r', encoding='utf-8') as f:
                return [json.loads(line) for line in f]
        else:
            # Original JSON loading
            with open(label_path, 'r') as f:
                datas = json.load(f)
    else:
        raise OSError("the label file does not exist")
    
    device = args.device
    
    # --- Model Loading (Unchanged) ---
    use_local_blip = False
    blip_processor = None
    if args.extract_feature_model == 'blip':
        blip_path = os.path.join(args.blip_model_path, "blip-itm-large-coco-ssd")
        if blip_path and os.path.exists(blip_path): # Added exist check for safety
            use_local_blip = True
            print(f"Loading local BLIP from {blip_path}")
            model = BlipForImageTextRetrieval.from_pretrained(blip_path, local_files_only=True).to(device)
            blip_processor = BlipProcessor.from_pretrained(blip_path, local_files_only=True)
        else:
            print("Loading BLIP from LAVIS hub...")
            model, vis_processors, text_processors = load_model_and_preprocess("blip_image_text_matching", "large", device=device, is_eval=True)
    elif args.extract_feature_model == 'clip':
        clip_path = os.path.join(args.clip_model_path, "clip-vit-base-patch32-ssd")
        model = CLIPModel.from_pretrained(clip_path or "openai/clip-vit-base-patch32", local_files_only=bool(clip_path))
        model.to(device)
        processor = CLIPProcessor.from_pretrained(clip_path or "openai/clip-vit-base-patch32", local_files_only=bool(clip_path))
    else:
        raise ValueError("model not support")

    # --- Output Directory Setup ---
    if not os.path.exists(os.path.join(args.output_file,args.dataset_name)):
        os.makedirs(os.path.join(args.output_file,args.dataset_name), exist_ok=True)
    out_score_path = os.path.join(args.output_file,args.dataset_name,args.extract_feature_model)
    if not os.path.exists(out_score_path):
        os.makedirs(out_score_path, exist_ok=True)
    
    # Define file paths. 
    # NOTE: Output format is now JSON Lines (one JSON array per line).
    score_file_path = os.path.join(out_score_path,'scores.json')
    frame_file_path = os.path.join(out_score_path,'frames.json')

    # --- Resume Logic (New) ---
    processed_count = 0
    if os.path.exists(score_file_path):
        with open(score_file_path, 'r') as f:
            # Count lines to determine how many videos were already processed
            processed_count = sum(1 for _ in f)
        print(f"Found existing output file. {processed_count} videos already processed. Resuming...")

    total_videos = len(datas)

    # Open files in APPEND mode ('a') outside the loop
    # Using buffering=1 might help, but explicit flush is safer for this use case
    with open(score_file_path, 'a') as f_score, open(frame_file_path, 'a') as f_frame:
        
        for idx, data in enumerate(datas):
            # Skip already processed videos
            if idx < processed_count:
                continue

            print(f"[{idx + 1}/{total_videos}] Processing video: {data.get('video_path', data.get('videoID', 'unknown'))}")
            
            # --- Data Preparation (Unchanged) ---
            try:
                text = data['question']
            except:
                text = data['prompt']

            if args.dataset_name == 'longvideobench':
                video = os.path.join(video_path, data["video_path"])
            elif args.dataset_name == 'videomme':
                video = os.path.join(video_path, data["videoID"]+'.mp4')
            elif args.dataset_name == 'mlvu':
                video = data["video_path"]
            elif args.dataset_name == 'mmvu':
                video = data["video_path"]
            elif args.dataset_name == 'videommmu':
                video = data["video_path"]
            elif args.dataset_name == 'lvbench':
                video = data["video_path"]
            elif args.dataset_name == 'mvbench':
                video = data["video_path"]
            elif args.dataset_name == 'fcmbench':
                video = data["video_path"]
            else:
                raise ValueError("dataset_name: longvideobench or videomme or mlvu or mmvu or videommmu or lvbench or mvbench or fcmbench")
                
            try:
                # Add basic error handling for bad video files so script doesn't die
                if not os.path.exists(video):
                    print(f"Warning: Video file not found: {video}")
                    # Write empty lists to maintain index alignment or handle as you see fit
                    # Here we write empty to keep line count consistent with dataset
                    score = []
                    frame_num = []
                else:
                    # duration = data['duration']
                    vr = VideoReader(video, ctx=cpu(0), num_threads=1)
                    fps = vr.get_avg_fps()
                    frame_nums = int(len(vr)/int(fps))

                    score = []
                    frame_num = []

                    # --- Feature Extraction (Unchanged Logic) ---
                    if args.extract_feature_model == 'blip':
                        if use_local_blip:
                            for j in range(frame_nums):
                                raw_image = np.array(vr[j*int(fps)])
                                raw_image = Image.fromarray(raw_image)
                                inputs = blip_processor(images=raw_image, text=text, return_tensors="pt").to(device)
                                with torch.no_grad():
                                    blip_output = model(**inputs, return_dict=True)
                                itm_scores = getattr(blip_output, "itm_scores", None)
                                if itm_scores is None:
                                    itm_scores = getattr(blip_output, "logits_per_image", None)
                                if itm_scores is None:
                                    itm_scores = blip_output[0]
                                blip_scores = torch.nn.functional.softmax(itm_scores, dim=1)
                                score.append(blip_scores[:, 1].item())
                                frame_num.append(j*int(fps))
                        else:
                            txt = text_processors["eval"](text)
                            for j in range(frame_nums):
                                raw_image = np.array(vr[j*int(fps)])
                                raw_image = Image.fromarray(raw_image)
                                img = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
                                with torch.no_grad():
                                    blip_output = model({"image": img, "text_input": txt}, match_head="itm")              
                                blip_scores = torch.nn.functional.softmax(blip_output, dim=1)
                                score.append(blip_scores[:, 1].item())
                                frame_num.append(j*int(fps))

                    elif args.extract_feature_model == 'clip':
                        inputs_text = processor(text=text, return_tensors="pt", padding=True,truncation=True).to(device)
                        text_features = model.get_text_features(**inputs_text)
                        for j in range(frame_nums):
                            raw_image = np.array(vr[j*int(fps)])
                            raw_image = Image.fromarray(raw_image)
                            inputs_image = processor(images=raw_image, return_tensors="pt", padding=True).to(device)
                            with torch.no_grad():
                                image_features = model.get_image_features(**inputs_image)
                            clip_score = torch.nn.CosineSimilarity(dim=-1)(text_features, image_features)
                            score.append(clip_score.item())
                            frame_num.append(j*int(fps))

            except Exception as e:
                print(f"Error processing video {idx}: {e}")
                # Optional: Write empty lists or None to indicate failure, ensuring line alignment
                score = []
                frame_num = []

            # --- Incremental Writing (New) ---
            # Write the current result as a single line JSON
            f_score.write(json.dumps(score) + "\n")
            f_frame.write(json.dumps(frame_num) + "\n")
            
            # Flush to disk immediately to ensure data safety
            f_score.flush()
            f_frame.flush()

    print("Processing complete.")

if __name__ == '__main__':
    args = parse_arguments()
    main(args)