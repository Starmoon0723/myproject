import math

def calculate_qwen_tokens(
    target_frames,       # 你想设置的帧数 (nframes)
    orig_h, orig_w,      # 原始视频/图片的分辨率
    
    # 以下是你 provided 的参数配置
    max_pixels=786432,         # 你的 max_pixels
    total_pixels=234881024,    # 你的 total_pixels
    min_pixels=49152,          # 你的 min_pixels
):
    # --- 常量定义 (Qwen2.5/3-VL) ---
    PATCH_SIZE = 16
    MERGE_SIZE = 2
    FACTOR = PATCH_SIZE * MERGE_SIZE  # 32
    FRAME_FACTOR = 2  # 时间维度每2帧合并
    
    # 1. 确保帧数是对齐的 (必须是2的倍数)
    nframes = round(target_frames / FRAME_FACTOR) * FRAME_FACTOR
    if nframes != target_frames:
        print(f"Warning: Frames adjusted from {target_frames} to {nframes}")

    # 2. 计算单帧允许的最大像素 (动态限制)
    # 逻辑来源：fetch_video 中的 max_pixels 计算
    limit_by_total = (total_pixels / nframes) * FRAME_FACTOR
    
    # 取三者中的最小值：
    # A. 硬性上限 max_pixels
    # B. 总量限制分配到单帧的量 limit_by_total
    # C. 代码中还有一个硬编码的 VIDEO_FRAME_MAX_PIXELS，通常等于 max_pixels
    actual_max_pixels = min(max_pixels, limit_by_total)
    
    # 确保不低于最小值
    actual_max_pixels = max(actual_max_pixels, int(min_pixels * 1.05))

    # 3. 模拟 Smart Resize (核心步骤)
    # 计算缩放比例
    current_pixels = orig_h * orig_w
    scale = math.sqrt(actual_max_pixels / current_pixels)
    
    # 计算缩放后的长宽 (必须是32的倍数)
    # 逻辑：先缩放，除以32取整，再乘回32
    new_h = max(FACTOR, round(orig_h * scale / FACTOR) * FACTOR)
    new_w = max(FACTOR, round(orig_w * scale / FACTOR) * FACTOR)
    
    # 4. 二次检查 (Corner Case)
    # 如果缩放后依然超过上限（因为取整可能变大），需要缩小
    if new_h * new_w > actual_max_pixels:
        scale = math.sqrt(actual_max_pixels / (new_h * new_w))
        new_h = math.floor(new_h * scale / FACTOR) * FACTOR
        new_w = math.floor(new_w * scale / FACTOR) * FACTOR
    
    # 5. 计算 Token 数量
    # 时间维度压缩 / 2
    # 空间维度压缩 / 32 / 32
    spatial_tokens = (new_h // FACTOR) * (new_w // FACTOR)
    temporal_blocks = nframes // FRAME_FACTOR
    
    total_tokens = temporal_blocks * spatial_tokens
    
    print(f"--- 预计算结果 ---")
    print(f"输入: {nframes}帧, 原分辨率 {orig_h}x{orig_w}")
    print(f"限制: 单帧最大像素 {int(actual_max_pixels)}")
    print(f"缩放: {new_h}x{new_w} (Factor={FACTOR})")
    print(f"结构: {temporal_blocks} (时间块) x {spatial_tokens} (空间Token/块)")
    print(f"最终 Token 总量: {total_tokens}")
    
    return total_tokens

# --- 测试案例 ---
# 假设你想做对比实验，目标是控制在 10000 token 左右

# 实验A：少帧 (8帧)
tokens_a = calculate_qwen_tokens(target_frames=8, orig_h=1080, orig_w=1920)

# 实验B：多帧 (64帧)
tokens_b = calculate_qwen_tokens(target_frames=64, orig_h=1080, orig_w=1920)