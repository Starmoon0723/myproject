# QfinVidCore/perception/keyframe_extraction.py
"""
从视频中提取关键帧。
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Union

from QfinVidCore.core.VideoEntity import VideoEntity
from QfinVidCore.core.config import get_config
from QfinVidCore.utils.runtime import run_cmd


Pathlike = Union[str, Path]


def keyframe_extraction(
    input_video: Union[VideoEntity, Pathlike],
    output_dir: Optional[Pathlike] = None,
    n_frames: Optional[int] = None,
    method: Optional[str] = None,
    **kwargs
) -> List[Path]:
    """
    从视频中提取关键帧。
    
    Args:
        input_video: 输入视频实体或路径
        output_dir: 输出目录，如果为None则在输入文件同目录生成
        n_frames: 要提取的关键帧数量，默认为配置文件中的设置
        method: 提取方法，默认为配置文件中的设置
        **kwargs: 传递给提取算法的其他参数
    
    Returns:
        List[Path]: 提取的关键帧文件路径列表
    """
    # 处理输入视频
    if isinstance(input_video, VideoEntity):
        input_path = input_video.uri
    else:
        input_path = Path(input_video).resolve()
    
    # 获取配置参数
    if n_frames is None:
        n_frames = get_config("perception.keyframe_extraction.n_frames", 5)
    if method is None:
        method = get_config("perception.keyframe_extraction.method", "uniform")
    
    # 处理输出目录
    if output_dir is None:
        output_dir_config = get_config("perception.keyframe_extraction.output_dir")
        if output_dir_config:
            output_dir = Path(output_dir_config).resolve()
        else:
            output_dir = input_path.parent / f"{input_path.stem}_keyframes"
    else:
        output_dir = Path(output_dir).resolve()
    
    # 确保输出目录存在
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 构建输出文件模板
    output_template = str(output_dir / f"keyframe_%06d.png")
    
    if method == "uniform":
        # 使用ffmpeg的select滤镜均匀采样
        args = [
            "ffmpeg",
            "-i", str(input_path),
            "-vf", f"select=not(mod(n\,{max(1, int(input_video.num_frames / n_frames) if isinstance(input_video, VideoEntity) else 30)})),setpts=N/FRAME_RATE/TB",
            "-vsync", "vfr",
        ]
    else:
        # 默认使用均匀采样
        args = [
            "ffmpeg",
            "-i", str(input_path),
            "-vf", f"select=not(mod(n\,30)),setpts=N/FRAME_RATE/TB",
            "-vsync", "vfr",
        ]
    
    # 添加覆盖参数
    overwrite = get_config("dependencies.ffmpeg.default_args.overwrite", True)
    if overwrite:
        args.append("-y")
    
    args.append(output_template)
    
    # 执行命令
    run_cmd(args, check=True)
    
    # 收集输出文件
    output_files = list(output_dir.glob("*.png"))
    output_files.sort()
    
    # 如果提取的帧数量超过n_frames，只返回前n_frames个
    if len(output_files) > n_frames:
        output_files = output_files[:n_frames]
    
    return output_files
