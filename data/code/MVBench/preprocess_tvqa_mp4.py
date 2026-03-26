# preprocess_tvqa_mp4.py
import os
import glob
from moviepy.editor import ImageSequenceClip

root = "/data/oceanus_share/shangshouduo-jk/project/datasets/MVBench/video/unpacked/tvqa/frames_fps3_hq/"
for frame_dir in glob.glob(os.path.join(root, "*")):
    if os.path.isdir(frame_dir):
        mp4_path = frame_dir + ".mp4"
        if not os.path.exists(mp4_path):
            frames = sorted(glob.glob(os.path.join(frame_dir, "*.jpg")))
            if frames:
                print(f"Generating {mp4_path}")
                clip = ImageSequenceClip(frames, fps=3)
                clip.write_videofile(mp4_path, codec="libx264", logger=None, audio=False)