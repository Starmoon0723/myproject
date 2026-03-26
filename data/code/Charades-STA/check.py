import os
import pandas as pd

video_dir = "/data/oceanus_share/shangshouduo-jk/project/datasets/Charades-STA/video"
df = pd.read_parquet("/data/oceanus_share/shangshouduo-jk/project/datasets/Charades-STA/data/test-00000-of-00001.parquet")

print(df.head())
print("Total test samples:", len(df))

missing = []
for vid in df["video"].unique():
    if not os.path.exists(os.path.join(video_dir, f"{vid}")):
        missing.append(vid)

print(f"Missing videos: {len(missing)}")
