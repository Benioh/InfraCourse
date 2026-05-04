from __future__ import annotations
from _toy_data import RAW, make_ppm, make_text

for i, color in enumerate([(255, 0, 0), (0, 180, 0), (0, 0, 255)], 1):
    make_ppm(RAW / "flickr8k" / f"image_{i}.ppm", *color)
    make_text(
        RAW / "flickr8k" / f"image_{i}.txt",
        f"一张教学图片 {i}，用于 image-text caption。",
    )
print("已生成 Flickr8K 练习子集")
