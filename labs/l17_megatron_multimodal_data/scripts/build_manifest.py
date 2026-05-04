from __future__ import annotations
from _toy_data import RAW, MANIFEST, write_jsonl

rows = []
for image in sorted((RAW / "flickr8k").glob("*.ppm")):
    rows.append(
        {
            "id": image.stem,
            "task": "image_text_caption",
            "files": {"image": str(image), "text": str(image.with_suffix(".txt"))},
            "split": "train",
        }
    )
for wav in sorted((RAW / "librispeech").glob("*.wav")):
    rows.append(
        {
            "id": wav.stem,
            "task": "audio_text_asr",
            "files": {"audio": str(wav), "text": str(wav.with_suffix(".txt"))},
            "split": "train",
        }
    )
for wav in sorted((RAW / "esc50").glob("*.wav")):
    rows.append(
        {
            "id": wav.stem,
            "task": "audio_label",
            "files": {"audio": str(wav), "label": str(wav.with_suffix(".label"))},
            "split": "train",
        }
    )
write_jsonl(MANIFEST, rows)
print(MANIFEST)
