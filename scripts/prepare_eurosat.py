"""Extract the EuroSAT parquet shards into an image tree plus a manifest."""
import io
import os
import sys

sys.path.insert(0, os.getcwd())
import pyarrow.parquet as pq
from PIL import Image
from mlkit.manifest import sha256, write_manifest, assert_no_split_overlap

RAW, OUT = "data/raw/eurosat", "data/processed/eurosat"
SPLITS = {"train": "train", "validation": "val", "test": "test"}
CLASSES = ["AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
           "Pasture", "PermanentCrop", "Residential", "River", "SeaLake"]


def main():
    rows = []
    for src_split, out_split in SPLITS.items():
        table = pq.read_table(f"{RAW}/{src_split}.parquet")
        has_names = "filename" in table.column_names
        images = table.column("image").to_pylist()
        labels = table.column("label").to_pylist()
        names = table.column("filename").to_pylist() if has_names else None

        for i, (img, label) in enumerate(zip(images, labels)):
            cls = CLASSES[label]
            directory = os.path.join(OUT, out_split, cls)
            os.makedirs(directory, exist_ok=True)

            fname = os.path.basename(names[i]) if names else f"{out_split}_{i}.jpg"
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                fname += ".jpg"
            path = os.path.join(directory, fname)
            if not os.path.exists(path):
                Image.open(io.BytesIO(img["bytes"])).convert("RGB").save(path, quality=95)
            rows.append({"path": path, "label": cls, "split": out_split,
                         "sha256": sha256(path), "source": f"EuroSAT_RGB/{src_split}"})
        print(f"  {src_split}: {len(images)} images", flush=True)

    assert_no_split_overlap(rows, "sha256")
    os.makedirs("data/manifests", exist_ok=True)
    write_manifest(rows, "data/manifests/eurosat.csv")

    counts = {}
    for r in rows:
        counts.setdefault(r["split"], 0)
        counts[r["split"]] += 1
    print("EuroSAT prepared:", len(rows), "images", counts)


if __name__ == "__main__":
    main()
