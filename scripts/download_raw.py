"""Download raw datasets into data/raw/. Idempotent: existing files are skipped."""
import os, sys, json, urllib.request, hashlib

RAW = "data/raw"
HF = "https://huggingface.co/api/datasets/blanchon/EuroSAT_RGB/parquet/default"


def log(msg):
    print(msg, flush=True)


def download(url, dest):
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        log(f"  skip (exists): {dest}")
        return
    tmp = dest + ".part"
    with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    os.replace(tmp, dest)
    log(f"  ok: {dest} ({os.path.getsize(dest)/1e6:.1f} MB)")


def eurosat():
    log("EuroSAT parquet shards:")
    os.makedirs(f"{RAW}/eurosat", exist_ok=True)
    for split in ["train", "validation", "test"]:
        download(f"{HF}/{split}/0.parquet", f"{RAW}/eurosat/{split}.parquet")


def mitdb():
    """Fetch the MIT-BIH records used by the DS1/DS2 inter-patient split."""
    import wfdb
    # Paced records 102, 104, 107, 217 are excluded by convention.
    DS1 = [101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122,
           124, 201, 203, 205, 207, 208, 209, 215, 220, 223, 230]
    DS2 = [100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210,
           212, 213, 214, 219, 221, 222, 228, 231, 232, 233, 234]
    out = f"{RAW}/mitdb"
    os.makedirs(out, exist_ok=True)
    records = DS1 + DS2
    log(f"MIT-BIH records ({len(records)}):")
    for i, rec in enumerate(records, 1):
        marker = os.path.join(out, f"{rec}.dat")
        if os.path.exists(marker):
            log(f"  [{i}/{len(records)}] skip {rec}")
            continue
        wfdb.dl_database("mitdb", out, records=[str(rec)], annotators=["atr"])
        log(f"  [{i}/{len(records)}] ok {rec}")
    json.dump({"DS1": DS1, "DS2": DS2}, open(f"{out}/splits.json", "w"), indent=2)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "eurosat"):
        eurosat()
    if which in ("all", "mitdb"):
        mitdb()
    log("DOWNLOADS COMPLETE")
