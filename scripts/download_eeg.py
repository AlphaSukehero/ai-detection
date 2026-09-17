"""Fetch the smallest EEG subsets that still permit an honest evaluation.

Deliberately not the full corpora. CHB-MIT is roughly 40 GB and ds004504
about 2 GB; neither is needed to establish whether this pipeline works. What
is needed is enough of each to train and, more importantly, to hold out a
test split that was never seen -- for CHB-MIT that means seizure-containing
records plus genuine seizure-free background from the same patient, and for
ds004504 a balanced handful of subjects per diagnosis.

Scope is a deliberate limitation, not an oversight, and it bounds the claims
anything downstream may make: a model trained on one epilepsy patient has not
been shown to generalise to another, and 12 subjects cannot establish a
dementia biomarker. Both are recorded in the dataset card.

    python scripts/download_eeg.py            # both, default subset
    python scripts/download_eeg.py seizure    # CHB-MIT only
    python scripts/download_eeg.py alzheimer  # ds004504 only
"""
import os
import sys
import urllib.error
import urllib.request

# The PhysioNet S3 mirror, not physionet.org itself. Measured from here the
# main site served large files at roughly 10 kB/s -- about an hour per 35 MB
# recording -- while the mirror sustained ~280 kB/s for byte-identical
# content. Same data, same paths, 28x the throughput.
CHB_BASE = "https://physionet-open.s3.amazonaws.com/chbmit/1.0.0"
ADF_BASE = "https://s3.amazonaws.com/openneuro.org/ds004504"
RAW = "data/raw"

# One patient. chb01's seizure records, plus seizure-free records from the
# same patient for the negative class -- same electrodes, same amplifier, same
# session conditions, so the classifier cannot separate the classes on
# recording setup instead of on physiology.
CHB_SUBJECT = "chb01"
CHB_SEIZURE_FILES = ["chb01_03.edf", "chb01_04.edf", "chb01_15.edf",
                     "chb01_16.edf", "chb01_18.edf", "chb01_21.edf",
                     "chb01_26.edf"]
CHB_BASELINE_FILES = ["chb01_01.edf", "chb01_02.edf", "chb01_05.edf",
                      "chb01_06.edf"]

# Balanced across diagnosis so the split cannot be won by guessing the
# majority class. Group labels come from participants.tsv (A=Alzheimer's,
# C=control, F=frontotemporal dementia); F is skipped to keep this binary.
# Controls are sub-037..065 -- an earlier list took sub-066..071, which are F,
# and prep silently dropped every one; tests now check these against the file.
ADF_SUBJECTS = {
    "A": ["sub-001", "sub-002", "sub-003", "sub-004", "sub-005", "sub-006"],
    "C": ["sub-037", "sub-038", "sub-039", "sub-040", "sub-041", "sub-042"],
}


def fetch(url, dest):
    """Download unless already present. Returns True if the file is there."""
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        print(f"  have {os.path.basename(dest)}")
        return True
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + ".part"
    try:
        with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
            while chunk := r.read(1 << 20):
                f.write(chunk)
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        # Leave no partial file behind: a truncated EDF reads as a valid but
        # short recording, which is worse than an absent one.
        if os.path.exists(tmp):
            os.remove(tmp)
        print(f"  FAILED {os.path.basename(dest)}: {e}")
        return False
    os.rename(tmp, dest)
    print(f"  got  {os.path.basename(dest)} "
          f"({os.path.getsize(dest) / 1e6:.1f} MB)")
    return True


def download_seizure():
    out = f"{RAW}/chbmit/{CHB_SUBJECT}"
    print(f"CHB-MIT {CHB_SUBJECT} -> {out}")
    ok = fetch(f"{CHB_BASE}/{CHB_SUBJECT}/{CHB_SUBJECT}-summary.txt",
               f"{out}/{CHB_SUBJECT}-summary.txt")
    for name in CHB_SEIZURE_FILES + CHB_BASELINE_FILES:
        ok &= fetch(f"{CHB_BASE}/{CHB_SUBJECT}/{name}", f"{out}/{name}")
    return ok


def download_alzheimer():
    out = f"{RAW}/ds004504"
    print(f"ds004504 -> {out}")
    ok = fetch(f"{ADF_BASE}/participants.tsv", f"{out}/participants.tsv")
    for group, subjects in ADF_SUBJECTS.items():
        for sub in subjects:
            name = f"{sub}_task-eyesclosed_eeg.set"
            ok &= fetch(f"{ADF_BASE}/{sub}/eeg/{name}",
                        f"{out}/{sub}/eeg/{name}")
            print(f"    ^ group {group}")
    return ok


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    ok = True
    if which in ("all", "seizure"):
        ok &= download_seizure()
    if which in ("all", "alzheimer"):
        ok &= download_alzheimer()
    if not ok:
        raise SystemExit("some downloads failed; re-run to resume")
    print("done")


if __name__ == "__main__":
    main()
