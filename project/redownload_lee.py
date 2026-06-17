"""Delete corrupted Lee2019 session files and re-download via MOABB."""
import os, scipy.io, moabb, mne
moabb.set_download_dir(r"g:\MI_opendata")
mne.set_config("MNE_DATA", r"g:\MI_opendata")
mne.set_log_level("WARNING")
from moabb.datasets import Lee2019_MI

BASE = ("G:/MI_opendata/MNE-lee2019-mi-data/gigadb-datasets/live/pub"
        "/10.5524/100001_101000/100542")

# Find and delete bad files
deleted = []
for subj in range(42, 55):
    for sess in [1, 2]:
        f = f"{BASE}/session{sess}/s{subj}/sess0{sess}_subj{subj:02d}_EEG_MI.mat"
        if not os.path.exists(f):
            deleted.append(f"MISSING S{subj} sess{sess}")
            continue
        try:
            scipy.io.loadmat(f)
        except Exception:
            os.remove(f)
            deleted.append(f"DELETED S{subj} sess{sess}: {f}")
            print(f"Deleted: {f}")

print(f"\nDeleted {len(deleted)} bad files. Re-downloading...")

# Force re-download by loading each subject (MOABB downloads missing files)
ds = Lee2019_MI()
for subj in range(42, 55):
    try:
        print(f"Downloading S{subj}...", flush=True)
        data = ds.get_data(subjects=[subj])
        print(f"  S{subj} OK")
    except Exception as e:
        print(f"  S{subj} FAILED: {e}")

print("Done.")
