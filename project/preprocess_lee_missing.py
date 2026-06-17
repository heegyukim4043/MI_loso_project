"""
Preprocess Lee2019_MI subjects 42-54 and merge with existing lee2019.npz
"""
import os
import sys
import numpy as np
import mne
import moabb
from moabb.datasets import Lee2019_MI

sys.path.insert(0, r"g:\MI_opendata")
from preprocess_data import preprocess_dataset, SAVE_DIR

mne.set_log_level("WARNING")
moabb.set_download_dir(r"g:\MI_opendata")
mne.set_config("MNE_DATA", r"g:\MI_opendata")

EXISTING_PATH = os.path.join(SAVE_DIR, "lee2019.npz")
MISSING_PATH  = os.path.join(SAVE_DIR, "lee2019_missing.npz")

# Preprocess only missing subjects 42-54
lee_dataset = Lee2019_MI()
preprocess_dataset(
    dataset=lee_dataset,
    subjects=[42] + list(range(44, 55)),   # 43 already in npz
    sfreq_target=100,
    save_path=MISSING_PATH,
    n_ica=20,
    per_session_ica=True,
)

# Merge
print("\nMerging with existing data...")
d_exist  = np.load(EXISTING_PATH,  allow_pickle=True)
d_new    = np.load(MISSING_PATH,   allow_pickle=True)

X_all    = np.concatenate([d_exist["X"],        d_new["X"]],        axis=0)
y_all    = np.concatenate([d_exist["y"],        d_new["y"]],        axis=0)
subs_all = np.concatenate([d_exist["subjects"], d_new["subjects"]], axis=0)

np.savez_compressed(
    EXISTING_PATH,
    X=X_all,
    y=y_all,
    subjects=subs_all,
    sfreq=d_exist["sfreq"],
    ch_names=d_exist["ch_names"],
)
print(f"Merged → {EXISTING_PATH}")
print(f"X={X_all.shape}, subjects={np.unique(subs_all).shape[0]}")
