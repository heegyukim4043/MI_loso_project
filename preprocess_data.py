"""
Preprocessing pipeline for Cho2017 and Lee2019_MI datasets.

Steps per subject:
  1. Load raw EEG via MOABB
  2. Resample  (Cho2017 → 128 Hz, Lee2019_MI → 100 Hz)
  3. Fit ICA on 1-40 Hz copy  →  ICLabel, remove eye/muscle comps ≥ 90 %
  4. Apply ICA to 8-30 Hz bandpass-filtered data
  5. Epoch [2, 4] s post-stimulus onset  (2-second window)
  6. Save  preprocessed/cho2017.npz  and  preprocessed/lee2019.npz

Output arrays (per file):
  X        : (N, C, T)  float32   — epoched EEG
  y        : (N,)       int64     — 0 = left_hand, 1 = right_hand
  subjects : (N,)       int64     — subject id for each trial
  sfreq    : scalar               — sampling frequency after resample
  ch_names : (C,)       object    — channel names
"""

import os
import numpy as np
import mne
import moabb
from moabb.datasets import Cho2017, Lee2019_MI

mne.set_log_level("WARNING")
moabb.set_download_dir(r"g:\MI_opendata")
mne.set_config("MNE_DATA", r"g:\MI_opendata")

SAVE_DIR = r"g:\MI_opendata\preprocessed"
os.makedirs(SAVE_DIR, exist_ok=True)

# ICLabel component labels (order matches model output)
ICLABEL_NAMES = [
    "brain", "muscle artifact", "eye blink",
    "heart beat", "line noise", "channel noise", "other",
]
ARTIFACT_LABELS = {"eye blink", "muscle artifact"}


# ─────────────────────────────────────────────────────────────────────────────
# ICA + ICLabel
# ─────────────────────────────────────────────────────────────────────────────

def remove_artifacts_ica(raw, n_components=20, threshold=0.90):
    """
    Fit extended-infomax ICA on a 1-100 Hz, CAR-referenced copy,
    label with ICLabel, then subtract eye-blink / muscle components
    above *threshold* from the already-filtered *raw* (in-place).

    ICLabel requirements (addressed here):
      - Bandpass 1-100 Hz for ICA fitting
      - Common Average Reference (CAR)
      - Extended infomax algorithm
    """
    try:
        from mne_icalabel import label_components
    except ImportError:
        print("    [WARN] mne-icalabel not found — skipping ICA")
        return raw

    # Prepare a copy that meets ICLabel requirements.
    # Called BEFORE resampling so original sfreq (512/1000 Hz) ensures Nyquist > 100 Hz.
    raw_ica_fit = raw.copy().filter(1.0, 100.0, verbose=False)
    raw_ica_fit.set_eeg_reference("average", projection=False, verbose=False)

    ica = mne.preprocessing.ICA(
        n_components=n_components,
        method="infomax",
        fit_params=dict(extended=True),
        random_state=42,
        max_iter="auto",
        verbose=False,
    )
    ica.fit(raw_ica_fit, verbose=False)

    ic_labels = label_components(raw_ica_fit, ica, method="iclabel")  # no warnings now
    labels = ic_labels["labels"]          # list of str
    probs  = ic_labels["y_pred_proba"]    # (n_components, n_classes)

    # y_pred_proba: (n_components,) — probability of the predicted label
    exclude = [
        idx for idx, (lbl, prob) in enumerate(zip(labels, probs))
        if lbl in ARTIFACT_LABELS and float(prob) >= threshold
    ]

    if exclude:
        print(f"    Removing {len(exclude)} ICA comp(s): "
              f"{[labels[i] for i in exclude]}")
        ica.exclude = exclude
        ica.apply(raw, verbose=False)
    else:
        print(f"    No artifact components ≥ {threshold*100:.0f}%")

    return raw


# ─────────────────────────────────────────────────────────────────────────────
# Event extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_epochs(raw, tmin=2.0, tmax=4.0):
    """
    Return X (n_trials, C, T) and y (n_trials,) from *raw*.
    y: 0 = left_hand, 1 = right_hand.
    """
    events, event_id = mne.events_from_annotations(raw, verbose=False)

    # Build  code → binary-label  mapping
    label_map = {}
    for name, code in event_id.items():
        n = name.lower()
        if "left" in n or n in ("1", "769"):
            label_map[code] = 0
        elif "right" in n or n in ("2", "770"):
            label_map[code] = 1

    if len(label_map) < 2:
        # Fallback: sort by code, first=0, second=1
        codes = sorted(set(events[:, 2]))[:2]
        label_map = {codes[0]: 0, codes[1]: 1}
        print(f"    [WARN] fallback label mapping: {label_map}")

    sel_event_id = {k: v for k, v in event_id.items() if v in label_map}

    epochs = mne.Epochs(
        raw, events, event_id=sel_event_id,
        tmin=tmin, tmax=tmax,
        baseline=None, preload=True, verbose=False,
    )

    X = epochs.get_data().astype(np.float32)          # (N, C, T)
    y = np.array(
        [label_map[e] for e in epochs.events[:, 2]],
        dtype=np.int64,
    )
    return X, y, list(epochs.ch_names)


# ─────────────────────────────────────────────────────────────────────────────
# Per-subject pipeline
# ─────────────────────────────────────────────────────────────────────────────

def process_subject(raw, sfreq_target, n_ica=20, tmin=2.0, tmax=4.0):
    """
    Pipeline (ICA before resample so Nyquist ≥ 100 Hz for ICLabel):
      1. ICA on original sfreq  (1-100 Hz, CAR, extended infomax)
      2. Resample to sfreq_target
      3. Bandpass 8-30 Hz
      4. Epoch [tmin, tmax]
    """
    # 1. ICA artifact removal at original (high) sampling rate
    raw = remove_artifacts_ica(raw, n_components=n_ica)

    # 2. Resample to target frequency
    if raw.info["sfreq"] != sfreq_target:
        raw.resample(sfreq_target, verbose=False)

    # 3. Bandpass 8-30 Hz
    raw.filter(8.0, 30.0, verbose=False)

    # 4. Epoch [tmin, tmax]
    X, y, ch_names = extract_epochs(raw, tmin=tmin, tmax=tmax)
    return X, y, ch_names


# ─────────────────────────────────────────────────────────────────────────────
# Dataset-level runner
# ─────────────────────────────────────────────────────────────────────────────

def process_session(runs, sfreq_target, n_ica, session_label=""):
    """
    Concat all runs within one session → ICA → resample → filter → epoch.
    Returns X (N, C, T), y (N,), ch_names.
    """
    runs = [r.copy() for r in runs]
    if len(runs) > 1:
        raw = mne.concatenate_raws(runs)
        print(f"    [{session_label}] Concatenated {len(runs)} runs for ICA")
    else:
        raw = runs[0]

    # EEG 채널만 유지 (EMG, Stim 등 제거)
    raw.pick("eeg", verbose=False)

    # ICA at original (high) sfreq
    raw = remove_artifacts_ica(raw, n_components=n_ica)

    # Resample → bandpass
    if raw.info["sfreq"] != sfreq_target:
        raw.resample(sfreq_target, verbose=False)
    raw.filter(8.0, 30.0, verbose=False)

    return extract_epochs(raw, tmin=0.5, tmax=2.5)


def preprocess_dataset(dataset, subjects, sfreq_target, save_path,
                       n_ica=20, per_session_ica=False):
    """
    Loop over subjects, apply ICA, epoch, save.

    per_session_ica=False  →  concat ALL sessions per subject, fit ICA once
                               (Cho2017: 1 session → same result)
    per_session_ica=True   →  fit ICA independently per session
                               (Lee2019: session1 / session2 separately)
    """
    all_X, all_y, all_subs = [], [], []
    ch_names_out = None

    for subj in subjects:
        print(f"  Subject {subj:02d} ...", flush=True)
        try:
            raw_dict = dataset.get_data(subjects=[subj])
        except Exception as e:
            print(f"    [ERROR] load failed: {e}")
            continue

        # raw_dict: {subj: {session_name: {run_name: Raw}}}
        sessions = raw_dict[subj]
        subj_X, subj_y = [], []

        try:
            if per_session_ica:
                # ── Session-wise ICA (Lee2019) ─────────────────────────────
                for sess_name, sess_data in sessions.items():
                    runs = list(sess_data.values())
                    X, y, ch_names = process_session(
                        runs, sfreq_target, n_ica, session_label=sess_name
                    )
                    subj_X.append(X)
                    subj_y.append(y)
                    if ch_names_out is None:
                        ch_names_out = ch_names
            else:
                # ── Subject-wise ICA (Cho2017) ─────────────────────────────
                all_runs = [r for sd in sessions.values() for r in sd.values()]
                X, y, ch_names = process_session(
                    all_runs, sfreq_target, n_ica, session_label="all"
                )
                subj_X.append(X)
                subj_y.append(y)
                if ch_names_out is None:
                    ch_names_out = ch_names

        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"    [ERROR] processing failed: {e}")

        if not subj_X or all(x is None for x in subj_X):
            print(f"    [SKIP] no valid epochs")
            continue

        X_s = np.concatenate([x for x in subj_X if x is not None], axis=0)
        y_s = np.concatenate([y for y in subj_y if y is not None], axis=0)
        all_X.append(X_s)
        all_y.append(y_s)
        all_subs.append(np.full(len(y_s), subj, dtype=np.int64))
        print(f"    → {len(y_s)} trials  "
              f"(left={np.sum(y_s==0)}, right={np.sum(y_s==1)})")

    X_all   = np.concatenate(all_X,   axis=0)
    y_all   = np.concatenate(all_y,   axis=0)
    subs_all = np.concatenate(all_subs, axis=0)

    np.savez_compressed(
        save_path,
        X=X_all,
        y=y_all,
        subjects=subs_all,
        sfreq=np.float32(sfreq_target),
        ch_names=np.array(ch_names_out, dtype=object),
    )
    print(f"\n  Saved → {save_path}")
    print(f"  Shape : X={X_all.shape}, y={y_all.shape}, "
          f"subjects={np.unique(subs_all).shape[0]} subjs\n")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["cho2017", "lee2019", "both"],
                        default="both")
    args = parser.parse_args()

    # ── Cho2017 ───────────────────────────────────────────────────────────────
    cho_save = os.path.join(SAVE_DIR, "cho2017.npz")
    if args.dataset not in ("cho2017", "both"):
        pass
    elif os.path.exists(cho_save):
        print(f"[Cho2017] already exists at {cho_save}, skipping.\n")
    else:
        print("=" * 60)
        print("[Cho2017]  52 subjects | 512 Hz → 128 Hz | 8-30 Hz | [0.5-2.5] s")
        print("           ICA: subject-wise concat")
        print("=" * 60)
        cho_dataset = Cho2017()
        preprocess_dataset(
            dataset=cho_dataset,
            subjects=list(range(1, 53)),
            sfreq_target=128,
            save_path=cho_save,
            n_ica=20,
            per_session_ica=False,   # 1 session → subject-wise = session-wise
        )

    # ── Lee2019_MI ────────────────────────────────────────────────────────────
    lee_save = os.path.join(SAVE_DIR, "lee2019.npz")
    if args.dataset not in ("lee2019", "both"):
        pass
    elif os.path.exists(lee_save):
        print(f"[Lee2019_MI] already exists at {lee_save}, skipping.\n")
    else:
        print("=" * 60)
        print("[Lee2019_MI]  54 subjects | 1000 Hz → 100 Hz | 8-30 Hz | [0.5-2.5] s")
        print("             ICA: session-wise (session1 / session2 separately)")
        print("=" * 60)
        lee_dataset = Lee2019_MI()
        preprocess_dataset(
            dataset=lee_dataset,
            subjects=list(range(1, 55)),
            sfreq_target=100,
            save_path=lee_save,
            n_ica=20,
            per_session_ica=True,    # session1, session2 each get their own ICA
        )

    print("All preprocessing done.")
