"""
ERD (Event-Related Desynchronization) analysis per subject.

Loads raw MOABB data with a wider epoch window (-1 to 4 s) so that
the pre-stimulus baseline (-1 to 0 s) is available, then applies
time-frequency analysis (multitaper) in mu (8-13 Hz) and beta (13-30 Hz).

Usage:
    python analyze_erd.py --dataset cho2017
    python analyze_erd.py --dataset lee2019
    python analyze_erd.py --dataset both --subjects 1 2 3   # subset only
    python analyze_erd.py --dataset cho2017 --plot           # save ERD plots

Output (saved to results/):
    erd_raw_<dataset>_<ts>.csv      -- trial-level rows
    erd_summary_<dataset>_<ts>.csv  -- subject-level summary
"""

import os
import sys
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import moabb
from mne.time_frequency import tfr_multitaper
from datetime import datetime

warnings.filterwarnings("ignore")
mne.set_log_level("WARNING")
moabb.set_download_dir(r"g:\MI_opendata")
mne.set_config("MNE_DATA", r"g:\MI_opendata")

RESULTS_DIR = r"g:\MI_opendata\results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ERD parameters
BASELINE   = (-1.0, 0.0)   # pre-stimulus reference
ACTIVE     = (0.5, 2.5)    # MI active window
TMIN, TMAX = -1.0, 4.0    # epoch window (must cover baseline + active)
FREQ_BANDS = {"mu": (8, 13), "beta": (13, 30)}
PICKS_CHO  = ["C3", "Cz", "C4", "CP3", "CP4"]
PICKS_LEE  = ["C3", "Cz", "C4", "CP3", "CP4"]   # same names in Lee2019


# ---------------------------------------------------------------------------
# ERD computation  (your original function, slightly adapted)
# ---------------------------------------------------------------------------

def compute_subject_erd(
    epochs,
    picks,
    baseline=BASELINE,
    active=ACTIVE,
    freq_bands=FREQ_BANDS,
):
    """
    Compute ERD strength per condition / band / channel for one subject.

    Returns a DataFrame with columns:
        condition, band, channel, erd_percent_mean, erd_strength
    """
    avail = [p for p in picks if p in epochs.ch_names]
    if not avail:
        print(f"    [WARN] none of {picks} found; skipping")
        return pd.DataFrame()

    ep = epochs.copy().pick(avail)
    rows = []

    # Split by class label (left / right)
    event_names = list(epochs.event_id.keys())
    subsets = {name: ep[name] for name in event_names if name in ep.event_id}
    if not subsets:
        subsets = {"all": ep}

    for cond, sub_ep in subsets.items():
        if len(sub_ep) == 0:
            continue

        for band, (fmin, fmax) in freq_bands.items():
            freqs   = np.arange(fmin, fmax + 1, dtype=float)
            n_cyc   = np.full(len(freqs), 4 if band == "mu" else 6)

            power = tfr_multitaper(
                sub_ep,
                freqs=freqs,
                n_cycles=n_cyc,
                use_fft=True,
                return_itc=False,
                average=False,
                decim=2,
                picks=avail,
                verbose=False,
            )
            power.apply_baseline(baseline=baseline, mode="percent")

            times       = power.times
            active_mask = (times >= active[0]) & (times <= active[1])

            # mean over [trials, freqs, active-time]  -> (n_channels,)
            band_mean = power.data[:, :, :, active_mask].mean(axis=(0, 2, 3))

            for ch, raw_val in zip(avail, band_mean):
                rows.append(dict(
                    condition       = cond,
                    band            = band,
                    channel         = ch,
                    erd_percent_mean= round(float(raw_val * 100), 3),
                    erd_strength    = round(float(-raw_val * 100), 3),
                ))

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Load one subject from MOABB and build MNE Epochs with wide window
# ---------------------------------------------------------------------------

def load_subject_epochs(dataset, subj: int, sfreq_target: int):
    """
    Load raw data for one subject, resample, bandpass 1-40 Hz,
    and epoch [-1, 4] s around each MI cue.

    Returns MNE Epochs or None on failure.
    """
    try:
        raw_dict = dataset.get_data(subjects=[subj])
    except Exception as e:
        print(f"    [ERROR] load failed: {e}")
        return None

    sessions = raw_dict[subj]
    all_epochs = []

    for sess_name, sess_data in sessions.items():
        runs = list(sess_data.values())
        raws = [r.copy() for r in runs]
        raw  = mne.concatenate_raws(raws) if len(raws) > 1 else raws[0]

        # Keep only EEG channels
        raw.pick("eeg", verbose=False)

        # Resample + light bandpass (for TFR we keep broader range)
        if raw.info["sfreq"] != sfreq_target:
            raw.resample(sfreq_target, verbose=False)
        raw.filter(1.0, 40.0, verbose=False)

        # Epoch around events
        events, event_id = mne.events_from_annotations(raw, verbose=False)

        # Build left/right label map
        label_map = {}
        for name, code in event_id.items():
            n = name.lower()
            if "left"  in n or n in ("1", "769"): label_map[code] = "left_hand"
            elif "right" in n or n in ("2", "770"): label_map[code] = "right_hand"

        if len(label_map) < 2:
            codes = sorted(set(events[:, 2]))[:2]
            label_map = {codes[0]: "left_hand", codes[1]: "right_hand"}

        sel_event_id = {v: k for k, v in label_map.items()}   # name -> code

        try:
            ep = mne.Epochs(
                raw, events,
                event_id={v: k for k, v in label_map.items()},
                tmin=TMIN, tmax=TMAX,
                baseline=None, preload=True, verbose=False,
            )
            all_epochs.append(ep)
        except Exception as e:
            print(f"    [WARN] epoch failed ({sess_name}): {e}")

    if not all_epochs:
        return None

    return mne.concatenate_epochs(all_epochs, verbose=False)


# ---------------------------------------------------------------------------
# Per-dataset runner
# ---------------------------------------------------------------------------

def run_erd_analysis(
    dataset_name: str,
    subjects=None,
    do_plot: bool = False,
):
    from moabb.datasets import Cho2017, Lee2019_MI

    if dataset_name == "cho2017":
        dataset      = Cho2017()
        sfreq_target = 128
        picks        = PICKS_CHO
        all_subjects = list(range(1, 53))
    else:
        dataset      = Lee2019_MI()
        sfreq_target = 100
        picks        = PICKS_LEE
        all_subjects = list(range(1, 55))

    if subjects:
        all_subjects = [s for s in all_subjects if s in subjects]

    print(f"\n{'='*60}")
    print(f" ERD Analysis -- {dataset_name.upper()}  ({len(all_subjects)} subjects)")
    print(f" Channels : {picks}")
    print(f" Baseline : {BASELINE} s    Active : {ACTIVE} s")
    print(f"{'='*60}")

    all_rows = []
    summary_rows = []

    for subj in all_subjects:
        print(f"  Subject {subj:02d} ...", end=" ", flush=True)
        epochs = load_subject_epochs(dataset, subj, sfreq_target)
        if epochs is None:
            print("SKIP")
            continue

        df = compute_subject_erd(epochs, picks=picks)
        if df.empty:
            print("SKIP (no channels)")
            continue

        df["subject"] = subj
        df["dataset"] = dataset_name
        all_rows.append(df)

        # Per-band summary for this subject (mean over conditions & channels)
        for band in FREQ_BANDS:
            mu_df = df[df["band"] == band]
            summary_rows.append(dict(
                dataset   = dataset_name,
                subject   = subj,
                band      = band,
                erd_mean  = round(mu_df["erd_strength"].mean(), 3),
                erd_std   = round(mu_df["erd_strength"].std(),  3),
                erd_c3    = round(mu_df[mu_df["channel"] == "C3"]["erd_strength"].mean(), 3)
                            if "C3" in mu_df["channel"].values else np.nan,
                erd_cz    = round(mu_df[mu_df["channel"] == "Cz"]["erd_strength"].mean(), 3)
                            if "Cz" in mu_df["channel"].values else np.nan,
                erd_c4    = round(mu_df[mu_df["channel"] == "C4"]["erd_strength"].mean(), 3)
                            if "C4" in mu_df["channel"].values else np.nan,
            ))

        mu_str   = df[df["band"] == "mu"  ]["erd_strength"].mean()
        beta_str = df[df["band"] == "beta"]["erd_strength"].mean()
        print(f"mu={mu_str:+.1f}%  beta={beta_str:+.1f}%")

        # Optional: save TFR plot per subject
        if do_plot:
            _plot_erd(df, subj, dataset_name)

    if not all_rows:
        print("No results.")
        return

    all_df    = pd.concat(all_rows, ignore_index=True)
    summ_df   = pd.DataFrame(summary_rows)
    ts        = datetime.now().strftime("%Y%m%d_%H%M%S")

    raw_path  = os.path.join(RESULTS_DIR, f"erd_raw_{dataset_name}_{ts}.csv")
    summ_path = os.path.join(RESULTS_DIR, f"erd_summary_{dataset_name}_{ts}.csv")
    all_df.to_csv(raw_path,  index=False)
    summ_df.to_csv(summ_path, index=False)
    print(f"\nSaved: {raw_path}")
    print(f"Saved: {summ_path}")

    # Print top / bottom 5 subjects by mu ERD
    mu_summ = summ_df[summ_df["band"] == "mu"].sort_values("erd_mean", ascending=False)
    print(f"\n  Top-5  mu ERD (strong ERD):")
    print(mu_summ.head(5)[["subject", "erd_mean", "erd_c3", "erd_cz", "erd_c4"]].to_string(index=False))
    print(f"\n  Bot-5  mu ERD (weak ERD):")
    print(mu_summ.tail(5)[["subject", "erd_mean", "erd_c3", "erd_cz", "erd_c4"]].to_string(index=False))


def _plot_erd(df: pd.DataFrame, subj: int, dataset_name: str):
    """Bar chart of ERD strength by channel and band."""
    plot_dir = os.path.join(RESULTS_DIR, f"erd_plots_{dataset_name}")
    os.makedirs(plot_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, band in zip(axes, ["mu", "beta"]):
        sub = df[df["band"] == band]
        pivot = sub.groupby(["channel", "condition"])["erd_strength"].mean().unstack()
        pivot.plot(kind="bar", ax=ax, colormap="coolwarm")
        ax.set_title(f"S{subj:02d} {band} ERD strength (%)")
        ax.set_xlabel(""); ax.set_ylabel("ERD strength (+= ERD, -= ERS)")
        ax.axhline(0, color="k", linewidth=0.8)
        ax.legend(title="condition", fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"S{subj:02d}_erd.png"), dpi=100)
    plt.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",  choices=["cho2017", "lee2019", "both"],
                        default="cho2017")
    parser.add_argument("--subjects", type=int, nargs="*", default=None,
                        help="Subset of subject IDs (default: all)")
    parser.add_argument("--plot", action="store_true",
                        help="Save per-subject ERD bar plots")
    args = parser.parse_args()

    datasets = (["cho2017", "lee2019"] if args.dataset == "both"
                else [args.dataset])

    for ds in datasets:
        run_erd_analysis(ds, subjects=args.subjects, do_plot=args.plot)
