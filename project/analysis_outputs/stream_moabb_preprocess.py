"""Stream MOABB MI datasets into compact downsampled NPZ files.

This script is designed for large MOABB files. It processes one
subject/session unit at a time, saves compact epochs, and can optionally
delete the raw downloaded .mat file after a successful save.

Examples:

  # Cho2017 subject 1, keep raw cache
  .venv\\Scripts\\python analysis_outputs\\stream_moabb_preprocess.py --dataset cho --subjects 1

  # Lee2019 subject 1 session 1, keep raw cache
  .venv\\Scripts\\python analysis_outputs\\stream_moabb_preprocess.py --dataset lee --subjects 1 --sessions 1

  # Delete raw .mat after save. This removes files from the MNE cache.
  .venv\\Scripts\\python analysis_outputs\\stream_moabb_preprocess.py --dataset lee --subjects 1 --sessions 1 --delete-raw
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import mne
import numpy as np
from moabb.datasets import Cho2017, Lee2019_MI


def parse_int_list(value: str) -> list[int]:
    out: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, stop = [int(x) for x in part.split("-", 1)]
            out.extend(range(start, stop + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


def make_dataset(name: str, sessions: tuple[int, ...] | None = None):
    if name == "cho":
        return Cho2017()
    if name == "lee":
        if sessions is None:
            sessions = (1,)
        return Lee2019_MI(train_run=True, test_run=False, sessions=sessions)
    raise ValueError(f"Unknown dataset: {name}")


def raw_file_paths(dataset, subject: int) -> list[Path]:
    paths = dataset.data_path(subject)
    if isinstance(paths, (str, Path)):
        paths = [paths]
    return [Path(p) for p in paths]


def safe_delete_raw(paths: list[Path], allow_delete: bool, delete_tmp: bool) -> list[str]:
    deleted: list[str] = []
    if not allow_delete:
        return deleted

    home_mne = (Path.home() / "mne_data").resolve()
    targets: list[Path] = []
    for path in paths:
        resolved = path.resolve()
        if home_mne not in resolved.parents:
            raise RuntimeError(f"Refusing to delete outside MNE cache: {resolved}")
        if resolved.suffix.lower() != ".mat":
            raise RuntimeError(f"Refusing to delete non-.mat raw file: {resolved}")
        targets.append(resolved)
        if delete_tmp:
            targets.extend(p.resolve() for p in resolved.parent.glob("tmp*") if p.is_file())

    for target in targets:
        if target.exists():
            target.unlink()
            deleted.append(str(target))
    return deleted


def extract_epochs_from_raw(raw: mne.io.BaseRaw, event_id: dict[str, int], tmin: float, tmax: float, sfreq: float):
    eeg_picks = mne.pick_types(raw.info, eeg=True, stim=False, eog=False, emg=False, exclude=[])
    if len(eeg_picks) == 0:
        raise RuntimeError("No EEG channels found")

    stim_candidates = [ch for ch in raw.ch_names if ch.upper().startswith("STI")]
    if not stim_candidates:
        raise RuntimeError("No stim channel found")

    events = mne.find_events(raw, stim_channel=stim_candidates[0], shortest_event=0, verbose=False)
    keep = np.isin(events[:, 2], list(event_id.values()))
    events = events[keep]
    if len(events) == 0:
        raise RuntimeError("No matching MI events found")

    raw_proc = raw.copy().pick(eeg_picks)
    raw_proc.filter(8.0, 30.0, method="iir", verbose=False)
    epochs = mne.Epochs(
        raw_proc,
        events,
        event_id=event_id,
        tmin=tmin,
        tmax=tmax,
        baseline=None,
        preload=True,
        verbose=False,
    )
    epochs.resample(sfreq, npad="auto", verbose=False)
    X = epochs.get_data(copy=True).astype("float32")
    y_names = np.asarray(epochs.events[:, 2])
    inv = {v: k for k, v in event_id.items()}
    labels = np.asarray([inv[int(v)] for v in y_names])
    y = np.asarray([0 if label == "left_hand" else 1 for label in labels], dtype="int64")
    return X, y, labels, epochs.ch_names, float(epochs.info["sfreq"])


def process_unit(dataset_name: str, subject: int, session: int | None, args) -> dict:
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    if args.skip_existing and dataset_name == "lee" and session is not None:
        session_name = str(session - 1)
        existing = sorted(
            (out_root / dataset_name / f"sub-{subject:02d}").glob(
                f"{dataset_name}_sub-{subject:02d}_ses-{session_name}_run-*_sfreq{int(args.sfreq)}_mi.npz"
            )
        )
        if existing:
            return {
                "dataset": dataset_name,
                "subject": subject,
                "session_arg": session,
                "raw_paths": [],
                "written": [],
                "skipped": [str(p) for p in existing],
                "deleted": [],
                "skip_stage": "preload",
            }

    sessions = (session,) if dataset_name == "lee" and session is not None else None
    dataset = make_dataset(dataset_name, sessions)
    raw_paths = raw_file_paths(dataset, subject)
    nested = dataset._get_single_subject_data(subject)

    written: list[str] = []
    skipped: list[str] = []

    for session_name, runs in nested.items():
        for run_name, raw in runs.items():
            X, y, labels, ch_names, actual_sfreq = extract_epochs_from_raw(
                raw, dataset.event_id, args.tmin, args.tmax, args.sfreq
            )
            out_dir = out_root / dataset_name / f"sub-{subject:02d}"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_file = out_dir / f"{dataset_name}_sub-{subject:02d}_ses-{session_name}_run-{run_name}_sfreq{int(actual_sfreq)}_mi.npz"
            if args.skip_existing and out_file.exists():
                skipped.append(str(out_file))
                continue
            np.savez_compressed(
                out_file,
                X=X,
                y=y,
                labels=labels,
                ch_names=np.asarray(ch_names),
                sfreq=np.asarray([actual_sfreq], dtype="float32"),
                dataset=np.asarray([dataset_name]),
                subject=np.asarray([subject], dtype="int64"),
                session=np.asarray([session_name]),
                run=np.asarray([run_name]),
                tmin=np.asarray([args.tmin], dtype="float32"),
                tmax=np.asarray([args.tmax], dtype="float32"),
            )
            written.append(str(out_file))
            del X, y, labels, raw
            gc.collect()

    deleted = safe_delete_raw(raw_paths, args.delete_raw, args.delete_tmp)
    return {
        "dataset": dataset_name,
        "subject": subject,
        "session_arg": session,
        "raw_paths": [str(p) for p in raw_paths],
        "written": written,
        "skipped": skipped,
        "deleted": deleted,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["cho", "lee", "both"], required=True)
    parser.add_argument("--subjects", required=True, help="Comma/range list, e.g. 1,2,5-10")
    parser.add_argument("--sessions", default="1", help="Lee2019 sessions, e.g. 1,2. Ignored for Cho2017.")
    parser.add_argument("--sfreq", type=float, default=100.0)
    parser.add_argument("--tmin", type=float, default=0.5)
    parser.add_argument("--tmax", type=float, default=2.5)
    parser.add_argument("--output-dir", default="moabb_streamed_npz")
    parser.add_argument("--delete-raw", action="store_true")
    parser.add_argument("--delete-tmp", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    datasets = ["cho", "lee"] if args.dataset == "both" else [args.dataset]
    subjects = parse_int_list(args.subjects)
    sessions = parse_int_list(args.sessions)

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    manifest_path = out_root / "stream_manifest.json"
    if args.skip_existing and manifest_path.exists():
        manifest: list[dict] = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        manifest = []
    for dataset_name in datasets:
        for subject in subjects:
            if dataset_name == "lee":
                for session in sessions:
                    manifest.append(process_unit(dataset_name, subject, session, args))
            else:
                manifest.append(process_unit(dataset_name, subject, None, args))

    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"Wrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()
