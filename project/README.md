# MI LOSO Project

Motor imagery (MI) EEG classification with LOSO (leave-one-subject-out) evaluation on Cho2017 and Lee2019_MI (OpenBMI). The repo includes preprocessing, deep models, traditional baselines, and a full pipeline harness with resume support.

## Datasets (not included)
This repository does **not** include datasets or preprocessed arrays. Download is handled via MOABB in `preprocess_data.py`.

Excluded directories (via `.gitignore`):
- `MNE-gigadb-data/`
- `MNE-lee2019-mi-data/`
- `preprocessed/`
- `results/`

## Environment
Conda env (example):
- Python 3.10
- PyTorch 2.1.2+cu118
- MNE 1.7.0
- MOABB 1.1.0
- mne-icalabel 0.6.0

## Preprocessing

```bash
C:/Users/Bio_lab_HG/miniforge3/envs/mi_spdnet/python.exe g:/MI_opendata/preprocess_data.py --dataset cho2017
C:/Users/Bio_lab_HG/miniforge3/envs/mi_spdnet/python.exe g:/MI_opendata/preprocess_data.py --dataset lee2019
```

Key parameters (current code):
- ICA (extended infomax + ICLabel, threshold 0.90) before resampling
- Bandpass 8?30 Hz
- Epoch window [0.5, 2.5] s
- Cho2017: resample 128 Hz, subject-wise ICA
- Lee2019: resample 100 Hz, session-wise ICA

## LOSO Training

```bash
# SPDNet
C:/Users/Bio_lab_HG/miniforge3/envs/mi_spdnet/python.exe g:/MI_opendata/train_loso.py --dataset cho2017 --model spdnet

# MIN2Net
C:/Users/Bio_lab_HG/miniforge3/envs/mi_spdnet/python.exe g:/MI_opendata/train_loso.py --dataset lee2019 --model min2net

# Resume a run
C:/Users/Bio_lab_HG/miniforge3/envs/mi_spdnet/python.exe g:/MI_opendata/train_loso.py --dataset lee2019 --model spdnet --run_id 20260410_2100 --resume
```

## Traditional Baseline (MRFBCSP + LDA)

```bash
C:/Users/Bio_lab_HG/miniforge3/envs/mi_spdnet/python.exe g:/MI_opendata/mrfbcsp_loso.py --dataset lee2019
```

## Full Pipeline Harness

```bash
C:/Users/Bio_lab_HG/miniforge3/envs/mi_spdnet/python.exe g:/MI_opendata/pipeline.py \
  --datasets cho2017,lee2019 \
  --models spdnet,min2net \
  --augment_models spdnet \
  --include_mrfbcsp \
  --run_id 20260410_2100
```

Outputs are written under:
```
results/runs/<run_id>/
```
with a `run_manifest.json` log.

## Notes
- Results are appended per subject to allow safe interruption and resume.
- See `PROJECT_LOG.md` for detailed experiment logs.
