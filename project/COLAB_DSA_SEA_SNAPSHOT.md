# Colab T4 Run: DSA+SEA+Snapshot Cross-Dataset

Use this when `cho2017.npz` and `lee2019.npz` are already on Google Drive.
This flow does not download MOABB data and does not run preprocessing.

Target runtime: `T4 GPU`.

## 1. Required Data

Expected Drive folder:

```text
/content/drive/MyDrive/MI_loso_project/project/crossdata/preprocessed_sfreq100/
```

Required files:

```text
cho2017.npz
lee2019.npz
```

Expected shapes:

```text
cho2017: X=(10520, 64, 201), subjects=52, sfreq=100
lee2019: X=(5400, 62, 201), subjects=54, sfreq=100
```

## 2. Colab Setup

```python
from google.colab import drive
drive.mount('/content/drive')
```

```bash
cd /content
git clone https://github.com/heegyukim4043/MI_loso_project.git
cd /content/MI_loso_project/project/crossdata/models
```

If the repo already exists in the current runtime:

```bash
cd /content/MI_loso_project
git pull
cd /content/MI_loso_project/project/crossdata/models
```

Set data path:

```python
import os
os.environ["MI_PREPROCESSED_DIR"] = "/content/drive/MyDrive/MI_loso_project/project/crossdata/preprocessed_sfreq100"
os.environ["MI_BACKUP_DIR"] = "/content/drive/MyDrive/MI_loso_project/colab_results/dsa_sea_snapshot_20260628"
os.environ["MI_N_TIMES"] = "201"
```

Verify:

```bash
ls -lh "$MI_PREPROCESSED_DIR"
```

## 3. Priority Runs

Run CSPNet first, split by direction to reduce Colab interruption risk.

```bash
python manage_colab_dsa_sea_snapshot_20260628.py \
  --models cspnet \
  --direction cho2lee \
  --preprocessed_dir "$MI_PREPROCESSED_DIR" \
  --backup_dir "$MI_BACKUP_DIR"
```

```bash
python manage_colab_dsa_sea_snapshot_20260628.py \
  --models cspnet \
  --direction lee2cho \
  --preprocessed_dir "$MI_PREPROCESSED_DIR" \
  --backup_dir "$MI_BACKUP_DIR"
```

Then EEGNet:

```bash
python manage_colab_dsa_sea_snapshot_20260628.py \
  --models eegnet \
  --direction both \
  --preprocessed_dir "$MI_PREPROCESSED_DIR" \
  --backup_dir "$MI_BACKUP_DIR"
```

Then Conformer:

```bash
python manage_colab_dsa_sea_snapshot_20260628.py \
  --models conformer \
  --direction both \
  --preprocessed_dir "$MI_PREPROCESSED_DIR" \
  --backup_dir "$MI_BACKUP_DIR"
```

Use `--force` only when rerunning after a failed or interrupted attempt.

## 4. Outputs

CSV outputs:

```text
/content/MI_loso_project/project/crossdata/results/
```

Logs:

```text
/content/MI_loso_project/project/crossdata/results/runs/
```

Live summary:

```text
/content/MI_loso_project/project/crossdata/colab_dsa_sea_snapshot_20260628.md
```

With `--backup_dir`, the runner copies the live summary and every matching result CSV to Drive after each run and at queue completion. The backup folder should contain one summary plus up to six CSV files:

```text
/content/drive/MyDrive/MI_loso_project/colab_results/dsa_sea_snapshot_20260628/
```
