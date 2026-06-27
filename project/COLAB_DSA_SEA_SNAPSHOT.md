# Colab Run: DSA+SEA+Snapshot Cross-Dataset

This is the priority run for filling the missing cross-dataset Snapshot cells:

- CSPNet + DatasetEA + SubjectEA + Snapshot
- EEGNet + DatasetEA + SubjectEA + Snapshot
- Conformer + DatasetEA + SubjectEA + Snapshot

The runner is resumable. It skips a direction if the expected CSV already exists with complete `snap_acc` rows.

## 1. Setup

Mount Google Drive and keep the repository plus preprocessed data on Drive so outputs survive runtime resets.

```python
from google.colab import drive
drive.mount('/content/drive')
```

```bash
cd /content/drive/MyDrive
git clone https://github.com/heegyukim4043/MI_loso_project.git
cd MI_loso_project/project/crossdata/models
```

If the repo already exists:

```bash
cd /content/drive/MyDrive/MI_loso_project
git pull
cd project/crossdata/models
```

Install only missing packages. A typical Colab runtime already has PyTorch, NumPy, pandas, and scikit-learn.

```bash
pip install moabb mne pyriemann einops
```

## 2. Data Path

`cross_dataset.py` expects:

```text
cho2017.npz
lee2019.npz
```

under `MI_PREPROCESSED_DIR`.

Recommended Drive layout:

```text
/content/drive/MyDrive/MI_loso_project/project/crossdata/preprocessed_sfreq100/cho2017.npz
/content/drive/MyDrive/MI_loso_project/project/crossdata/preprocessed_sfreq100/lee2019.npz
```

Then set:

```bash
export MI_PREPROCESSED_DIR=/content/drive/MyDrive/MI_loso_project/project/crossdata/preprocessed_sfreq100
export MI_N_TIMES=201
```

## 3. Priority Runs

### Priority 1: CSPNet

Run this first. It is the most important missing cell.

```bash
python manage_colab_dsa_sea_snapshot_20260628.py \
  --models cspnet \
  --direction both \
  --preprocessed_dir "$MI_PREPROCESSED_DIR"
```

If the runtime is unstable, split by direction:

```bash
python manage_colab_dsa_sea_snapshot_20260628.py \
  --models cspnet \
  --direction cho2lee \
  --preprocessed_dir "$MI_PREPROCESSED_DIR"

python manage_colab_dsa_sea_snapshot_20260628.py \
  --models cspnet \
  --direction lee2cho \
  --preprocessed_dir "$MI_PREPROCESSED_DIR"
```

### Priority 2: EEGNet

```bash
python manage_colab_dsa_sea_snapshot_20260628.py \
  --models eegnet \
  --direction both \
  --preprocessed_dir "$MI_PREPROCESSED_DIR"
```

### Priority 3: Conformer

Run only after CSPNet and EEGNet are complete, or when Colab assigns an A100/L4.

```bash
python manage_colab_dsa_sea_snapshot_20260628.py \
  --models conformer \
  --direction both \
  --preprocessed_dir "$MI_PREPROCESSED_DIR"
```

## 4. One-Shot All Models

Use this only on a stable Pro+ runtime:

```bash
python manage_colab_dsa_sea_snapshot_20260628.py \
  --models cspnet eegnet conformer \
  --direction both \
  --preprocessed_dir "$MI_PREPROCESSED_DIR"
```

## 5. Outputs

Raw CSVs are written under:

```text
project/crossdata/results/
```

Logs:

```text
project/crossdata/results/runs/
```

Live summary:

```text
project/crossdata/colab_dsa_sea_snapshot_20260628.md
```

Expected CSV names:

```text
loso_results_20260628_colab_dsa_sea_snapshot_cspnet_cross_cho2017_to_lee2019_cspnet.csv
loso_results_20260628_colab_dsa_sea_snapshot_cspnet_cross_lee2019_to_cho2017_cspnet.csv
loso_results_20260628_colab_dsa_sea_snapshot_eegnet_cross_cho2017_to_lee2019_eegnet.csv
loso_results_20260628_colab_dsa_sea_snapshot_eegnet_cross_lee2019_to_cho2017_eegnet.csv
loso_results_20260628_colab_dsa_sea_snapshot_conformer_cross_cho2017_to_lee2019_conformer.csv
loso_results_20260628_colab_dsa_sea_snapshot_conformer_cross_lee2019_to_cho2017_conformer.csv
```

## 6. Push Results Back

After runs finish:

```bash
cd /content/drive/MyDrive/MI_loso_project
git status
git add project/crossdata/results project/crossdata/colab_dsa_sea_snapshot_20260628.md
git commit -m "Add DSA SEA Snapshot cross-dataset results"
git push origin master
```

If Git identity is missing:

```bash
git config user.name "heegyukim4043"
git config user.email "55726335+heegyukim4043@users.noreply.github.com"
```
