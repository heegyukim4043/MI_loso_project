import os
import mne
import moabb

# Set download directory to current folder
mne.set_config('MNE_DATA', r'g:\MI_opendata')
moabb.set_download_dir(r'g:\MI_opendata')

from moabb.datasets import Cho2017

print("Downloading Cho2017 dataset to g:\\MI_opendata ...")
print("Subjects: 52, Task: Left/Right hand motor imagery, Channels: 64 EEG @ 512Hz")
print("-" * 60)

dataset = Cho2017()

# Download all 52 subjects
subjects = list(range(1, 53))
data = dataset.get_data(subjects=subjects)

print("-" * 60)
print(f"Download complete. {len(data)} subjects loaded.")
print(f"Files saved to: g:\\MI_opendata")
