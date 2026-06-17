import os
import mne
import moabb

# Set download directory
mne.set_config('MNE_DATA', r'g:\MI_opendata')
moabb.set_download_dir(r'g:\MI_opendata')

from moabb.datasets import Lee2019_MI

print("Downloading Lee2019_MI dataset to g:\\MI_opendata ...")
print("Subjects: 54, Task: Left/Right hand motor imagery, Channels: 62 EEG @ 1000Hz, Sessions: 2")
print("-" * 60)

dataset = Lee2019_MI()

# Download one subject at a time to avoid MemoryError
subjects = list(range(48, 55))  # subjects 1-47 already downloaded
success = []
failed = []

for s in subjects:
    try:
        print(f"Downloading subject {s}...")
        data = dataset.get_data(subjects=[s])
        success.append(s)
        del data  # free memory immediately
        print(f"  Subject {s} done.")
    except Exception as e:
        print(f"  Subject {s} FAILED: {e}")
        failed.append(s)

print("-" * 60)
print(f"Done. Success: {success}")
if failed:
    print(f"Failed: {failed}")
