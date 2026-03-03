# Google Colab Migration & Execution Plan

This guide outlines the steps to successfully migrate the DynaDetect-WM project to Google Colab and run the full 50-epoch experiments. Moving to Colab will give you access to high-end GPUs (like the A100 or V100) which are highly recommended for the larger datasets (VGGFace2 and CheXpert).

## Phase 1: Preparation & Setup

### 1. Push to GitHub
If you haven't already, push this local repository to a Git hosting provider (GitHub, GitLab, etc.).
```bash
git remote add origin <YOUR_GITHUB_REPO_URL>
git push -u origin main
```
*(Alternatively, you can zip the `dd2-wm` folder—excluding the `data/` and `venv/` directories—and upload it directly to Google Drive).*

### 2. Dataset Management (Crucial for Colab)
Colab instances reset after your session ends. Moving large datasets repeatedly over the network can be slow.
*   **CIFAR-100 & GTSRB:** These will download automatically via `torchvision` scripts. You do not need to upload them.
*   **VGGFace2 & CheXpert:** Since these are massive (2.3GB and 11GB) and require credentials:
    1.  Zip your local `VGGFace2` and `CheXpert` folders.
    2.  Upload the `.zip` files to your Google Drive. 
    3.  *Note: We will extract them directly into Colab's local high-speed disk during the run to prevent slow Google Drive I/O bottlenecks during training.*

---

## Phase 2: Google Colab Notebook Setup

Create a new Google Colab notebook and follow these steps in separate cells.

### Step 1: Enable GPU
In the Colab menu, go to **Runtime > Change runtime type** and select **GPU** (T4, V100, or A100 if available).

### Step 2: Mount Google Drive
Mount your Drive so you can save checkpoints and results permanently, and access your zipped datasets.
```python
from google.colab import drive
drive.mount('/content/drive')
```

### Step 3: Get the Code
Clone the repository or copy the unzipped code folder from your Drive.
```python
# If using GitHub:
!git clone <YOUR_GITHUB_REPO_URL>
%cd dd2-wm

# OR if you uploaded a zip to Drive:
# !cp /content/drive/MyDrive/dd2-wm.zip /content/
# !unzip -q /content/dd2-wm.zip
# %cd dd2-wm
```

### Step 4: Install Dependencies
```python
!pip install -r requirements.txt
```

### Step 5: Transfer and Extract Large Datasets
Copy the dataset zip files from your Drive to the Colab local disk, then extract them into the `data/` folder. This ensures high-speed read access during training.
```python
# Make data directory
!mkdir -p data

# Example for CheXpert
!cp /content/drive/MyDrive/CheXpert.zip /content/dd2-wm/data/
!unzip -q /content/dd2-wm/data/CheXpert.zip -d /content/dd2-wm/data/

# Example for VGGFace2
!cp /content/drive/MyDrive/VGGFace2.zip /content/dd2-wm/data/
!unzip -q /content/dd2-wm/data/VGGFace2.zip -d /content/dd2-wm/data/
```

---

## Phase 3: Running the Full-Scale Experiments

Now you are ready to execute the full pipeline. You can run these commands directly in Colab cells.

### Experiment 1: Baseline Clean Training (50 Epochs)
```python
# CIFAR-100
!python train.py --dataset cifar100 --epochs 50 --batch-size 128

# GTSRB
!python train.py --dataset gtsrb --epochs 50 --batch-size 128

# VGGFace2
!python train.py --dataset vggface --epochs 50 --batch-size 64

# CheXpert
!python train.py --dataset chexpert --epochs 50 --batch-size 64
```

### Experiment 2: Generate Poisons & Detect (Phase 2)
```python
!python run_phase2.py --dataset cifar100 --num-poisons 500 --model-path checkpoints/best_cifar100_resnet18.pth
# Repeat for gtsrb, vggface, chexpert
```

### Experiment 3: Generate Reference Signatures (Phase 3)
```python
!python run_phase3.py --dataset cifar100 --num-poisons 500 --model-path checkpoints/best_cifar100_resnet18.pth
# Repeat for gtsrb, vggface, chexpert
```

### Experiment 4: Simulate Unauthorized Training & Trace (Phases 4 & 5)
```python
!python run_phase4.py --dataset cifar100 --num-poisons 500 --epochs 50 --auth-model-path checkpoints/best_cifar100_resnet18.pth
# Repeat for gtsrb, vggface, chexpert
```

---

## Phase 4: Backup Results
Once an experiment finishes, make sure to copy your `checkpoints/` and `results/` folders back to Google Drive so you don't lose them when the Colab session disconnects!
```python
!cp -r /content/dd2-wm/checkpoints/ /content/drive/MyDrive/dd2-wm_backup/
!cp -r /content/dd2-wm/results/ /content/drive/MyDrive/dd2-wm_backup/
```
