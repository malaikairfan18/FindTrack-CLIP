# Guide: Running Updated FindTrack on Kaggle

This guide walks you through running the upgraded 3-stage FindTrack pipeline (with customizable CLIP reranker modes: `mask_crop`, `object_box_crop`, `full_frame`) on Kaggle using T4 GPUs.

---

## 1. Preparation & Setup

### Step 1: Uploading Code
1. Download `FindTrack_Updated.zip` from your local workspace scratch directory (`C:\Users\Giga TECH\.gemini\antigravity\scratch\FindTrack_Updated.zip`).
2. Open a new Kaggle Notebook.
3. Click on the **+ Add Input** button in the top right panel.
4. Upload `FindTrack_Updated.zip` as a Kaggle Dataset (e.g. name it `findtrack-updated`).

### Step 2: Extracting Code
In the first cell of your Kaggle notebook, extract the zip archive:
```bash
!unzip /kaggle/input/findtrack-updated/FindTrack_Updated.zip -d /kaggle/working/FindTrack
```

### Step 3: Run Automation Setup
Navigate into the directory and run the setup script to install all dependencies, download the Alpha-CLIP weights, and automatically link/combine your datasets:
```python
%cd /kaggle/working/FindTrack
!python kaggle_setup.py
```

---

## 2. Running Experiments (for your Thesis/FYP)

You can run and evaluate the pipeline with different configurations using the command-line arguments. The setup script combines the split dataset inputs into `/kaggle/working/MeViS_combined` and `/kaggle/working/YTVOS_combined`.

### Experiment 1: Baseline FindTrack (Full Frame / Attention Masked)
This replicates the original FindTrack paper baseline where similarity is computed over the full frame with the mask as the alpha/attention channel:
```bash
!python run_mevis.py --mode full_frame --w_finder 0.5 --w_clip 0.5 --dataset_path /kaggle/working/MeViS_combined
```

### Experiment 2: CLIP Reranking with Object Box Crops
This crops the bounding box containing the object (with a 10% context padding), resizes it, and sends it to CLIP:
```bash
!python run_mevis.py --mode object_box_crop --w_finder 0.5 --w_clip 0.5 --dataset_path /kaggle/working/MeViS_combined
```

### Experiment 3: CLIP Reranking with Mask Crops (Best Mode)
This crops the bounding box containing the object, sets all background pixels outside the mask to black, resizes, and runs CLIP (strongly focuses CLIP's attention on the exact segmented target):
```bash
!python run_mevis.py --mode mask_crop --w_finder 0.5 --w_clip 0.5 --dataset_path /kaggle/working/MeViS_combined
```

### Experiment 4: Optimizing Score Fusion Weights
Evaluate how varying the fusion balance affects performance.
- **Finder-heavy** (Focus on segmentation/mask quality):
  ```bash
  !python run_mevis.py --mode mask_crop --w_finder 0.7 --w_clip 0.3 --dataset_path /kaggle/working/MeViS_combined
  ```
- **CLIP-heavy** (Focus on language-visual query alignment):
  ```bash
  !python run_mevis.py --mode mask_crop --w_finder 0.3 --w_clip 0.7 --dataset_path /kaggle/working/MeViS_combined
  ```

---

## 3. Dataset Path Configuration
The dataset paths are automatically linked and combined by `kaggle_setup.py`:
- **MeViS Combined Dataset**: `/kaggle/working/MeViS_combined`
- **Ref-YouTube-VOS Combined Dataset**: `/kaggle/working/YTVOS_combined`

To run Ref-YouTube-VOS evaluations:
```bash
!python run_ytvos.py --mode mask_crop --w_finder 0.5 --w_clip 0.5 --dataset_path /kaggle/working/YTVOS_combined
```

