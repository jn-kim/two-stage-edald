## Datasets

We validate our method on four standard semantic segmentation benchmarks: **CamVid**, **ADE-Bed**, **Cityscapes**, and **PASCAL Context**. This directory contains the preprocessing scripts for each dataset.
Please download the raw datasets using the links below and run the corresponding Python scripts.

### 1. CamVid

**Download:**
- Dataset: [SegNet-Tutorial repository](https://github.com/alexgkendall/SegNet-Tutorial)

**Usage:**
```bash
python preprocess/camvid.py \
    --input_dir /path/to/SegNet-Tutorial/CamVid \
    --output_dir datasets/CamVid_256
```

`--input_dir`: Path to the downloaded original CamVid dataset  
`--output_dir`: Path where preprocessed data will be saved (final files are saved to `{output_dir}/real/train` and `{output_dir}/real/test`)

---

### 2. ADE-Bed

We curated this dataset by selecting images from the bedroom category of the [ADE20K](https://ade20k.csail.mit.edu/) dataset.

**Download:**
- Dataset: [ADE_Bed_256.tar.gz](https://drive.google.com/file/d/1L9rOV-BgFaIHc3JFC1fjSZINtcEpkJ_8/view?usp=drive_link)

**Usage:**

```bash
tar -xzvf ADE_Bed_256.tar.gz -C datasets/
rm ADE_Bed_256.tar.gz
```

---

### 3. Cityscapes

**Download:**
- Labels: [gtFine_trainvaltest.zip (241MB)](https://www.cityscapes-dataset.com/file-handling/?packageID=1)
- Images: [leftImg8bit_trainvaltest.zip (11GB)](https://www.cityscapes-dataset.com/file-handling/?packageID=3)

Extract the downloaded zip files before running the preprocessing script.

**Usage:**

The `--input_dir` should point to the Cityscapes root directory that contains `gtFine` and `leftImg8bit` folders:
```
Cityscapes/
├── gtFine/
└── leftImg8bit/
```

```bash
python preprocess/cityscapes.py \
    --input_dir /path/to/Cityscapes \
    --output_dir datasets/Cityscapes_256
```

`--input_dir`: Path to the extracted Cityscapes dataset directory (should contain `leftImg8bit` and `gtFine` folders)  
`--output_dir`: Path where preprocessed data will be saved (final files are saved to `{output_dir}/real/train` and `{output_dir}/real/val`)

---

### 4. PASCAL Context

**Download:**
- Images: [PASCAL VOC 2012 (Kaggle)](https://www.kaggle.com/datasets/huanghanchina/pascal-voc-2012/data)
- Labels: [trainval.tar.gz](https://cs.stanford.edu/~roozbeh/pascal-context/trainval.tar.gz)

Extract the downloaded files before running the preprocessing script.

**Directory Structure:**

After extracting `pascal2012.zip`:
```
pascal2012/
└── VOC2012/
    ├── Annotations/
    ├── ImageSets/
    ├── JPEGImages/
    ├── SegmentationClass/
    ├── SegmentationObject/
    └── VOC2012/
```

After extracting `trainval.tar.gz`:
```
trainval/
├── labels.txt
└── trainval/
```

The `pascal_train.txt` and `pascal_val.txt` split files are provided in the `preprocess/` directory.

**Usage:**
```bash
python preprocess/pascal_context.py \
    --image_dir /path/to/VOC2012/JPEGImages \
    --mat_dir /path/to/trainval/trainval \
    --train_txt preprocess/pascal_train.txt \
    --val_txt preprocess/pascal_val.txt \
    --output_dir datasets/Pascal_context_256
```

`--image_dir`: Path to PASCAL VOC 2012 JPEG images directory  
`--mat_dir`: Path to extracted trainval directory (should contain .mat files)  
`--train_txt`: Path to train split file (provided in `preprocess/pascal_train.txt`)  
`--val_txt`: Path to validation split file (provided in `preprocess/pascal_val.txt`)  
`--output_dir`: Path where preprocessed data will be saved (final files are saved to `{output_dir}/real/train` and `{output_dir}/real/val`)

---

## Notes
- All scripts resize images to 256x256 size.
- Labels are converted and saved in `.npy` format.
- Ignore labels for each dataset are converted to 255.