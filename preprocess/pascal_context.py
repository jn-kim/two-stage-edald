import os
import argparse
import shutil
import tempfile
import glob
import numpy as np
import scipy.io as sio
from PIL import Image
from tqdm import tqdm


CLASS_IDX = {
    'aeroplane': 2, 'bicycle': 23, 'bird': 25, 'boat': 31, 'bottle': 34,
    'bus': 45, 'car': 59, 'cat': 65, 'chair': 72, 'cow': 98,
    'table': 397, 'dog': 113, 'horse': 207, 'motorbike': 258, 'person': 284,
    'pottedplant': 308, 'sheep': 347, 'sofa': 368, 'train': 416, 'tvmonitor': 427,
    'sky': 360, 'grass': 187, 'ground': 189, 'road': 324, 'building': 44,
    'tree': 420, 'water': 445, 'mountain': 259, 'wall': 440, 'floor': 158,
    'track': 415, 'keyboard': 220, 'ceiling': 68
}
ALLOWED = np.array(list(CLASS_IDX.values()), dtype=np.int16)
ORIG2NEW = {orig: new for new, orig in enumerate(ALLOWED)}


def load_label_map(mat_path):
    mat = sio.loadmat(mat_path)
    if 'LabelMap' in mat:
        return mat['LabelMap']
    for v in mat.values():
        if isinstance(v, np.ndarray):
            return v
    raise ValueError(f'No ndarray found in {mat_path}')


def process_mat_to_npy(mat_dir, npy_dir):
    os.makedirs(npy_dir, exist_ok=True)
    
    mat_paths = sorted(glob.glob(os.path.join(mat_dir, '*.mat')))
    if not mat_paths:
        return 0
    
    for mat_path in tqdm(mat_paths, desc="Processing .mat files"):
        fname = os.path.splitext(os.path.basename(mat_path))[0]
        label = load_label_map(mat_path).astype(np.int16)
        ignore = label.copy()
        ignore[~np.isin(ignore, ALLOWED)] = 255
        
        remap = ignore.copy()
        for orig, new in ORIG2NEW.items():
            remap[ignore == orig] = new
        
        npy_path = os.path.join(npy_dir, fname + '.npy')
        np.save(npy_path, remap.astype(np.uint8))
    
    return len(mat_paths)


def resize_image(input_path, output_path, size, resample_method):
    image = Image.open(input_path).convert("RGB")
    resized_image = image.resize(size, resample=resample_method)
    resized_image.save(output_path)


def resize_label_npy(npy_path, output_path, size, resample_method):
    label = np.load(npy_path).astype(np.uint8)
    label_img = Image.fromarray(label, mode='L')
    label_resized = label_img.resize(size, resample=resample_method)
    np.save(output_path, np.array(label_resized, dtype=np.uint8))


def read_split_file(txt_path):
    with open(txt_path, "r") as f:
        return set(line.strip() for line in f.readlines())


def preprocess_pascal_context(input_dir, train_txt, val_txt, output_base_dir, image_dir=None, mat_dir=None):
    if image_dir is None:
        image_dir = os.path.join(input_dir, "JPEGImages")
    if mat_dir is None:
        mat_dir = os.path.join(input_dir, "trainval")
    
    if not os.path.exists(image_dir):
        raise ValueError(f"Image directory does not exist: {image_dir}")
    if not os.path.exists(mat_dir):
        raise ValueError(f"Label directory does not exist: {mat_dir}")
    
    train_ids = read_split_file(train_txt)
    val_ids = read_split_file(val_txt)
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_npy_dir = os.path.join(tmp_dir, "npy")
        tmp_resized_dir = os.path.join(tmp_dir, "resized")
        os.makedirs(tmp_npy_dir, exist_ok=True)
        os.makedirs(tmp_resized_dir, exist_ok=True)
        
        process_mat_to_npy(mat_dir, tmp_npy_dir)
        
        image_paths = {os.path.splitext(os.path.basename(p))[0]: p
                      for p in glob.glob(os.path.join(image_dir, '*.jpg'))}
        npy_paths = {os.path.splitext(os.path.basename(p))[0]: p
                     for p in glob.glob(os.path.join(tmp_npy_dir, '*.npy'))}
        common_keys = sorted(set(image_paths) & set(npy_paths))
        
        for key in tqdm(common_keys, desc="Resizing images and labels"):
            img = Image.open(image_paths[key]).convert('RGB')
            img_resized = img.resize((256, 256), resample=Image.BILINEAR)
            img_resized.save(os.path.join(tmp_resized_dir, f'{key}.jpg'), quality=95)
            
            label = np.load(npy_paths[key]).astype(np.uint8)
            label_img = Image.fromarray(label, mode='L')
            label_resized = label_img.resize((256, 256), resample=Image.NEAREST)
            np.save(os.path.join(tmp_resized_dir, f'{key}.npy'),
                    np.array(label_resized, dtype=np.uint8))
        
        output_train_dir = os.path.join(output_base_dir, "real", "train")
        output_val_dir = os.path.join(output_base_dir, "real", "val")
        os.makedirs(output_train_dir, exist_ok=True)
        os.makedirs(output_val_dir, exist_ok=True)
        
        for key in common_keys:
            jpg_path = os.path.join(tmp_resized_dir, f'{key}.jpg')
            npy_path = os.path.join(tmp_resized_dir, f'{key}.npy')
            
            if not os.path.exists(jpg_path) or not os.path.exists(npy_path):
                continue
            
            if key in train_ids:
                shutil.copy2(jpg_path, os.path.join(output_train_dir, f'{key}.jpg'))
                shutil.copy2(npy_path, os.path.join(output_train_dir, f'{key}.npy'))
            elif key in val_ids:
                shutil.copy2(jpg_path, os.path.join(output_val_dir, f'{key}.jpg'))
                shutil.copy2(npy_path, os.path.join(output_val_dir, f'{key}.npy'))


def main():
    parser = argparse.ArgumentParser(description="Preprocess Pascal-Context dataset")
    parser.add_argument("--input_dir", type=str, default="/path/to/pascal_context",
                        help="Input directory (base directory)")
    parser.add_argument("--image_dir", type=str, default="/path/to/pascal_context/VOC2012/JPEGImages",
                        help="Directory containing JPEG images")
    parser.add_argument("--mat_dir", type=str, default="/path/to/pascal_context/trainval",
                        help="Directory containing .mat label files")
    parser.add_argument("--train_txt", type=str, default="/path/to/preprocess/pascal_train.txt",
                        help="Path to pascal_train.txt file")
    parser.add_argument("--val_txt", type=str, default="/path/to/preprocess/pascal_val.txt",
                        help="Path to pascal_val.txt file")
    parser.add_argument("--output_dir", type=str, default="datasets/Pascal_context_256",
                        help="Base path for final organized structure (files will be saved to {output_dir}/real/train and {output_dir}/real/val)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.image_dir):
        raise ValueError(f"Image directory does not exist: {args.image_dir}")
    if not os.path.exists(args.mat_dir):
        raise ValueError(f"Label directory does not exist: {args.mat_dir}")
    if not os.path.exists(args.train_txt):
        raise ValueError(f"Train split file does not exist: {args.train_txt}")
    if not os.path.exists(args.val_txt):
        raise ValueError(f"Val split file does not exist: {args.val_txt}")
    
    preprocess_pascal_context(args.input_dir, args.train_txt, args.val_txt, args.output_dir, 
                             image_dir=args.image_dir, mat_dir=args.mat_dir)
    print(f"Preprocessing completed. Output: {args.output_dir}/real")


if __name__ == "__main__":
    main()

