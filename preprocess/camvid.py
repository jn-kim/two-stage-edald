import os
import argparse
import shutil
import tempfile
import numpy as np
from glob import glob
from PIL import Image
from tqdm import tqdm


def resize_image(input_path, output_path, size, resample_method):
    image = Image.open(input_path)
    resized_image = image.resize(size, resample=resample_method)
    resized_image.save(output_path)


def process_folder(input_folder, output_folder, resample_method, size=(256, 256)):
    os.makedirs(output_folder, exist_ok=True)
    
    image_paths = glob(os.path.join(input_folder, "*.png"))
    if not image_paths:
        return 0
    
    for image_path in tqdm(image_paths, desc=f"Processing {os.path.basename(input_folder)}"):
        filename = os.path.basename(image_path)
        output_path = os.path.join(output_folder, filename)
        resize_image(image_path, output_path, size, resample_method)
    
    return len(image_paths)


def convert_labels_to_npy(input_dir, output_dir, old_ignore_label=11, new_ignore_label=255):
    os.makedirs(output_dir, exist_ok=True)
    
    label_paths = glob(os.path.join(input_dir, "*.png"))
    if not label_paths:
        return 0
    
    for label_path in tqdm(label_paths, desc=f"Converting labels in {os.path.basename(input_dir)}"):
        label_map = np.array(Image.open(label_path), dtype=np.uint8)
        label_map[label_map == old_ignore_label] = new_ignore_label
        filename = os.path.basename(label_path).replace(".png", ".npy")
        output_path = os.path.join(output_dir, filename)
        np.save(output_path, label_map)
    
    return len(label_paths)


def preprocess_camvid(input_dir, output_dir):
    image_folders = ["train", "test"]
    label_folders = ["trainannot", "testannot"]
    
    for folder in image_folders:
        input_folder = os.path.join(input_dir, folder)
        output_folder = os.path.join(output_dir, folder)
        if os.path.exists(input_folder):
            process_folder(input_folder, output_folder, Image.BILINEAR)
    
    for folder in label_folders:
        input_folder = os.path.join(input_dir, folder)
        output_folder = os.path.join(output_dir, folder)
        if os.path.exists(input_folder):
            process_folder(input_folder, output_folder, Image.NEAREST)
    
    for folder in label_folders:
        input_folder = os.path.join(output_dir, folder)
        output_folder = os.path.join(output_dir, f"{folder}_npy")
        if os.path.exists(input_folder):
            convert_labels_to_npy(input_folder, output_folder)


def organize_final_structure(preprocessed_dir, output_base_dir, cleanup_intermediate=False):
    splits = ["train", "test"]
    
    for split in splits:
        img_source = os.path.join(preprocessed_dir, split)
        label_source = os.path.join(preprocessed_dir, f"{split}annot_npy")
        output_dir = os.path.join(output_base_dir, "real", split)
        os.makedirs(output_dir, exist_ok=True)
        
        if os.path.exists(img_source):
            img_files = glob(os.path.join(img_source, "*.png"))
            for img_file in tqdm(img_files, desc=f"Copying {split} images"):
                filename = os.path.basename(img_file)
                dest_path = os.path.join(output_dir, filename)
                shutil.copy2(img_file, dest_path)
        
        if os.path.exists(label_source):
            npy_files = glob(os.path.join(label_source, "*.npy"))
            for npy_file in tqdm(npy_files, desc=f"Copying {split} labels"):
                filename = os.path.basename(npy_file)
                dest_path = os.path.join(output_dir, filename)
                shutil.copy2(npy_file, dest_path)
    
    if cleanup_intermediate:
        intermediate_folders = []
        for split in splits:
            intermediate_folders.extend([
                os.path.join(preprocessed_dir, split),
                os.path.join(preprocessed_dir, f"{split}annot"),
                os.path.join(preprocessed_dir, f"{split}annot_npy")
            ])
        
        for folder in intermediate_folders:
            if os.path.exists(folder):
                shutil.rmtree(folder)


def main():
    parser = argparse.ArgumentParser(description="Preprocess CamVid dataset from original to final structure")
    parser.add_argument(
        "--input_dir",
        type=str,
        default="/path/to/SegNet-Tutorial/CamVid",
        help="Path to the original CamVid dataset directory"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="datasets/CamVid_256",
        help="Base path for final organized structure (files will be saved to {output_dir}/real/train and {output_dir}/real/test)"
    )
    parser.add_argument(
        "--preprocessed_dir",
        type=str,
        default=None,
        help="Path to already preprocessed data (if provided, skips preprocessing and only organizes)"
    )
    parser.add_argument(
        "--keep_intermediate",
        action="store_true",
        help="Keep intermediate files in temporary directory (by default, intermediate files are deleted)"
    )
    
    args = parser.parse_args()
    
    if args.preprocessed_dir:
        if not os.path.exists(args.preprocessed_dir):
            raise ValueError(f"Preprocessed directory does not exist: {args.preprocessed_dir}")
        source_dir = args.preprocessed_dir
        cleanup = False
    else:
        if not os.path.exists(args.input_dir):
            raise ValueError(f"Input directory does not exist: {args.input_dir}")
        
        temp_dir = tempfile.mkdtemp(prefix="camvid_preprocess_")
        try:
            preprocess_camvid(args.input_dir, temp_dir)
            source_dir = temp_dir
            cleanup = not args.keep_intermediate
        except Exception as e:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            raise e
    
    organize_final_structure(source_dir, args.output_dir, cleanup_intermediate=cleanup)
    
    if args.preprocessed_dir is None and cleanup and os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)


if __name__ == "__main__":
    main()
