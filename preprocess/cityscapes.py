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


def resize_with_center_crop(input_path, output_path, target_size=256, resample_method=Image.BILINEAR):
    image = Image.open(input_path)
    original_size = image.size
    
    if original_size == (2048, 1024):
        resized = image.resize((512, 256), resample=resample_method)
    elif original_size == (512, 256):
        resized = image
    else:
        scale_factor = 256 / original_size[1]
        new_width = int(original_size[0] * scale_factor)
        resized = image.resize((new_width, 256), resample=resample_method)
    
    if resized.size[0] >= target_size:
        left = (resized.size[0] - target_size) // 2
        top = 0
        cropped = resized.crop((left, top, left + target_size, top + target_size))
    else:
        cropped = resized
    
    cropped.save(output_path)


def process_city_folder(input_folder, output_folder, resample_method, size=(256, 256)):
    os.makedirs(output_folder, exist_ok=True)
    
    for city_folder in os.listdir(input_folder):
        city_input_path = os.path.join(input_folder, city_folder)
        if not os.path.isdir(city_input_path):
            continue
        
        city_output_path = os.path.join(output_folder, city_folder)
        os.makedirs(city_output_path, exist_ok=True)
        
        image_paths = glob(os.path.join(city_input_path, "*.png"))
        for image_path in tqdm(image_paths, desc=f"{city_folder}"):
            filename = os.path.basename(image_path)
            output_path = os.path.join(city_output_path, filename)
            resize_image(image_path, output_path, size, resample_method)


def process_city_images(input_folder, output_folder, use_center_crop=False, resample_method=None, size=(256, 256)):
    os.makedirs(output_folder, exist_ok=True)
    
    for city_folder in os.listdir(input_folder):
        city_input_path = os.path.join(input_folder, city_folder)
        if not os.path.isdir(city_input_path):
            continue
        
        city_output_path = os.path.join(output_folder, city_folder)
        os.makedirs(city_output_path, exist_ok=True)
        
        image_paths = glob(os.path.join(city_input_path, "*.png"))
        for image_path in tqdm(image_paths, desc=f"{city_folder}"):
            filename = os.path.basename(image_path)
            output_path = os.path.join(city_output_path, filename)
            if use_center_crop:
                resample = resample_method if resample_method else Image.BILINEAR
                resize_with_center_crop(image_path, output_path, target_size=size[0], resample_method=resample)
            else:
                resize_image(image_path, output_path, size, resample_method)


def convert_labels_to_npy(input_dir, output_dir, old_ignore_label=19, new_ignore_label=255):
    os.makedirs(output_dir, exist_ok=True)
    
    total_count = 0
    for city_folder in os.listdir(input_dir):
        city_input_path = os.path.join(input_dir, city_folder)
        if not os.path.isdir(city_input_path):
            continue
        
        city_output_path = os.path.join(output_dir, city_folder)
        os.makedirs(city_output_path, exist_ok=True)
        
        label_paths = glob(os.path.join(city_input_path, "*_labelIds.png"))
        for label_path in tqdm(label_paths, desc=f"Converting {city_folder}"):
            label_map = np.array(Image.open(label_path), dtype=np.uint8)
            label_map[label_map == old_ignore_label] = new_ignore_label
            filename = os.path.basename(label_path).replace(".png", ".npy")
            output_path = os.path.join(city_output_path, filename)
            np.save(output_path, label_map)
            total_count += 1
    
    return total_count


def preprocess_cityscapes(input_dir, output_dir):
    splits = ["train", "val"]
    
    for split in splits:
        img_input = os.path.join(input_dir, "leftImg8bit", split)
        img_output = os.path.join(output_dir, "leftImg8bit", split)
        if os.path.exists(img_input):
            process_city_images(img_input, img_output, use_center_crop=True, resample_method=Image.BILINEAR)
        
        label_input = os.path.join(input_dir, "gtFine", split)
        label_output = os.path.join(output_dir, "gtFine", split)
        if os.path.exists(label_input):
            process_city_images(label_input, label_output, use_center_crop=True, resample_method=Image.NEAREST)
    
    for split in splits:
        label_input = os.path.join(output_dir, "gtFine", split)
        label_output = os.path.join(output_dir, "gtFine_npy", split)
        if os.path.exists(label_input):
            convert_labels_to_npy(label_input, label_output)


def organize_final_structure(preprocessed_dir, output_base_dir, cleanup_intermediate=False):
    splits = ["train", "val"]
    
    for split in splits:
        img_source = os.path.join(preprocessed_dir, "leftImg8bit", split)
        label_source = os.path.join(preprocessed_dir, "gtFine_npy", split)
        output_dir = os.path.join(output_base_dir, "real", split)
        os.makedirs(output_dir, exist_ok=True)
        
        if os.path.exists(img_source):
            img_files = glob(os.path.join(img_source, "**/*.png"), recursive=True)
            for img_file in tqdm(img_files, desc=f"Copying {split} images"):
                filename = os.path.basename(img_file)
                dest_path = os.path.join(output_dir, filename)
                shutil.copy2(img_file, dest_path)
        
        if os.path.exists(label_source):
            npy_files = glob(os.path.join(label_source, "**/*.npy"), recursive=True)
            for npy_file in tqdm(npy_files, desc=f"Copying {split} labels"):
                filename = os.path.basename(npy_file)
                dest_path = os.path.join(output_dir, filename)
                shutil.copy2(npy_file, dest_path)
    
    if cleanup_intermediate:
        intermediate_folders = []
        for split in splits:
            intermediate_folders.extend([
                os.path.join(preprocessed_dir, "leftImg8bit", split),
                os.path.join(preprocessed_dir, "gtFine", split),
                os.path.join(preprocessed_dir, "gtFine_npy", split)
            ])
        
        for folder in intermediate_folders:
            if os.path.exists(folder):
                shutil.rmtree(folder)


def main():
    parser = argparse.ArgumentParser(description="Preprocess Cityscapes dataset from original to final structure")
    parser.add_argument(
        "--input_dir",
        type=str,
        default="/path/to/Cityscapes",
        help="Path to the original Cityscapes dataset directory"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="datasets/Cityscapes_256",
        help="Base path for final organized structure (files will be saved to {output_dir}/real/train and {output_dir}/real/val)"
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
        
        temp_dir = tempfile.mkdtemp(prefix="cityscapes_preprocess_")
        try:
            preprocess_cityscapes(args.input_dir, temp_dir)
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

