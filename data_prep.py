#!/usr/bin/env python
"""
Data Preparation Script for Bird Detection Project
This script extracts ZIP files and organizes data for training and testing.
"""

import os
import zipfile
import shutil
import argparse

def extract_zip_files(data_dir):
    """Extract all ZIP files in the data directory."""
    print(f"Extracting ZIP files in {data_dir}...")
    
    # Create folders for extracted data
    os.makedirs(os.path.join(data_dir, 'models-klim-extracted'), exist_ok=True)
    os.makedirs(os.path.join(data_dir, 'test-skagen-extracted'), exist_ok=True)
    
    # Extract Klim model files - handle case sensitivity
    klim_dir = None
    for dir_name in os.listdir(data_dir):
        if dir_name.lower() == 'models-klim':
            klim_dir = os.path.join(data_dir, dir_name)
            break
    
    if klim_dir and os.path.exists(klim_dir):
        print(f"Found Klim directory: {klim_dir}")
        zip_files = [f for f in os.listdir(klim_dir) if f.endswith('.zip')]
        print(f"Found {len(zip_files)} ZIP files: {zip_files}")
        
        for file in zip_files:
            zip_path = os.path.join(klim_dir, file)
            extract_path = os.path.join(data_dir, 'models-klim-extracted', os.path.splitext(file)[0])
            os.makedirs(extract_path, exist_ok=True)
            
            print(f"Extracting {zip_path} to {extract_path}...")
            try:
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    file_list = zip_ref.namelist()
                    print(f"ZIP file contains {len(file_list)} files")
                    zip_ref.extractall(extract_path)
                print(f"Extraction of {file} complete")
                
                # List contents to verify extraction
                extracted_files = os.listdir(extract_path)
                print(f"Extracted {len(extracted_files)} items to {extract_path}")
            except Exception as e:
                print(f"Error extracting {file}: {e}")
                import traceback
                traceback.print_exc()
    else:
        print("Klim directory not found")
    
    # Similarly for Skagen
    skagen_dir = None
    for dir_name in os.listdir(data_dir):
        if dir_name.lower() == 'test-skagen':
            skagen_dir = os.path.join(data_dir, dir_name)
            break
    
    if skagen_dir and os.path.exists(skagen_dir):
        print(f"Found Skagen directory: {skagen_dir}")
        zip_files = [f for f in os.listdir(skagen_dir) if f.endswith('.zip')]
        print(f"Found {len(zip_files)} ZIP files: {zip_files}")
        
        for file in zip_files:
            zip_path = os.path.join(skagen_dir, file)
            base_name = os.path.splitext(file)[0]
            # Adjust the base name to match what the test function is looking for
            if base_name.lower().startswith('testm'):
                extract_name = base_name
            else:
                extract_name = f"testm{base_name.split('-')[-1] if '-' in base_name else '1'}"
            
            extract_path = os.path.join(data_dir, 'test-skagen-extracted', extract_name)
            os.makedirs(extract_path, exist_ok=True)
            
            print(f"Extracting {zip_path} to {extract_path}...")
            try:
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    file_list = zip_ref.namelist()
                    print(f"ZIP file contains {len(file_list)} files")
                    zip_ref.extractall(extract_path)
                print(f"Extraction of {file} complete")
                
                # List contents to verify extraction
                extracted_files = os.listdir(extract_path)
                print(f"Extracted {len(extracted_files)} items to {extract_path}")
            except Exception as e:
                print(f"Error extracting {file}: {e}")
                import traceback
                traceback.print_exc()
    else:
        print("Skagen directory not found")

def prepare_k_fold_data(data_dir, k_fold_num=1):
    """Prepare k-fold data for training."""
    print(f"Preparing k-fold data (fold {k_fold_num})...")
    
    # First check the standard path
    k_fold_dir = os.path.join(data_dir, 'k-fold', f'k{k_fold_num}')
    
    # If that doesn't exist, try other possible paths
    if not os.path.exists(k_fold_dir):
        print(f"Standard k-fold directory {k_fold_dir} not found, searching for alternatives...")
        
        # Try searching directly in the k-fold directory
        k_fold_base = os.path.join(data_dir, 'k-fold')
        if os.path.exists(k_fold_base):
            # List all directories and find those that might match our k-fold
            possible_dirs = []
            for item in os.listdir(k_fold_base):
                item_path = os.path.join(k_fold_base, item)
                if os.path.isdir(item_path) and (item.lower() == f'k{k_fold_num}' or item.lower() == f'fold{k_fold_num}'):
                    possible_dirs.append(item_path)
            
            if possible_dirs:
                k_fold_dir = possible_dirs[0]
                print(f"Found alternative k-fold directory: {k_fold_dir}")
            else:
                print(f"No matching k-fold directory found in {k_fold_base}")
                print(f"Available directories: {os.listdir(k_fold_base)}")
                return
    
    if not os.path.exists(k_fold_dir):
        print(f"K-fold directory {k_fold_dir} not found!")
        return
    
    # Create output directories
    train_img_dir = os.path.join(data_dir, 'prepared', 'images', 'train')
    train_ann_dir = os.path.join(data_dir, 'prepared', 'annotations', 'train')
    val_img_dir = os.path.join(data_dir, 'prepared', 'images', 'val')
    val_ann_dir = os.path.join(data_dir, 'prepared', 'annotations', 'val')
    
    os.makedirs(train_img_dir, exist_ok=True)
    os.makedirs(train_ann_dir, exist_ok=True)
    os.makedirs(val_img_dir, exist_ok=True)
    os.makedirs(val_ann_dir, exist_ok=True)
    
    # Copy training data
    src_train_dir = os.path.join(k_fold_dir, 'train')
    if os.path.exists(src_train_dir):
        print(f"Copying training data from {src_train_dir}")
        for file in os.listdir(src_train_dir):
            src_path = os.path.join(src_train_dir, file)
            if file.endswith('.jpg') or file.endswith('.jpeg') or file.endswith('.png'):
                dst_path = os.path.join(train_img_dir, file)
                shutil.copy2(src_path, dst_path)
            elif file.endswith('.txt'):
                dst_path = os.path.join(train_ann_dir, file)
                shutil.copy2(src_path, dst_path)
    else:
        print(f"Training directory not found: {src_train_dir}")
    
    # Copy validation data (look for "valid" or "val")
    val_dir_names = ['valid', 'val', 'validation']
    found_val_dir = False
    
    for val_name in val_dir_names:
        src_val_dir = os.path.join(k_fold_dir, val_name)
        if os.path.exists(src_val_dir):
            found_val_dir = True
            print(f"Copying validation data from {src_val_dir}")
            for file in os.listdir(src_val_dir):
                src_path = os.path.join(src_val_dir, file)
                if file.endswith('.jpg') or file.endswith('.jpeg') or file.endswith('.png'):
                    dst_path = os.path.join(val_img_dir, file)
                    shutil.copy2(src_path, dst_path)
                elif file.endswith('.txt'):
                    dst_path = os.path.join(val_ann_dir, file)
                    shutil.copy2(src_path, dst_path)
    
    if not found_val_dir:
        print(f"No validation directory found in {k_fold_dir}")
    
    # Count files
    train_imgs = len([f for f in os.listdir(train_img_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])
    train_anns = len([f for f in os.listdir(train_ann_dir) if f.endswith('.txt')])
    val_imgs = len([f for f in os.listdir(val_img_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])
    val_anns = len([f for f in os.listdir(val_ann_dir) if f.endswith('.txt')])
    
    print(f"Training data: {train_imgs} images, {train_anns} annotations")
    print(f"Validation data: {val_imgs} images, {val_anns} annotations")

def prepare_test_data(data_dir, model_num=1):
    """Prepare test data for evaluation."""
    print(f"Preparing test data for model {model_num}...")
    
    # Try different possible test directory patterns
    test_dir = None
    possible_patterns = [
        f"testm{model_num}",
        f"test-mm{model_num}",
        f"test_m{model_num}",
        f"test{model_num}"
    ]
    
    # Check in test-skagen-extracted
    extracted_dir = os.path.join(data_dir, 'test-skagen-extracted')
    if os.path.exists(extracted_dir):
        for pattern in possible_patterns:
            for item in os.listdir(extracted_dir):
                if pattern.lower() in item.lower():
                    test_dir = os.path.join(extracted_dir, item)
                    print(f"Found test directory: {test_dir}")
                    break
            if test_dir:
                break
    
    # Also check directly in the Test-Skagen directory
    if not test_dir:
        for dir_name in os.listdir(data_dir):
            if dir_name.lower() == 'test-skagen':
                skagen_dir = os.path.join(data_dir, dir_name)
                # See if there are non-zip files we can use
                for item in os.listdir(skagen_dir):
                    item_path = os.path.join(skagen_dir, item)
                    if os.path.isdir(item_path) and any(pattern.lower() in item.lower() for pattern in possible_patterns):
                        test_dir = item_path
                        print(f"Found test directory directly in Test-Skagen: {test_dir}")
                        break
                if test_dir:
                    break
    
    if not test_dir:
        print(f"Test directory for model {model_num} not found! Tried patterns: {possible_patterns}")
        return
    
    # Create output directories
    test_img_dir = os.path.join(data_dir, 'prepared', 'images', 'test')
    test_ann_dir = os.path.join(data_dir, 'prepared', 'annotations', 'test')
    
    os.makedirs(test_img_dir, exist_ok=True)
    os.makedirs(test_ann_dir, exist_ok=True)
    
    # Find and copy test data
    images_copied = 0
    annotations_copied = 0
    
    # Recursive function to handle nested directories
    def copy_files_recursive(directory):
        nonlocal images_copied, annotations_copied
        for item in os.listdir(directory):
            item_path = os.path.join(directory, item)
            if os.path.isdir(item_path):
                # Recursively process directories
                copy_files_recursive(item_path)
            elif item.endswith(('.jpg', '.jpeg', '.png')):
                dst_path = os.path.join(test_img_dir, item)
                shutil.copy2(item_path, dst_path)
                images_copied += 1
            elif item.endswith('.txt'):
                dst_path = os.path.join(test_ann_dir, item)
                shutil.copy2(item_path, dst_path)
                annotations_copied += 1
    
    copy_files_recursive(test_dir)
    
    print(f"Test data: {images_copied} images, {annotations_copied} annotations copied")

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Data Preparation for Bird Detection')
    
    parser.add_argument('--data-dir', type=str, default='data',
                        help='Base directory containing the data')
    parser.add_argument('--extract', action='store_true',
                        help='Extract ZIP files')
    parser.add_argument('--prepare-train', action='store_true',
                        help='Prepare training data')
    parser.add_argument('--prepare-test', action='store_true',
                        help='Prepare test data')
    parser.add_argument('--k-fold', type=int, default=1,
                        help='K-fold number to use (1-6)')
    parser.add_argument('--model-num', type=int, default=1,
                        help='Model number for test data (1 or 3)')
    
    return parser.parse_args()

def main():
    """Main function."""
    args = parse_arguments()
    
    # Create prepared data directory
    os.makedirs(os.path.join(args.data_dir, 'prepared', 'images', 'train'), exist_ok=True)
    os.makedirs(os.path.join(args.data_dir, 'prepared', 'annotations', 'train'), exist_ok=True)
    os.makedirs(os.path.join(args.data_dir, 'prepared', 'images', 'val'), exist_ok=True)
    os.makedirs(os.path.join(args.data_dir, 'prepared', 'annotations', 'val'), exist_ok=True)
    os.makedirs(os.path.join(args.data_dir, 'prepared', 'images', 'test'), exist_ok=True)
    os.makedirs(os.path.join(args.data_dir, 'prepared', 'annotations', 'test'), exist_ok=True)
    
    if args.extract:
        extract_zip_files(args.data_dir)
    
    if args.prepare_train:
        prepare_k_fold_data(args.data_dir, args.k_fold)
    
    if args.prepare_test:
        prepare_test_data(args.data_dir, args.model_num)
    
    if not (args.extract or args.prepare_train or args.prepare_test):
        print("No actions specified. Use --extract, --prepare-train, or --prepare-test flags.")

if __name__ == '__main__':
    main()