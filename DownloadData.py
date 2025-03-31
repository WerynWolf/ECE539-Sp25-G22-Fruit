import random
import kagglehub
import os
import shutil
import numpy as np
from tensorflow.keras.datasets import cifar100
from PIL import Image

allowed_fruits = {"apple", "banana", "strawberry", "kiwi", "peach", "plum", "cherry", "pear", "mango", "limes"}

def ds1():
    dataset_path = kagglehub.dataset_download("moltean/fruits")
    print("Path to dataset files:", dataset_path)

    base_fruits_dir = os.path.join(dataset_path, "fruits-360_original-size", "fruits-360-original-size")

    if not os.path.exists(base_fruits_dir):
        print("Expected base folder not found. Available directories in the dataset path:")
        print(os.listdir(dataset_path))
        print(base_fruits_dir)
        raise Exception(f"Expected directory '{base_fruits_dir}' not found.")

    splits = ["test", "training", "validation"]

    dest_base_dir = os.getcwd()
    destination_folder_name = "data"
    data_dir = os.path.join(dest_base_dir, destination_folder_name)
    os.makedirs(data_dir, exist_ok=True)

    for split in splits:
        split_dir = None
        for candidate in [split, split.capitalize(), split.upper()]:
            candidate_path = os.path.join(base_fruits_dir, candidate)
            if os.path.exists(candidate_path):
                split_dir = candidate_path
                break
        if split_dir is None:
            print(f"Split folder for '{split}' not found in '{base_fruits_dir}'. Skipping this split...")
            continue

        for folder in os.listdir(split_dir):
            folder_path = os.path.join(split_dir, folder)
            if os.path.isdir(folder_path):
                fruit_candidate = folder.split()[0].lower()
                if fruit_candidate in allowed_fruits:
                    fruit_folder_name = fruit_candidate.capitalize()
                    target_fruit_dir = os.path.join(data_dir, fruit_folder_name)
                    os.makedirs(target_fruit_dir, exist_ok=True)

                    for file_name in os.listdir(folder_path):
                        src_file = os.path.join(folder_path, file_name)
                        dst_file = os.path.join(target_fruit_dir, file_name)
                        counter = 1
                        original_dst = dst_file
                        while os.path.exists(dst_file):
                            file_root, file_ext = os.path.splitext(original_dst)
                            dst_file = f"{file_root}_{counter}{file_ext}"
                            counter += 1
                        shutil.move(src_file, dst_file)

                    if not os.listdir(folder_path):
                        os.rmdir(folder_path)
                    print(f"Moved files from '{folder}' in split '{split}' to '{target_fruit_dir}'")
                else:
                    print(f"Skipped folder '{folder}' (not an allowed fruit)")


def ds2():
    dataset_path = kagglehub.dataset_download("sshikamaru/fruit-recognition")
    print("Path to dataset files:", dataset_path)

    base_fruits_dir = dataset_path

    if not os.path.exists(base_fruits_dir):
        print("Expected base folder not found. Available directories in the dataset path:")
        print(os.listdir(dataset_path))
        raise Exception(f"Expected directory '{base_fruits_dir}' not found.")

    splits = ["train"]

    dest_base_dir = os.getcwd()
    destination_folder_name = "data"
    new_split_dir = os.path.join(dest_base_dir, destination_folder_name)
    os.makedirs(new_split_dir, exist_ok=True)

    for split in splits:
        split_dir = None
        for candidate in [split, split.capitalize(), split.upper()]:
            candidate_path = os.path.join(base_fruits_dir, candidate)
            if os.path.exists(candidate_path):
                split_dir = candidate_path
                break
        if split_dir is None:
            print(f"Split folder for '{split}' not found in '{base_fruits_dir}'. Skipping...")
            continue

        nested_train = os.path.join(split_dir, "train")
        if os.path.exists(nested_train) and os.path.isdir(nested_train):
            print("Found nested 'train' folder. Using its contents.")
            split_dir = nested_train

        for folder in os.listdir(split_dir):
            folder_path = os.path.join(split_dir, folder)
            if os.path.isdir(folder_path):
                fruit_candidate = folder.split()[0].lower()
                if fruit_candidate in allowed_fruits:
                    fruit_folder_name = fruit_candidate.capitalize()
                    target_fruit_dir = os.path.join(new_split_dir, fruit_folder_name)
                    os.makedirs(target_fruit_dir, exist_ok=True)

                    for file_name in os.listdir(folder_path):
                        src_file = os.path.join(folder_path, file_name)
                        dst_file = os.path.join(target_fruit_dir, file_name)
                        counter = 1
                        original_dst_file = dst_file
                        while os.path.exists(dst_file):
                            file_root, file_ext = os.path.splitext(original_dst_file)
                            dst_file = f"{file_root}_{counter}{file_ext}"
                            counter += 1
                        shutil.move(src_file, dst_file)

                    if not os.listdir(folder_path):
                        os.rmdir(folder_path)
                    print(f"Moved files from '{folder}' to '{target_fruit_dir}'")
                else:
                    print(f"Skipped folder '{folder}' (not an allowed fruit)")


def ds3():
    train_ratio, val_ratio, test_ratio = (0.7, 0.15, 0.15)

    source_dir = kagglehub.dataset_download("aelchimminut/fruits262")
    dest_dir = os.getcwd()

    training_dir = os.path.join(dest_dir, "training")
    validation_dir = os.path.join(dest_dir, "validation")
    test_dir = os.path.join(dest_dir, "test")
    for d in [training_dir, validation_dir, test_dir]:
        os.makedirs(d, exist_ok=True)

    fruits_base = os.path.join(source_dir, 'Fruit-262')

    for fruit_folder in os.listdir(fruits_base):
        fruit_folder_path = os.path.join(fruits_base, fruit_folder)
        if os.path.isdir(fruit_folder_path):
            if allowed_fruits and fruit_folder.lower() not in allowed_fruits:
                print(f"Skipping folder '{fruit_folder}' as it is not in allowed fruits.")
                continue

            fruit_train_dir = os.path.join(training_dir, fruit_folder)
            fruit_val_dir = os.path.join(validation_dir, fruit_folder)
            fruit_test_dir = os.path.join(test_dir, fruit_folder)
            os.makedirs(fruit_train_dir, exist_ok=True)
            os.makedirs(fruit_val_dir, exist_ok=True)
            os.makedirs(fruit_test_dir, exist_ok=True)

            files = [f for f in os.listdir(fruit_folder_path)
                     if os.path.isfile(os.path.join(fruit_folder_path, f))]
            random.shuffle(files)

            total = len(files)
            train_count = int(total * train_ratio)
            val_count = int(total * val_ratio)
            test_count = total - train_count - val_count

            train_files = files[:train_count]
            val_files = files[train_count:train_count + val_count]
            test_files = files[train_count + val_count:]

            for file in train_files:
                src = os.path.join(fruit_folder_path, file)
                dst = os.path.join(fruit_train_dir, file)
                shutil.move(src, dst)

            for file in val_files:
                src = os.path.join(fruit_folder_path, file)
                dst = os.path.join(fruit_val_dir, file)
                shutil.move(src, dst)

            for file in test_files:
                src = os.path.join(fruit_folder_path, file)
                dst = os.path.join(fruit_test_dir, file)
                shutil.move(src, dst)

            print(f"Processed '{fruit_folder}': {len(train_files)} training, {len(val_files)} validation")


def backgrounds():
    selected_categories = [5, 6, 9, 10]
    coarse_label_names = {
        5: 'household_electrical_devices',
        6: 'household_furniture',
        9: 'large_man_made_outdoor_things',
        10: 'large_natural_outdoor_scenes'
    }
    (train_images, train_labels), (_, _) = cifar100.load_data(label_mode='coarse')

    indices = np.where(np.isin(train_labels, selected_categories))[0]
    print(f"Found {len(indices)} images in the selected categories in the training set.")

    output_dir = 'background'
    os.makedirs(output_dir, exist_ok=True)
    for cat in selected_categories:
        os.makedirs(os.path.join(output_dir, coarse_label_names[cat]), exist_ok=True)

    per_category_limit = 50
    counters = {cat: 0 for cat in selected_categories}

    for idx in indices:
        cat = int(train_labels[idx][0])
        if counters[cat] < per_category_limit:
            img = Image.fromarray(train_images[idx])
            filename = os.path.join(output_dir, coarse_label_names[cat], f"{counters[cat]:04d}.png")
            img.save(filename)
            counters[cat] += 1

