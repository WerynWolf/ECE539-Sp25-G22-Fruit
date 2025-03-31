import os
import random
from PIL import Image
from tqdm import tqdm


def remove_white_background(image, threshold=240):
    image = image.convert("RGBA")
    datas = image.getdata()
    new_data = []
    for item in datas:
        if item[0] >= threshold and item[1] >= threshold and item[2] >= threshold:
            new_data.append((255, 255, 255, 0))
        else:
            new_data.append(item)
    image.putdata(new_data)
    return image


def apply_random_shear(image, probability=0.3, shear_range=(-0.3, 0.3)):
    if random.random() < probability:
        shear_factor = random.uniform(*shear_range)
        width, height = image.size
        try:
            resample_method = Image.Resampling.BICUBIC
        except AttributeError:
            resample_method = Image.BICUBIC
        image = image.transform(
            (width, height),
            Image.AFFINE,
            (1, shear_factor, 0, 0, 1, 0),
            resample=resample_method
        )
    return image


def composite_single_fruit_with_mask(fruit_image_path, background_image_path, output_image_path, output_mask_path):
    background = Image.open(background_image_path).convert("RGBA")
    composite = background.copy()
    bg_width, bg_height = background.size

    fruit = Image.open(fruit_image_path).convert("RGBA")
    fruit = remove_white_background(fruit)
    fruit = apply_random_shear(fruit, probability=0.3, shear_range=(-0.3, 0.3))

    desired_scale = random.uniform(0.2, 0.5)
    target_width = int(bg_width * desired_scale)
    target_height = int(bg_height * desired_scale)
    fruit_width, fruit_height = fruit.size
    fruit_aspect = fruit_width / fruit_height
    if target_width / target_height > fruit_aspect:
        new_fruit_height = target_height
        new_fruit_width = int(fruit_aspect * target_height)
    else:
        new_fruit_width = target_width
        new_fruit_height = int(target_width / fruit_aspect)

    try:
        resample_method = Image.Resampling.LANCZOS
    except AttributeError:
        resample_method = Image.ANTIALIAS
    fruit = fruit.resize((new_fruit_width, new_fruit_height), resample_method)

    max_x = bg_width - new_fruit_width
    max_y = bg_height - new_fruit_height
    x = random.randint(0, max(0, max_x))
    y = random.randint(0, max(0, max_y))
    composite.paste(fruit, (x, y), fruit)

    fruit_alpha = fruit.split()[-1]
    threshold_value = 128
    fruit_mask = fruit_alpha.point(lambda p: 255 if p > threshold_value else 0)
    mask = Image.new("L", (bg_width, bg_height), 0)
    mask.paste(fruit_mask, (x, y))

    if output_image_path.lower().endswith(('.jpg', '.jpeg')):
        composite = composite.convert("RGB")
    composite.save(output_image_path)
    mask.save(output_mask_path)




def composite_multiple_fruits_with_mask(fruit_image_paths, background_image_path, output_image_path, output_mask_path):
    background = Image.open(background_image_path).convert("RGBA")
    composite = background.copy()
    bg_width, bg_height = background.size
    mask = Image.new("L", (bg_width, bg_height), 0)

    for fruit_image_path in fruit_image_paths:
        fruit = Image.open(fruit_image_path).convert("RGBA")
        fruit = remove_white_background(fruit)
        fruit = apply_random_shear(fruit, probability=0.3, shear_range=(-0.3, 0.3))

        desired_scale = random.uniform(0.15, 0.3)
        target_width = int(bg_width * desired_scale)
        target_height = int(bg_height * desired_scale)
        fruit_width, fruit_height = fruit.size
        fruit_aspect = fruit_width / fruit_height
        if target_width / target_height > fruit_aspect:
            new_fruit_height = target_height
            new_fruit_width = int(fruit_aspect * target_height)
        else:
            new_fruit_width = target_width
            new_fruit_height = int(target_width / fruit_aspect)

        try:
            resample_method = Image.Resampling.LANCZOS
        except AttributeError:
            resample_method = Image.ANTIALIAS
        fruit = fruit.resize((new_fruit_width, new_fruit_height), resample_method)

        max_x = bg_width - new_fruit_width
        max_y = bg_height - new_fruit_height
        x = random.randint(0, max(0, max_x))
        y = random.randint(0, max(0, max_y))
        composite.paste(fruit, (x, y), fruit)

        fruit_alpha = fruit.split()[-1]
        threshold_value = 128
        fruit_mask = fruit_alpha.point(lambda p: 255 if p > threshold_value else 0)
        mask.paste(fruit_mask, (x, y))

    if output_image_path.lower().endswith(('.jpg', '.jpeg')):
        composite = composite.convert("RGB")
    composite.save(output_image_path)
    mask.save(output_mask_path)



def get_all_images(root_folder, valid_extensions=('.png', '.jpg', '.jpeg')):
    image_paths = []
    for dirpath, _, filenames in os.walk(root_folder):
        for filename in filenames:
            if filename.lower().endswith(valid_extensions):
                image_paths.append(os.path.join(dirpath, filename))
    return image_paths


def main():
    fruits_root = "data"
    backgrounds_root = "background"
    output_folder = "composited"
    masks_folder = "masks"

    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(masks_folder, exist_ok=True)

    background_images = get_all_images(backgrounds_root)
    if not background_images:
        print("No background images found in:", backgrounds_root)
        return

    fruit_types = [d for d in os.listdir(fruits_root) if os.path.isdir(os.path.join(fruits_root, d))]
    if not fruit_types:
        print("No fruit type folders found in:", fruits_root)
        return

    total_composites = 0
    fruit_info = {}
    for fruit_type in fruit_types:
        fruit_type_folder = os.path.join(fruits_root, fruit_type)
        fruit_images = [f for f in os.listdir(fruit_type_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not fruit_images:
            continue
        fruit_info[fruit_type] = fruit_images
        total_composites += len(fruit_images) * len(background_images)
        if len(fruit_images) >= 2:
            total_composites += len(background_images)

    progress_bar = tqdm(total=total_composites, desc="Processing composites", unit="img")

    for fruit_type in fruit_types:
        if fruit_type not in fruit_info:
            continue
        fruit_type_folder = os.path.join(fruits_root, fruit_type)
        fruit_images = fruit_info[fruit_type]

        output_type_folder = os.path.join(output_folder, fruit_type)
        masks_type_folder = os.path.join(masks_folder, fruit_type)
        os.makedirs(output_type_folder, exist_ok=True)
        os.makedirs(masks_type_folder, exist_ok=True)

        for fruit_image in fruit_images:
            fruit_image_path = os.path.join(fruit_type_folder, fruit_image)
            fruit_base = os.path.splitext(fruit_image)[0]
            for bg_path in background_images:
                bg_base = os.path.splitext(os.path.basename(bg_path))[0]
                output_image_name = f"comp_{fruit_type}_{fruit_base}_{bg_base}.png"
                output_mask_name = f"mask_{fruit_type}_{fruit_base}_{bg_base}.png"
                output_image_path = os.path.join(output_type_folder, output_image_name)
                output_mask_path = os.path.join(masks_type_folder, output_mask_name)
                composite_single_fruit_with_mask(fruit_image_path, bg_path, output_image_path, output_mask_path)
                progress_bar.update(1)

        if len(fruit_images) >= 2:
            for bg_path in background_images:
                bg_base = os.path.splitext(os.path.basename(bg_path))[0]
                num_to_select = random.randint(2, min(3, len(fruit_images)))
                selected = random.sample(fruit_images, num_to_select)
                selected_paths = [os.path.join(fruit_type_folder, f) for f in selected]
                fruits_label = "_".join([os.path.splitext(f)[0] for f in selected])
                output_image_name = f"comp_{fruit_type}_{bg_base}_mult_{fruits_label}.png"
                output_mask_name = f"mask_{fruit_type}_{bg_base}_mult_{fruits_label}.png"
                output_image_path = os.path.join(output_type_folder, output_image_name)
                output_mask_path = os.path.join(masks_type_folder, output_mask_name)
                composite_multiple_fruits_with_mask(selected_paths, bg_path, output_image_path, output_mask_path)
                progress_bar.update(1)

    progress_bar.close()


if __name__ == "__main__":
    main()
