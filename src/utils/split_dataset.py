import os
import shutil
import random

# function to clear target directory
def clear_directory(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)

# split the dataset into train/test/split in a new directory
def split_dataset(
        source_dir='raw_data',
        dest_dir='data',
        splits=(0.7, 0.15, 0.15),
        seed=42
):
    assert sum(splits) == 1.0, "Splits must sum to 1.0"
    random.seed(seed)
    split_names = ['train', 'val', 'test']
    class_names = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]

    # clear and recreate destination directories
    for split in split_names:
        for cls in class_names:
            clear_directory(os.path.join(dest_dir, split, cls))

    # shuffle and copy files
    for cls in class_names:
        class_path = os.path.join(source_dir, cls)
        images = os.listdir(class_path)
        random.shuffle(images)
        total = len(images)
        train_end = int(splits[0] * total)
        val_end = train_end + int(splits[1] * total)

        subsets = {
            'train': images[:train_end],
            'val': images[train_end:val_end],
            'test': images[val_end:]
        }

        for split, files in subsets.items():
            for file in files:
                src = os.path.join(source_dir, cls, file)
                dst = os.path.join(dest_dir, split, cls, file)
                shutil.copy2(src, dst)

# main loop
if __name__ == "__main__":
    split_dataset()