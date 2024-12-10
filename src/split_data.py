import shutil
import os
import random

# Define the paths
input_dir = '../data/music_genre_dataset/mel_spectrograms'
output_dirs = {
    'train': '../data/dataset/train',
    'val': '../data/dataset/val',
    'test': '../data/dataset/test',
}

# Create the output directories
for dir_path in output_dirs.values():
    os.makedirs(dir_path, exist_ok=True)

# Define the split ratios
train_ratio = 0.70
val_ratio = 0.15

# Initialize counters for each split
data_counts = {
    'train': 0,
    'val': 0,
    'test': 0
}

# Iterate through each genre
for genre in os.listdir(input_dir):
    genre_path = os.path.join(input_dir, genre)
    if os.path.isdir(genre_path):
        # Get all file names for the current genre
        files = [f for f in os.listdir(genre_path) if os.path.isfile(os.path.join(genre_path, f))]
        random.shuffle(files)

        # Calculate the split indices
        total_files = len(files)
        train_end = int(total_files * train_ratio)
        val_end = train_end + int(total_files * val_ratio)

        # Split the files
        train_files = files[:train_end]
        val_files = files[train_end:val_end]
        test_files = files[val_end:]

        # Update data counts
        data_counts['train'] += len(train_files)
        data_counts['val'] += len(val_files)
        data_counts['test'] += len(test_files)

        # Move the files to the corresponding directories
        for split, file_list in zip(['train', 'val', 'test'], [train_files, val_files, test_files]):
            genre_output_dir = os.path.join(output_dirs[split], genre)
            os.makedirs(genre_output_dir, exist_ok=True)
            for file_name in file_list:
                src = os.path.join(genre_path, file_name)
                dst = os.path.join(genre_output_dir, file_name)
                shutil.copy(src, dst)

# Print data counts
print("Dataset split completed successfully.")
print(f"Training samples: {data_counts['train']}")
print(f"Validation samples: {data_counts['val']}")
print(f"Testing samples: {data_counts['test']}")
