import os
import shutil
import random

# Set paths
original_dataset_dir = "C:/Users/Kaushik/OneDrive/Desktop/Major Project/archive/lung_colon_image_set/lung_image_sets"
classes = ['LUAD', 'LUSC', 'NORMAL']  # Renamed classes

# Output folders
base_dir = 'data'
train_dir = os.path.join(base_dir, 'train')
test_dir = os.path.join(base_dir, 'test')

# Create directory structure
for dir_path in [train_dir, test_dir]:
    os.makedirs(dir_path, exist_ok=True)
    for cls in classes:
        os.makedirs(os.path.join(dir_path, cls), exist_ok=True)

# Split ratio
split_ratio = 0.8

# Move files
for cls in classes:
    src = os.path.join(original_dataset_dir, cls)
    files = os.listdir(src)
    random.shuffle(files)
    split_index = int(len(files) * split_ratio)
    
    train_files = files[:split_index]
    test_files = files[split_index:]
    
    for f in train_files:
        shutil.copy(os.path.join(src, f), os.path.join(train_dir, cls, f))
    for f in test_files:
        shutil.copy(os.path.join(src, f), os.path.join(test_dir, cls, f))

print("✅ Dataset split into train and test folders.")
