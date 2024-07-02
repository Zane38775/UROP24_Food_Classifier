import os
from sklearn.model_selection import train_test_split

data_dir = 'data/food-101/images'

image_paths = []
labels = []
for class_name in os.listdir(data_dir):
    class_dir = os.path.join(data_dir, class_name)
    if os.path.isdir(class_dir):
        for img_file in os.listdir(class_dir):
            image_paths.append(os.path.join(class_dir, img_file))
            labels.append(class_name)
            
train_paths, tmp_paths, train_labels, tmp_labels = train_test_split(
    image_paths, labels, test_size=0.4, random_state=42, stratify=labels)

val_paths, test_paths, val_labels, test_labels = train_test_split(
    tmp_paths, tmp_labels, test_size=0.5, random_state=42, stratify=tmp_labels)

print(f"Train set size: {len(train_paths)}")
print(f"Validation set size: {len(val_paths)}")  
print(f"Test set size: {len(test_paths)}")

