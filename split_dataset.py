import os
import shutil
from sklearn.model_selection import train_test_split

DATASET_DIR = "dataset/images"           
OUTPUT_DIR = "dataset_split"      
TEST_SIZE = 0.2                   
RANDOM_STATE = 42                 

train_dir = os.path.join(OUTPUT_DIR, "train")
val_dir = os.path.join(OUTPUT_DIR, "val")

os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

for class_name in os.listdir(DATASET_DIR):
    class_path = os.path.join(DATASET_DIR, class_name)
    if not os.path.isdir(class_path):
        continue
    images = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if len(images) == 0:
        print(f"⚠️ No images found in {class_name}, skipping...")
        continue

    train_imgs, val_imgs = train_test_split(images, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
    os.makedirs(os.path.join(val_dir, class_name), exist_ok=True)

    for img in train_imgs:
        shutil.copy(os.path.join(class_path, img), os.path.join(train_dir, class_name, img))
    for img in val_imgs:
        shutil.copy(os.path.join(class_path, img), os.path.join(val_dir, class_name, img))

    print(f"✅ {class_name}: {len(train_imgs)} train, {len(val_imgs)} val images")

print("\n🎉 Dataset split complete! Check the 'dataset_split/train' and 'dataset_split/val' folders.")
