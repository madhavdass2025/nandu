from datasets import load_dataset
import os
from PIL import Image

def download_and_save_dataset():
    # Load the dataset from Hugging Face
    dataset = load_dataset("Sohaibsoussi/NIH-Chest-X-ray-dataset-small", split="train")

    # Create directories if they don't exist
    os.makedirs("data/train/NORMAL", exist_ok=True)
    os.makedirs("data/train/PNEUMONIA", exist_ok=True)
    os.makedirs("data/test/NORMAL", exist_ok=True)
    os.makedirs("data/test/PNEUMONIA", exist_ok=True)

    normal_train_count = 0
    pneumonia_train_count = 0
    normal_test_count = 0
    pneumonia_test_count = 0

    for i, item in enumerate(dataset):
        image = item['image']
        labels = item['labels']

        # We'll consider a label of [0] as "NORMAL" and any other label as "PNEUMONIA"
        if labels == [0]:
            label = "NORMAL"
            if normal_train_count < 10:
                image_path = os.path.join("data/train", label, f"image_{i}.png")
                normal_train_count += 1
            elif normal_test_count < 5:
                image_path = os.path.join("data/test", label, f"image_{i}.png")
                normal_test_count += 1
            else:
                continue
        else:
            label = "PNEUMONIA"
            if pneumonia_train_count < 10:
                image_path = os.path.join("data/train", label, f"image_{i}.png")
                pneumonia_train_count += 1
            elif pneumonia_test_count < 5:
                image_path = os.path.join("data/test", label, f"image_{i}.png")
                pneumonia_test_count += 1
            else:
                continue

        # Convert to RGB if necessary and save the image
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image.save(image_path)

        if normal_train_count >= 10 and pneumonia_train_count >= 10 and normal_test_count >= 5 and pneumonia_test_count >= 5:
            break

if __name__ == "__main__":
    download_and_save_dataset()