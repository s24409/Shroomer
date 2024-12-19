import os
import shutil
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torchvision.datasets.folder import default_loader
import pandas as pd
from PIL import Image
import pickle
import warnings
from tqdm import tqdm

# Suppress warnings
warnings.filterwarnings("ignore")

class CustomDataset(Dataset):
    def __init__(self, dataframe, image_dir, transform, class_to_idx):
        self.df = dataframe.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform
        self.labels = self.df['class_id'].map(class_to_idx).values
        self.image_paths = self.df['image_path'].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_paths[idx])

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found at {img_path}")

        image = default_loader(img_path)
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return image, label

def preprocess_images(metadata, original_image_dir, preprocessed_image_dir, image_size):
    failed_images = []
    for idx, row in tqdm(metadata.iterrows(), total=len(metadata), desc="Preprocessing images"):
        img_path = os.path.join(original_image_dir, row['image_path'])
        preprocessed_img_path = os.path.join(preprocessed_image_dir, row['image_path'])
        os.makedirs(os.path.dirname(preprocessed_img_path), exist_ok=True)

        if not os.path.exists(img_path):
            print(f"Original file not found: {img_path}")
            failed_images.append(row['image_path'])
            continue

        try:
            image = Image.open(img_path).convert('RGB')
            image = image.resize(image_size, resample=Image.BILINEAR)
            image.save(preprocessed_img_path)
        except Exception as e:
            print(f"Error processing image {img_path}: {e}")
            failed_images.append(row['image_path'])

    if failed_images:
        print(f"Failed to process {len(failed_images)} images.")
        with open('failed_images.txt', 'w') as f:
            for item in failed_images:
                f.write(f"{item}\n")

def main():
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Hyperparameters
    batch_size = 64
    num_epochs = 18
    learning_rate = 0.001
    image_size = (224, 224)
    num_workers = 8

    # Paths
    train_metadata_path = 'DF20-train_metadata_PROD-2.csv'
    image_dir = 'DF20_300'
    preprocessed_image_dir = 'DF20_300_preprocessed'
    model_path = 'model_gpu_test.pth'  # We will use this for resuming training if it exists

    # Load train metadata
    train_metadata = pd.read_csv(train_metadata_path)
    train_metadata['class_id'] = train_metadata['class_id'].astype(str)

    # Create class_to_idx mapping
    classes = sorted(train_metadata['class_id'].unique())
    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}
    num_classes = len(classes)

    # Save class_to_idx mapping
    with open('class_to_idx.pkl', 'wb') as f:
        pickle.dump(class_to_idx, f)

    # Check if preprocessing needed
    if not os.path.exists(preprocessed_image_dir) or len(os.listdir(preprocessed_image_dir)) == 0:
        print("Preprocessing images...")
        shutil.rmtree(preprocessed_image_dir, ignore_errors=True)
        os.makedirs(preprocessed_image_dir, exist_ok=True)
        preprocess_images(train_metadata, image_dir, preprocessed_image_dir, image_size)
    else:
        print("Skipping preprocessing as preprocessed images already exist.")

    # Filter out missing/failed files
    failed_images_path = 'failed_images.txt'
    if os.path.exists(failed_images_path):
        with open(failed_images_path, 'r') as f:
            failed_images = f.read().splitlines()
        train_metadata = train_metadata[~train_metadata['image_path'].isin(failed_images)].reset_index(drop=True)

    # Data transformations
    data_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # Create dataset and dataloader
    dataset = CustomDataset(train_metadata, preprocessed_image_dir, transform=data_transforms, class_to_idx=class_to_idx)
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )

    # Load model
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)

    # Load previous weights if they exist
    start_epoch = 10
    if os.path.exists(model_path):
        print("Loading model from previous checkpoint...")
        model.load_state_dict(torch.load(model_path))
        # If you have a saved optimizer state and epoch number, you can load them too here
        # For example:
        # checkpoint = torch.load('checkpoint.pth')
        # model.load_state_dict(checkpoint['model_state_dict'])
        # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # start_epoch = checkpoint['epoch'] + 1
    else:
        print("No previous checkpoint found. Starting training from scratch.")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(start_epoch, num_epochs):
        model.train()
        running_loss = 0.0
        with tqdm(data_loader, desc=f"Epoch [{epoch+1}/{num_epochs}]", unit="batch") as tepoch:
            for inputs, labels in tepoch:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                if tepoch.n > 0:
                    tepoch.set_postfix(loss=(running_loss / tepoch.n))

        avg_loss = running_loss / len(data_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {avg_loss:.4f}")

        # Save model checkpoint at the end of each epoch
        torch.save(model.state_dict(), model_path)

    print("Training complete.")

if __name__ == '__main__':
    main()
