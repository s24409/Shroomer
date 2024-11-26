import os
import torch
from torchvision import models, transforms
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image
import pickle
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Define the CustomDataset class
class CustomDataset(torch.utils.data.Dataset):
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
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

def evaluate_model():
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Paths
    model_path = 'model_gpu_test.pth'
    test_metadata_path = 'DF20-public_test_metadata_PROD-2.csv'
    test_image_dir = 'DF20_300'  # Adjust if necessary
    batch_size = 64
    image_size = (224, 224)

    # Load class_to_idx mapping
    if not os.path.exists('class_to_idx.pkl'):
        print("Error: 'class_to_idx.pkl' not found. Please ensure the training script has been run and the file exists.")
        return
    with open('class_to_idx.pkl', 'rb') as f:
        class_to_idx = pickle.load(f)
    print("class_to_idx.pkl has been loaded.")
    idx_to_class = {idx: cls_name for cls_name, idx in class_to_idx.items()}
    num_classes = len(class_to_idx)

    # Load the model architecture
    model = models.resnet50(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()  # Set to evaluation mode

    # Load test metadata
    test_metadata = pd.read_csv(test_metadata_path)
    test_metadata['class_id'] = test_metadata['class_id'].astype(str)

    # Filter test_metadata to include only classes seen during training
    test_metadata = test_metadata[test_metadata['class_id'].isin(class_to_idx.keys())].reset_index(drop=True)

    # Check if test dataset is empty after filtering
    if len(test_metadata) == 0:
        print("No matching classes found in test dataset after filtering. Evaluation cannot proceed.")
        return
    else:
        print(f"Number of test samples: {len(test_metadata)}")

    # Data transformations (ensure they match validation)
    data_transforms = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # Create the test dataset and DataLoader
    test_dataset = CustomDataset(test_metadata, test_image_dir, transform=data_transforms, class_to_idx=class_to_idx)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # Evaluate the model on the test data
    correct = 0
    total = 0

    # Use mixed precision for faster inference if CUDA is available
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Testing"):
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=(scaler is not None)):
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Debug: Print sample predictions and labels
            if total <= 5 * batch_size:
                print(f"Predicted: {predicted.cpu().numpy()}")
                print(f"Actual: {labels.cpu().numpy()}")

    accuracy = correct / total if total > 0 else 0
    print(f'Test accuracy: {accuracy:.4f}')

if __name__ == '__main__':
    evaluate_model()
