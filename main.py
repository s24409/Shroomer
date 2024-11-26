import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision import models
from tqdm import tqdm

def main():
    # Verify CUDA availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    assert torch.cuda.is_available(), "CUDA is not available. Please check your PyTorch installation."

    # Set random seed for reproducibility
    torch.manual_seed(42)
    if device.type == 'cuda':
        torch.cuda.manual_seed(42)

    # Hyperparameters
    batch_size = 128  # Adjust based on your GPU memory
    num_classes = 10  # Example number of classes
    num_epochs = 15    # Number of epochs for training
    learning_rate = 0.001

    # Generate synthetic dataset
    num_samples = 10000
    input_size = (3, 224, 224)  # Image dimensions for ResNet

    # Create synthetic data tensors
    inputs = torch.randn(num_samples, *input_size)
    labels = torch.randint(0, num_classes, (num_samples,))

    # Create TensorDataset and DataLoader
    dataset = TensorDataset(inputs, labels)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    # Load ResNet50 model
    model = models.resnet50(pretrained=False)  # Use pretrained=True if you want to use pretrained weights
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)

    # Confirm that model is on GPU
    assert next(model.parameters()).is_cuda, "Model is not on GPU"

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Use mixed precision training
    from torch.cuda.amp import GradScaler, autocast
    scaler = GradScaler()

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_corrects = 0

        # Progress bar
        train_loader_tqdm = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', leave=False)

        for inputs_batch, labels_batch in train_loader_tqdm:
            # Move data to GPU
            inputs_batch = inputs_batch.to(device, non_blocking=True)
            labels_batch = labels_batch.to(device, non_blocking=True)

            # Confirm that inputs and labels are on GPU
            assert inputs_batch.is_cuda, "Inputs are not on GPU"
            assert labels_batch.is_cuda, "Labels are not on GPU"

            optimizer.zero_grad()

            with autocast():
                outputs = model(inputs_batch)
                loss = criterion(outputs, labels_batch)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # Statistics
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs_batch.size(0)
            running_corrects += torch.sum(preds == labels_batch.data)

            # Update progress bar
            train_loader_tqdm.set_postfix(loss=loss.item())

        epoch_loss = running_loss / num_samples
        epoch_acc = running_corrects.double() / num_samples
        print(f'Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

    print("Training complete.")

    # Save the model
    torch.save(model.state_dict(), 'model_gpu_test.pth')
    print("Model saved to 'model_gpu_test.pth'.")

if __name__ == "__main__":
    main()
