import json
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2
import random
import csv
import os
import pickle
import pandas as pd
import pydicom
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import functional as TF
from PIL import Image
import numpy as np

## assisting script
def plot_confusion_matrix(cm, class_names):
    fig, ax = plt.subplots(figsize=(8, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    return fig


def save_checkpoint(model, optimizer, scheduler, epoch, fold, path, best_f1):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'epoch': epoch,
        'fold': fold,
        'best_f1': best_f1
    }
    torch.save(checkpoint, path)
    print(f"Checkpoint saved at {path}")
    
    
def load_checkpoint(path, model, optimizer, scheduler):
    if os.path.exists(path):
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        fold = checkpoint['fold']  # Load the last completed fold
        best_f1 = checkpoint['best_f1']
        print(f"Checkpoint loaded from {path}")
        return start_epoch, fold, best_f1, True
    return 0, 0, 0.0, False

def find_latest_checkpoint(model_output_dir):
    """
    Find the latest available checkpoint in the directory.
    Looks for checkpoints named in the format 'checkpoint_fold_X.pth' where X is the fold number.
    """
    for fold in range(5, 0, -1):  # Check from fold_5 to fold_1
        checkpoint_path = os.path.join(model_output_dir, f"checkpoint_fold_{fold}.pth")
        if os.path.exists(checkpoint_path):
            print(f"Found checkpoint: {checkpoint_path}")
            return checkpoint_path, fold
    print("No checkpoint found. Starting training from scratch.")
    return None, None

np.random.seed(0)
random.seed(0)
torch.manual_seed(0)


csv_path = r"/pfs/work7/workspace/scratch/ma_mkleinma-thesis/training_splits/grouped_data.csv"
image_folder = r"/pfs/work7/workspace/scratch/ma_mkleinma-thesis/rsna-pneumonia-detection-challenge/stage_2_train_images"
splits_path = r"/pfs/work7/workspace/scratch/ma_mkleinma-thesis/training_splits/splits_balanced_fix.pkl"
cm_output_dir = r"/pfs/work7/workspace/scratch/ma_mkleinma-thesis/trained_models/30_epochs_baseline_transformer_b_16/seed_0/confusion_matrix"
model_output_dir = r"/pfs/work7/workspace/scratch/ma_mkleinma-thesis/trained_models/30_epochs_baseline_transformer_b_16/seed_0/"



data = pd.read_csv(csv_path)
with open(splits_path, 'rb') as f:
    splits = pickle.load(f)
    
class PneumoniaDataset(Dataset):
    def __init__(self, dataframe, image_folder, transform=None):
        self.data = dataframe
        self.image_folder = image_folder
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_path = os.path.join(self.image_folder, f"{row['patientId']}.dcm")
        label = row['Target']
        
        # Load DICOM file and convert to PIL image
        dicom = pydicom.dcmread(image_path)
        image = dicom.pixel_array
        image = Image.fromarray(image).convert("RGB")

        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(label, dtype=torch.long)


# Define transformations for the training and validation sets
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalize with ImageNet stats
])


device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

fold = 0

# Training loop        
# preemptive loading
model = torch.hub.load('B-cos/B-cos-v2', 'standard_simple_vit_b_patch16_224', pretrained=True)        
model.linear_head.linear = torch.nn.Linear(in_features=768, out_features=2, bias=True)
optimizer = optim.Adam(model.parameters(), lr=1e-5)  #adjusted to lower learning rate due to transformer

criterion = nn.CrossEntropyLoss()
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
start_epoch, start_fold, best_f1, checkpoint_exists = 0, 0, 0.0, False

latest_checkpoint_path, latest_fold = find_latest_checkpoint(model_output_dir)
if latest_checkpoint_path:
    start_epoch, start_fold, best_f1, checkpoint_exists = load_checkpoint(
        latest_checkpoint_path, model, optimizer, scheduler)


for current_fold, (train_idx, val_idx) in enumerate(splits):
    fold = current_fold + 1
    print(f"Training fold {fold}...")
    if checkpoint_exists and fold < start_fold:
        print(f"Skipping fold {current_fold} as it's already completed.")
        continue  # Skip completed folds
                
    #if we dont start from checkpoint: initialize new model to train
    if not checkpoint_exists or current_fold != start_fold:
        model = torch.hub.load('B-cos/B-cos-v2', 'standard_simple_vit_b_patch16_224', pretrained=True)        
        model.linear_head.linear = torch.nn.Linear(in_features=768, out_features=2, bias=True)
        optimizer = optim.Adam(model.parameters(), lr=1e-5)  
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
        model = model.to(device)
        checkpoint_exists = False

        
    log_dir = os.path.join(model_output_dir, f"tensorboard_logs_fold_{fold}")
    log_writer = SummaryWriter(log_dir=log_dir)
    

    # Split data for the current fold
    train_data = data.iloc[train_idx]
    val_data = data.iloc[val_idx]
    
    train_dataset = PneumoniaDataset(train_data, image_folder, transform=transform)
    val_dataset = PneumoniaDataset(val_data, image_folder, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    # Training and validation loop for each fold
    num_epochs = 30
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        if (epoch < start_epoch and start_epoch < num_epochs):
            print(f"Skipping epoch {epoch}, resuming from checkpoint at epoch {start_epoch}.")
            continue
        
        if fold == start_fold:
            checkpoint_exists = False 
            start_epoch = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        
        train_loss = running_loss / len(train_loader.dataset)
        train_accuracy = correct / total
        
        log_writer.add_scalar('Loss/Train', train_loss, epoch)
        log_writer.add_scalar('Accuracy/Train', train_accuracy, epoch)
        
        print(f"Training Accuracy: {train_accuracy:.4f}")

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        all_preds = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * images.size(0)
                preds = torch.argmax(outputs, dim=1)
                
                all_probs.extend(torch.softmax(outputs, dim=1)[:, 1].cpu().numpy().flatten())  
                all_preds.extend(preds.cpu().numpy().flatten())  
                all_labels.extend(labels.cpu().numpy().flatten())

                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_loss /= len(val_loader.dataset)
        val_accuracy = val_correct / val_total
        scheduler.step(val_loss)
        
        precision = precision_score(all_labels, all_preds)
        recall = recall_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds)
        auc = roc_auc_score(all_labels, all_probs)
        
        log_writer.add_scalar('Loss/Validation', val_loss, epoch)
        log_writer.add_scalar('Accuracy/Validation', val_accuracy, epoch)
        log_writer.add_scalar('Metrics/Precision', precision, epoch)
        log_writer.add_scalar('Metrics/Recall', recall, epoch)
        log_writer.add_scalar('Metrics/F1', f1, epoch)
        log_writer.add_scalar('Metrics/AUC', auc, epoch)
        
        current_lr = optimizer.param_groups[0]['lr']
        log_writer.add_scalar('Learning_Rate', current_lr, epoch)

        cm = confusion_matrix(all_labels, all_preds)
        class_names = ['No Pneumonia', 'Pneumonia']
        cm_figure = plot_confusion_matrix(cm, class_names)
        log_writer.add_figure('Confusion_Matrix', cm_figure, epoch)

        
        if (f1 > best_f1):
            best_f1 = f1
            torch.save(model.state_dict(), os.path.join(model_output_dir, f"pneumonia_detection_model_trans_base_bestf1_{fold}.pth"))
            cm_file_path = os.path.join(cm_output_dir, f"confusion_matrix_best_f1_{fold}.json")
            with open(cm_file_path, 'w') as cm_file:
                json.dump({'confusion_matrix': cm.tolist()}, cm_file, indent=4)
            print(f"Confusion Matrix for Fold {fold} saved at {cm_file_path}")

        if epoch == num_epochs - 1:
            cm = confusion_matrix(all_labels, all_preds)
            cm_file_path = os.path.join(cm_output_dir, f"confusion_matrix_fold_{fold}.json")
            with open(cm_file_path, 'w') as cm_file:
                json.dump({'confusion_matrix': cm.tolist()}, cm_file, indent=4)
            print(f"Confusion Matrix for Fold {fold} saved at {cm_file_path}")
            
        save_checkpoint_path = os.path.join(model_output_dir, f"checkpoint_fold_{fold}.pth")
        save_checkpoint(model, optimizer, scheduler, epoch, fold, save_checkpoint_path, best_f1)

        print(f"Fold {fold}, Epoch {epoch+1}/{num_epochs}, "
            f"Val Acc: {val_accuracy:.4f}, "
            f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")
        
    print(f"Finished training fold {fold}.\n")
    
    model_path = f"pneumonia_detection_model_fold_{fold}_transformer_baseline.pth"
    torch.save(model.state_dict(),  os.path.join(model_output_dir, model_path))
    log_writer.close()



