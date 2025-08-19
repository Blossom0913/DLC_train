import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import os
from datetime import datetime


from data_load import *

# Custom dataset class
class MouseDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Modified 1D CNN model for single time-step classification
class BehaviorCNN(nn.Module):
    def __init__(self, input_size=8, num_classes=4):
        super(BehaviorCNN, self).__init__()
        # Since we're not using windowing, we'll treat the 8 features as a single "sequence"
        # and use 1D convolutions to learn feature interactions
        self.conv1 = nn.Conv1d(1, 64, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        # x shape: (batch_size, input_size) - reshape to (batch_size, 1, input_size)
        x = x.unsqueeze(1)  # Add channel dimension
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool(x).squeeze(-1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

def save_model(model, experiment_type):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"cnn_{experiment_type}_{timestamp}.pth"
    torch.save(model.state_dict(), model_name)
    print(f"Model saved as {model_name}")
    return model_name

# Update the compute_metrics function
def compute_metrics(y_true, y_pred, num_classes):
    acc = 100 * accuracy_score(y_true, y_pred)
    weighted_f1 = 100 * f1_score(y_true, y_pred, average='weighted')
    macro_f1 = 100 * f1_score(y_true, y_pred, average='macro')
    
    # Calculate per-class F1 scores
    class_f1 = f1_score(y_true, y_pred, average=None, labels=range(num_classes)) * 100
    best_class_f1 = np.max(class_f1)
    worst_class_f1 = np.min(class_f1)
    
    return {
        'accuracy': acc,
        'weighted_f1': weighted_f1,
        'macro_f1': macro_f1,
        'best_class_f1': best_class_f1,
        'worst_class_f1': worst_class_f1,
        'class_f1': class_f1
    }

def run_experiment(X_raw, y_raw, batch_size, epochs, device, 
                   num_classes, include_base=True, experiment_name="cnn"):
    # Ensure X_raw and y_raw have the same length
    if len(X_raw) != len(y_raw):
        min_len = min(len(X_raw), len(y_raw))
        print(f"Warning: X_raw and y_raw have different lengths ({len(X_raw)} vs {len(y_raw)}). Trimming to {min_len}.")
        X_raw = X_raw[:min_len]
        y_raw = y_raw[:min_len]
    
    # Data preparation
    mask = np.isin(y_raw, [0, 1, 2, 3]) if include_base else np.isin(y_raw, [1, 2, 3])
    X_raw_exp, y_raw_exp = X_raw[mask], y_raw[mask]
    
    # Rest of the code remains the same...
    if not include_base: 
        y_raw_exp -= 1
    
    # Standardize features directly (no windowing)
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw_exp)
    
    # Split data using random stratified splitting (70% train, 15% val, 15% test)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y_raw_exp, test_size=0.3, random_state=42, stratify=y_raw_exp)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
    
    # Create datasets
    train_dataset = MouseDataset(X_train, y_train)
    val_dataset = MouseDataset(X_val, y_val)
    test_dataset = MouseDataset(X_test, y_test)
    
    # Create dataloaders
    loaders = {
        'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
        'val': DataLoader(val_dataset, batch_size=batch_size, shuffle=False),
        'test': DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    }
    
    # Model setup
    model = BehaviorCNN(input_size=X_raw.shape[1], num_classes=num_classes).to(device)
    classes = np.unique(y_raw_exp)
    class_weights = compute_class_weight('balanced', classes=classes, y=y_raw_exp)
    criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(class_weights)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    
    # Training loop
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_val_acc = 0
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        for inputs, labels in loaders['train']:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        train_acc = 100 * train_correct / train_total
        history['train_loss'].append(train_loss / len(loaders['train']))
        history['train_acc'].append(train_acc)
        
        # Validation phase
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for inputs, labels in loaders['val']:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_acc = 100 * val_correct / val_total
        history['val_loss'].append(val_loss / len(loaders['val']))
        history['val_acc'].append(val_acc)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model_path = save_model(model, f"{experiment_name}_{num_classes}class")
        
        print(f"Epoch {epoch+1}/{epochs} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")
    
    # Load best model for testing
    model.load_state_dict(torch.load(model_path))
    
    # Test evaluation
    test_loss, test_acc, test_f1, class_acc, test_metrics = evaluate_model(
        model, loaders['test'], criterion, device, num_classes
    )

    
    
    # Update the benchmark printing
    print("\n=== FINAL BENCHMARK RESULTS ===")
    print(f"Accuracy (%): {test_metrics['accuracy']:.2f}")
    print(f"Weighted F1: {test_metrics['weighted_f1']:.2f}")
    print(f"Macro F1: {test_metrics['macro_f1']:.2f}")
    print(f"Best Class F1: {test_metrics['best_class_f1']:.2f}")
    print(f"Worst Class F1: {test_metrics['worst_class_f1']:.2f}")
    print("Per-class F1 Scores:")
    for i, f1 in enumerate(test_metrics['class_f1']):
        print(f"  Class {i}: {f1:.2f}%")
    
    # Print detailed classification report
    print("\nClassification Report (test):")
    print(classification_report(test_metrics['y_true'], test_metrics['y_pred'], digits=4))
    
    return {
        'history': history,
        'test_metrics': test_metrics,
        'model_path': model_path
    }

def evaluate_model(model, loader, criterion, device, num_classes):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0
    class_correct = [0] * num_classes
    class_total = [0] * num_classes
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Update class-wise counts
            for i in range(num_classes):
                idx = (labels == i)
                class_correct[i] += (preds[idx] == labels[idx]).sum().item()
                class_total[i] += idx.sum().item()
    
    # Calculate metrics
    avg_loss = total_loss / len(loader)
    accuracy = 100 * accuracy_score(all_labels, all_preds)
    f1 = 100 * f1_score(all_labels, all_preds, average='weighted')
    class_accuracy = [100 * class_correct[i] / class_total[i] if class_total[i] > 0 else 0 
                      for i in range(num_classes)]
    
    test_metrics = compute_metrics(
        y_true=np.array(all_labels), y_pred=np.array(all_preds), num_classes=num_classes
    )
    
    # Add y_true and y_pred for classification report
    test_metrics['y_true'] = all_labels
    test_metrics['y_pred'] = all_preds
    
    return avg_loss, accuracy, f1, class_accuracy, test_metrics

def main():
    # Hyperparameters
    CONFIG = {
        'batch_size': 256,
        'epochs': 50,
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    }
    
    # Load data
    X_raw, y_raw = load_mouse_data(
        r"E:\2025Fall\DeepLabVideo\code\source\data\dataset58\feature8_58.xlsx",
        r"E:\2025Fall\DeepLabVideo\code\source\data\dataset58\merged_labels.xlsx"
    )
    
    # Run experiments
    results = {}
    for include_base, num_classes in [(True, 4), (False, 3)]:
        exp_type = f"{num_classes}class"
        print(f"\n=== STARTING {exp_type.upper()} EXPERIMENT ===")
        history, model_path = run_experiment(
            X_raw, y_raw, 
            experiment_name="cnn",
            num_classes=num_classes,
            include_base=include_base,
            **CONFIG
        )
        results[exp_type] = {
            'history': history,
            'model_path': model_path
        }

if __name__ == "__main__":
    main()