import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score

from data_load import *

# 自定义数据集类
class MouseDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# 模型定义
class BehaviorLSTM(nn.Module):
    def __init__(self, input_size=8, hidden_size=128, num_layers=2, num_classes=4):
        super(BehaviorLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(hidden_size*2, 64)  # 双向输出需要乘以2
        self.fc2 = nn.Linear(64, num_classes)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.dropout(out)
        out = torch.relu(self.fc1(out))
        out = self.fc2(out)
        return out

def preprocess_data(X, y, window_length=10):
    X_windowed = []
    y_windowed = []
    for i in range(len(X) - window_length + 1):
        X_windowed.append(X[i:i+window_length])
        y_windowed.append(y[i+window_length-1])
    if len(X_windowed) == 0:
        raise ValueError("窗口长度大于输入数据长度")
    if X_windowed[0].shape != (window_length, X.shape[1]):
        raise ValueError(f"窗口形状异常，期望: {(window_length, X.shape[1])}，实际: {X_windowed[0].shape}")
    return np.array(X_windowed), np.array(y_windowed)

def plot_and_save_training_curves(history, save_path='training_curves.png'):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_acc'], label='Training Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title('Accuracy Curves')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(linestyle='--', alpha=0.5)
    plt.subplot(1, 2, 2)
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Loss Curves')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"训练曲线已保存至: {save_path}")

def calculate_class_accuracy(model, loader, device, num_classes=4):
    model.eval()
    class_correct = [0] * num_classes
    class_total = [0] * num_classes
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            for i in range(len(labels)):
                label = labels[i].item()
                pred = predicted[i].item()
                if 0 <= label < num_classes:
                    class_correct[label] += (pred == label)
                    class_total[label] += 1
    class_accuracy = []
    for i in range(num_classes):
        if class_total[i] > 0:
            class_accuracy.append(100 * class_correct[i] / class_total[i])
        else:
            class_accuracy.append(0.0)
    return class_accuracy

def evaluate_model(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    avg_loss = total_loss / len(test_loader)
    accuracy = 100 * np.sum(np.array(all_preds) == np.array(all_labels)) / len(all_labels)
    f1 = 100 * f1_score(all_labels, all_preds, average='weighted')
    return avg_loss, accuracy, f1

def run_lstm_experiment(X_raw, y_raw, window_length, batch_size, epochs, device, num_classes, include_base=True):
    # Mask and remap labels as before
    if include_base:
        print("\n=== 4-class experiment (base, aggression, social, nonsocial) ===")
        mask = np.isin(y_raw, [0, 1, 2, 3])
    else:
        print("\n=== 3-class experiment (aggression, social, nonsocial) ===")
        mask = np.isin(y_raw, [1, 2, 3])
    X_raw_exp = X_raw[mask]
    y_raw_exp = y_raw[mask]
    if not include_base:
        y_raw_exp = y_raw_exp - 1

    # Split indices for train/val/test
    total_len = len(X_raw_exp)
    train_end = int(total_len * 0.7)
    val_end = int(total_len * 0.85)

    X_train_raw, X_val_raw, X_test_raw = X_raw_exp[:train_end], X_raw_exp[train_end:val_end], X_raw_exp[val_end:]
    y_train_raw, y_val_raw, y_test_raw = y_raw_exp[:train_end], y_raw_exp[train_end:val_end], y_raw_exp[val_end:]

    # Windowing
    X_train_windows, y_train_windows = preprocess_data(X_train_raw, y_train_raw, window_length)
    X_val_windows, y_val_windows = preprocess_data(X_val_raw, y_val_raw, window_length)
    X_test_windows, y_test_windows = preprocess_data(X_test_raw, y_test_raw, window_length)

    # Standardize
    scaler = StandardScaler()
    num_features = X_train_windows.shape[-1]
    X_train_windows = scaler.fit_transform(X_train_windows.reshape(-1, num_features)).reshape(X_train_windows.shape)
    X_val_windows = scaler.transform(X_val_windows.reshape(-1, num_features)).reshape(X_val_windows.shape)
    X_test_windows = scaler.transform(X_test_windows.reshape(-1, num_features)).reshape(X_test_windows.shape)

    # DataLoader
    train_dataset = MouseDataset(X_train_windows, y_train_windows)
    val_dataset = MouseDataset(X_val_windows, y_val_windows)
    test_dataset = MouseDataset(X_test_windows, y_test_windows)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Model
    model = BehaviorLSTM(input_size=num_features, num_classes=num_classes).to(device)
    classes = np.arange(num_classes)
    criterion = nn.CrossEntropyLoss(
        weight=torch.FloatTensor(
            compute_class_weight('balanced', classes=classes, y=np.concatenate([y_train_windows, classes]))
        ).to(device)
    )

    # modified optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

    # Add these variables before the training loop
    best_val_loss = float('inf')
    patience = 10
    counter = 0

    # Training loop with validation
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    for epoch in range(epochs):
        model.train()
        train_loss, correct, total = 0, 0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        train_acc = 100 * correct / total
        history['train_loss'].append(train_loss / len(train_loader))
        history['train_acc'].append(train_acc)

        # Validation
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        val_acc = 100 * val_correct / val_total
        history['val_loss'].append(val_loss / len(val_loader))
        history['val_acc'].append(val_acc)


        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            torch.save(model.state_dict(), "best_model.pth")
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

        if (epoch+1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

    plot_and_save_training_curves(history, save_path=f'training_curves_{"4class" if include_base else "3class"}.png')

    # Final evaluation on test set
    avg_loss, accuracy, f1 = evaluate_model(model, test_loader, criterion, device)
    print(f"Test Accuracy: {accuracy:.2f}%")
    print(f"Test F1 Score: {f1:.2f}%")
    class_accuracy = calculate_class_accuracy(model, test_loader, device, num_classes)
    print("Per-class accuracy:")
    for i, acc in enumerate(class_accuracy):
        print(f"  Class {i + (0 if include_base else 1)}: {acc:.2f}%")

def main():
    WINDOW_LENGTH = 10
    BATCH_SIZE = 256
    EPOCHS = 50
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    feature_path = r"E:\2025Fall\DeepLabVideo\code\source\data\dataset58\feature8_58.xlsx"
    label_path = r"E:\2025Fall\DeepLabVideo\code\source\data\dataset58\merged_labels.xlsx"
    X_raw, y_raw = load_mouse_data(feature_path, label_path)

    # Ensure X_raw and y_raw have the same length
    if len(X_raw) != len(y_raw):
        min_len = min(len(X_raw), len(y_raw))
        print(f"Warning: X_raw and y_raw have different lengths ({len(X_raw)} vs {len(y_raw)}). Trimming to {min_len}.")
        X_raw = X_raw[:min_len]
        y_raw = y_raw[:min_len]

    # 4-class experiment
    run_lstm_experiment(X_raw, y_raw, WINDOW_LENGTH, batch_size=BATCH_SIZE, epochs=EPOCHS, device=DEVICE, num_classes=4, include_base=True)

    # 3-class experiment
    run_lstm_experiment(X_raw, y_raw, WINDOW_LENGTH, batch_size=BATCH_SIZE, epochs=EPOCHS, device=DEVICE, num_classes=3, include_base=False)

if __name__ == "__main__":
    main()