import os
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score, confusion_matrix

def save_model_weights(model, path):
    torch.save(model.state_dict(), path)

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def calculate_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    specificity = recall_score(y_true, y_pred, pos_label=0)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred)
    
    return accuracy, recall, specificity, f1, auc

def plot_metric(metric_name, values, save_path):
    plt.figure()
    plt.plot(values)
    plt.title(f'{metric_name} over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel(metric_name)
    plt.grid(True)
    plt.savefig(save_path, dpi=300, facecolor='white')
    plt.close()

def plot_all_metrics(train_losses, val_losses, train_accuracies, val_accuracies, metrics, script_dir):
    plot_metric('Train Loss', train_losses, os.path.join(script_dir, 'train_loss.png'))
    plot_metric('Validation Loss', val_losses, os.path.join(script_dir, 'val_loss.png'))
    plot_metric('Train Accuracy', train_accuracies, os.path.join(script_dir, 'train_accuracy.png'))
    plot_metric('Validation Accuracy', val_accuracies, os.path.join(script_dir, 'val_accuracy.png'))
    
    metric_names = ['Accuracy', 'Recall', 'Specificity', 'F1 Score', 'AUC']
    for metric_name, values in zip(metric_names, metrics):
        plot_metric(metric_name, values, os.path.join(script_dir, f'{metric_name.lower().replace(" ", "_")}.png'))
