import os
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import torch
import torch.nn as nn
import torch.optim as optim
from data_loader import create_data_loaders
from model import ConvNet
from train import train_model, evaluate_model
from utils import calculate_metrics, plot_metric, save_model_weights, get_device, plot_all_metrics
import sys

def create_and_train():
    script_dir = os.path.dirname(os.path.realpath(__file__))
    data_dir = './data'
    batch_size = 32
    num_classes = 2
    num_epochs = 10
    patience = 10
    learning_rate = 0.001
    
    device = get_device()
    train_loader, val_loader, test_loader = create_data_loaders(data_dir, batch_size, device)
    
    model = ConvNet(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    model, train_losses, val_losses, train_accuracies, val_accuracies, metrics = train_model(
        model, train_loader, val_loader, criterion, optimizer, device, num_epochs, patience)
    
    test_loss, test_accuracy = evaluate_model(model, test_loader, criterion, device)
    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')
    
    save_model_weights(model, os.path.join(script_dir, 'bone_fracture_detection_model_weights.pth'))
    print("Model weights saved successfully!")

    # Plot metrics
    plot_all_metrics(train_losses, val_losses, train_accuracies, val_accuracies, [metrics['accuracy'], metrics['recall'], metrics['specificity'], metrics['f1'], metrics['auc']], script_dir)


def test_model():
    # Load model
    model = ConvNet(num_classes=2)
    model.load_state_dict(torch.load('./models/bone_fracture_detection_model_weights.pth'))
    model.eval()

    # Load data
    data_dir = './data'
    batch_size = 32
    device = get_device()
    _, _, test_loader = create_data_loaders(data_dir, batch_size, device)

    # Evaluate model
    criterion = nn.CrossEntropyLoss()
    test_loss, test_accuracy = evaluate_model(model, test_loader, criterion, device)
    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')

    # Calculate metrics
    test_predictions = []
    test_targets = []
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            test_predictions.extend(predicted.tolist())
            test_targets.extend(labels.tolist())
    accuracy, recall, specificity, f1, auc = calculate_metrics(test_targets, test_predictions)
    print(f'Accuracy: {accuracy:.4f}, Recall: {recall:.4f}, Specificity: {specificity:.4f}, F1 Score: {f1:.4f}, AUC: {auc:.4f}')

    # Plot confusion matrix
    cm = confusion_matrix(test_targets, test_predictions)
    plt.figure()
    sys.heatmap(cm, annot=True, fmt='g', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig('./confusion_matrix.png', dpi=300, facecolor='white')
    plt.close()

    print("Model evaluation completed successfully!")
    


if __name__ == '__main__':
    test_model()
