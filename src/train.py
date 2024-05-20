import torch
import torch.optim as optim
from sklearn.metrics import accuracy_score
from model import ConvNet
from utils import calculate_metrics

def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=20, patience=5):
    best_val_loss = float('inf')
    counter = 0
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    metrics = {'accuracy': [], 'recall': [], 'specificity': [], 'f1': [], 'auc': []}

    for epoch in range(num_epochs):
        model.train()
        epoch_train_losses = []
        epoch_train_predictions = []
        epoch_train_targets = []
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_train_losses.append(loss.item())
            _, predicted = torch.max(outputs, 1)
            epoch_train_predictions.extend(predicted.tolist())
            epoch_train_targets.extend(labels.tolist())
        train_loss = sum(epoch_train_losses) / len(epoch_train_losses)
        train_accuracy = accuracy_score(epoch_train_targets, epoch_train_predictions)
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        
        model.eval()
        epoch_val_losses = []
        epoch_val_predictions = []
        epoch_val_targets = []
        with torch.no_grad():
            for images, labels in val_loader:
                outputs = model(images)
                loss = criterion(outputs, labels)
                epoch_val_losses.append(loss.item())
                _, predicted = torch.max(outputs, 1)
                epoch_val_predictions.extend(predicted.tolist())
                epoch_val_targets.extend(labels.tolist())
        val_loss = sum(epoch_val_losses) / len(epoch_val_losses)
        val_accuracy = accuracy_score(epoch_val_targets, epoch_val_predictions)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping triggered.")
                break

        # Calculate and store metrics for each epoch
        accuracy, recall, specificity, f1, auc = calculate_metrics(epoch_val_targets, epoch_val_predictions)
        metrics['accuracy'].append(accuracy)
        metrics['recall'].append(recall)
        metrics['specificity'].append(specificity)
        metrics['f1'].append(f1)
        metrics['auc'].append(auc)

    return model, train_losses, val_losses, train_accuracies, val_accuracies, metrics

def evaluate_model(model, test_loader, criterion, device):
    model.eval()
    test_losses = []
    test_predictions = []
    test_targets = []
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_losses.append(loss.item())
            _, predicted = torch.max(outputs, 1)
            test_predictions.extend(predicted.tolist())
            test_targets.extend(labels.tolist())
    test_loss = sum(test_losses) / len(test_losses)
    test_accuracy = accuracy_score(test_targets, test_predictions)
    return test_loss, test_accuracy
