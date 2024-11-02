import torch
import torch.nn as nn
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
from plot import ConfusionMatrixPlotter


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=2, padding=0)  # Output: (32, 6, 2, 8)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)       # Output: (32, 6, 1, 4)
        
        self.conv2 = nn.Conv2d(6, 16, kernel_size=(1, 2), padding=0)  # Output: (32, 16, 1, 3)
        self.pool2 = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))  # Output: (32, 16, 1, 1)

        # Correcting the input size for the fully connected layer
        self.fc1 = nn.Linear(16 * 1 * 1, 120)  # Input size is now 16
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 5)  # Output for 5 categories

    def forward(self, x):
        x = self.pool1(torch.relu(self.conv1(x)))  # After pool1: (32, 6, 1, 4)
        x = self.pool2(torch.relu(self.conv2(x)))  # After pool2: (32, 16, 1, 1)
        
        x = x.view(x.size(0), -1)  # Flatten for fully connected layers
        x = torch.relu(self.fc1(x))  # Input size now matches (32, 16)
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)  # Final output
        return x

def train(model, device, train_loader, optimizer, criterion, epoch, train_losses, train_accuracies):
    model.train()
    correct = 0
    total_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

    train_losses.append(total_loss / len(train_loader))
    train_accuracies.append(100. * correct / len(train_loader.dataset))

def val(model, device, val_loader, criterion, val_losses, val_accuracies, val_precisions, val_recalls, val_f1s):
    model.eval()
    val_loss = 0
    correct = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            val_loss += criterion(output, target).item()

            pred = output.argmax(dim=1, keepdim=True)
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

            correct += pred.eq(target.view_as(pred)).sum().item()

    val_loss /= len(val_loader)
    accuracy = 100. * correct / len(val_loader.dataset)

    # Calculate precision, recall, and F1-score
    precision = precision_score(all_targets, all_preds, average='weighted')
    recall = recall_score(all_targets, all_preds, average='weighted')
    f1 = f1_score(all_targets, all_preds, average='weighted')

    val_losses.append(val_loss)
    val_accuracies.append(accuracy)
    val_precisions.append(precision)
    val_recalls.append(recall)
    val_f1s.append(f1)

    print('\nval set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'
          'Precision: {:.2f}, Recall: {:.2f}, F1-score: {:.2f}\n'.format(
        val_loss, correct, len(val_loader.dataset), accuracy, precision, recall, f1))
    return accuracy

def test(model, device, val_loader, labels, save_path=None):
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)

            pred = output.argmax(dim=1, keepdim=True)
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

    cm = confusion_matrix(all_targets, all_preds)
    cm_plotter = ConfusionMatrixPlotter(labels=labels, cm=cm, model_name="LeNet")
    cm_plotter.plot(save_path)

