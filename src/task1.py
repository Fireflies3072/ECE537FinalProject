import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils import training_device, read_model, save_model
from model import Classifier
from dataset import NetworkDataset

import os
from tqdm import tqdm

# Parameters
num_epoch = 100
batch_size = 64
in_dim = 512
hidden_dim = 128
out_dim = 5

base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(base_path, 'model')

def main():
    # determine the device
    device = training_device()

    dataset_train = NetworkDataset()
    dataloader_train = DataLoader(dataset_train, batch_size)
    dataset_test = NetworkDataset()
    dataloader_test = DataLoader(dataset_test, batch_size)

    # Define model and optimizer
    model = Classifier(in_dim, hidden_dim, out_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    # read model
    epoch, test_loss = read_model(os.path.join(model_path, 'task1.pt'), model, optimizer)
    best_accuracy = -test_loss[0]

    # Define loss function
    loss_fn = nn.CrossEntropyLoss()

    while True:
        # Train model
        model.train()
        progress_bar = tqdm(dataloader_train, desc=f'Training (epoch: {epoch})')
        for packet, label in progress_bar:
            packet, label = packet.to(device), label.to(device)
            pred = model(packet)
            pred_label = torch.argmax(pred, dim=1)
            loss = loss_fn(pred, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            progress_bar.set_postfix({'loss': loss.item()})

        # Test model
        model.eval()
        correct = 0
        total = len(dataset_test)
        with torch.no_grad():
            for packet, label in tqdm(dataloader_test, desc='Testing'):
                packet, label = packet.to(device), label.to(device)
                pred = model(packet)
                pred_label = torch.argmax(pred, dim=1)
                correct += len(label[pred_label == label])
        accuracy = correct / total
        print(f'epoch: {epoch}  test_accuracy: {accuracy * 100:.2f}%')

        # Save model
        save_model(os.path.join(model_path, 'task1_latest.pt'), model, optimizer, epoch, test_loss, False)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            save_model(os.path.join(model_path, 'task1_best.pt'), model, None, epoch, test_loss, False)

        test_loss.append(-accuracy)
        epoch = epoch + 1

if __name__ == '__main__':
    main()
