import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils import training_device, read_model, save_model, PredictionStatistics
from model import Classifier
from dataset import NetworkDataset

import os
from tqdm import tqdm
import json
import matplotlib.pyplot as plt

# Parameters
num_class = 5
num_epoch = 100
batch_size = 64
in_dim = 512
hidden_dim = 128
out_dim = num_class
learning_rate = 0.0001
tolerate_epoch = 5

base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(base_path, 'model')
data_path = os.path.join(base_path, 'data', 'data.json')

def main():
    # determine the device
    device = training_device()

    dataset_train = NetworkDataset(data_path, packet_length=in_dim, split_ratio=(0, 0.9))
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    dataset_test = NetworkDataset(data_path, packet_length=in_dim, split_ratio=(0.9, 1))
    dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)

    # Define model and optimizer
    model = Classifier(in_dim, hidden_dim, out_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # read model
    epoch, test_loss = read_model(os.path.join(model_path, 'task1.pt'), model, optimizer)
    best_f1_score = -test_loss[0]
    tolerate_count = 0

    # Define loss function
    loss_fn = nn.CrossEntropyLoss()

    # Define statistics
    statistics = PredictionStatistics(num_class)
    stat_log = []

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
        statistics.reset()
        with torch.no_grad():
            for packet, label in tqdm(dataloader_test, desc='Testing'):
                packet, label = packet.to(device), label.to(device)
                pred = model(packet)
                pred_label = torch.argmax(pred, dim=1)
                statistics.add(pred_label, label)
        precision = statistics.get_precision()
        recall = statistics.get_recall()
        accuracy = statistics.get_accuracy()
        f1_score = statistics.get_f1_score()
        print(f'epoch: {epoch}  test_precision: {precision * 100:.2f}%  test_recall: {recall * 100:.2f}%  test_accuracy: {accuracy * 100:.2f}%  test_f1_score: {f1_score * 100:.2f}%')
        stat_log.append([epoch, precision, recall, accuracy, f1_score])
        test_loss.append(-f1_score)

        # Save model
        save_model(os.path.join(model_path, 'task1_latest.pt'), model, optimizer, epoch, test_loss, False)
        if f1_score > best_f1_score:
            tolerate_count = 0
            best_f1_score = f1_score
            save_model(os.path.join(model_path, 'task1_best.pt'), model, None, epoch, test_loss, False)
        else:
            tolerate_count += 1
            if tolerate_count >= tolerate_epoch:
                break

        epoch = epoch + 1

    # Save statistics
    with open(os.path.join(model_path, 'task1_stat.json'), 'w') as f:
        json.dump(stat_log, f, indent=4)
    # Plot statistics
    plt.figure(figsize=(8, 5))
    epoch_list = [item[0] for item in stat_log]
    precision_list = [item[1] for item in stat_log]
    recall_list = [item[2] for item in stat_log]
    accuracy_list = [item[3] for item in stat_log]
    f1_score_list = [item[4] for item in stat_log]
    plt.plot(epoch_list, precision_list, label='precision', color='red')
    plt.plot(epoch_list, recall_list, label='recall', color='green')
    plt.plot(epoch_list, accuracy_list, label='accuracy', color='blue')
    plt.plot(epoch_list, f1_score_list, label='f1_score', color='purple')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.title('Task 1 Statistics')
    plt.legend()
    plt.savefig(os.path.join(model_path, 'task1_stat.png'))
    plt.show()

if __name__ == '__main__':
    main()
