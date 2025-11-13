import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils import training_device, read_model, save_model, PredictionStatistics
from model import Classifier
from dataset import NetworkDataset

import os
from tqdm import tqdm
import json
import copy
import matplotlib.pyplot as plt

# Parameters
packet_length = 1456
num_class = 5
num_epoch = 100
batch_size = 32
hidden_dim = 128
learning_rate = 0.0001
tolerate_epoch = 5

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_dir = os.path.join(base_dir, 'model')
data_path = os.path.join(base_dir, 'data', 'data.json')

def main():
    # Create directory
    os.makedirs(model_dir, exist_ok=True)
    
    # Determine the device
    device = training_device()

    dataset_train = NetworkDataset(data_path, packet_length=packet_length, split_ratio=(0, 0.9))
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    dataset_test = NetworkDataset(data_path, packet_length=packet_length, split_ratio=(0.9, 1))
    dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)

    # Define model and optimizer
    model = Classifier(packet_length, hidden_dim, num_class).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # read model
    epoch, test_loss = read_model(os.path.join(model_dir, 'task1_latest.pt'), model, optimizer)
    best_f1_score = -test_loss[0]
    best_epoch = 0
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
        print(f'epoch: {epoch}  precision: {precision.mean() * 100:.2f}%  recall: {recall.mean() * 100:.2f}%  accuracy: {accuracy.mean() * 100:.2f}%  f1_score: {f1_score.mean() * 100:.2f}%')
        stat_log.append({'epoch': epoch, 'precision': precision, 'recall': recall, 'accuracy': accuracy, 'f1_score': f1_score})
        test_loss.append(-f1_score.mean().item())

        # Save model
        save_model(os.path.join(model_dir, 'task1_latest.pt'), model, optimizer, epoch, test_loss, False)
        if f1_score.mean() > best_f1_score:
            tolerate_count = 0
            best_f1_score = f1_score.mean()
            best_epoch = epoch
            save_model(os.path.join(model_dir, 'task1_best.pt'), model, None, epoch, test_loss, False)
        else:
            tolerate_count += 1
            if tolerate_count >= tolerate_epoch:
                break

        epoch = epoch + 1

    # Save statistics
    with open(os.path.join(model_dir, 'task1_stat.json'), 'w') as f:
        stat_log_json = copy.deepcopy(stat_log)
        for item in stat_log_json:
            item['precision'] = list(item['precision'])
            item['recall'] = list(item['recall'])
            item['accuracy'] = list(item['accuracy'])
            item['f1_score'] = list(item['f1_score'])
        json.dump(stat_log_json, f, indent=4)
    
    # Print best statistics
    print(f'\nBest epoch: {best_epoch}')
    best_stat = None
    for i in range(len(stat_log)):
        if stat_log[i]['epoch'] == best_epoch:
            best_stat = stat_log[i]
            break
    if best_stat is not None:
        for i in range(num_class):
            print(f'\nClass {i}: {dataset_test.labels[i]}')
            print(f'    Precision: {best_stat["precision"][i] * 100:.2f}%')
            print(f'    Recall: {best_stat["recall"][i] * 100:.2f}%')
            print(f'    Accuracy: {best_stat["accuracy"][i] * 100:.2f}%')
            print(f'    F1 Score: {best_stat["f1_score"][i] * 100:.2f}%')
    else:
        print('No best stat found')
    
    # Plot statistics
    plt.figure(figsize=(8, 5))
    epoch_list = [item['epoch'] for item in stat_log]
    precision_list = [item['precision'].mean() * 100 for item in stat_log]
    recall_list = [item['recall'].mean() * 100 for item in stat_log]
    accuracy_list = [item['accuracy'].mean() * 100 for item in stat_log]
    f1_score_list = [item['f1_score'].mean() * 100 for item in stat_log]
    plt.plot(epoch_list, precision_list, label='precision (%)', color='red')
    plt.plot(epoch_list, recall_list, label='recall (%)', color='green')
    plt.plot(epoch_list, accuracy_list, label='accuracy (%)', color='blue')
    plt.plot(epoch_list, f1_score_list, label='f1_score (%)', color='purple')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.title('Task 1 Statistics')
    plt.legend()
    plt.savefig(os.path.join(model_dir, 'task1_stat.png'))
    plt.show()

if __name__ == '__main__':
    main()
