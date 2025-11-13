import torch
from torch.utils.data import DataLoader

from utils import training_device, read_model, save_model, PredictionStatistics, calculate_gradient_penalty
from model import Classifier, Generator, Discriminator
from dataset import NetworkDataset

import os
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
import copy

# Parameters
packet_length = 1456
num_class = 5
num_epoch = 100
batch_size = 64
latent_size = 91
hidden_dim = 128
learning_rate_G = 0.0001
learning_rate_D = 0.0004
tolerate_epoch = 5
lambda_gp = 10
data_label = 3
data_label_str = 'youtube'

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_dir = os.path.join(base_dir, 'model')
data_path = os.path.join(base_dir, 'data', 'data_youtube.json')

def main():
    # Create directory
    os.makedirs(model_dir, exist_ok=True)
    
    # Determine the device
    device = training_device()

    dataset_train = NetworkDataset(data_path, packet_length=packet_length)
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)

    # Define model and optimizer
    G = Generator(latent_size).to(device)
    D = Discriminator(packet_length).to(device)
    G_optimizer = torch.optim.Adam(G.parameters(), lr=learning_rate_G, betas=(0.0, 0.9))
    D_optimizer = torch.optim.Adam(D.parameters(), lr=learning_rate_D, betas=(0.0, 0.9))
    classifier1 = Classifier(packet_length, hidden_dim, num_class).to(device)
    classifier2 = Classifier(packet_length, hidden_dim, num_class).to(device)

    # read model
    _, _ = read_model(os.path.join(model_dir, 'task1_best.pt'), classifier1)
    _, _ = read_model(os.path.join(model_dir, 'task2_best.pt'), classifier2)
    classifier1.eval()
    classifier2.eval()
    epoch, test_loss = read_model(os.path.join(model_dir, 'task3_latest.pt'), [G, D], [G_optimizer, D_optimizer])
    best_f1_score = -test_loss[0]
    best_epoch = 0
    tolerate_count = 0

    # Define statistics
    statistics1 = PredictionStatistics(num_class)
    statistics2 = PredictionStatistics(num_class)
    stat_log1 = []
    stat_log2 = []

    while True:
        # Train model
        G.train()
        D.train()
        progress_bar = tqdm(dataloader_train, desc=f'Training (epoch: {epoch})')
        for real_packet, _ in progress_bar:
            # Real packet loss
            real_packet = real_packet.to(device)
            D_loss_real = D(real_packet).mean()
            # Fake packet loss
            z = torch.randn(real_packet.shape[0], latent_size).to(device)
            fake_packet = G(z)
            D_loss_fake = D(fake_packet).mean()
            # Gradient penalty
            gradient_penalty = calculate_gradient_penalty(real_packet, fake_packet, D, device)
            # D loss
            D_loss = -D_loss_real + D_loss_fake + gradient_penalty * lambda_gp
            # Optimize D
            D_optimizer.zero_grad()
            D_loss.backward()
            D_optimizer.step()

            # Fake packet loss
            z = torch.randn(batch_size, latent_size).to(device)
            fake_packet = G(z)
            G_loss = -D(fake_packet).mean()
            # Optimize G
            G_optimizer.zero_grad()
            G_loss.backward()
            G_optimizer.step()

            # Print information
            progress_bar.set_postfix({'D_loss': D_loss.item(), 'G_loss': G_loss.item()})

        # Test model
        G.eval()
        statistics1.reset()
        statistics2.reset()
        with torch.no_grad():
            label = torch.full((batch_size,), data_label, dtype=torch.int64)
            for i in tqdm(range(10), desc='Testing'):
                z = torch.randn(batch_size, latent_size).to(device)
                packet = G(z).detach()
                packet = torch.round((packet + 1) * 127.5) / 127.5 - 1
                pred1 = classifier1(packet)
                pred2 = classifier2(packet)
                pred1_label = torch.argmax(pred1, dim=1)
                pred2_label = torch.argmax(pred2, dim=1)
                statistics1.add(pred1_label, label)
                statistics2.add(pred2_label, label)
        precision1 = statistics1.get_precision()
        recall1 = statistics1.get_recall()
        accuracy1 = statistics1.get_accuracy()
        f1_score1 = statistics1.get_f1_score()
        precision2 = statistics2.get_precision()
        recall2 = statistics2.get_recall()
        accuracy2 = statistics2.get_accuracy()
        f1_score2 = statistics2.get_f1_score()
        precision = (precision1 + precision2) / 2
        recall = (recall1 + recall2) / 2
        accuracy = (accuracy1 + accuracy2) / 2
        f1_score = (f1_score1 + f1_score2) / 2
        print(f'epoch: {epoch}  precision: {precision[data_label] * 100:.2f}%  recall: {recall[data_label] * 100:.2f}%  accuracy: {accuracy[data_label] * 100:.2f}%  f1_score: {f1_score[data_label] * 100:.2f}%')
        stat_log1.append({'epoch': epoch, 'precision': precision1, 'recall': recall1, 'accuracy': accuracy1, 'f1_score': f1_score1})
        stat_log2.append({'epoch': epoch, 'precision': precision2, 'recall': recall2, 'accuracy': accuracy2, 'f1_score': f1_score2})
        test_loss.append(-f1_score[data_label].item())

        # Save model
        save_model(os.path.join(model_dir, 'task3_latest.pt'), [G, D], [G_optimizer, D_optimizer], epoch, test_loss, False)
        if f1_score[data_label] > best_f1_score:
            tolerate_count = 0
            best_f1_score = f1_score[data_label]
            best_epoch = epoch
            save_model(os.path.join(model_dir, 'task3_best.pt'), [G], None, epoch, test_loss, False)
        else:
            tolerate_count += 1
            if tolerate_count >= tolerate_epoch:
                break

        epoch = epoch + 1

    # Save statistics
    with open(os.path.join(model_dir, 'task3_stat1.json'), 'w') as f:
        stat_log1_json = copy.deepcopy(stat_log1)
        for item in stat_log1_json:
            item['precision'] = list(item['precision'])
            item['recall'] = list(item['recall'])
            item['accuracy'] = list(item['accuracy'])
            item['f1_score'] = list(item['f1_score'])
        json.dump(stat_log1_json, f, indent=4)
    with open(os.path.join(model_dir, 'task3_stat2.json'), 'w') as f:
        stat_log2_json = copy.deepcopy(stat_log2)
        for item in stat_log2_json:
            item['precision'] = list(item['precision'])
            item['recall'] = list(item['recall'])
            item['accuracy'] = list(item['accuracy'])
            item['f1_score'] = list(item['f1_score'])
        json.dump(stat_log2_json, f, indent=4)
    
    # Print best statistics
    print(f'\nBest epoch: {best_epoch}')
    best_stat1, best_stat2 = None, None
    for i in range(len(stat_log1)):
        if stat_log1[i]['epoch'] == best_epoch:
            best_stat1 = stat_log1[i]
            best_stat2 = stat_log2[i]
            break
    if best_stat1 is not None and best_stat2 is not None:
        print(f'\nClass {data_label}: {data_label_str} (Classifier 1)')
        print(f'    Precision: {best_stat1["precision"][data_label] * 100:.2f}%')
        print(f'    Recall: {best_stat1["recall"][data_label] * 100:.2f}%')
        print(f'    Accuracy: {best_stat1["accuracy"][data_label] * 100:.2f}%')
        print(f'    F1 Score: {best_stat1["f1_score"][data_label] * 100:.2f}%')
        print(f'\nClass {data_label}: {data_label_str} (Classifier 2)')
        print(f'    Precision: {best_stat2["precision"][data_label] * 100:.2f}%')
        print(f'    Recall: {best_stat2["recall"][data_label] * 100:.2f}%')
        print(f'    Accuracy: {best_stat2["accuracy"][data_label] * 100:.2f}%')
        print(f'    F1 Score: {best_stat2["f1_score"][data_label] * 100:.2f}%')
    else:
        print('No best stat found')
    
    # Plot statistics
    plt.figure(figsize=(14, 5))
    epoch_list1 = [item['epoch'] for item in stat_log1]
    precision_list1 = [item['precision'][data_label] * 100 for item in stat_log1]
    recall_list1 = [item['recall'][data_label] * 100 for item in stat_log1]
    accuracy_list1 = [item['accuracy'][data_label] * 100 for item in stat_log1]
    f1_score_list1 = [item['f1_score'][data_label] * 100 for item in stat_log1]
    epoch_list2 = [item['epoch'] for item in stat_log2]
    precision_list2 = [item['precision'][data_label] * 100 for item in stat_log2]
    recall_list2 = [item['recall'][data_label] * 100 for item in stat_log2]
    accuracy_list2 = [item['accuracy'][data_label] * 100 for item in stat_log2]
    f1_score_list2 = [item['f1_score'][data_label] * 100 for item in stat_log2]
    plt.subplot(1, 2, 1)
    plt.plot(epoch_list1, precision_list1, label='precision (%)', color='red')
    plt.plot(epoch_list1, recall_list1, label='recall (%)', color='green')
    plt.plot(epoch_list1, accuracy_list1, label='accuracy (%)', color='blue')
    plt.plot(epoch_list1, f1_score_list1, label='f1_score (%)', color='purple')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.title('Task 3 Statistics 1')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(epoch_list2, precision_list2, label='precision (%)', color='red')
    plt.plot(epoch_list2, recall_list2, label='recall (%)', color='green')
    plt.plot(epoch_list2, accuracy_list2, label='accuracy (%)', color='blue')
    plt.plot(epoch_list2, f1_score_list2, label='f1_score (%)', color='purple')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.title('Task 3 Statistics 2')
    plt.legend()
    plt.savefig(os.path.join(model_dir, 'task3_stat.png'))
    plt.show()

if __name__ == '__main__':
    main()
