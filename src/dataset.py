import torch
from torch.utils.data import Dataset

import json
import base64

class NetworkDataset(Dataset):
    def __init__(self, data_path, packet_length=512, split_ratio=(0, 1)):
        self.packet_length = packet_length
        self.split_ratio = split_ratio
        with open(data_path, 'r') as f:
            self.raw_data = json.load(f)
        self.labels = self.raw_data.keys()
        self.data = self._preprocess_data()
    
    def _preprocess_data(self):
        data = []
        for i in range(len(self.labels)):
            label = self.labels[i]
            start_idx = int(len(self.raw_data[label]) * self.split_ratio[0])
            end_idx = int(len(self.raw_data[label]) * self.split_ratio[1])
            for str_packet in self.raw_data[label][start_idx:end_idx]:
                # Decode and truncate to packet length
                raw_packet = base64.b64decode(str_packet)[:self.packet_length]
                # Pad with zeros if packet length is less than packet_length
                raw_packet = raw_packet.rjust(self.packet_length, b'\x00')
                # Convert each byte to float -1 to 1
                packet = [(b / 127.5 - 1.0) for b in raw_packet]
                # Convert to tensor
                packet_tensor = torch.tensor(packet, dtype=torch.float32)
                label_tensor = torch.tensor(i, dtype=torch.int64)
                data.append((packet_tensor, label_tensor))
        return data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        packet, label = self.data[idx]
        return packet, label
