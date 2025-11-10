import os
import torch
import platform
import numpy as np

# Device and training utils
def training_device(cuda=True):
    # Basic platform info
    print(f'Operating system: {platform.platform()}')
    # CPU info via platform where available (best-effort)
    try:
        processor = platform.processor() or platform.machine()
        if processor:
            print(f'CPU: {processor}')
    except Exception:
        pass
    # CUDA availability
    if cuda:
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            print('Support CUDA')
            print(f'Current CUDA device: {torch.cuda.get_device_name(torch.cuda.current_device())}')
        else:
            print('Not support CUDA')
        print()
        return torch.device('cuda' if cuda_available else 'cpu')
    else:
        print()
        return torch.device('cpu')


def save_model(path, model, optimizer=None, epoch=None, test_loss=None, save_simplied_model=True):
    os.makedirs(os.path.split(path)[0], exist_ok=True)
    if isinstance(model, (list, tuple)):
        model_state_dict = [m.state_dict() for m in model]
    else:
        model_state_dict = [model.state_dict()]
    state = {'model': model_state_dict}
    if optimizer is not None:
        if isinstance(optimizer, (list, tuple)):
            optimizer_state_dict = [o.state_dict() for o in optimizer]
        else:
            optimizer_state_dict = [optimizer.state_dict()]
        state['optimizer'] = optimizer_state_dict
    if epoch is not None:
        state['epoch'] = epoch
    if test_loss is not None:
        if isinstance(test_loss, list):
            test_loss = min(test_loss)
        state['test_loss'] = test_loss
    torch.save(state, path)
    if save_simplied_model:
        state_simple = {'model': model_state_dict}
        name, suffix = os.path.splitext(path)
        torch.save(state_simple, f'{name}_{epoch}{suffix}')


def read_model(path, model, optimizer=None):
    epoch = 1
    test_loss = [100000]
    if os.path.isfile(path):
        checkpoint = torch.load(path)
        if 'model' in checkpoint:
            if isinstance(model, (list, tuple)):
                if 'model2' in checkpoint:
                    model[0].load_state_dict(checkpoint['model'])
                    model[1].load_state_dict(checkpoint['model2'])
                else:
                    for i in range(len(checkpoint['model'])):
                        if model[i]:
                            model[i].load_state_dict(checkpoint['model'][i])
            else:
                if isinstance(checkpoint['model'], list):
                    model.load_state_dict(checkpoint['model'][0])
                else:
                    model.load_state_dict(checkpoint['model'])
        if (optimizer is not None) and ('optimizer' in checkpoint):
            if isinstance(optimizer, (list, tuple)):
                if 'optimizer2' in checkpoint:
                    optimizer[0].load_state_dict(checkpoint['optimizer'])
                    optimizer[1].load_state_dict(checkpoint['optimizer2'])
                else:
                    for i in range(len(checkpoint['optimizer'])):
                        optimizer[i].load_state_dict(checkpoint['optimizer'][i])
            else:
                if isinstance(checkpoint['optimizer'], list):
                    optimizer.load_state_dict(checkpoint['optimizer'][0])
                else:
                    optimizer.load_state_dict(checkpoint['optimizer'])
        if 'epoch' in checkpoint:
            epoch = checkpoint['epoch'] + 1
        if 'test_loss' in checkpoint:
            test_loss.append(checkpoint['test_loss'])
    return epoch, test_loss

def calculate_gradient_penalty(real_image, fake_image, D, device):
    alpha = torch.rand(real_image.shape[0], 1, 1, 1).to(device)
    interpolate = (alpha * real_image + (1 - alpha) * fake_image).detach().requires_grad_(True)
    interpolate_score = D(interpolate)
    gradient = torch.autograd.grad(interpolate_score, interpolate, torch.ones_like(interpolate_score), True, True)[0]
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_penalty = torch.mean((torch.norm(gradient, 2, 1) - 1) ** 2)
    return gradient_penalty

class PredictionStatistics():
    def __init__(self, num_class):
        self.num_class = num_class
        self.stat = np.zeros((num_class, 4), dtype=np.int64)
    
    def add(self, pred, expected):
        for i in range(self.num_class):
            tp = torch.where(pred[torch.where(expected == i)] == i)[0].shape[0]
            fp = torch.where(pred[torch.where(expected != i)] == i)[0].shape[0]
            fn = torch.where(pred[torch.where(expected == i)] != i)[0].shape[0]
            tn = torch.where(pred[torch.where(expected != i)] != i)[0].shape[0]
            self.stat[i] += [tp, fp, fn, tn]
    
    def reset(self):
        self.stat = np.zeros((self.num_class, 4), dtype=np.int64)
    
    def get_tp(self):
        return self.stat[:, 0]
    
    def get_fp(self):
        return self.stat[:, 1]
    
    def get_fn(self):
        return self.stat[:, 2]
    
    def get_tn(self):
        return self.stat[:, 3]
    
    def get_precision(self):
        tp = self.get_tp()
        fp = self.get_fp()
        precision = tp / (tp + fp)
        return precision
    
    def get_recall(self):
        tp = self.get_tp()
        fn = self.get_fn()
        recall = tp / (tp + fn)
        return recall

    def get_accuracy(self):
        tp = self.get_tp()
        fp = self.get_fp()
        fn = self.get_fn()
        tn = self.get_tn()
        accuracy = (tp + tn) / (tp + fp + fn + tn)
        return accuracy
    
    def get_f1_score(self):
        precision = self.get_precision()
        recall = self.get_recall()
        f1_score = 2 * (precision * recall) / (precision + recall)
        return f1_score
