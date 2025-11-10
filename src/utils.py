import os
import torch
import platform

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
