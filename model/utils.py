import torch
import torch.nn as nn
from torch.autograd import grad as torch_grad


def gradient_penalty(net, real_data, generated_data, LAMBDA=10, device=torch.device('cuda')):
    assert isinstance(net, nn.Module)
    batch_size = real_data.size()[0]

    # Calculate interpolationsubsampling_rate=20
    alpha = torch.rand(batch_size, 1).requires_grad_()
    alpha = alpha.expand_as(real_data).to(device)
    interpolated = alpha * real_data.data + (1 - alpha) * generated_data.data

    # Calculate probability of interpolated examples
    prob_interpolated = net(interpolated)

    # Calculate gradients of probabilities with respect to examples
    gradients = torch_grad(outputs=prob_interpolated, inputs=interpolated,
                           grad_outputs=torch.ones(prob_interpolated.size()).to(device),
                           create_graph=True, retain_graph=True)[0]

    # Gradients have shape (batch_size, num_channels, img_width, img_height),
    # so flatten to easily take norm per example in batch
    gradients = gradients.view(batch_size, -1)

    # Return gradient penalty
    return LAMBDA * ((gradients.norm(2, dim=1) - 1) ** 2).mean()


def orthogonal_regularization(net, reg_coef=1e-4, device=torch.device('cuda')):
    assert isinstance(net, nn.Module)

    reg = 0
    for layer in net.modules():
        if isinstance(layer, torch.nn.Linear):
            prod = torch.matmul(torch.transpose(layer.weight, 0, 1), layer.weight)
            reg += torch.sum(torch.square(prod * (1 - torch.eye(prod.shape[0]).to(device))))

    return reg * reg_coef


def soft_update(net, target_net, tau=0.005):
    assert isinstance(net, torch.nn.Module) and isinstance(target_net, torch.nn.Module)

    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)