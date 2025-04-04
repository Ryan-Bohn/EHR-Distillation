import numpy as np
import torch.nn as nn

def epoch(mode, dataloader, net, criterion, optimizer=None, device=None):
    loss_avg, acc_avg, num_exp = 0, 0, 0

    if device is None:
        device = next(net.parameters()).device
    else:
        net = net.to(device)

    if mode == 'train':
        net.train()
    else:
        net.eval()

    for i_batch, datum in enumerate(dataloader):
        feat = datum[0].to(device)
        lab = datum[1].to(device)
        n_b = lab.shape[0]

        output = net(feat)
        if isinstance(criterion, nn.MSELoss):
            lab = lab.view(-1, 1)
        loss = criterion(output, lab)
        acc = np.sum(np.equal(np.argmax(output.cpu().data.numpy(), axis=-1), lab.cpu().data.numpy()))

        loss_avg += loss.item()*n_b
        acc_avg += acc
        num_exp += n_b

        if mode == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    loss_avg /= num_exp
    acc_avg /= num_exp

    return loss_avg, acc_avg