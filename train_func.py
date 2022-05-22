import torch

'Computes and stores the average and current value.'
class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

' network training function '
def train_net(net, device, loader, optimizer, loss_f,  batch_size):
    net.train()
    train_loss = AverageMeter()
    for batch_idx, (input, gt, weights) in enumerate(loader):
        input, gt = input.to(device), gt.to(device) # Send data to GPU
        output = net(input) # Forward
        loss = loss_f(output, gt) # Loss calculation
        train_loss.update(loss.item(), output.size(0)) # Update the record
        # Back propagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(' Train_Loss: ' + str(round(train_loss.avg, 6)), end=" ")
    return train_loss.avg

' network validating function '
def val_net(net, device, loader, loss_f, batch_size):
    net.eval()
    val_loss = AverageMeter()
    with torch.no_grad():
        for batch_idx, (input, gt, gt ) in enumerate(loader):
            input, gt = input.to(device), gt.to(device) # Send data to GPU
            output = net(input) # Forward
            loss = loss_f(output, gt) # Loss calculation
            val_loss.update(loss.item(), output.size(0)) # Update the record
    print(' Val_loss: ' + str(round(val_loss.avg, 6)))
    return val_loss.avg