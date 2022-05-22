import torch
import time
from tqdm import tqdm
from optparse import OptionParser
import os
import csv

from Network import UNet
from dataset_read import get_dataloaders
from train_func import train_net , val_net

' Definition of the needed parameters '
def get_args():
    parser = OptionParser()
    parser.add_option('-e', '--epochs', dest='epochs', default=100, type='int', help='number of epochs')
    parser.add_option('-b', '--batch size', dest='batch_size', default=32, type='int', help='batch size')
    parser.add_option('-l', '--learning rate', dest='lr', default=0.01, type='float', help='learning rate')
    parser.add_option('-r', '--root', dest='root', default="", help='root directory')
    parser.add_option('-i', '--input', dest='input', default='train_in', help='folder of input')
    parser.add_option('-g', '--ground truth', dest='gt', default='train_gt', help='folder of ground truth')
    parser.add_option('-s', '--model', dest='model', default='model_weights', help='folder for model/weights')
    parser.add_option('-v', '--val percentage', dest='val_perc', default=0.05 ,type='float', help='validation percentage')
    (options, args) = parser.parse_args()
    return options

' Run of the training and validation '
def setup_and_run_train(dir_input, dir_gt, dir_model, val_perc, batch_size, epochs, lr):
    time_start = time.time()
    # Use GPU or not
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    # Create the model
    net = UNet().to(device)
    net.train()
    net = torch.nn.DataParallel(net, device_ids=list(range(torch.cuda.device_count()))).to(device)
    # Load the dataset
    train_loader, val_loader = get_dataloaders(dir_input, dir_gt, val_perc, batch_size)
    # Definition of the optimizer
    optimizer = torch.optim.Adam(net.parameters(),lr=lr)
    # Definition of the loss function
    loss_f = torch.nn.L1Loss()
    # Set the header for csv
    header = ['epoch', 'learning rate', 'train loss','val loss', 'time cost now/second']
    best_loss = 1000000
    # Ready to use the tqdm (A Fast, Extensible Progress Bar for Python and CLI)
    for epoch in tqdm(range(epochs)):
        if args.lr > 0.000001:
            if epoch % 1 == 0:
                args.lr = args.lr * 0.85
        print('\ Learning rate = ' , round(args.lr,6), end= ' ')
        # Get training loss function and validating loss function
        train_loss = train_net(net, device, train_loader, optimizer, loss_f, batch_size)
        val_loss = val_net(net, device, val_loader, loss_f, batch_size)
        # Get time cost now
        time_cost_now = time.time() - time_start
        # Set the values for csv
        values = [epoch+1, args.lr, train_loss, val_loss, time_cost_now]
        # Save epoch, learning rate, train loss, val loss and time cost now to a csv
        if not os.path.exists(args.root + args.model + '/', ):
            os.makedirs(args.root + args.model + '/', )
        path_csv = dir_model + "loss and others" + ".csv"
        if os.path.isfile(path_csv) == False:
            file = open(path_csv, 'w', newline='')
            writer = csv.writer(file)
            writer.writerow(header)
            writer.writerow(values)
        else:
            file = open(path_csv, 'a', newline='')
            writer = csv.writer(file)
            writer.writerow(values)
        file.close()
        # Save model
        if train_loss < best_loss:
            best_loss = train_loss
            torch.save({
                    'epoch': epoch + 1,
                    'state_dict': net.state_dict(),
                    'loss': train_loss,
                    'optimizer' : optimizer.state_dict(),
                }, dir_model + "weights" + ".pth")
    time_all = time.time() - time_start
    print("Total time %.4f seconds for training" % (time_all))

' Run the application '
if __name__ == "__main__":
    args = get_args()
    setup_and_run_train(
            dir_input=args.root + args.input + '/',
            dir_gt = args.root +args.gt+'/',
            dir_model=args.root + args.model + '/',
            val_perc = args.val_perc,
            batch_size = args.batch_size,
            epochs = args.epochs,
            lr = args.lr)