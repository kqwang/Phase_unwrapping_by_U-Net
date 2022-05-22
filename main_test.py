import os
from optparse import OptionParser
import torch
import time
import os.path
import scipy.io as sio

from Network import UNet
from dataset_read import get_dataloader_for_test

' Definition of the needed parameters '
def get_args():
    parser = OptionParser()
    parser.add_option('-e', '--result', dest='result', default="Results_real", help='folder of results')
    parser.add_option('-r', '--root', dest='root', default="", help='root directory')
    parser.add_option('-m', '--model', dest='model', default='model_weights/weights.pth', help='folder for model/weights')
    parser.add_option('-i', '--input', dest='input', default="test_in_real", help='folder of input')
    parser.add_option('-g', '--gt', dest='gt', default="test_gt_real", help='folder of ground truth')
    (options, args) = parser.parse_args()
    return options

' Pass inputs through the Res-UNet '
def get_results(load_weights, dir_input, dir_gt, resultdir):
    # Use GPU or not
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    # Create the model
    net = UNet().to(device)
    net.eval()
    net = torch.nn.DataParallel(net, device_ids=list(range(torch.cuda.device_count()))).to(device)
    # Load old weights
    checkpoint = torch.load(load_weights, map_location='cpu')
    net.load_state_dict(checkpoint['state_dict'])
    # Load the dataset
    loader = get_dataloader_for_test(dir_input, dir_gt)
    # If resultdir does not exists make folder
    if not os.path.exists(resultdir):
        os.makedirs(resultdir)
    net.eval()
    with torch.no_grad():
        time1 = 0
        for (input, gt , gt) in loader:
            input, gt = input.to(device), gt.to(device)
            for th in range(0, len(input)):
                time_start = time.time()
                input1 = input[th]
                input1 = torch.unsqueeze(input1, 0)
                output = net(input1)[0]
                input_numpy = input1.squeeze(0).squeeze(0).cpu().numpy()
                output_numpy = output.squeeze(0).cpu().numpy()
                gt_numpy = gt[th].squeeze(0).cpu().numpy()
                filename = resultdir + (str(th + 1).zfill(6)) + '-results.mat'
                sio.savemat(filename, {'input': input_numpy, 'output': output_numpy, 'gt': gt_numpy})
                print("Result saved as {}".format(filename))

                time_end = time.time()
                time1 = time1 + (time_end - time_start)
        print('totally cost', time1)

' Run the application '
if __name__ == '__main__':
    args = get_args()
    get_results(load_weights = args.root + args.model,
                dir_input = args.root +args.input+"/",
                dir_gt=args.root + args.gt + "/",
                resultdir=args.root + args.result + "/")