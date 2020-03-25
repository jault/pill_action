from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import torch.optim as optim
from torch.utils.data import DataLoader
from pill_dataset import VideoFolder
import os
import json
import matplotlib.pyplot as plt
import numpy as np


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(75, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        output = F.sigmoid(x)
        return output


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    losses = []
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.binary_cross_entropy(output, target.float())
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    return sum(losses)/len(losses)


def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.binary_cross_entropy(output, target.float(), reduction='sum').item()  # sum up batch loss
            pred = output >= 0.5  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return test_loss


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    dataset = VideoFolder('data')
    # Split dataset
    dsize = len(dataset)
    print(dsize)
    trainset, valset = torch.utils.data.random_split(dataset, [11946, 2987])

    train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, **kwargs)
    val_loader = DataLoader(valset, batch_size=args.batch_size, shuffle=True, **kwargs)

    model = Net().to(device)
    summary(model, (32, 75))
    optimizer = optim.Adam(model.parameters(), lr=args.lr)


    train_losses = []
    test_losses = [test(args, model, device, val_loader)]
    for epoch in range(1, args.epochs + 1):
        train_losses.append(train(args, model, device, train_loader, optimizer, epoch))
        test_losses.append(test(args, model, device, val_loader))

    plt.plot(train_losses)
    plt.show()
    plt.plot(test_losses)
    plt.show()

    test_video(model, device, 'examples/coat')
    test_video(model, device, 'examples/dog')
    test_video(model, device, 'examples/circles')
    test_video(model, device, 'examples/pushup')
    test_video(model, device, 'examples/wave')

    if args.save_model:
        torch.save(model.state_dict(), "model.pt")


def test_video(model, device, video_path):
    predictions = []
    for root, _, fnames in sorted(os.walk(video_path, followlinks=True)):
        for fname in sorted(fnames):
            if '.mp4' not in fname:
                path = os.path.join(root, fname)
                with open(path) as f:
                    try:
                        keypoints = json.load(f)['people'][0]['pose_keypoints_2d']
                    except:
                        keypoints = [-1] * 75
                keypoints = torch.FloatTensor(keypoints).to(device)
                model.eval()
                predictions.append(model(keypoints).cpu().detach().numpy()[0])

    predsecs = []
    frames = []
    jsonit = []
    i = 1
    for pred in predictions:
        if len(frames) != 30:
            frames.append(pred)
        else:
            predsecs.append(sum(frames)/30)
            jsonit.append([i*1.0, sum(frames)/30])
            i += 1
            frames = [pred]
    jsonit = {'pilltaking':jsonit}
    video_name = video_path.split('/')[1]
    video_json = json.dumps(jsonit)
    print(video_json)
    plt.ylim((0.0, 1.0))
    plt.plot(predsecs)
    plt.show()


if __name__ == '__main__':
    main()
