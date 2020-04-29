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
from torch.nn import Conv2d, Dropout, MaxPool2d


class FullNet(nn.Module):
    def __init__(self):
        super(FullNet, self).__init__()
        self.dropout1 = nn.Dropout2d(0.4)
        self.pool = MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(100, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 16)

        self.out = nn.Linear(16, 1)


    def forward(self, json):
        x = json

        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.fc4(x)
        x = F.relu(x)
        x = self.dropout1(x)

        x = self.out(x)

        output = F.sigmoid(x)
        return output

class RNNet(nn.Module):
    def __init__(self):
        super(RNNet, self).__init__()

        self.lstm = nn.LSTM(input_size=50, hidden_size=64, num_layers=3, batch_first=True, dropout=0.8)

        self.out = nn.Linear(64, 1)


    def forward(self, json):
        x, _ = self.lstm(json, None)

        x = F.sigmoid(self.out(x[:, -1, :]))
        return x


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    losses = []
    for batch_idx, (data, target) in enumerate(train_loader):
        json = data.to(device)
        #video = data[1].to(device)
        target = target.to(device)
        optimizer.zero_grad()
        output = model(json)
        loss = F.binary_cross_entropy(output, target.float())
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    print('Avg train loss:', sum(losses)/len(losses))
    return sum(losses)/len(losses)


def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            json = data.to(device)
            #video = data[1].to(device)
            target = target.to(device)
            output = model(json)
            test_loss += F.binary_cross_entropy(output, target.float(), reduction='sum').item()  # sum up batch loss
            pred = output >= 0.5
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return test_loss


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=30, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 6, 'pin_memory': True} if use_cuda else {}

    trainset = VideoFolder('data')
    valset = VideoFolder('data', is_validation=True)
    print('train size', len(trainset), 'val size', len(valset))

    train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, **kwargs)
    val_loader = DataLoader(valset, batch_size=args.batch_size, shuffle=True, **kwargs)

    #model = FullNet().to(device)
    #summary(model, (500,), batch_size=32)
    model = RNNet().to(device)
    #summary(model, (20,50), batch_size=32)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)


    train_losses = []
    test_losses = []
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
    frames = [[-1] * 50, [-1] * 50, [-1] * 50, [-1] * 50, [-1] * 50,
              [-1] * 50, [-1] * 50, [-1] * 50, [-1] * 50, [-1] * 50,
                [-1] * 50, [-1] * 50, [-1] * 50, [-1] * 50, [-1] * 50,
              [-1] * 50, [-1] * 50, [-1] * 50, [-1] * 50, [-1] * 50,
              [-1] * 50, [-1] * 50, [-1] * 50, [-1] * 50, [-1] * 50,
              [-1] * 50, [-1] * 50, [-1] * 50, [-1] * 50, [-1] * 50,
              [-1] * 50, [-1] * 50, [-1] * 50, [-1] * 50, [-1] * 50,
              [-1] * 50, [-1] * 50, [-1] * 50, [-1] * 50, [-1] * 50,
              [-1] * 50, [-1] * 50, [-1] * 50, [-1] * 50, [-1] * 50,
              [-1] * 50, [-1] * 50, [-1] * 50, [-1] * 50, [-1] * 50,
              [-1] * 50, [-1] * 50, [-1] * 50, [-1] * 50, [-1] * 50,
              [-1] * 50, [-1] * 50, [-1] * 50, [-1] * 50, [-1] * 50]
    for root, _, fnames in sorted(os.walk(video_path, followlinks=True)):
        for fname in sorted(fnames):
            if '.mp4' not in fname:
                path = os.path.join(root, fname)
                with open(path) as f:
                    frames.pop(0)
                    try:
                        keys = np.asarray(json.load(f)['people'][0]['pose_keypoints_2d'])
                        keys = keys.reshape(-1, 3)[:, 0:2].flatten()
                        frames.append(keys)
                    except:
                        frames.append([-1] * 50)
                model.eval()

                model_inp = []
                missing_frame = False
                for i in range(60):
                    if i !=0 and i%5==0: continue
                    if -1 in frames[i]: missing_frame = True
                    model_inp.append(frames[i])
                if missing_frame:
                    predictions.append(0)
                else:
                    #predictions.append(model(torch.FloatTensor(np.asarray(frames).flatten()).to(device)).cpu().detach().numpy()[0])
                    predictions.append(model(torch.FloatTensor(np.asarray([model_inp])).to(device)).cpu().detach().numpy()[0][0])
    predsecs = []
    frames = []
    jsonit = []
    i = 1
    x = []
    for pred in predictions:
        if len(frames) != 15:
            frames.append(pred)
        else:
            x.append(i*0.5)
            predsecs.append(sum(frames)/15)
            jsonit.append([i*0.5, sum(frames)/15])
            i += 1
            frames = [pred]
    jsonit = {'pilltaking':jsonit}
    video_name = video_path.split('/')[1]
    video_json = json.dumps(jsonit)
    print(video_json)
    plt.ylim((0.0, 1.0))
    plt.plot(x, predsecs)
    plt.show()


if __name__ == '__main__':
    main()
