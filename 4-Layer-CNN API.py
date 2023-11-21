# miniforgeをONにする場合は、 conda activate catechcv1 とコマンドに入力

import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn import metrics

from torchvision import transforms

# 初回の学習時は標準化をかけていなかったので注意。
# transform_test = transforms.Compose(
# [
# transforms.ToTensor(),
# transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
# ]
# )

# test_data = torchvision.datasets.CIFAR10(
# "./cifar-10", train=False, download=True, transform=transform_test
# )


test_data = torchvision.datasets.CIFAR10(
    "./cifar-10",
    train=False,
    download=True,
    transform=torchvision.transforms.ToTensor(),
)


# シードの固定
torch.manual_seed(4931)


class SimpleCNN(nn.Module):
    def __init__(self, dropout_rate=0.2):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, padding=1
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(in_features=64 * 8 * 8, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=10)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = nn.ReLU()(self.conv1(x))
        x = self.pool(nn.ReLU()(self.conv2(x)))
        x = self.pool(x)
        x = x.view(-1, 64 * 8 * 8)
        x = self.dropout(x)
        x = nn.ReLU()(self.fc1(x))
        x = self.fc2(x)
        return x


net = SimpleCNN()

net.eval()

# Load the state dictionary into the model
model_state_dict_path = (
    "/Users/asato/Desktop/CA Tech CV課題/CA Tech CV 1/simple_cnn_model_state_dict.pth"
)


# Load the state dictionary into the model without DataParallel
state_dict = torch.load(model_state_dict_path, map_location=torch.device("cpu"))

state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

net.load_state_dict(state_dict)

test_loader = torch.utils.data.DataLoader(test_data)

label_list = []
pred_list = []


with torch.no_grad():
    for data, label in test_loader:
        output = net(data)  # Removed the .to("cuda") part
        _, pred = torch.max(output, 1)

        label_list.extend(label.cpu().numpy().tolist())
        pred_list.extend(pred.cpu().numpy().tolist())


print(metrics.classification_report(pred_list, label_list))

score = metrics.accuracy_score(label_list, pred_list)
print(score)
