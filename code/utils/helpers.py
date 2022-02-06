import datetime


def get_time():
    return datetime.datetime.now().strftime("%m-%d-%Y_%H:%M:%S")


#### using a CNN to extract summary statistics

from torch import nn
import torch.nn.functional as F 

class SummaryNet(nn.Module): 

    def __init__(self): 
        super().__init__()
        # 2D convolutional layer
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2)
        # maxpooling reduces length of time series
        self.pool = nn.MaxPool2d(kernel_size=5, stride=5)
        # Fully connected layer taking as input the 6 flattened output arrays from the maxpooling layer
        self.fc = nn.Linear(in_features=6*17*16, out_features=8) 

    def forward(self, x):
        x = x.view(-1, 1, 80, 85)
        x = x.float()
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(-1, 6*17*16)
        x = F.relu(self.fc(x))
        return x

