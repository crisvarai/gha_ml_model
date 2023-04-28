import logging
import torch
from torch import nn

def block(c_in, c_out, k=3, p=1, s=1, pk=2, ps=2):
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, k, padding=p, stride=s),
        nn.ReLU(),
        nn.MaxPool2d(pk, stride=ps)
    )

def block2(c_in, c_out):
    return nn.Sequential(
        nn.Linear(c_in, c_out),
        nn.ReLU()
    )

class Model(nn.Module):
    def __init__(self, n_channels=3, n_outputs=4):
        super().__init__()
        self.conv1 = block(n_channels, 8)
        self.conv2 = block(8, 16)
        self.conv3 = block(16, 32)
        self.conv4 = block(32, 64)
        self.fc1 = block2(64*6*6, 100)
        self.fc2 = nn.Linear(100, n_outputs)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x
    
if __name__ == "__main__":
    BATCH_SIZE = 64
    CHAN = 3
    IMG_SIZE = 100
    x = torch.randn(BATCH_SIZE, CHAN, IMG_SIZE, IMG_SIZE)
    model = Model()
    out = model(x)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s-%(name)s-%(levelname)s-%(message)s")
    logging.info(out.shape)