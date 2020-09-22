import torch.nn as nn
import torch.nn.functional as F
import torch


class SoundModel(nn.Module):

    def __init__(self):
        
        super().__init__()
        self.bn = nn.BatchNorm1d(66)
        self.drop_layer = nn.Dropout(p=0.5)
        self.liner_1 = nn.Linear(66, 512)
        self.liner_2 = nn.Linear(512, 1024)
        self.liner_3 = nn.Linear(1024, 2048)
        self.liner_4 = nn.Linear(2048, 1024)
        self.liner_5 = nn.Linear(1024, 256)
        self.liner_6 = nn.Linear(256, 128)
        self.liner_7 = nn.Linear(128, 2)


    def init_weights(self, filepath):
        checkpoint = torch.load(filepath)
        self.load_state_dict(checkpoint['model_state_dict'])

    def forward(self, X):

        X = self.bn(X)

        X = F.relu(self.liner_1(X))
        X = F.relu(self.liner_2(X))
        X = self.drop_layer(X)
        X = F.relu(self.liner_3(X))
        X = self.drop_layer(X)
        X = F.relu(self.liner_4(X))
        X = self.drop_layer(X)
        X = F.relu(self.liner_5(X))
        X = F.relu(self.liner_6(X))

        X = F.sigmoid(self.liner_7(X))

        return X