import torch.nn as nn
import torch.nn.functional as F
import torch


class GenderModel(nn.Module):

    def __init__(self):
        
        super(GenderModel , self).__init__()
        # self.out = nn.Linear(128, 128)
        self.c1 = nn.Conv1d(input_ 4, 32, kernel_size=5)
        self.c2 = nn.Conv1d(32, 128, kernel_size=4)      
        self.c3 = nn.Conv1d(64, 128, kernel_size=4) 
        self.c4 = nn.Conv1d(128, 256, kernel_size=4) 
        

        # self.bn1 = nn.BatchNorm1d(64) 
        # self.p1 = nn.AvgPool1d(64)
        self.out1 = nn.Linear(256, 1024)
        self.out2 = nn.Linear(1024, 2)


    def init_weights(self, filepath):
        checkpoint = torch.load(filepath)
        self.load_state_dict(checkpoint['model_state_dict'])

    def forward(self, X):
        print(X)
        print(X.shape)
        # X = self.out(X)
        
        X = F.relu(self.c1(X))
        X = F.relu(self.c2(X))
        X = F.relu(self.c3(X))
        X = F.relu(self.c4(X))

        # X = self.bn1(X)
        # X = self.p1(X)
        # X = X.view(X.size(0), -1)
        X = F.relu(self.out1(X))
        X = F.softmax(self.out2(X))

        
        return X


        

