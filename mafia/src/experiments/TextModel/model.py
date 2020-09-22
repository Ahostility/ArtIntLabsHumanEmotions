import torch.nn as nn
import torch.nn.functional as F
import torch


class NLPMOdel(nn.Module):

    def __init__(self, hidden_dim, layer_dim):
        super(NLPMOdel, self).__init__()
        
        self.hidden_dim = hidden_dim
        
        self.layer_dim = layer_dim

        self.emb = nn.Embedding(3000, 60)
        self.lstm_1 = nn.LSTM(input_size=60, hidden_size=hidden_dim, num_layers=layer_dim, batch_first=True)
        # self.lstm_2 = nn.LSTM(input_size=64, hidden_size=128, batch_first=True, num_layers=128)
        # self.lstm_3 = nn.LSTM(input_size=128, hidden_size=128, batch_first=True, num_layers=128)
        # self.lstm_4 = nn.LSTM(input_size=128, hidden_size=128, batch_first=True, num_layers=64)
        
        # self.liner_1 = nn.Linear(128, 256)
        # self.liner_2 = nn.Linear(256, 128)
        # self.liner_3 = nn.Linear(128, 64)
        self.liner_4 = nn.Linear(hidden_dim, 2)


    def init_weights(self, filepath):
        checkpoint = torch.load(filepath)
        self.load_state_dict(checkpoint['model_state_dict'])

    def forward(self, X):

        X = self.emb(X.long())

        h0 = torch.zeros(self.layer_dim, X.size(0), self.hidden_dim).requires_grad_().cuda()
        c0 = torch.zeros(self.layer_dim, X.size(0), self.hidden_dim).requires_grad_().cuda()

        out, (hn, cn) = self.lstm_1(X, (h0.detach(), c0.detach()))
        # lstm_out, (ht, ct) = self.lstm_2(lstm_out)
        # lstm_out, (ht, ct) = self.lstm_3(lstm_out)
        # lstm_out, (ht, ct) = self.lstm_4(lstm_out)
        # print(ht[-1])

        # print(ht.shape)
        # X = self.liner_1(ht[-1])
        # X = F.relu(self.liner_2(X))
        # X = F.relu(self.liner_3(ht[-1]))

        X = F.sigmoid(self.liner_4(out[:, -1, :]))

        return X