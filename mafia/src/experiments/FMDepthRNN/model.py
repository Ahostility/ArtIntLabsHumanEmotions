import torch

class FMDepthRNN(torch.nn.Module):

    def __init__(self,
            input_dim: int = 2,
            hidden_size: int = 136,
            num_layers: int = 1,
            output_dim: int = 68):

        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers  = num_layers
        self.rnn = torch.nn.RNN(input_dim, hidden_size, num_layers, nonlinearity='tanh', batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, output_dim)

    def __len__(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):
        h0 = self.init_hidden(x)
        out, hn = self.rnn(x, h0)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

    def init_hidden(self, x):
        return torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
