import torch
import torch.nn.functional as F

def init_weights(model: torch.nn.Module, filepath: str, device_name: str = 'cpu'):
    device = torch.device(device_name)
    checkpoint = torch.load(filepath, map_location=device)
    weights = checkpoint['model_state_dict']
    model.load_state_dict(weights)
    model.eval()

class FMDepthRNN(torch.nn.Module):

    def __init__(self, weights_path: str):
        super().__init__()
        input_dim, output_dim = 2, 68
        self.hidden_size = 136
        self.num_layers  = 1
        self.rnn = torch.nn.RNN(input_dim, self.hidden_size, self.num_layers, nonlinearity='tanh', batch_first=True)
        self.fc = torch.nn.Linear(self.hidden_size, output_dim)
        init_weights(self, weights_path)

    def init_hidden(self, x):
        return torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)

    def forward(self, x):
        x = torch.FloatTensor(x).unsqueeze(0)
        with torch.no_grad():
            h0 = self.init_hidden(x)
            out, hn = self.rnn(x, h0)
            out = out[:, -1, :]
            out = self.fc(out)
            out = out.detach().squeeze(0)
            torch.cuda.empty_cache()
            return out.numpy()

class EmoNet(torch.nn.Module):

    def __init__(self, weights_path: str):
        super().__init__()
        input_size, hidden_size, output_size = 163, 256, 7
        self.bn1 = torch.nn.BatchNorm1d(input_size)
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.bn2 = torch.nn.BatchNorm1d(hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, output_size)
        init_weights(self, weights_path)

    def forward(self, x):
        x = torch.FloatTensor(x).unsqueeze(0)
        with torch.no_grad():
            x = self.bn1(x)
            x = self.fc1(x)
            x = F.tanh(x)
            x = self.bn2(x)
            x = self.fc2(x)
            x = F.softmax(x)
            x = x.detach().squeeze(0)
            torch.cuda.empty_cache()
            return x.numpy()
