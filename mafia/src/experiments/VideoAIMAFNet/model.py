import torch
import torch.nn.functional as F

class VideoAIMAFNet(torch.nn.Module):

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.act1 = torch.nn.Tanh()
        self.bn2 = torch.nn.BatchNorm1d(hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, hidden_size)
        self.act2 = torch.nn.Tanh()
        self.bn3 = torch.nn.BatchNorm1d(hidden_size)
        self.fc3 = torch.nn.Linear(hidden_size, 2)

    def __len__(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act1(x)
        x = self.bn2(x)
        x = self.fc2(x)
        x = self.act2(x)
        x = self.bn3(x)
        x = self.fc3(x)
        return x

    # def init_weights(self, filepath: str):
    #     checkpoint = torch.load(filepath)
    #     self.load_state_dict(checkpoint['model_state_dict'])
