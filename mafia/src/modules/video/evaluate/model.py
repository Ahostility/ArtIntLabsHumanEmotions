import torch

class VideoAIMAFNet(torch.nn.Module):

    def __init__(self, hidden_size: int, p1: float, p2: float):
        super().__init__()
        self.fc1 = torch.nn.Linear(8, hidden_size)
        self.bn1 = torch.nn.BatchNorm1d(hidden_size)
        self.dp1 = torch.nn.Dropout(p1)
        self.fc2 = torch.nn.Linear(hidden_size, hidden_size)
        self.bn2 = torch.nn.BatchNorm1d(hidden_size)
        self.dp2 = torch.nn.Dropout(p2)
        self.fc3 = torch.nn.Linear(hidden_size, 2)

    def __len__(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.tanh(x)
        x = self.bn1(x)
        x = self.dp1(x)

        x = self.fc2(x)
        x = torch.tanh(x)
        x = self.bn2(x)
        x = self.dp2(x)

        x = self.fc3(x)
        return x

    def init_weights(self, filepath: str):
        checkpoint = torch.load(filepath)
        self.load_state_dict(checkpoint['model_state_dict'])
