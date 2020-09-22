import torch

class EmoNet(torch.nn.Module):

    def __init__(self,
            input_size: int = 163,
            hidden_size: int = 256,
            output_size: int = 7):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size

    def __len__(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

class EmoNet0(EmoNet):

    def __init__(self, hidden_size: int):
        super().__init__(hidden_size=hidden_size)
        self.bn1 = torch.nn.BatchNorm1d(self.input_size)
        self.fc1 = torch.nn.Linear(self.input_size, hidden_size)
        self.act1 = torch.nn.Tanh()
        self.bn2 = torch.nn.BatchNorm1d(hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, self.output_size)

    def forward(self, x):
        x = self.bn1(x)
        x = self.fc1(x)
        x = self.act1(x)
        x = self.bn2(x)
        x = self.fc2(x)
        return x
