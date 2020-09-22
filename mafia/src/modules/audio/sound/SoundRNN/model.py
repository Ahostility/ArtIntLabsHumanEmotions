# Import Libraries
import torch

# Create RNN Model
class AudioRNN(torch.nn.Module):
    def __init__(self, 
                n_steps: int = 1600, 
                n_inputs: int = 13, 
                n_neurons: int = 128, 
                n_outputs: int = 2):
        super(AudioRNN, self).__init__()
        
        self.n_neurons = n_neurons
        self.n_steps = n_steps
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        
        self.basic_rnn = torch.nn.RNN(self.n_inputs, self.n_neurons) 
        
        self.FC = torch.nn.Linear(self.n_neurons, self.n_outputs)
        
    def init_hidden(self,x):
        
        return torch.zeros(1, x.size(1), self.n_neurons, device=x.device)
        
    def forward(self, X):
        
        hidden = self.init_hidden(X)
        
        rnn_out, hn = self.basic_rnn(X, hidden)    
        rnn_out = rnn_out[:, -1, :]  
        out = self.FC(rnn_out)
        
        return out

