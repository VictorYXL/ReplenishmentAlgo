import torch
import torch.nn as nn
import torch.nn.functional as F

class MLPMMOEAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(MLPMMOEAgent, self).__init__()
        self.args = args
        self.num_experts = 5

        self.fc1 = nn.Linear(input_shape, args.hidden_dim)
        self.fc2 = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.fc3 = nn.Linear(args.hidden_dim, args.n_actions * self.num_experts)
        
        self.gate = nn.Linear(args.hidden_dim, self.num_experts)
        
    def init_hidden(self):
        return None

    def forward(self, inputs, h=None):
        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))
        
        a = F.softmax(self.gate(x), dim=-1).squeeze() 
        
        y = self.fc3(x).reshape(-1, x.shape[1], self.num_experts, self.args.n_actions) 
        # print(a.shape, y.shape)
        output = (a.unsqueeze(-1) * y).sum(-2)
        
        return output, None
