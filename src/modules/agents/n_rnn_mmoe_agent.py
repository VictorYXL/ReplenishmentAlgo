import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LayerNorm

from utils.th_utils import orthogonal_init_


class NRNNMMOEAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(NRNNMMOEAgent, self).__init__()
        self.args = args

        self.num_experts = 5
        self.fc1 = nn.Linear(input_shape, args.hidden_dim)
        self.rnn = nn.GRUCell(args.hidden_dim, args.hidden_dim)
        # self.fc2 = nn.Linear(args.hidden_dim, args.n_actions)
        self.fc2 = nn.Linear(args.hidden_dim, args.n_actions * self.num_experts)
        self.gate = nn.Linear(args.hidden_dim, self.num_experts)

        if getattr(args, "use_layer_norm", False):
            self.layer_norm = LayerNorm(args.hidden_dim)

        if getattr(args, "use_orthogonal", False):
            orthogonal_init_(self.fc1)
            orthogonal_init_(self.fc2, gain=args.gain)
        if getattr(args, "dropout", False):
            self.dp = nn.Dropout(0.2)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        b, a, e = inputs.size()

        inputs = inputs.view(-1, e)
        if getattr(self.args, "dropout", False):
            inputs = self.dp(inputs)

        x = F.relu(self.fc1(inputs), inplace=True)
        h_in = hidden_state.reshape(-1, self.args.hidden_dim)
        hh = self.rnn(x, h_in)

        if getattr(self.args, "use_layer_norm", False):
            q = self.fc2(self.layer_norm(hh))
        else:
            q = self.fc2(hh)
        hh = hh.view(b, a, -1)
        alpha = F.softmax(self.gate(hh), dim=-1).squeeze()
        q = q.reshape(b, a, self.num_experts, self.args.n_actions) 
        # print(alpha.shape, y.shape)
        q = (alpha.unsqueeze(-1) * q).sum(-2)

        return q, hh
        # return q.view(b, a, -1), hh.view(b, a, -1)