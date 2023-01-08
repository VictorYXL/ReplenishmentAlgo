import torch.nn as nn
import torch.nn.functional as F

from utils.input_utils import build_critic_inputs, get_critic_input_shape


class MLPCritic(nn.Module):
    def __init__(self, scheme, args):
        super(MLPCritic, self).__init__()
        self.args = args
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.input_seq_str = (
            f"{args.critic_input_seq_str}_{self.n_agents}_{self.n_actions}"
        )

        input_shape = get_critic_input_shape(self.input_seq_str, scheme)
        self.fc1 = nn.Linear(input_shape, args.hidden_dim)
        self.fc2 = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.fc3 = nn.Linear(args.hidden_dim, 1)

    def forward(self, batch):
        inputs, bs, max_t = build_critic_inputs(self.input_seq_str, batch)
        x = F.relu(self.fc1(inputs)).view(-1, self.args.hidden_dim)
        x = F.relu(self.fc2(x))
        q = self.fc3(x)
        return q.view(bs, -1, self.n_agents, 1)
