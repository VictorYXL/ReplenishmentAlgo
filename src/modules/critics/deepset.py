import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.input_utils import build_critic_inputs, get_critic_input_shape
from utils.th_utils import orthogonal_init_


class DeepSet(nn.Module):
    """A deep set."""

    def __init__(
        self, scheme, args, pool_type="avg",
    ):
        super(DeepSet, self).__init__()
        self.args = args
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.obs_agent_id = args.obs_agent_id
        self.pool_type = pool_type

        self.input_seq_str = (
            f"{args.critic_input_seq_str}_{self.n_agents}_{self.n_actions}"
        )
        input_shape = get_critic_input_shape(self.input_seq_str, scheme)
        self.output_type = "v"

        self.fc1 = nn.Linear(input_shape, args.hidden_dim)
        self.fc2 = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.fc3 = nn.Linear(args.hidden_dim, 1)

        if getattr(args, "use_layer_norm", False):
            self.layer_norm = nn.LayerNorm(args.hidden_dim)

        if getattr(args, "use_orthogonal", False):
            orthogonal_init_(self.fc1)
            orthogonal_init_(self.fc2, gain=args.gain)

    def forward(self, batch):
        inputs, bs, max_t = build_critic_inputs(self.input_seq_str, batch)
        h1 = F.relu(self.fc1(inputs.flatten(0, 1)))
        h1_summed = torch.sum(h1, 1)
        h2 = F.relu(self.fc2(h1_summed))

        # Compute Q
        if getattr(self.args, "use_layer_norm", False):
            q = self.fc3(self.layer_norm(h2))
        else:
            q = self.fc3(h2)
        q = q / self.n_agents / (self.args.hidden_dim ** 0.5)
        return torch.repeat_interleave(
            q.reshape(bs, max_t, 1, -1), repeats=self.n_agents, dim=2
        )
