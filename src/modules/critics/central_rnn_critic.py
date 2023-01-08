import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LayerNorm

from utils.th_utils import orthogonal_init_


class CentralRNNVCritic(nn.Module):
    def __init__(self, scheme, args):
        super(CentralRNNVCritic, self).__init__()

        self.args = args
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents

        input_shape = self._get_input_shape(scheme)
        self.output_type = "v"

        # Set up network layers
        self.fc1 = nn.Linear(input_shape, args.hidden_dim)
        self.rnn = nn.GRUCell(args.hidden_dim, args.hidden_dim)
        self.fc2 = nn.Linear(args.hidden_dim, 1)

        if getattr(args, "use_layer_norm", False):
            self.layer_norm = LayerNorm(args.hidden_dim)

        if getattr(args, "use_orthogonal", False):
            orthogonal_init_(self.fc1)
            orthogonal_init_(self.fc2, gain=args.gain)

        self.hidden_states = None

    def init_hidden(self, batch_size):
        hidden_states = self.fc1.weight.new(1, self.args.hidden_dim).zero_()
        self.hidden_states = hidden_states.unsqueeze(0).expand(
            batch_size, self.n_agents, -1
        )  # bav

    def forward(self, batch, t):
        inputs, bs, max_t = self._build_inputs(batch, t=t)
        x = F.relu(self.fc1(inputs)).view(-1, self.args.hidden_dim)
        h_in = self.hidden_states.reshape(-1, self.args.hidden_dim)
        hh = self.rnn(x, h_in)
        if getattr(self.args, "use_layer_norm", False):
            q = self.fc2(self.layer_norm(hh))
        else:
            q = self.fc2(hh)
        self.hidden_states = hh.view(bs, self.n_agents, -1)
        return q.view(bs, self.n_agents, -1)

    def _build_inputs(self, batch, t=None):
        bs = batch.batch_size
        max_t = batch.max_seq_length if t is None else 1
        ts = slice(None) if t is None else slice(t, t + 1)
        inputs = []
        # state
        inputs.append(batch["state"][:, ts].unsqueeze(2).repeat(1, 1, self.n_agents, 1))

        # observations
        if self.args.obs_individual_obs:
            inputs.append(
                batch["obs"][:, ts]
                .view(bs, max_t, -1)
                .unsqueeze(2)
                .repeat(1, 1, self.n_agents, 1)
            )

        # last actions
        if self.args.obs_last_action:
            if t == 0:
                last_actions = torch.zeros_like(batch["actions_onehot"][:, 0:1])
            elif isinstance(t, int):
                last_actions = batch["actions_onehot"][:, slice(t - 1, t)]
            else:
                last_actions = torch.cat(
                    [
                        torch.zeros_like(batch["actions_onehot"][:, 0:1]),
                        batch["actions_onehot"][:, :-1],
                    ],
                    dim=1,
                )
            last_actions = last_actions.view(bs, max_t, 1, -1)
            inputs.append(last_actions.repeat(1, 1, self.n_agents, 1))

        inputs.append(
            torch.eye(self.n_agents, device=batch.device)
            .unsqueeze(0)
            .unsqueeze(0)
            .expand(bs, max_t, -1, -1)
        )

        inputs = torch.cat(inputs, dim=-1)

        return inputs, bs, max_t

    def _get_input_shape(self, scheme):
        # state
        input_shape = scheme["state"]["vshape"]
        # observations
        if self.args.obs_individual_obs:
            input_shape += scheme["obs"]["vshape"] * self.n_agents
        # last actions
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0] * self.n_agents
        input_shape += self.n_agents
        return input_shape
