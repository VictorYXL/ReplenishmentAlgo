import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
# from replenishment.efficient_env import EfficientReplenishmentEnv
from torch.optim import Adam

from components.episode_buffer import EpisodeBatch


class StochasticCapacityDynamicsModel(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1, num_layers=2):
        super().__init__()
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.mu_layer = torch.nn.Linear(hidden_size, output_size)
        self.log_std_layer = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x, deterministic=False):
        r_out, (h_n, h_c) = self.lstm(x, None)
        mu = self.mu_layer(r_out[:, -1])
        if deterministic:
            return mu
        else:
            log_std = self.log_std_layer(r_out[:, -1])
            log_std = torch.clamp(log_std, -2, 20)
            std = torch.exp(log_std)
            dist = torch.distributions.Normal(mu, std)
            return dist.rsample()


class DeterministicCapacityDynamicsModel(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1, num_layers=2):
        super().__init__()
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        r_out, (h_n, h_c) = self.lstm(x, None)
        return self.fc(r_out[:, -1])


class ContextLearner:
    def __init__(self, logger, args):
        self.args = args
        self.logger = logger
        self.context_model = DeterministicCapacityDynamicsModel(
            input_size=3, hidden_size=64, output_size=3, num_layers=2
        )
        self.context_optimiser = Adam(
            params=self.context_model.parameters(), lr=args.lr
        )
        self.context_batch_size = 512
        self.hist_len = 7
        self.log_stats_t = -self.args.learner_log_interval - 1
        self.capacity_dynamics = None
        self.max_storage_capacity = int(
            re.findall(r"c(.*)d", args.env_args["map_name"])[0]
        )

    def generate_C_trajectories(self, agent_ids, aug_type="noise", prob=0.1):
        B, T, _, _ = self.capacity_dynamics.shape
        device = self.capacity_dynamics.device
        n_trajectories = len(agent_ids)
        real_dynamics = torch.empty(n_trajectories, T, 3).to(device)
        fake_dynamics = torch.empty(n_trajectories, T, 3).to(device)
        pred_dynamics = torch.empty(n_trajectories, T, 3).to(device)
        from_Batches = [0] * len(agent_ids)
        self.context_model.eval()

        for i in range(n_trajectories):
            from_Batch = np.random.choice(B)
            from_Batches[i] = from_Batch
            real_dynamics[i] = self.capacity_dynamics[from_Batch].sum(-1)

        if aug_type == "noise":
            noise = torch.randn_like(real_dynamics) * 0.1
            noise[torch.rand_like(real_dynamics) >= prob] = 0.0
            fake_dynamics = real_dynamics + noise
            fake_dynamics[:, :, 0].clip_(0.0, 1.0)
            fake_dynamics[:, :, 1].clip_(0.0, 1000.0)
            fake_dynamics[:, :, 2].clip_(0.0, 1.0)
            # fake_dynamics = real_dynamics
        elif aug_type == "pred":
            pred_dynamics[:, : self.hist_len] = real_dynamics[:, : self.hist_len]
            for j in range(self.hist_len, T):
                pred_dynamics[:, j] = self.context_model(
                    real_dynamics[:, (j - self.hist_len) : j]
                )
            mask = (torch.rand_like(real_dynamics) < prob).float()
            fake_dynamics = pred_dynamics * mask + real_dynamics * (1 - mask)
        else:
            raise NotImplementedError

        for i in range(n_trajectories):
            fake_dynamics[i] -= self.capacity_dynamics[
                from_Batches[i], :, :, agent_ids[i] - 1
            ]

        return (fake_dynamics.transpose(1, 2) * self.max_storage_capacity).int()

    def train(self, C_trjactories: np.array, t_env: int):
        B, T, _, _ = C_trjactories.shape
        # device = next(self.context_model.parameters()).device

        # shape = [B, T, 3, N]
        # self.capacity_dynamics = (
        #     torch.from_numpy(C_trjactories).to(device) / self.max_storage_capacity
        # )
        self.capacity_dynamics = (
            torch.from_numpy(C_trjactories) / self.max_storage_capacity
        )

        # sum_capacity_dynamics = self.capacity_dynamics.sum(dim=-1)
        # D = self.hist_len
        # L = B * (T - D)
        # dataset_c = torch.zeros(size=[L, D, 3]).to(device)
        # dataset_c_prime = torch.zeros(size=[L, 3]).to(device)
        # shuffled_pos = torch.randperm(L).tolist()
        # for i in range(B):
        #     for j in range(T - D):
        #         dataset_c[shuffled_pos[i * B + j]] = sum_capacity_dynamics[i, j : j + D]
        #         dataset_c_prime[shuffled_pos[i * B + j]] = sum_capacity_dynamics[
        #             i, j + D
        #         ]
        # train_x = dataset_c[: int(L * 0.8)]
        # train_y = dataset_c_prime[: int(L * 0.8)]
        # train_dataset = torch.utils.data.TensorDataset(train_x, train_y)
        # val_x = dataset_c[int(L * 0.8) :]
        # val_y = dataset_c_prime[int(L * 0.8) :]
        # val_dataset = torch.utils.data.TensorDataset(val_x, val_y)

        # for _ in range(20):
        #     self.context_model.train()
        #     train_dataloader = torch.utils.data.DataLoader(
        #         train_dataset, batch_size=self.context_batch_size, shuffle=True
        #     )
        #     total_train_loss = 0
        #     total_train_num = 0
        #     for i, (x, y) in enumerate(train_dataloader):
        #         pred = self.context_model(x)
        #         loss = torch.nn.functional.mse_loss(pred, y)
        #         total_train_loss += loss.item()
        #         total_train_num += len(x)
        #         self.context_optimiser.zero_grad()
        #         loss.backward()
        #         self.context_optimiser.step()

        #     self.context_model.eval()
        #     val_dataloader = torch.utils.data.DataLoader(
        #         val_dataset, batch_size=self.context_batch_size, shuffle=False
        #     )
        #     total_val_loss = 0
        #     total_val_num = 0
        #     for i, (x, y) in enumerate(val_dataloader):
        #         with torch.no_grad():
        #             pred = self.context_model(x)
        #             loss = torch.nn.functional.mse_loss(pred, y)
        #         total_val_loss += loss.item()
        #         total_val_num += len(x)

        # if t_env - self.log_stats_t >= self.args.learner_log_interval:
        #     self.logger.log_stat(
        #         "c_train_loss", total_train_loss / total_train_num, t_env
        #     )
        #     self.logger.log_stat("c_val_loss", total_val_loss / total_val_num, t_env)
        #     self.log_stats_t = t_env

    def cuda(self):
        self.context_model.cuda()

    def save_models(self, path):
        torch.save(self.context_model.state_dict(), "{}/context.th".format(path))
        torch.save(
            self.context_optimiser.state_dict(), "{}/context_opt.th".format(path)
        )

    def load_models(self, path):
        self.context_model.load_state_dict(
            torch.load(
                "{}/context.th".format(path), map_location=lambda storage, loc: storage
            )
        )
        self.context_optimiser.load_state_dict(
            torch.load(
                "{}/context_opt.th".format(path),
                map_location=lambda storage, loc: storage,
            )
        )

    def run_visualize(self, path):
        B, T, _ = self.capacity_dynamics.shape
        D = self.hist_len
        device = self.capacity_dynamics.device
        fake_dynamics = torch.empty(B, T, 3).to(device)
        for i in range(B):
            fake_dynamics[i, :D] = self.capacity_dynamics[i, :D]
        with torch.no_grad():
            for i in range(D, T):
                fake_dynamics[:, i] = self.context_model(
                    self.capacity_dynamics[:, i - D : i]
                )
        for i in range(B):
            sns.lineplot(x=range(T), y=fake_dynamics[i, :, 0].cpu().numpy())
            sns.lineplot(x=range(T), y=self.capacity_dynamics[i, :, 0].cpu().numpy())
            plt.legend([f"utilization_pred_{i}", f"utilization_gt_{i}"])
            plt.savefig(f"{path}/utilization_{i}.jpg", dpi=800)
            plt.clf()
            sns.lineplot(x=range(T), y=fake_dynamics[i, :, 1].cpu().numpy())
            sns.lineplot(x=range(T), y=self.capacity_dynamics[i, :, 1].cpu().numpy())
            plt.legend([f"unloading_pred_{i}", f"unloading_gt_{i}"])
            plt.savefig(f"{path}/unloading_{i}.jpg", dpi=800)
            plt.clf()
            sns.lineplot(x=range(T), y=fake_dynamics[i, :, 2].cpu().numpy())
            sns.lineplot(x=range(T), y=self.capacity_dynamics[i, :, 2].cpu().numpy())
            plt.legend([f"excess_pred_{i}", f"excess_gt_{i}"])
            plt.savefig(f"{path}/excess_{i}.jpg", dpi=800)
            plt.clf()
