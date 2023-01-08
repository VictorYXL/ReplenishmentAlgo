import os
from collections import defaultdict
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# from replenishment.efficient_env import EfficientReplenishmentEnv
# from replenishment.efficient_env.utility.tools import SimulationTracker

# from replenishment.n100_local.render.inventory_renderer import AsciiWorldRenderer
from components.episode_buffer import EpisodeBatch
from envs import REGISTRY as env_REGISTRY
from learners import REGISTRY as le_REGISTRY
from utils.timehelper import TimeStat


class EpisodeRunner:
    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.batch_size = 1

        self.env = env_REGISTRY[self.args.env](**self.args.env_args)
        self.episode_limit = self.env.episode_limit
        self.t = 0

        self.t_env = 0

        self.train_returns = []
        self.test_returns = []
        self.train_profits = []
        self.test_profits = []
        self.train_stats = {}
        self.test_stats = {}

        # self.time_stats = defaultdict(lambda: TimeStat(1000))

        # Log the first run
        self.log_train_stats_t = -1000000

    def setup(self, scheme, groups, preprocess, mac):
        self.new_batch = partial(
            EpisodeBatch,
            scheme,
            groups,
            self.batch_size,
            self.episode_limit + 1,
            preprocess=preprocess,
            device=self.args.device,
        )
        self.mac = mac

    def get_env_info(self):
        return self.env.get_env_info()

    def save_replay(self):
        pass

    def close_env(self):
        self.env.close()

    def reset(self, test_mode=False):
        self.batch = self.new_batch()
        self.env.switch_mode("eval" if test_mode else "train")
        self.env.reset()
        self.t = 0

    def run(self, test_mode=False, visulize_mode=False):
        self.reset(test_mode=test_mode)

        terminated = False
        episode_return = 0
        self.mac.init_hidden(batch_size=self.batch_size)
        save_probs = getattr(self.args, "save_probs", False)
        while not terminated:

            pre_transition_data = {
                "state": [self.env.get_state()],
                "avail_actions": [self.env.get_avail_actions()],
                "obs": [self.env.get_obs()],
                # "individual_rewards": [np.zeros(self.args.n_agents)]
            }
            # with self.time_stats["batch_update"]:
            self.batch.update(pre_transition_data, ts=self.t)

            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch of size 1
            # with self.time_stats["select_action"]:
            if save_probs:
                actions, probs = self.mac.select_actions(
                    self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode
                )
            else:
                actions = self.mac.select_actions(
                    self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode
                )
            # with self.time_stats["env_step"]:
            reward, terminated, env_info = self.env.step(
                actions[0].flatten().cpu().numpy()
            )
            episode_return += reward

            post_transition_data = {
                "actions": actions,
                "reward": [(reward,)],
                "terminated": [(terminated != env_info.get("episode_limit", False),)],
                "probs": probs
                # "individual_rewards": env_info["individual_rewards"]
            }
            self.batch.update(post_transition_data, ts=self.t)

            self.t += 1

        last_data = {
            "state": [self.env.get_state()],
            "avail_actions": [self.env.get_avail_actions()],
            "obs": [self.env.get_obs()],
        }
        self.batch.update(last_data, ts=self.t)

        # Select actions in the last stored state
        if save_probs:
            actions, probs = self.mac.select_actions(
                self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode
            )
            self.batch.update({"actions": actions, "probs": probs}, ts=self.t)
        else:
            actions = self.mac.select_actions(
                self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode
            )
            self.batch.update({"actions": actions}, ts=self.t)

        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns
        cur_profits = self.test_profits if test_mode else self.train_profits
        log_prefix = "test_" if test_mode else ""
        cur_stats.update(
            {
                k: cur_stats.get(k, 0) + env_info.get(k, 0)
                for k in set(cur_stats) | set(env_info)
            }
        )
        cur_stats["n_episodes"] = 1 + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = self.t + cur_stats.get("ep_length", 0)

        if not test_mode:
            self.t_env += self.t

        if not visulize_mode:
            cur_returns.append(episode_return)
            cur_profits.append(
                self.env.get_profit() / self.t * (self.episode_limit)
            )
            if test_mode and (len(self.test_returns) == self.args.test_nepisode):
                self._log(cur_returns, cur_profits, cur_stats, log_prefix)
                # for key, value in self.time_stats.items():
                #     self.logger.log_stat(f"{key}_time_mean", value.mean, self.t_env)
            elif self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
                self._log(cur_returns, cur_profits, cur_stats, log_prefix)
                # for key, value in self.time_stats.items():
                #     self.logger.log_stat(f"{key}_time_mean", value.mean, self.t_env)
                if hasattr(self.mac.action_selector, "epsilon"):
                    self.logger.log_stat(
                        "epsilon", self.mac.action_selector.epsilon, self.t_env
                    )
                self.log_train_stats_t = self.t_env

        return self.batch

    def _log(self, returns, profits, stats, prefix):
        self.logger.log_stat(prefix + "return_mean", np.mean(returns), self.t_env)
        self.logger.log_stat(prefix + "return_std", np.std(returns), self.t_env)
        returns.clear()

        self.logger.log_stat(prefix + "profit_mean", np.mean(profits), self.t_env)
        self.logger.log_stat(prefix + "profit_std", np.std(profits), self.t_env)
        profits.clear()

        for k, v in stats.items():
            if k not in ["n_episodes", "individual_rewards"]:
                self.logger.log_stat(
                    prefix + k + "_mean", v / stats["n_episodes"], self.t_env
                )
        stats.clear()

    def run_visualize(self, visual_outputs_path, t):
        pass
        # # self.reset(test_mode=False)
        # self.mac.init_hidden(batch_size=self.batch_size)
        # self.run(test_mode=False, visulize_mode=True)
        # os.makedirs(f"{visual_outputs_path}/{t}", exist_ok=True)
        # env: EfficientReplenishmentEnv = self.env._env.env.env
        # env.tracker.render(f"{visual_outputs_path}/plot_{t}.png")
        # env.tracker.render_sku(f"{visual_outputs_path}/{t}")
        # # - 商品角度
        # #     - [ ] demand/in transit/sale曲线
        # #     - [ ] 收益
        # #     - [ ] action 选择
        # #     - [ ] 各种reward
        # #     - [ ] sale mean变化

        # analysis_oytputs_path = visual_outputs_path.replace("visual", "analysis")
        # os.makedirs(f"{analysis_oytputs_path}/{t}", exist_ok=True)

        # sku_dfs = []
        # for i in range(1, self.args.n_agents + 1):
        #     # for k, v in env.sku_monitor.items():
        #     #     print(k, np.array(v).shape)
        #     df = pd.DataFrame.from_dict(
        #         {
        #             **{
        #                 k: np.array(v)[:, i - 1] / env.max_capacity
        #                 for k, v in env.sku_monitor.items()
        #             },
        #             **{
        #                 k: np.cumsum(np.array(v)[:, i - 1] / 1e6)
        #                 for k, v in env.reward_monitor.items()
        #             },
        #         }
        #     )
        #     df["excess_ratio"] *= env.max_capacity
        #     df["action"] *= env.max_capacity
        #     df["oracle_demand_mean"] = float(env.demand_mean[i - 1] / env.max_capacity)
        #     df["demand/oracle_demand_mean"] = df["demand"] / df["oracle_demand_mean"]
        #     df["replenish/oracle_demand_mean"] = (
        #         df["replenish_amount"] / df["oracle_demand_mean"]
        #     )
        #     df["day"] = range(env.episode_len)
        #     df["sku"] = f"sku{i}"

        #     plt.clf()
        #     plt.cla()
        #     plt.title(f"SKU{i}_behaviour_part1")
        #     fig, ax = plt.subplots(2, 1, figsize=(28, 20))
        #     ax[0].set_title(
        #         f"SKU{i} Demand and Sales Status (normed with MaxCapacity {env.max_capacity})"
        #     )
        #     ax[0].set_ybound(0, 1)
        #     dfm = df[
        #         ["day", "demand", "sales", "oracle_demand_mean", "replenish_amount"]
        #     ].melt("day", var_name="cols", value_name="vals")
        #     sns.lineplot(data=dfm, x="day", y="vals", hue="cols", ax=ax[0])

        #     ax[1].set_title(
        #         f"SKU{i} Stock Amount Status (normed with MaxCapacity {env.max_capacity})"
        #     )
        #     ax[1].set_ybound(0, 1)
        #     dfm = df[
        #         ["day", "replenish_amount", "unloading_amount", "excess_amount"]
        #     ].melt("day", var_name="cols", value_name="vals")
        #     sns.lineplot(data=dfm, x="day", y="vals", hue="cols", ax=ax[1])
        #     fig.savefig(f"{analysis_oytputs_path}/{t}/SKU{i}_behaviour_part1.png")

        #     plt.clf()
        #     plt.cla()
        #     plt.title(f"SKU{i}_behaviour_part2")
        #     fig, ax = plt.subplots(2, 1, figsize=(28, 20))
        #     ax[0].set_title(f"SKU{i} Action Status")
        #     dfm = df[["day", "action"]].melt("day", var_name="cols", value_name="vals")
        #     sns.lineplot(data=dfm, x="day", y="vals", hue="cols", ax=ax[0])

        #     ax[1].set_title(f"SKU{i} Stock and Transit Status")
        #     dfm = df[
        #         ["day", "in_stocks_begin", "in_stocks_end", "in_transits", "demand"]
        #     ].melt("day", var_name="cols", value_name="vals")
        #     sns.lineplot(data=dfm, x="day", y="vals", hue="cols", ax=ax[1])
        #     fig.savefig(f"{analysis_oytputs_path}/{t}/SKU{i}_behaviour_part2.png")

        #     plt.clf()
        #     plt.cla()
        #     plt.title(f"SKU{i}_behaviour_part3")
        #     fig, ax = plt.subplots(2, 1, figsize=(28, 20))
        #     ax[0].set_title(f"SKU{i} Relative Demand and Replenishment Status")
        #     dfm = df[
        #         ["day", "demand/oracle_demand_mean", "replenish/oracle_demand_mean"]
        #     ].melt("day", var_name="cols", value_name="vals")
        #     sns.lineplot(data=dfm, x="day", y="vals", hue="cols", ax=ax[0])

        #     ax[1].set_title(f"SKU{i} Return Status")
        #     dfm = df[["day"] + list(env.reward_monitor.keys())].melt(
        #         "day", var_name="cols", value_name="vals"
        #     )
        #     sns.lineplot(data=dfm, x="day", y="vals", hue="cols", ax=ax[1])
        #     fig.savefig(f"{analysis_oytputs_path}/{t}/SKU{i}_behaviour_part3.png")

        #     sku_dfs.append(df)

        # # - 商店角度
        # #     - [ ] 库存周转率
        # #     - [ ] 商品排序
        # # - [ ] 进货数量排序
        # # - [ ] 占用率排序
        # # - [ ] 销量排序

        # df = pd.DataFrame.from_dict(
        #     {
        #         "sales/oracle_demand_mean": np.argsort(
        #             np.array(env.sku_monitor["sales"]) / env.demand_mean, axis=1
        #         )[:, 0],
        #         "sku_storage_utilization": np.argsort(
        #             np.array(env.sku_monitor["in_stocks_begin"]), axis=1
        #         )[:, 0],
        #         **{
        #             k: np.argsort(np.array(v), axis=1)[:, 0]
        #             for k, v in env.reward_monitor.items()
        #         },
        #         "day": np.array(range(400)),
        #     }
        # )
        # dfm = df.melt("day", var_name="cols", value_name="vals")
        # plt.clf()
        # plt.cla()
        # plt.title("Store_behaviour_part1")
        # sns.lineplot(data=dfm, x="day", y="vals", hue="cols")
        # fig.savefig(f"{analysis_oytputs_path}/{t}/Store_behaviour_part1.png")
