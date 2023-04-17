import os
from collections import defaultdict
from functools import partial
from multiprocessing import Pipe, Process

import numpy as np
import torch.nn.functional as F

import wandb
from components.episode_buffer import EpisodeBatch
from envs import REGISTRY as env_REGISTRY
from utils.timehelper import TimeStat


# Based (very) heavily on SubprocVecEnv from OpenAI Baselines
# https://github.com/openai/baselines/blob/master/baselines/common/vec_env/subproc_vec_env.py
class ParallelRunner:
    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.batch_size = self.args.batch_size_run
        # Make subprocesses for the envs
        # 创建管道，用于一读一写的双工情况，返回值为两个connection对象，其中一个用于读，一个用于写
        self.parent_conns, self.worker_conns = zip(
            *[Pipe() for _ in range(self.batch_size)]
        )
        env_fn = env_REGISTRY[self.args.env]
        # self.n_warehouse = env_fn.warehouse_count()
        env_args = [self.args.env_args.copy() for _ in range(self.batch_size)]

        for i in range(len(env_args)):
            env_args[i]["seed"] += i

        # 创建进程列表self.ps，其中每个元素都是一个通信，首先听取来自parent_conns的信息，然后将信息采用env_worker函数来发送信息给环境，进行交互
        self.ps = [
            Process(
                # env_workder需要两个参数，第一个是说明pipe的一端，第二个是说明采用的环境
                target=env_worker,
                args=(worker_conn, CloudpickleWrapper(partial(env_fn, **env_arg))),
            )
            for env_arg, worker_conn in zip(env_args, self.worker_conns)
        ]
        for p in self.ps:
            p.daemon = True
            p.start()

        self.parent_conns[0].send(("get_env_info", None))
        self.env_info = self.parent_conns[0].recv()
        self.episode_limit = self.env_info["episode_limit"]
        self.t = 0

        self.t_env = 0

        self.train_returns = []
        self.test_returns = []
        self.train_profits = []
        self.test_profits = []
        self.train_stats = {}
        self.test_stats = {}

        # self.time_stats = defaultdict(lambda: TimeStat(1000))
        self.log_train_stats_t = -100000

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
        self.scheme = scheme
        self.groups = groups
        self.preprocess = preprocess

    def get_env_info(self):
        return self.env_info

    def save_replay(self):
        pass

    def close_env(self):
        for parent_conn in self.parent_conns:
            parent_conn.send(("close", None))

    def reset(self, test_mode=False):
        self.batch = self.new_batch()

        # Reset the envs
        for parent_conn in self.parent_conns:
            parent_conn.send(("switch_mode", "eval" if test_mode else "train"))

        for parent_conn in self.parent_conns:
            parent_conn.send(("reset", None))

        pre_transition_data = {
            "state": [],
            "avail_actions": [],
            "obs": [],
            "mean_action": [],
        }
        # Get the obs, state and avail_actions back
        for parent_conn in self.parent_conns:
            # data是env_worker与环境交互后所得到的东西, 这里在等待交互后从环境听取信息
            data = parent_conn.recv()
            pre_transition_data["state"].append(data["state"])
            pre_transition_data["avail_actions"].append(data["avail_actions"])
            pre_transition_data["obs"].append(data["obs"])
            pre_transition_data["mean_action"].append(
                np.zeros([1, self.args.n_agents, self.args.n_actions])
            )

        self.batch.update(pre_transition_data, ts=0)

        self.t = 0
        # 在多线程之中，经过的总的时间步数
        self.env_steps_this_run = 0
        self.C_trajectories = []

    # visual_outputs_path如果不是none，那么隔一段时间就会开始一次视觉render
    def run(self, test_mode=False, visual_outputs_path=None):
        self.reset(test_mode=test_mode)

        all_terminated = False
        episode_returns = [0 for _ in range(self.batch_size)]
        episode_lengths = [0 for _ in range(self.batch_size)]
        episode_balance = [0 for _ in range(self.batch_size)]
        episode_individual_returns = np.zeros([self.batch_size, self.args.n_agents])

        self.mac.init_hidden(batch_size=self.batch_size)
        terminated = [False for _ in range(self.batch_size)]
        # 这里是还没有到terminate状态的env的序号
        envs_not_terminated = [
            b_idx for b_idx, termed in enumerate(terminated) if not termed
        ]
        final_env_infos = (
            []
        )  # may store extra stats like battle won. this is filled in ORDER OF TERMINATION

        save_probs = getattr(self.args, "save_probs", False)
        while True:

            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch for each un-terminated env
            # TODO:这两个不是一样的？参数也是一样的

            if save_probs:
                actions, probs = self.mac.select_actions(
                    self.batch,
                    t_ep=self.t,
                    t_env=self.t_env,
                    bs=envs_not_terminated,
                    test_mode=test_mode,
                )
            else:
                actions = self.mac.select_actions(
                    self.batch,
                    t_ep=self.t,
                    t_env=self.t_env,
                    bs=envs_not_terminated,
                    test_mode=test_mode,
                )
                
            cpu_actions = actions.to("cpu").numpy()

            # Update the actions taken
            actions_chosen = {
                "actions": actions.unsqueeze(1).to("cpu"),
            }
            if save_probs:
                actions_chosen["probs"] = probs.unsqueeze(1).to("cpu")

            self.batch.update(
                actions_chosen, bs=envs_not_terminated, ts=self.t, mark_filled=False
            )

            # Send actions to each env
            action_idx = 0
            for idx, parent_conn in enumerate(self.parent_conns):
                if idx in envs_not_terminated:  # We produced actions for this env
                    if not terminated[
                        idx
                    ]:  # Only send the actions to the env if it hasn't terminated
                        parent_conn.send(("step", cpu_actions[action_idx]))
                    action_idx += 1  # actions is not a list over every env

            # Update envs_not_terminated
            envs_not_terminated = [
                b_idx for b_idx, termed in enumerate(terminated) if not termed
            ]
            all_terminated = all(terminated)
            if all_terminated:
                break

            # Post step data we will insert for the current timestep
            post_transition_data = {
                "reward": [],
                "terminated": [],
                "individual_rewards": [],
                "cur_balance": []
            }
            # Data for the next step we will insert in order to select an action
            pre_transition_data = {
                "state": [],
                "avail_actions": [],
                "obs": [],
                "mean_action": [],
            }

            # Receive data back for each unterminated env
            for idx, parent_conn in enumerate(self.parent_conns):
                if not terminated[idx]:
                    data = parent_conn.recv()
                    # print(f"data: {data}")
                    # Remaining data for this current timestep
                    post_transition_data["reward"].append((data["reward"],))
                    post_transition_data["individual_rewards"].append(
                        data["info"]["individual_rewards"]
                    )
                    post_transition_data["cur_balance"].append(
                        data["info"]["cur_balance"]
                    )
                    # print(f"cur_balance: {data['info']['cur_balance'].shape}")
                    episode_returns[idx] += data["reward"]
                    if self.args.n_agents > 1:
                        episode_individual_returns[idx] += data["info"][
                            "individual_rewards"
                        ]
                    else:
                        episode_individual_returns[idx] += data["info"][
                            "individual_rewards"
                        ][0]
                    # 这个就是每个交互得到的balance，但是好像并没有进行处理？
                    episode_balance[idx] = data["info"]["cur_balance"]
                    # print(f"episode_balance: {len(episode_balance)}")
                    episode_lengths[idx] += 1
                    if not test_mode:
                        self.env_steps_this_run += 1

                    env_terminated = False
                    # 如果是结束状态，那么就将结束的info添加，用于最后的统计结果
                    if data["terminated"]:
                        final_env_infos.append(data["info"])
                    if data["terminated"] and not data["info"].get(
                        "episode_limit", False
                    ):
                        env_terminated = True
                    terminated[idx] = data["terminated"]
                    post_transition_data["terminated"].append((env_terminated,))

                    # Data for the next timestep needed to select an action
                    pre_transition_data["state"].append(data["state"])
                    pre_transition_data["avail_actions"].append(data["avail_actions"])
                    pre_transition_data["obs"].append(data["obs"])
                    pre_transition_data["mean_action"].append(
                        F.one_hot(actions[idx], self.env_info["n_actions"])
                        .float()
                        .mean(dim=0)
                        .view(1, 1, -1)
                        .repeat(1, self.args.n_agents, 1)
                        .cpu()
                        .numpy()
                    )

            # Add post_transiton data into the batch
            self.batch.update(
                post_transition_data,
                bs=envs_not_terminated,
                ts=self.t,
                mark_filled=False,
            )

            # Move onto the next timestep
            self.t += 1

            # Add the pre-transition data
            self.batch.update(
                pre_transition_data, bs=envs_not_terminated, ts=self.t, mark_filled=True
            )

        if not test_mode:
            self.t_env += self.env_steps_this_run
        # Get profit for each env
        episode_profits = []
        for parent_conn in self.parent_conns:
            # get_profit函数返回的，就是在整次交互中所得到的balance
            parent_conn.send(("get_profit", None))
        # 前一步送出了请求，这一步来接收
        for parent_conn in self.parent_conns:
            episode_profit = parent_conn.recv()
            # 将episode_profit，用真实得到的profit除以时间t，然后再乘以最大时间长度，得到一个类似于最大时间长度的总量？
            episode_profits.append(episode_profit / self.t * (self.episode_limit))

        for parent_conn in self.parent_conns:
            parent_conn.send(("get_C_trajectory", None))
        for parent_conn in self.parent_conns:
            C_trajectory = parent_conn.recv()
            self.C_trajectories.append(C_trajectory)

        # Get stats back for each env
        for parent_conn in self.parent_conns:
            parent_conn.send(("get_stats", None))

        env_stats = []
        for parent_conn in self.parent_conns:
            env_stat = parent_conn.recv()
            env_stats.append(env_stat)

        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns
        cur_profits = self.test_profits if test_mode else self.train_profits
        # log_prefix = "test_" if test_mode else ""
        if test_mode:
            log_prefix = "test" if visual_outputs_path is not None else "val"
        else:
            log_prefix = "train"
        infos = [cur_stats] + final_env_infos

        cur_stats.update(
            {
                k: sum(d.get(k, 0) for d in infos)
                for k in set.union(*[set(d) for d in infos])
            }
        )
        cur_stats["n_episodes"] = self.batch_size + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = sum(episode_lengths) + cur_stats.get("ep_length", 0)
        cur_stats['max_in_stock_sum_max'] = max([d['max_in_stock_sum'] for d in final_env_infos])
        cur_stats['max_in_stock_sum_min'] = min([d['max_in_stock_sum'] for d in final_env_infos])
        cur_stats['max_in_stock_sum_mean'] = sum([d['max_in_stock_sum'] for d in final_env_infos])/len([d['max_in_stock_sum'] for d in final_env_infos])

        cur_returns.extend(episode_returns)
        cur_profits.extend(episode_profits)

        n_test_runs = (
            max(1, self.args.test_nepisode // self.batch_size) * self.batch_size
        )
        if test_mode and (len(self.test_returns) == n_test_runs):
            # print("if")
            self._log(
                cur_returns,
                episode_individual_returns,
                cur_profits,
                cur_stats,
                log_prefix,
            )
            if visual_outputs_path is not None:
                for idx, parent_conn in enumerate(self.parent_conns):
                    output_path = os.path.join(visual_outputs_path, f"batch_{idx}")
                    parent_conn.send(("visualize_render", output_path))

        elif self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
            self._log(
                cur_returns,
                episode_individual_returns,
                cur_profits,
                cur_stats,
                log_prefix,
            )
            # for key, value in self.time_stats.items():
            #     self.logger.log_stat(f"{key}_time_mean", value.mean, self.t_env)
            if hasattr(self.mac.action_selector, "epsilon"):
                self.logger.log_stat(
                    "epsilon", self.mac.action_selector.epsilon, self.t_env
                )
            self.log_train_stats_t = self.t_env

        return self.batch

    def get_C_trajectories(self):
        return np.array(self.C_trajectories)
    
    # TODO:注意看这个函数有没有问题,因为统计出来的利润不对
    def get_overall_avg_balance(self):
        for parent_conn in self.parent_conns:
            parent_conn.send(("get_profit", None))
        cur_balances = []
        for parent_conn in self.parent_conns:
            cur_balances.append(parent_conn.recv())
        cur_balances = np.array(cur_balances)
        # 为了展示每个商店的profit而设计的
        n_store = 3
        # 将每一层的每个sku都独立的看作智能体？
        n_skus = int(self.args.n_agents/n_store)
        cur_store_balances = []
        for i in range(n_store):
            store_profit = cur_balances[:,i*n_skus:(i+1)*n_skus]
            cur_store_balances.append(np.mean(np.sum(store_profit,axis = 1)))

        # cur_balances中每个元素都是拉平之后的数组。其中，前num_sku个数字，就是store1的balance
        # TODO:怎么每个store的balance都是一样的？测试的时候是不是没有必要parallel runner？
        return np.mean(np.sum(cur_balances, axis = 1)), cur_store_balances

    def _log(self, returns, individual_returns, profits, stats, prefix):
        self.logger.log_stat(prefix + "_return_mean", np.mean(returns), self.t_env)
        self.logger.log_stat(prefix + "_return_std", np.std(returns), self.t_env)
        returns.clear()

        self.logger.log_stat(prefix + "_profit_mean", np.mean(profits), self.t_env)
        self.logger.log_stat(prefix + "_profit_std", np.std(profits), self.t_env)
        profits.clear()

        if self.args.use_wandb and self.args.n_agents <= 100:
            for i in range(self.args.n_agents):
                wandb.log(
                    {
                        f"SKUReturn/joint_{prefix}_sku{i+1}_mean": individual_returns[
                            :, i
                        ].mean()
                    },
                    step=self.t_env,
                )

            for i in range(self.args.n_agents):
                for parent_conn in self.parent_conns:
                    parent_conn.send(("get_reward_dict", None))
                reward_dicts = []
                for parent_conn in self.parent_conns:
                    reward_dicts.append(parent_conn.recv())

                for parent_conn in self.parent_conns:
                    parent_conn.send(("get_profit", None))
                cur_balances = []
                for parent_conn in self.parent_conns:
                    cur_balances.append(parent_conn.recv())
                wandb.log(
                    {
                        f"SKUReturn_{k}/joint_{prefix}_sku{i+1}_mean": np.mean(
                            [np.array(rd[k])[:, i].sum() / 1e6 for rd in reward_dicts]
                        )
                        for k in reward_dicts[0].keys()
                    },
                    step=self.t_env,
                )
                wandb.log(
                    {
                        f"SKUBalance/joint_{prefix}_sku{i+1}_mean": np.mean(
                            np.array(cur_balances)[:, i]
                        )
                    },
                    step=self.t_env,
                )
            wandb.log(
                    {
                        f"SumBalance/joint_{prefix}_sum": np.mean(
                            np.sum(np.array(cur_balances), 1)
                        )
                    },
                    step=self.t_env,
            )    

        if self.args.use_wandb:
            wandb.log(
                    {
                        f"instock_sum/{prefix}_max_in_stock_sum_mean": stats['max_in_stock_sum_mean'],
                        f"instock_sum/{prefix}_max_in_stock_sum_min": stats['max_in_stock_sum_min'],
                        f"instock_sum/{prefix}_max_in_stock_sum_max": stats['max_in_stock_sum_max'],
                    },
                    step=self.t_env,
            )    
        self.logger.log_stat(
            prefix + "_max_in_stock_sum_mean", stats['max_in_stock_sum_mean'], self.t_env
        )
        self.logger.log_stat(
            prefix + "_max_in_stock_sum_min", stats['max_in_stock_sum_min'], self.t_env
        )
        self.logger.log_stat(
            prefix + "_max_in_stock_sum_max", stats['max_in_stock_sum_max'], self.t_env
        )

        for k, v in stats.items():
            if k not in ["n_episodes", "individual_rewards"]:
                self.logger.log_stat(
                    prefix + "_" + k + "_mean", v / stats["n_episodes"], self.t_env
                )

        stats.clear()


def env_worker(remote, env_fn):
    # Make environment
    env = env_fn.x()
    while True:
        # remote先进入一个等待状态，等待收听东西，然后接收到了东西之后，经过环境处理再send出去
        cmd, data = remote.recv()
        if cmd == "step":
            actions = data
            # Take a step in the environment
            # 这里的reward是一个int。代表着batch['reward']中，只有一个int值，所有智能体共享一个reward
            reward, terminated, env_info = env.step(actions)
            # Return the observations, avail_actions and state to make the next action
            state = env.get_state()
            avail_actions = env.get_avail_actions()
            obs = env.get_obs()
            # obs中每个智能体加上自己所在的层数
            
            remote.send(
                {
                    # Data for the next timestep needed to pick an action
                    "state": state,
                    "avail_actions": avail_actions,
                    "obs": obs,
                    # Rest of the data for the current timestep
                    "reward": reward,
                    "terminated": terminated,
                    "info": env_info,
                }
            )
        elif cmd == "reset":
            env.reset()
            remote.send(
                {
                    "state": env.get_state(),
                    "avail_actions": env.get_avail_actions(),
                    "obs": env.get_obs(),
                }
            )
        elif cmd == "close":
            env.close()
            remote.close()
            break
        elif cmd == "get_env_info":
            remote.send(env.get_env_info())
        elif cmd == "get_stats":
            remote.send(env.get_stats())
        elif cmd == "switch_mode":
            mode = data
            env.switch_mode(mode)
        # 查看这个函数
        elif cmd == "get_profit":
            remote.send(env.get_profit())
        elif cmd == "get_C_trajectory":
            remote.send(env.get_C_trajectory())
        elif cmd == "get_reward_dict":
            remote.send(env._env.reward_monitor)
        elif cmd == "visualize_render":
            env.visualize_render(data)
        else:
            raise NotImplementedError


class CloudpickleWrapper:
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
    """

    def __init__(self, x):
        self.x = x

    def __getstate__(self):
        import cloudpickle

        return cloudpickle.dumps(self.x)

    def __setstate__(self, ob):
        import pickle

        self.x = pickle.loads(ob)