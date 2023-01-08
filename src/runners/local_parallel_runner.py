import copy
import os
from collections import defaultdict
from functools import partial
from multiprocessing import Pipe, Process

import numpy as np
import torch.nn.functional as F
from matplotlib import use

import wandb
from components.episode_buffer import EpisodeBatch
from envs import REGISTRY as env_REGISTRY
from utils.timehelper import TimeStat


# Based (very) heavily on SubprocVecEnv from OpenAI Baselines
# https://github.com/openai/baselines/blob/master/baselines/common/vec_env/subproc_vec_env.py
class LocalParallelRunner:
    def __init__(self, args, use_wandb=False):
        self.args = copy.deepcopy(args)
        # self.logger = logger
        self.use_wandb = use_wandb
        self.batch_size = self.args.batch_size_run

        # ctx = torch.multiprocessing.get_context("spawn")
        # Make subprocesses for the envs
        self.parent_conns, self.worker_conns = zip(
            *[Pipe() for _ in range(self.batch_size)]
        )
        env_fn = env_REGISTRY[self.args.env]

        env_args = []
        for _ in args.agent_ids:
            env_arg = self.args.env_args.copy()
            env_args.append(env_arg)

        for i in range(len(env_args)):
            env_args[i]["seed"] += i

        self.ps = [
            Process(
                target=env_worker,
                args=(worker_conn, CloudpickleWrapper(partial(env_fn, **env_arg))),
            )
            for env_arg, worker_conn in zip(env_args, self.worker_conns)
        ]
        for p in self.ps:
            p.daemon = True
            p.start()

        for i in range(len(self.parent_conns)):
            self.parent_conns[i].send(("set_local_SKU", self.args.agent_ids[i]))

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

    def restart_ps(self, agent_ids):
        self.clean_ps()
        # ctx = torch.multiprocessing.get_context("spawn")
        self.args.agent_ids = agent_ids
        self.parent_conns, self.worker_conns = zip(
            *[Pipe() for _ in range(len(agent_ids))]
        )
        env_fn = env_REGISTRY[self.args.env]
        env_args = []
        for _ in agent_ids:
            env_arg = self.args.env_args.copy()
            env_args.append(env_arg)
        for i in range(len(agent_ids)):
            env_args[i]["seed"] += i

        self.ps = [
            Process(
                target=env_worker,
                args=(worker_conn, CloudpickleWrapper(partial(env_fn, **env_arg))),
            )
            for env_arg, worker_conn in zip(env_args, self.worker_conns)
        ]
        for p in self.ps:
            p.daemon = True
            p.start()
        self.batch_size = len(agent_ids)

    def clean_ps(self):
        for p in self.ps:
            p.terminate()
            # p.close()

    def get_env_info(self):
        return self.env_info

    def save_replay(self):
        pass

    def close_env(self):
        for parent_conn in self.parent_conns:
            parent_conn.send(("close", None))

    def reset(self, test_mode=False, C_trajectories=None):
        self.batch = self.new_batch()

        # Reset the envs
        for parent_conn in self.parent_conns:
            parent_conn.send(("switch_mode", "eval" if test_mode else "train"))

        assert C_trajectories is not None
        for i in range(len(self.parent_conns)):
            self.parent_conns[i].send(("set_C_trajectory", C_trajectories[i]))
            self.parent_conns[i].send(("set_local_SKU", self.args.agent_ids[i]))

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
            data = parent_conn.recv()
            pre_transition_data["state"].append(data["state"])
            pre_transition_data["avail_actions"].append(data["avail_actions"])
            pre_transition_data["obs"].append(data["obs"])
            pre_transition_data["mean_action"].append(
                np.zeros([1, self.args.n_agents, self.args.n_actions])
            )
        self.batch.update(pre_transition_data, ts=0)

        self.t = 0
        self.env_steps_this_run = 0

    def run(self, test_mode=False, C_trajectories=None, visualize=False, visual_outputs_path=None):

        self.reset(test_mode=test_mode, C_trajectories=C_trajectories)

        all_terminated = False
        episode_returns = [0 for _ in range(self.batch_size)]
        episode_lengths = [0 for _ in range(self.batch_size)]
        self.mac.init_hidden(batch_size=self.batch_size)
        terminated = [False for _ in range(self.batch_size)]
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
            # with self.time_stats["select_action"]:
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
            actions_chosen = {"actions": actions.unsqueeze(1).to("cpu")}
            if save_probs:
                actions_chosen["probs"] = probs.unsqueeze(1).to("cpu")

            self.batch.update(
                actions_chosen, bs=envs_not_terminated, ts=self.t, mark_filled=False
            )

            # with self.time_stats["env_step"]:
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
                if visualize:
                    for idx, parent_conn in enumerate(self.parent_conns):
                        parent_conn.send(("visualize", os.path.join(visual_outputs_path, 
                                            "sku" + str(self.args.agent_ids[idx]))))
                break

            # Post step data we will insert for the current timestep
            post_transition_data = {
                "reward": [],
                "cur_balance" : [],
                "terminated": [],
                "individual_rewards": [],
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
                    # Remaining data for this current timestep

                    post_transition_data["reward"].append((data["reward"],))
                    post_transition_data["cur_balance"].append((data["info"]["cur_balance"],))
                    post_transition_data["individual_rewards"].append(
                        data["info"]["individual_rewards"]
                    )

                    episode_returns[idx] += data["reward"]
                    episode_lengths[idx] += 1
                    if not test_mode:
                        self.env_steps_this_run += 1

                    env_terminated = False
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
            # with self.time_stats["batch_update"]:
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

        if self.use_wandb:
            for i in range(self.batch_size):
                wandb.log(
                    {f"SKUReturn/local_train_sku{i+1}_mean": episode_returns[i]},
                    step=wandb.run.step,
                )

        return self.batch


def env_worker(remote, env_fn):
    # Make environment
    env = env_fn.x()
    while True:
        cmd, data = remote.recv()
        if cmd == "step":
            actions = data
            # Take a step in the environment
            reward, terminated, env_info = env.step(actions)
            # Return the observations, avail_actions and state to make the next action
            state = env.get_state()
            avail_actions = env.get_avail_actions()
            obs = env.get_obs()
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
        elif cmd == "get_profit":
            remote.send(env.get_profit())
        elif cmd == "set_C_trajectory":
            C_trajectory = data
            env.set_C_trajectory(C_trajectory)
        elif cmd == "set_local_SKU":
            local_SKU = data
            env.set_local_SKU(local_SKU)
        elif cmd == "visualize":
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
