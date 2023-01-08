# import copy
# import datetime
# import glob
# import os
# import pprint
# import re
# import threading
# import time
# from collections import defaultdict
# from dataclasses import replace
# from os.path import abspath, dirname
# from random import random
# from types import SimpleNamespace as SN

# import numpy as np
# import pandas as pd
# import seaborn as sns
# import torch
# from matplotlib import pyplot as plt

# import runners
# import wandb
# from components.episode_buffer import ReplayBuffer
# from components.transforms import OneHot
# from controllers import REGISTRY as mac_REGISTRY
# from envs import REGISTRY as env_REGISTRY
# from learners import REGISTRY as le_REGISTRY
# from runners import REGISTRY as r_REGISTRY
# from utils.logging import Logger
# from utils.timehelper import time_left, time_str


# def run(_run, _config, _log):

#     # check args sanity
#     _config = args_sanity_check(_config, _log)
#     if _config["n_decoupled_iterations"] == 0:
#         _config["train_with_joint_data"] = True

#     args = SN(**_config)
#     args.device = "cuda" if args.use_cuda else "cpu"

#     # setup loggers
#     logger = Logger(_log)

#     _log.info("Experiment Parameters:")
#     tmp_config = {k: _config[k] for k in _config if k != "env_args"}
#     tmp_config.update(
#         {f"env_agrs.{k}": _config["env_args"][k] for k in _config["env_args"]}
#     )
#     print(
#         pd.Series(tmp_config, name="HyperParameter Value")
#         .transpose()
#         .sort_index()
#         .fillna("")
#         .to_markdown()
#     )

#     # configure tensorboard logger
#     ts = datetime.datetime.now().strftime("%m%dT%H%M")
#     unique_token = f"{_config['name']}_{_config['env_args']['map_name']}_seed{_config['seed']}_{ts}"
#     args.unique_token = unique_token
#     if args.use_tensorboard:
#         tb_logs_direc = os.path.join(
#             dirname(dirname(dirname(abspath(__file__)))), "results", "tb_logs"
#         )
#         tb_exp_direc = os.path.join(tb_logs_direc, "{}").format(unique_token)
#         logger.setup_tb(tb_exp_direc)
#     if args.use_wandb:
#         logger.setup_wandb(args)

#     # sacred is on by default
#     logger.setup_sacred(_run)

#     # Run and train
#     run_sequential(args=args, logger=logger)

#     # Clean up after finishing
#     print("Exiting Main")

#     print("Stopping all threads")
#     for t in threading.enumerate():
#         if t.name != "MainThread":
#             print("Thread {} is alive! Is daemon: {}".format(t.name, t.daemon))
#             t.join(timeout=1)
#             print("Thread joined")

#     print("Exiting script")

#     # Making sure framework really exits
#     # os._exit(os.EX_OK)


# def evaluate_sequential(args, runner):

#     for _ in range(args.test_nepisode):
#         runner.run(test_mode=True)

#     if args.save_replay:
#         runner.save_replay()

#     runner.close_env()


# def run_sequential(args, logger):
#     # assert args.decoupled_training == args.use_individual_envs

#     # Init runner so we can get env info
#     # args.decoupled_training = not args.decoupled_training
#     runner = r_REGISTRY[args.runner](args, logger=logger)
#     # args.decoupled_training = not args.decoupled_training

#     # Set up schemes and groups here
#     env_info = env_REGISTRY[args.env](**args.env_args).get_env_info()
#     args.n_agents = env_info["n_agents"]
#     args.n_actions = env_info["n_actions"]
#     args.state_shape = env_info["state_shape"]

#     runner.args.n_agents = env_info["n_agents"]
#     # Default/Base scheme
#     scheme = {
#         "state": {"vshape": env_info["state_shape"]},
#         "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
#         "mean_action": {
#             "vshape": (env_info["n_actions"],),
#             "group": "agents",
#             "dtype": torch.float,
#         },
#         "actions": {"vshape": (1,), "group": "agents", "dtype": torch.long},
#         "avail_actions": {
#             "vshape": (env_info["n_actions"],),
#             "group": "agents",
#             "dtype": torch.int,
#         },
#         "probs": {
#             "vshape": (env_info["n_actions"],),
#             "group": "agents",
#             "dtype": torch.float,
#         },
#         "reward": {"vshape": (1,)},
#         "individual_rewards": {"vshape": (1,), "group": "agents"},
#         "terminated": {"vshape": (1,), "dtype": torch.uint8},
#     }
#     groups = {"agents": args.n_agents}
#     preprocess = {"actions": ("actions_onehot", [OneHot(out_dim=args.n_actions)])}

#     buffer = ReplayBuffer(
#         scheme,
#         groups,
#         args.buffer_size,
#         env_info["episode_limit"] + 1,
#         preprocess=preprocess,
#         device="cpu" if args.buffer_cpu_only else args.device,
#     )

#     logger.console_logger.info("MDP Components:")
#     print(pd.DataFrame(buffer.scheme).transpose().sort_index().fillna("").to_markdown())

#     # Setup multiagent controller here
#     mac = mac_REGISTRY[args.mac](buffer.scheme, groups, args)

#     # Give runner the scheme
#     runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac)

#     # Set up local parallel runner
#     local_logger = copy.copy(logger)
#     if args.use_tensorboard:
#         tb_logs_direc = os.path.join(
#             dirname(dirname(abspath(__file__))), "results", "tb_logs"
#         )
#         tb_exp_direc = os.path.join(tb_logs_direc, "{}").format(args.unique_token)
#         logger.setup_tb(tb_exp_direc)
#     local_logger.use_wandb = False

#     local_args = copy.deepcopy(args)
#     # local_args.use_individual_envs = True
#     local_args.critic_input_seq_str = "o_la^i"
#     local_args.learner = "local_ppo_learner"
#     if local_args.max_individual_envs < 1:
#         local_args.batch_size_run = max(
#             1, int(local_args.max_individual_envs * args.n_agents)
#         )
#         local_args.batch_size = max(
#             1, int(local_args.max_individual_envs * args.n_agents)
#         )
#         local_args.buffer_size = max(
#             1, int(local_args.max_individual_envs * args.n_agents)
#         )
#     else:
#         local_args.batch_size_run = local_args.max_individual_envs
#         local_args.batch_size = local_args.max_individual_envs
#         local_args.buffer_size = local_args.max_individual_envs
#     local_args.agent_ids = [1] * (
#         args.max_individual_envs
#         if args.max_individual_envs >= 1
#         else max(1, int(args.n_agents * args.max_individual_envs))
#     )
#     local_args.n_agents = 1  # local_env_info["n_agents"]
#     local_runner = r_REGISTRY["local_parallel"](
#         args=local_args, use_wandb=args.use_wandb
#     )
#     local_env_info = local_runner.get_env_info()

#     local_args.n_actions = local_env_info["n_actions"]
#     local_args.state_shape = local_env_info["obs_shape"]
#     local_scheme = {
#         "state": {"vshape": local_env_info["obs_shape"], "group": "agents"},
#         "obs": {"vshape": local_env_info["obs_shape"], "group": "agents"},
#         "actions": {"vshape": (1,), "group": "agents", "dtype": torch.long},
#         "mean_action": {
#             "vshape": (env_info["n_actions"],),
#             "group": "agents",
#             "dtype": torch.float,
#         },
#         "avail_actions": {
#             "vshape": (local_env_info["n_actions"],),
#             "group": "agents",
#             "dtype": torch.int,
#         },
#         "probs": {
#             "vshape": (env_info["n_actions"],),
#             "group": "agents",
#             "dtype": torch.float,
#         },
#         "reward": {"vshape": (1,)},
#         "individual_rewards": {"vshape": (1,), "group": "agents"},
#         "terminated": {"vshape": (1,), "dtype": torch.uint8},
#     }
#     local_groups = {"agents": local_args.n_agents}
#     local_preprocess = {
#         "actions": ("actions_onehot", [OneHot(out_dim=local_args.n_actions)])
#     }
#     local_infos = {
#         "local_scheme": local_scheme,
#         "local_groups": local_groups,
#         "buffer_size": local_args.buffer_size,
#         "episode_limit": local_env_info["episode_limit"],
#         "local_preprocess": local_preprocess,
#         "local_args": local_args,
#     }
#     local_buffer = ReplayBuffer(
#         local_scheme,
#         local_groups,
#         local_args.buffer_size,
#         local_env_info["episode_limit"] + 1,
#         preprocess=local_preprocess,
#         device="cpu" if local_args.buffer_cpu_only else local_args.device,
#     )

#     local_mac = mac_REGISTRY[local_args.mac](
#         local_buffer.scheme, local_groups, local_args
#     )
#     local_mac.agent = mac.agent
#     local_runner.setup(
#         scheme=local_scheme,
#         groups=local_groups,
#         preprocess=local_preprocess,
#         mac=local_mac,
#     )
    
#     val4test_args = copy.deepcopy(local_args)
#     val4test_args.env_args['map_name'] = "n58c2000d21s400l100"
#     val4test_args.env_args['time_limit'] = 100
#     val4test_args.n_agents = 1
#     val4test_args.buffer_size =58
#     val4test_args.batch_size_run = 58
#     val4test_args.batch_size = 58
#     val4test_args.agent_ids = [i+1 for i in range(58)]
#     val4test_args.action_selector = "multinomial" #"multinomial"
#     val4test_args.save_probs = False

#     val4test_runner = r_REGISTRY["local_parallel"](args=val4test_args, use_wandb=False)
#     val4test_groups = {"agents": 1}
#     val4test_mac = mac_REGISTRY[local_args.mac](
#         local_buffer.scheme, val4test_groups, val4test_args
#     )
#     val4test_mac.agent = mac.agent
#     val4test_runner.setup(
#         scheme=local_scheme,
#         groups=val4test_groups,
#         preprocess=local_preprocess,
#         mac=val4test_mac,
#     )

#     val4train_args = copy.deepcopy(local_args)
#     val4train_args.env_args['map_name'] = "n58c2000d21s0l400"
#     val4train_args.env_args['time_limit'] = 400
#     val4train_args.n_agents = 1
#     val4train_args.buffer_size =58
#     val4train_args.batch_size_run = 58
#     val4train_args.batch_size = 58
#     val4train_args.agent_ids = [i+1 for i in range(58)]
#     val4train_args.action_selector = "multinomial" #epsilon_greedy
#     val4train_args.save_probs = False

#     val4train_runner = r_REGISTRY["local_parallel"](args=val4train_args, use_wandb=False)
#     val4train_groups = {"agents": 1}
#     val4train_mac = mac_REGISTRY[local_args.mac](
#         local_buffer.scheme, val4test_groups, val4train_args
#     )
#     val4train_mac.agent = mac.agent
#     val4test_runner.setup(
#         scheme=local_scheme,
#         groups=val4train_groups,
#         preprocess=local_preprocess,
#         mac=val4train_mac,
#     )
#     val4train_runner.setup(
#         scheme=local_scheme,
#         groups=val4train_groups,
#         preprocess=local_preprocess,
#         mac=val4test_mac,
#     )


#     context_learner = le_REGISTRY["context_learner"](logger, args)
#     local_learner = le_REGISTRY["local_ppo_learner"](
#         local_mac, local_buffer.scheme, local_logger, local_args
#     )
#     if args.use_cuda:
#         context_learner.cuda()
#         local_learner.cuda()

#     if args.visualize:
#         visual_runner = runners.episode_runner.EpisodeRunner(args=args, logger=logger)
#         visual_runner.setup(
#             scheme=scheme, groups=groups, preprocess=preprocess, mac=mac
#         )

#     # Learner
#     learner = le_REGISTRY[args.learner](mac, buffer.scheme, logger, args)
#     if args.use_cuda:
#         learner.cuda()

#     print("Actor Arch:", mac.agent)
#     if args.train_with_joint_data:
#         print("Critic Arch:", learner.critic)
#     if args.n_decoupled_iterations > 0:
#         print("Local Critic Arch:", local_learner.critic)

#     if args.checkpoint_path != "":
#         timesteps = []
#         timestep_to_load = 0

#         if not os.path.isdir(args.checkpoint_path):
#             logger.console_logger.info(
#                 "Checkpoint directiory {} doesn't exist".format(args.checkpoint_path)
#             )
#             return

#         # Go through all files in args.checkpoint_path
#         for name in os.listdir(args.checkpoint_path):
#             full_name = os.path.join(args.checkpoint_path, name)
#             # Check if they are dirs the names of which are numbers
#             if os.path.isdir(full_name) and name.isdigit():
#                 timesteps.append(int(name))

#         if args.load_step == 0:
#             # choose the max timestep
#             timestep_to_load = max(timesteps)
#         else:
#             # choose the timestep closest to load_step
#             timestep_to_load = min(timesteps, key=lambda x: abs(x - args.load_step))

#         model_path = os.path.join(args.checkpoint_path, str(timestep_to_load))

#         logger.console_logger.info("Loading model from {}".format(model_path))
#         local_learner.load_models(model_path)
#         context_learner.load_models(model_path)
#         runner.t_env = timestep_to_load

#         if args.evaluate:
#             runner.log_train_stats_t = runner.t_env
#             # if args.n_decoupled_iterations > 0:
#             #     runner.mac.load_state(local_learner.mac)
#             evaluate_sequential(args, logger, runner)
#             logger.log_stat("episode", runner.t_env, runner.t_env)
#             logger.print_recent_stats()
#             logger.console_logger.info("Finished Evaluation")
#             return

#     # start training
#     episode = 0
#     local_episode = 0
#     last_test_T = -args.test_interval - 1
#     last_log_T = 0
#     model_save_time = 0
#     visual_time = 0

#     start_time = time.time()
#     last_time = start_time

#     logger.console_logger.info("Beginning training for {} timesteps".format(args.t_max))

#     while local_runner.t_env <= args.t_max:  # for the_i in range(1):#

#         for _ in range(args.n_decoupled_iterations):
#             if not args.select_ls_randomly:
#                 if args.max_individual_envs < 1:
#                     n_splits = int(1 / args.max_individual_envs)
#                     if n_splits > args.n_agents:
#                         n_splits = args.n_agents
#                 else:
#                     n_splits = args.n_agents // args.max_individual_envs
#                 for agent_ids in np.array_split(list(range(args.n_agents)), n_splits,):
#                     rets = run_decoupled_training(
#                         episode,
#                         (agent_ids + 1).tolist(),
#                         local_runner,
#                         local_infos,
#                         context_learner,
#                         local_learner,
#                         local_args,
#                     )
#                 joint_local_ratio = 1
#                 runner.t_env += args.env_args["time_limit"]

#             else:
#                 agent_ids = np.random.choice(
#                     # args.n_agents,
#                     58,
#                     size=[
#                         args.max_individual_envs
#                         if args.max_individual_envs >= 1
#                         else max(1, int(args.n_agents * args.max_individual_envs))
#                     ],
#                     replace=False,
#                 )
#                 rets = run_decoupled_training(
#                     episode,
#                     (agent_ids + 1).tolist(),
#                     local_runner,
#                     local_infos,
#                     context_learner,
#                     local_learner,
#                     local_args,
#                 )
#                 joint_local_ratio = args.max_individual_envs / (
#                     args.n_agents if args.max_individual_envs >= 1 else 1
#                 )
#                 if args.max_individual_envs >= 1:
#                     joint_local_ratio = args.max_individual_envs / args.n_agents
#                 else:
#                     joint_local_ratio = args.max_individual_envs
#                     if args.n_agents * args.max_individual_envs < 1:
#                         joint_local_ratio = 1 / args.n_agents
#                 runner.t_env += int(joint_local_ratio * 1 * args.env_args["time_limit"])
#             if args.use_wandb:
#                 wandb.log(rets, step=local_runner.t_env)

#             local_episode += (
#                 args.max_individual_envs
#                 if args.max_individual_envs >= 1
#                 else max(1, int(args.n_agents * args.max_individual_envs))
#             )

#             episode += args.batch_size_run + joint_local_ratio * 1

#         # if (
#         #     args.visualize
#         #     and ((local_runner.t_env - visual_time) / args.visualize_interval >= 1.0)
#         # ):
#         #     visual_time = local_runner.t_env
#         #     visual_outputs_path = os.path.join(
#         #         args.local_results_path, args.unique_token, "visual_outputs"
#         #     )
#         #     logger.console_logger.info(
#         #         f"Saving visualizations to {visual_outputs_path}/{local_runner.t_env}"
#         #     )

#         #     visual_runner.run_visualize2(visual_outputs_path, local_runner.t_env)
#         #     if args.use_wandb:
#         #         imgs = sorted(
#         #             glob.glob(
#         #                 f"{visual_outputs_path}/{local_runner.t_env}/OuterSKUStoreUnit*.png"
#         #             ),
#         #             key=os.path.getmtime,
#         #         )[::-1]
#         #         wandb.log(
#         #             {
#         #                 f"SKUBehaviour/sku_{i+1}": wandb.Image(imgs[i])
#         #                 for i in range(args.n_agents)
#         #             },
#         #             step=local_runner.t_env,
#         #         )

#         # Execute test runs once in a while
#         # n_test_runs = 5
#         # if (local_runner.t_env - last_test_T) / args.test_interval >= 1.0:
#         #     logger.console_logger.info(
#         #         "t_env: {} / {}".format(local_runner.t_env, args.t_max)
#         #     )
#         #     logger.console_logger.info(
#         #         "Estimated time left: {}. Time passed: {}".format(
#         #             time_left(last_time, last_test_T, local_runner.t_env, args.t_max),
#         #             time_str(time.time() - start_time),
#         #         )
#         #     )
#         #     last_time = time.time()
#         #     last_test_T = local_runner.t_env
#         #     predefined_relative_start_days = [10]  # [10, 20, 30, 40, 50]
#         #     test_episodes_returns = np.empty(
#         #         [args.n_agents, len(predefined_relative_start_days)]
#         #     )
#         #     train_episodes_returns = np.empty(
#         #         [args.n_agents, len(predefined_relative_start_days)]
#         #     )

#         #     for k in range(len(predefined_relative_start_days)):
#         #         map_name = args.env_args["map_name"]
#         #         n_agents = int(re.findall(r"n(.*)c", map_name)[0])
#         #         max_capacity = int(re.findall(r"c(.*)d", map_name)[0])
#         #         hist_len = int(re.findall(r"d(.*)s", map_name)[0])
#         #         relative_start_day = predefined_relative_start_days[k]
#         #         sampler_seq_len = int(re.findall(r"l(.*)", map_name)[0])
#         #         new_map_name = f"n{n_agents}c{max_capacity}d{hist_len}s{relative_start_day}l{sampler_seq_len}"
#         #         test_runner.args.env_args["map_name"] = new_map_name
#         #         for i in range(n_test_runs):
#         #             C_trajectories = np.zeros(
#         #                 (10, 3, args.env_args["time_limit"] + 1), dtype=int
#         #             )
#         #             agent_ids = list(range(i * 10 + 1, (i + 1) * 10 + 1))
#         #             test_runner.restart_ps(agent_ids)
#         #             with torch.no_grad():
#         #                 episode_batch = test_runner.run(
#         #                     test_mode=True, C_trajectories=C_trajectories
#         #                 )
#         #             test_episodes_returns[np.array(agent_ids) - 1, k] = (
#         #                 episode_batch["reward"][:, :-1, 0].sum(1).cpu().numpy()
#         #             )
#         #     for k in range(len(predefined_relative_start_days)):
#         #         map_name = args.env_args["map_name"]
#         #         n_agents = int(re.findall(r"n(.*)c", map_name)[0])
#         #         max_capacity = int(re.findall(r"c(.*)d", map_name)[0])
#         #         hist_len = int(re.findall(r"d(.*)s", map_name)[0])
#         #         sampler_seq_len = int(re.findall(r"l(.*)", map_name)[0])
#         #         new_map_name = (
#         #             f"n{n_agents}c{max_capacity}d{hist_len}s*l{sampler_seq_len}"
#         #         )
#         #         test_runner.args.env_args["map_name"] = new_map_name
#         #         for i in range(n_test_runs):
#         #             C_trajectories = np.zeros(
#         #                 (10, 3, args.env_args["time_limit"] + 1), dtype=int
#         #             )
#         #             agent_ids = list(range(i * 10 + 1, (i + 1) * 10 + 1))
#         #             test_runner.restart_ps(agent_ids)
#         #             with torch.no_grad():
#         #                 episode_batch = test_runner.run(
#         #                     test_mode=False, C_trajectories=C_trajectories
#         #                 )
#         #             train_episodes_returns[np.array(agent_ids) - 1, k] = (
#         #                 episode_batch["reward"][:, :-1, 0].sum(1).cpu().numpy()
#         #             )
#         #     logger.console_logger.info(
#         #         f"test_return_mean: {test_episodes_returns.mean(1).mean(0)}"
#         #     )
#         #     logger.console_logger.info(
#         #         f"train_return_mean: {train_episodes_returns.mean(1).mean(0)}"
#         #     )
#         #     # logger.console_logger.info(f"test_sku_returns: {test_episodes_returns.mean(1)}")
#         #     # logger.console_logger.info(f"train_sku_returns: {ttain_episodes_returns.mean(1)}")

#         #     if args.use_wandb:
#         #         v1 = test_episodes_returns.mean(1)[i]
#         #         v2 = train_episodes_returns.mean(1)[i]
#         #         wandb.log(
#         #             {
#         #                 **{
#         #                     f"MixedEnvSKUReturn/test_return_mean_sku{k}": v1
#         #                     for i in range(args.n_agents)
#         #                 },
#         #                 **{
#         #                     f"MixedEnvSKUReturn/train_return_mean_sku{k}": v2
#         #                     for i in range(args.n_agents)
#         #                 },
#         #             },
#         #             step=local_runner.t_env,
#         #         )
#         logger.console_logger.info(
#             "t_env: {} / {}".format(local_runner.t_env, args.t_max)
#         )
#         logger.console_logger.info(
#             "Estimated time left: {}. Time passed: {}".format(
#                 time_left(last_time, last_test_T, local_runner.t_env, args.t_max),
#                 time_str(time.time() - start_time),
#             )
#         )
#         last_time = time.time()
#         last_test_T = local_runner.t_env

#         # do evaluation for 400~500 data
#         C_trajectories = np.zeros(
#             (58, 3, 100 + 1), dtype=int
#         )
#         agent_ids = [i+1 for i in range(58)]
#         val4test_runner.restart_ps(agent_ids)
#         with torch.no_grad():
#             val4test_episode_batch = val4test_runner.run(
#                 test_mode=False, C_trajectories=C_trajectories
#             )
        
#         # do evaludation for 0~100 data
#         C_trajectories = np.zeros(
#             (58, 3, 400 + 1), dtype=int
#         )
#         agent_ids = [i+1 for i in range(58)]
#         val4train_runner.restart_ps(agent_ids)
#         with torch.no_grad():
#             val4train_episode_batch = val4train_runner.run(
#                 test_mode=False, C_trajectories=C_trajectories
#             )
        
#         for agent in agent_ids:
#             logger.console_logger.info(
#                 "val4test_return_sku{} : {}".format(agent, val4test_episode_batch["reward"][agent-1, :-1, 0].cpu().numpy().sum())
#             )
#             logger.console_logger.info(
#                 "val4train_return_sku{} : {}".format(agent, val4train_episode_batch["reward"][agent-1, :-1, 0].cpu().numpy().sum())
#             )
#         logger.console_logger.info(
#             "val4test_return_sum : {}".format(val4test_episode_batch["reward"][:, :-1, 0].cpu().numpy().sum())
#         )
#         logger.console_logger.info(
#             "val4train_return_sum : {}".format(val4train_episode_batch["reward"][:, :-1, 0].cpu().numpy().sum())
#         )
        
#         if args.use_wandb:
#             # for agent in agent_ids:
#             wandb.log({
#                 **{
#                     f"MixedEnvSKUReturn/val4test_return_mean_sku{agent}": val4test_episode_batch["reward"][agent-1, :-1, 0].cpu().numpy().sum()
#                         for agent in agent_ids
#                 },
#                 **{
#                     f"MixedEnvSKUReturn/val4train_return_mean_sku{agent}": val4train_episode_batch["reward"][agent-1, :-1, 0].cpu().numpy().sum()
#                         for agent in agent_ids
#                 }
#                 },
#                 step=local_runner.t_env,
#             )

#             wandb.log(
#                 {
#                     f"Total_return/val4test_return_sum" : val4test_episode_batch["reward"][:, :-1, 0].cpu().numpy().sum(),
#                     f"Total_return/val4train_return_sum" : val4train_episode_batch["reward"][:, :-1, 0].cpu().numpy().sum(),
#                 },
#                 step=local_runner.t_env,
#             )

#         if args.save_model and (
#             runner.t_env - model_save_time >= args.save_model_interval
#             or model_save_time == 0
#         ):
#             model_save_time = runner.t_env
#             save_path = os.path.join(
#                 args.local_results_path, args.unique_token, "models", str(runner.t_env)
#             )
#             os.makedirs(save_path, exist_ok=True)
#             logger.console_logger.info(f"Saving models to {save_path}")

#             # learner should handle saving/loading -- delegate actor save/load to mac,
#             # use appropriate filenames to do critics, optimizer states
#             local_learner.save_models(save_path)
#             # context_learner.save_models(save_path)

#         if (local_runner.t_env - last_log_T) / args.log_interval >= 1.0:
#             local_logger.log_stat("episode", episode, local_runner.t_env)
#             local_logger.print_recent_stats()
#             last_log_T = local_runner.t_env

#     runner.close_env()
#     logger.console_logger.info("Finished Training")


# def args_sanity_check(config, _log):

#     # set CUDA flags
#     # config["use_cuda"] = True # Use cuda whenever possible!
#     if config["use_cuda"] and not torch.cuda.is_available():
#         config["use_cuda"] = False
#         _log.warning(
#             "CUDA flag use_cuda was switched OFF automatically because no CUDA devices are available!"
#         )

#     if config["test_nepisode"] < config["batch_size_run"]:
#         config["test_nepisode"] = config["batch_size_run"]
#     else:
#         config["test_nepisode"] = (
#             config["test_nepisode"] // config["batch_size_run"]
#         ) * config["batch_size_run"]

#     return config


# def run_decoupled_training(
#     episode_num, agent_ids, local_runner, local_infos, context_learner, learner, args,
# ):

#     local_buffer = ReplayBuffer(
#         local_infos["local_scheme"],
#         local_infos["local_groups"],
#         local_infos["buffer_size"],
#         local_infos["episode_limit"] + 1,
#         preprocess=local_infos["local_preprocess"],
#         device="cpu"
#         if local_infos["local_args"].buffer_cpu_only
#         else local_infos["local_args"].device,
#     )
#     if args.use_zero_like_context:
#         C_trajectories = np.zeros(
#             (len(agent_ids), 3, args.env_args["time_limit"] + 1), dtype=int
#         )
#     else:
#         C_trajectories = context_learner.generate_C_trajectories(
#             agent_ids, prob=args.fake_dynamics_prob, aug_type=args.aug_type
#         )
#         C_trajectories = C_trajectories.cpu().numpy()
#     local_runner.restart_ps(agent_ids)
#     with torch.no_grad():
#         episode_batch = local_runner.run(test_mode=False, C_trajectories=C_trajectories)
#         local_buffer.insert_episode_batch(episode_batch)
#     episode_sample = local_buffer[:]
#     max_ep_t = episode_sample.max_t_filled()
#     episode_sample = episode_sample[:, :max_ep_t]
#     episode_sample.to(args.device)
#     rets = learner.train(episode_sample, local_runner.t_env, episode_num)
#     local_runner.clean_ps()

#     return rets


import copy
import datetime
import glob
import os
import pprint
import re
import threading
import time
from collections import defaultdict
from dataclasses import replace
from os.path import abspath, dirname
from random import random
from types import SimpleNamespace as SN

import math
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from matplotlib import pyplot as plt

import runners
import wandb
from components.episode_buffer import ReplayBuffer
from components.transforms import OneHot
from controllers import REGISTRY as mac_REGISTRY
from envs import REGISTRY as env_REGISTRY
from learners import REGISTRY as le_REGISTRY
from runners import REGISTRY as r_REGISTRY
from utils.logging import Logger
from utils.timehelper import time_left, time_str


def run(_run, _config, _log):

    # check args sanity
    _config = args_sanity_check(_config, _log)
    if _config["n_decoupled_iterations"] == 0:
        _config["train_with_joint_data"] = True

    args = SN(**_config)
    args.device = "cuda" if args.use_cuda else "cpu"

    # setup loggers
    logger = Logger(_log)

    _log.info("Experiment Parameters:")
    tmp_config = {k: _config[k] for k in _config if k != "env_args"}
    tmp_config.update(
        {f"env_agrs.{k}": _config["env_args"][k] for k in _config["env_args"]}
    )
    print(
        pd.Series(tmp_config, name="HyperParameter Value")
        .transpose()
        .sort_index()
        .fillna("")
        .to_markdown()
    )

    # configure tensorboard logger
    ts = datetime.datetime.now().strftime("%m%dT%H%M")
    unique_token = f"{_config['name']}_{_config['env_args']['map_name']}_seed{_config['seed']}_{ts}"
    args.unique_token = unique_token
    if args.use_tensorboard:
        tb_logs_direc = os.path.join(
            dirname(dirname(dirname(abspath(__file__)))), "results", "tb_logs"
        )
        tb_exp_direc = os.path.join(tb_logs_direc, "{}").format(unique_token)
        logger.setup_tb(tb_exp_direc)
    if args.use_wandb:
        logger.setup_wandb(args)

    # sacred is on by default
    logger.setup_sacred(_run)

    # Run and train
    run_sequential(args=args, logger=logger)

    # Clean up after finishing
    print("Exiting Main")

    print("Stopping all threads")
    for t in threading.enumerate():
        if t.name != "MainThread":
            print("Thread {} is alive! Is daemon: {}".format(t.name, t.daemon))
            t.join(timeout=1)
            print("Thread joined")

    print("Exiting script")

    # Making sure framework really exits
    # os._exit(os.EX_OK)


def evaluate_sequential(args, runner):

    for _ in range(args.test_nepisode):
        runner.run(test_mode=True)

    if args.save_replay:
        runner.save_replay()

    runner.close_env()


def run_sequential(args, logger):
    # assert args.decoupled_training == args.use_individual_envs

    # Init runner so we can get env info
    # args.decoupled_training = not args.decoupled_training
    runner = r_REGISTRY[args.runner](args, logger=logger)
    # args.decoupled_training = not args.decoupled_training

    # Set up schemes and groups here
    env_info = env_REGISTRY[args.env](**args.env_args).get_env_info()
    args.n_agents = env_info["n_agents"]
    args.n_actions = env_info["n_actions"]
    args.state_shape = env_info["state_shape"]

    runner.args.n_agents = env_info["n_agents"]
    # Default/Base scheme
    scheme = {
        "state": {"vshape": env_info["state_shape"]},
        "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
        "mean_action": {
            "vshape": (env_info["n_actions"],),
            "group": "agents",
            "dtype": torch.float,
        },
        "actions": {"vshape": (1,), "group": "agents", "dtype": torch.long},
        "avail_actions": {
            "vshape": (env_info["n_actions"],),
            "group": "agents",
            "dtype": torch.int,
        },
        "probs": {
            "vshape": (env_info["n_actions"],),
            "group": "agents",
            "dtype": torch.float,
        },
        "reward": {"vshape": (1,)},
        "individual_rewards": {"vshape": (1,), "group": "agents"},
        "terminated": {"vshape": (1,), "dtype": torch.uint8},
    }
    groups = {"agents": args.n_agents}
    preprocess = {"actions": ("actions_onehot", [OneHot(out_dim=args.n_actions)])}

    buffer = ReplayBuffer(
        scheme,
        groups,
        args.buffer_size,
        env_info["episode_limit"] + 1,
        preprocess=preprocess,
        device="cpu" if args.buffer_cpu_only else args.device,
    )

    logger.console_logger.info("MDP Components:")
    print(pd.DataFrame(buffer.scheme).transpose().sort_index().fillna("").to_markdown())

    # Setup multiagent controller here
    mac = mac_REGISTRY[args.mac](buffer.scheme, groups, args)

    # Give runner the scheme
    runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac)

    # Set up local parallel runner
    local_logger = copy.copy(logger)
    if args.use_tensorboard:
        tb_logs_direc = os.path.join(
            dirname(dirname(abspath(__file__))), "results", "tb_logs"
        )
        tb_exp_direc = os.path.join(tb_logs_direc, "{}").format(args.unique_token)
        logger.setup_tb(tb_exp_direc)
    local_logger.use_wandb = False

    local_args = copy.deepcopy(args)
    if local_args.max_individual_envs < 1:
        local_args.batch_size_run = max(
            1, int(local_args.max_individual_envs * args.n_agents)
        )
        local_args.batch_size = max(
            1, int(local_args.max_individual_envs * args.n_agents)
        )
        local_args.buffer_size = max(
            1, int(local_args.max_individual_envs * args.n_agents)
        )
    else:
        local_args.batch_size_run = local_args.max_individual_envs
        local_args.batch_size = local_args.max_individual_envs
        local_args.buffer_size = local_args.max_individual_envs
    local_args.agent_ids = [1] * (
        args.max_individual_envs
        if args.max_individual_envs >= 1
        else max(1, int(args.n_agents * args.max_individual_envs))
    )
    local_args.n_agents = 1  # local_env_info["n_agents"]
    local_runner = r_REGISTRY["local_parallel"](
        args=local_args, use_wandb=args.use_wandb
    )
    local_env_info = local_runner.get_env_info()

    local_args.n_actions = local_env_info["n_actions"]
    local_args.state_shape = local_env_info["obs_shape"]
    local_scheme = {
        "state": {"vshape": local_env_info["obs_shape"], "group": "agents"},
        "obs": {"vshape": local_env_info["obs_shape"], "group": "agents"},
        "actions": {"vshape": (1,), "group": "agents", "dtype": torch.long},
        "mean_action": {
            "vshape": (env_info["n_actions"],),
            "group": "agents",
            "dtype": torch.float,
        },
        "avail_actions": {
            "vshape": (local_env_info["n_actions"],),
            "group": "agents",
            "dtype": torch.int,
        },
        "probs": {
            "vshape": (env_info["n_actions"],),
            "group": "agents",
            "dtype": torch.float,
        },
        "reward": {"vshape": (1,)},
        "cur_balance": {"vshape": (1,)},
        "individual_rewards": {"vshape": (1,), "group": "agents"},
        "terminated": {"vshape": (1,), "dtype": torch.uint8},
    }
    local_groups = {"agents": local_args.n_agents}
    local_preprocess = {
        "actions": ("actions_onehot", [OneHot(out_dim=local_args.n_actions)])
    }
    local_infos = {
        "local_scheme": local_scheme,
        "local_groups": local_groups,
        "buffer_size": local_args.buffer_size,
        "episode_limit": local_env_info["episode_limit"],
        "local_preprocess": local_preprocess,
        "local_args": local_args,
    }
    local_buffer = ReplayBuffer(
        local_scheme,
        local_groups,
        local_args.buffer_size,
        local_env_info["episode_limit"] + 1,
        preprocess=local_preprocess,
        device="cpu" if local_args.buffer_cpu_only else local_args.device,
    )

    local_mac = mac_REGISTRY[local_args.mac](
        local_buffer.scheme, local_groups, local_args
    )
    local_mac.agent = mac.agent
    local_runner.setup(
        scheme=local_scheme,
        groups=local_groups,
        preprocess=local_preprocess,
        mac=local_mac,
    )
    
    map_name = args.env_args["map_name"]
    n_agents = int(re.findall(r"n(.*)c", map_name)[0])
    max_capacity = int(re.findall(r"c(.*)d", map_name)[0])
    hist_len = int(re.findall(r"d(.*)s", map_name)[0])

    # test for timestep 400~500
    eval4test_args = copy.deepcopy(local_args)
    eval4test_args.env_args['map_name'] = f"n{n_agents}c{max_capacity}d{hist_len}s{400}l{100}" #"n58c1000d21s400l100"
    eval4test_args.env_args['time_limit'] = 100
    # eval4test_args.env_args['max_individual_envs'] = n_agents
    eval4test_args.action_selector = "epsilon_greedy" #"multinomial" #"epsilon_greedy"
    eval4test_args.save_probs = False
    eval4test_args.batch_size_run = eval4test_args.max_individual_envs
    eval4test_args.batch_size = eval4test_args.max_individual_envs
    eval4test_args.buffer_size = eval4test_args.max_individual_envs
    eval4test_args.agent_ids = [1] * eval4test_args.max_individual_envs
    eval4test_runner = r_REGISTRY["local_parallel"](args=eval4test_args, use_wandb=False)
    eval4test_mac = mac_REGISTRY[local_args.mac](
        local_buffer.scheme, local_groups, eval4test_args
    )
    eval4test_mac.agent = mac.agent
    eval4test_runner.setup(
        scheme=local_scheme,
        groups=local_groups,
        preprocess=local_preprocess,
        mac=eval4test_mac,
    )

    # test for timestep 0~400
    eval4train_args = copy.deepcopy(local_args)
    eval4train_args.env_args['map_name'] = f"n{n_agents}c{max_capacity}d{hist_len}s{300}l{100}" #"n58c1000d21s300l100"
    eval4train_args.env_args['time_limit'] = 100
    # eval4train_args.env_args['max_individual_envs'] = n_agents
    eval4train_args.action_selector = "epsilon_greedy" #"multinomial" #"epsilon_greedy"
    eval4train_args.save_probs = False
    eval4train_args.batch_size_run = eval4train_args.max_individual_envs #n_agents
    eval4train_args.batch_size = eval4train_args.max_individual_envs
    eval4train_args.buffer_size = eval4train_args.max_individual_envs
    eval4train_args.agent_ids = [1] * eval4train_args.max_individual_envs
    eval4train_runner = r_REGISTRY["local_parallel"](args=eval4train_args, use_wandb=False)
    eval4train_mac = mac_REGISTRY[local_args.mac](
        local_buffer.scheme, local_groups, eval4train_args
    )
    eval4train_mac.agent = mac.agent
    eval4train_runner.setup(
        scheme=local_scheme,
        groups=local_groups,
        preprocess=local_preprocess,
        mac=eval4train_mac,
    )
    
    context_learner = le_REGISTRY["context_learner"](logger, args)
    local_learner = le_REGISTRY["local_ppo_learner"](
        local_mac, local_buffer.scheme, local_logger, local_args
    )
    if args.use_cuda:
        context_learner.cuda()
        local_learner.cuda()

    if args.visualize:
        visual_runner = runners.episode_runner.EpisodeRunner(args=args, logger=logger)
        visual_runner.setup(
            scheme=scheme, groups=groups, preprocess=preprocess, mac=mac
        )

    # Learner
    learner = le_REGISTRY[args.learner](mac, buffer.scheme, logger, args)
    if args.use_cuda:
        learner.cuda()

    print("Actor Arch:", mac.agent)
    if args.train_with_joint_data:
        print("Critic Arch:", learner.critic)
    if args.n_decoupled_iterations > 0:
        print("Local Critic Arch:", local_learner.critic)

    if args.checkpoint_path != "":
        # timesteps = []
        # timestep_to_load = 0

        # if not os.path.isdir(args.checkpoint_path):
        #     logger.console_logger.info(
        #         "Checkpoint directiory {} doesn't exist".format(args.checkpoint_path)
        #     )
        #     return

        # # Go through all files in args.checkpoint_path
        # for name in os.listdir(args.checkpoint_path):
        #     full_name = os.path.join(args.checkpoint_path, name)
        #     # Check if they are dirs the names of which are numbers
        #     if os.path.isdir(full_name) and name.isdigit():
        #         timesteps.append(int(name))

        # if args.load_step == 0:
        #     # choose the max timestep
        #     timestep_to_load = max(timesteps)
        # else:
        #     # choose the timestep closest to load_step
        #     timestep_to_load = min(timesteps, key=lambda x: abs(x - args.load_step))

        # model_path = os.path.join(args.checkpoint_path, str(timestep_to_load))

        # logger.console_logger.info("Loading model from {}".format(model_path))
        # local_learner.load_models(model_path)
        # context_learner.load_models(model_path)
        # runner.t_env = timestep_to_load
        eval4test_runner.mac.load_models(args.checkpoint_path)
        eval4train_runner.mac.load_models(args.checkpoint_path)
        if args.evaluate:
            # runner.log_train_stats_t = runner.t_env
            # # if args.n_decoupled_iterations > 0:
            # #     runner.mac.load_state(local_learner.mac)
            # evaluate_sequential(args, logger, runner)
            # logger.log_stat("episode", runner.t_env, runner.t_env)
            # logger.print_recent_stats()
            C_trajectories = np.zeros(
                (n_agents, 3, 100 + 1), dtype=int
            )
            
            # for agent in range(1, 58+1):
            agent_ids = [i+1 for i in range(n_agents)]
            eval4test_runner.restart_ps(agent_ids)

            with torch.no_grad():
                eval4test_episode_batch = eval4test_runner.run(
                    test_mode=True, C_trajectories=C_trajectories, 
                    visualize=args.visualize, visual_outputs_path= os.path.join(
                        args.local_results_path, args.unique_token, "visual_outputs"
                    )
                )
            
            C_trajectories = np.zeros(
                (n_agents, 3, 400 + 1), dtype=int
            )
            agent_ids = [i+1 for i in range(n_agents)]
            eval4train_runner.restart_ps(agent_ids)
            with torch.no_grad():
                eval4train_episode_batch = eval4train_runner.run(
                    test_mode=True, C_trajectories=C_trajectories
                )

            eval4test_return = eval4test_episode_batch["reward"][:, :-1, 0].sum(1).cpu().numpy()#.mean()
            eval4train_return = eval4train_episode_batch["reward"][:, :-1, 0].sum(1).cpu().numpy()#.mean()
            eval4test_balance = eval4test_episode_batch["cur_balance"][:, -2, 0].cpu().numpy()
            eval4train_balance = eval4train_episode_batch["cur_balance"][:, -2, 0].cpu().numpy()

            for agent in range(1, n_agents+1):
                # logger.console_logger.info(
                #     f"eval4test_return_sku{agent}: {eval4test_return[agent-1]}"
                # )
                # logger.console_logger.info(
                #     f"eval4train_return_sku{agent}: {eval4train_return[agent-1]}"
                # )
                logger.console_logger.info(
                    f"eval4test_balance_sku{agent}, {eval4test_balance[agent-1]}"
                )
                # logger.console_logger.info(
                #     f"eval4train_balance_sku{agent}: {eval4train_balance[agent-1]}"
                # )
            logger.console_logger.info(
                    "eval4test_return_sum: {}".format(eval4test_return.sum()),
            )
            logger.console_logger.info(
                    "eval4train_return_sum: {}".format(eval4train_return.sum()),
            )
            logger.console_logger.info(
                    "eval4test_balance_sum: {}".format(eval4test_balance.sum()),
            )
            logger.console_logger.info(
                    "eval4train_balance_sum: {}".format(eval4train_balance.sum()),
            )
            logger.console_logger.info("Finished Evaluation")
            return

    # start training
    episode = 0
    local_episode = 0
    last_test_T = -args.test_interval - 1
    last_log_T = 0
    model_save_time = 0
    visual_time = 0
    val_best_balance = 0
    test_best_balance = 0

    start_time = time.time()
    last_time = start_time

    logger.console_logger.info("Beginning training for {} timesteps".format(args.t_max))

    while local_runner.t_env <= args.t_max:  # for the_i in range(1):#

        for _ in range(args.n_decoupled_iterations):
            if not args.select_ls_randomly:
                if args.target_agent:
                    agent_ids = [args.target_agent for _ in range(args.max_individual_envs)]
                    rets = run_decoupled_training(
                        episode,
                        agent_ids.tolist(),
                        local_runner,
                        local_infos,
                        context_learner,
                        local_learner,
                        local_args,
                    )
                    joint_local_ratio = args.max_individual_envs / (
                        args.n_agents if args.max_individual_envs >= 1 else 1
                    )
                    if args.max_individual_envs >= 1:
                        joint_local_ratio = args.max_individual_envs / args.n_agents
                    else:
                        joint_local_ratio = args.max_individual_envs
                        if args.n_agents * args.max_individual_envs < 1:
                            joint_local_ratio = 1 / args.n_agents
                    runner.t_env += int(joint_local_ratio * 1 * args.env_args["time_limit"])
                else:
                    if args.max_individual_envs < 1:
                        n_splits = int(1 / args.max_individual_envs)
                        if n_splits > args.n_agents:
                            n_splits = args.n_agents
                    else:
                        n_splits = args.n_agents // args.max_individual_envs
                    for agent_ids in np.array_split(list(range(args.n_agents)), n_splits,):
                        rets = run_decoupled_training(
                            episode,
                            (agent_ids + 1).tolist(),
                            local_runner,
                            local_infos,
                            context_learner,
                            local_learner,
                            local_args,
                        )
                    joint_local_ratio = 1
                    runner.t_env += args.env_args["time_limit"]
            else:
                agent_ids = np.random.choice(
                    # args.n_agents,
                    n_agents,
                    size=[
                        args.max_individual_envs
                        if args.max_individual_envs >= 1
                        else max(1, int(args.n_agents * args.max_individual_envs))
                    ],
                    # replace=False,
                    replace = True,
                )
                rets = run_decoupled_training(
                    episode,
                    (agent_ids + 1).tolist(),
                    local_runner,
                    local_infos,
                    context_learner,
                    local_learner,
                    local_args,
                )
                joint_local_ratio = args.max_individual_envs / (
                    args.n_agents if args.max_individual_envs >= 1 else 1
                )
                if args.max_individual_envs >= 1:
                    joint_local_ratio = args.max_individual_envs / args.n_agents
                else:
                    joint_local_ratio = args.max_individual_envs
                    if args.n_agents * args.max_individual_envs < 1:
                        joint_local_ratio = 1 / args.n_agents
                runner.t_env += int(joint_local_ratio * 1 * args.env_args["time_limit"])
            if args.use_wandb:
                wandb.log(rets, step=local_runner.t_env)

            local_episode += (
                args.max_individual_envs
                if args.max_individual_envs >= 1
                else max(1, int(args.n_agents * args.max_individual_envs))
            )

            episode += args.batch_size_run + joint_local_ratio * 1

        # if (
        #     args.visualize
        #     and ((local_runner.t_env - visual_time) / args.visualize_interval >= 1.0)
        # ):
        #     visual_time = local_runner.t_env
        #     visual_outputs_path = os.path.join(
        #         args.local_results_path, args.unique_token, "visual_outputs"
        #     )
        #     logger.console_logger.info(
        #         f"Saving visualizations to {visual_outputs_path}/{local_runner.t_env}"
        #     )

        #     visual_runner.run_visualize2(visual_outputs_path, local_runner.t_env)
        #     if args.use_wandb:
        #         imgs = sorted(
        #             glob.glob(
        #                 f"{visual_outputs_path}/{local_runner.t_env}/OuterSKUStoreUnit*.png"
        #             ),
        #             key=os.path.getmtime,
        #         )[::-1]
        #         wandb.log(
        #             {
        #                 f"SKUBehaviour/sku_{i+1}": wandb.Image(imgs[i])
        #                 for i in range(args.n_agents)
        #             },
        #             step=local_runner.t_env,
        #         )

        # Execute test runs once in a while
        # n_test_runs = 5
        # if (local_runner.t_env - last_test_T) / args.test_interval >= 1.0:
        #     logger.console_logger.info(
        #         "t_env: {} / {}".format(local_runner.t_env, args.t_max)
        #     )
        #     logger.console_logger.info(
        #         "Estimated time left: {}. Time passed: {}".format(
        #             time_left(last_time, last_test_T, local_runner.t_env, args.t_max),
        #             time_str(time.time() - start_time),
        #         )
        #     )
        #     last_time = time.time()
        #     last_test_T = local_runner.t_env
        #     predefined_relative_start_days = [10]  # [10, 20, 30, 40, 50]
        #     test_episodes_returns = np.empty(
        #         [args.n_agents, len(predefined_relative_start_days)]
        #     )
        #     train_episodes_returns = np.empty(
        #         [args.n_agents, len(predefined_relative_start_days)]
        #     )

        #     for k in range(len(predefined_relative_start_days)):
        #         map_name = args.env_args["map_name"]
        #         n_agents = int(re.findall(r"n(.*)c", map_name)[0])
        #         max_capacity = int(re.findall(r"c(.*)d", map_name)[0])
        #         hist_len = int(re.findall(r"d(.*)s", map_name)[0])
        #         relative_start_day = predefined_relative_start_days[k]
        #         sampler_seq_len = int(re.findall(r"l(.*)", map_name)[0])
        #         new_map_name = f"n{n_agents}c{max_capacity}d{hist_len}s{relative_start_day}l{sampler_seq_len}"
        #         test_runner.args.env_args["map_name"] = new_map_name
        #         for i in range(n_test_runs):
        #             C_trajectories = np.zeros(
        #                 (10, 3, args.env_args["time_limit"] + 1), dtype=int
        #             )
        #             agent_ids = list(range(i * 10 + 1, (i + 1) * 10 + 1))
        #             test_runner.restart_ps(agent_ids)
        #             with torch.no_grad():
        #                 episode_batch = test_runner.run(
        #                     test_mode=True, C_trajectories=C_trajectories
        #                 )
        #             test_episodes_returns[np.array(agent_ids) - 1, k] = (
        #                 episode_batch["reward"][:, :-1, 0].sum(1).cpu().numpy()
        #             )
        #     for k in range(len(predefined_relative_start_days)):
        #         map_name = args.env_args["map_name"]
        #         n_agents = int(re.findall(r"n(.*)c", map_name)[0])
        #         max_capacity = int(re.findall(r"c(.*)d", map_name)[0])
        #         hist_len = int(re.findall(r"d(.*)s", map_name)[0])
        #         sampler_seq_len = int(re.findall(r"l(.*)", map_name)[0])
        #         new_map_name = (
        #             f"n{n_agents}c{max_capacity}d{hist_len}s*l{sampler_seq_len}"
        #         )
        #         test_runner.args.env_args["map_name"] = new_map_name
        #         for i in range(n_test_runs):
        #             C_trajectories = np.zeros(
        #                 (10, 3, args.env_args["time_limit"] + 1), dtype=int
        #             )
        #             agent_ids = list(range(i * 10 + 1, (i + 1) * 10 + 1))
        #             test_runner.restart_ps(agent_ids)
        #             with torch.no_grad():
        #                 episode_batch = test_runner.run(
        #                     test_mode=False, C_trajectories=C_trajectories
        #                 )
        #             train_episodes_returns[np.array(agent_ids) - 1, k] = (
        #                 episode_batch["reward"][:, :-1, 0].sum(1).cpu().numpy()
        #             )
        #     logger.console_logger.info(
        #         f"test_return_mean: {test_episodes_returns.mean(1).mean(0)}"
        #     )
        #     logger.console_logger.info(
        #         f"train_return_mean: {train_episodes_returns.mean(1).mean(0)}"
        #     )
        #     # logger.console_logger.info(f"test_sku_returns: {test_episodes_returns.mean(1)}")
        #     # logger.console_logger.info(f"train_sku_returns: {ttain_episodes_returns.mean(1)}")

        #     if args.use_wandb:
        #         v1 = test_episodes_returns.mean(1)[i]
        #         v2 = train_episodes_returns.mean(1)[i]
        #         wandb.log(
        #             {
        #                 **{
        #                     f"MixedEnvSKUReturn/test_return_mean_sku{k}": v1
        #                     for i in range(args.n_agents)
        #                 },
        #                 **{
        #                     f"MixedEnvSKUReturn/train_return_mean_sku{k}": v2
        #                     for i in range(args.n_agents)
        #                 },
        #             },
        #             step=local_runner.t_env,
        #         )
        
        if (local_runner.t_env - last_test_T) / args.test_interval >= 1.0:
            
            logger.console_logger.info(
                "t_env: {} / {}".format(local_runner.t_env, args.t_max)
            )
            logger.console_logger.info(
                "Estimated time left: {}. Time passed: {}".format(
                    time_left(last_time, last_test_T, local_runner.t_env, args.t_max),
                    time_str(time.time() - start_time),
                )
            )
            last_time = time.time()
            last_test_T = local_runner.t_env

            n_test_runs = math.ceil(n_agents / eval4test_args.max_individual_envs)

            eval4test_return_list = []
            eval4train_return_list = []
            eval4test_balance_list = []
            eval4train_balance_list = []

            for k in range(n_test_runs):
                start_agent = k * eval4test_args.max_individual_envs + 1
                agents_num = min(eval4test_args.max_individual_envs, n_agents-k*eval4test_args.max_individual_envs)
                C_trajectories = np.zeros(
                    (agents_num, 3, 100 + 1), dtype=int
                )
                
                # for agent in range(1, 58+1):
                agent_ids = [i + start_agent for i in range(agents_num)]
                eval4test_runner.restart_ps(agent_ids)

                with torch.no_grad():
                    eval4test_episode_batch = eval4test_runner.run(
                        test_mode=True, C_trajectories=C_trajectories
                    )
                
                C_trajectories = np.zeros(
                    (agents_num, 3, 100 + 1), dtype=int
                )
                agent_ids = [i + start_agent for i in range(agents_num)]
                eval4train_runner.restart_ps(agent_ids)
                with torch.no_grad():
                    eval4train_episode_batch = eval4train_runner.run(
                        test_mode=True, C_trajectories=C_trajectories
                    )

                eval4test_return_sub = eval4test_episode_batch["reward"][:, :-1, 0].sum(1).cpu().numpy()#.mean()
                eval4train_return_sub = eval4train_episode_batch["reward"][:, :-1, 0].sum(1).cpu().numpy()#.mean()
                eval4test_balance_sub = eval4test_episode_batch["cur_balance"][:, -2, 0].cpu().numpy()
                eval4train_balance_sub = eval4train_episode_batch["cur_balance"][:, -2, 0].cpu().numpy()

                eval4test_return_list.append(eval4test_return_sub)
                eval4train_return_list.append(eval4train_return_sub)
                eval4test_balance_list.append(eval4test_balance_sub)
                eval4train_balance_list.append(eval4train_balance_sub)

            eval4test_return = np.concatenate(eval4test_return_list, axis=0)
            eval4train_return = np.concatenate(eval4train_return_list, axis=0)
            eval4test_balance = np.concatenate(eval4test_balance_list, axis=0)
            eval4train_balance = np.concatenate(eval4train_balance_list, axis=0)

            for agent in range(1, n_agents + 1):
                logger.console_logger.info(
                    "sku{} | test_return:{:.2f} | test_balance:{:.2f} | train_return:{:.2f} | train_balance:{:.2f} |".format(
                        agent, eval4test_return[agent - 1], eval4test_balance[agent - 1], 
                        eval4train_return[agent - 1], eval4train_balance[agent - 1]
                    )
                )

            logger.console_logger.info(
                    "eval4test_return_sum: {:.2f}".format(eval4test_return.sum()),
            )
            logger.console_logger.info(
                    "eval4train_return_sum: {:.2f}".format(eval4train_return.sum()),
            )
            logger.console_logger.info(
                    "eval4test_balance_sum: {:.2f}".format(eval4test_balance.sum()),
            )
            logger.console_logger.info(
                    "eval4train_balance_sum: {:.2f}".format(eval4train_balance.sum()),
            )
            logger.console_logger.info(
                    "val_best_balance: {:.2f}".format(val_best_balance),
            )
            logger.console_logger.info(
                    "test_best_balance: {:.2f}".format(test_best_balance),
            )

            if args.use_wandb:
                for agent in range(1, n_agents + 1):
                    wandb.log(
                        {
                        f"MixedEnvSKUReturn/eval4test_return_sku{agent}": eval4test_return[agent-1],
                        f"MixedEnvSKUReturn/eval4train_return_sku{agent}": eval4train_return[agent-1],
                        f"MixedEnvSKUReturn/eval4test_balance_sku{agent}": eval4test_balance[agent-1],
                        f"MixedEnvSKUReturn/eval4train_balance_sku{agent}": eval4train_balance[agent-1],
                        },
                        step=local_runner.t_env,
                    )
                wandb.log(
                    {
                    f"TotalReturn/eval4test_return_sum": eval4test_return.sum(),
                    f"TotalReturn/eval4train_return_sum": eval4train_return.sum(),
                    f"TotalReturn/eval4test_balance_sum": eval4test_balance.sum(),
                    f"TotalReturn/eval4train_balance_sum": eval4train_balance.sum(),
                    },
                    step=local_runner.t_env,
                )

        #save best model
        if args.save_model and val_best_balance < eval4train_balance.sum():
            val_best_balance = eval4train_balance.sum()
            test_best_balance = eval4test_balance.sum()
            model_save_time = local_runner.t_env
            save_path = os.path.join(
                        args.local_results_path, args.unique_token, "models", str(runner.t_env)
                    ) if os.getenv("AMLT_OUTPUT_DIR") is None else os.path.join(os.getenv("AMLT_OUTPUT_DIR"), "results", args.unique_token, "models", str(runner.t_env))
            save_path = save_path.replace('*', '_')
            max_model_path = save_path
            os.makedirs(save_path, exist_ok=True)
            logger.console_logger.info(f"Saving models to {save_path}")

            # learner should handle saving/loading -- delegate actor save/load to mac,
            # use appropriate filenames to do critics, optimizer states
            local_learner.save_models(save_path)


        if args.save_model and (
            local_runner.t_env - model_save_time >= args.save_model_interval
            or model_save_time == 0
        ):
            model_save_time = local_runner.t_env
            save_path = os.path.join(
                args.local_results_path, args.unique_token, "models", str(runner.t_env)
            ) if os.getenv("AMLT_OUTPUT_DIR") is None else os.path.join(os.getenv("AMLT_OUTPUT_DIR"), "results", args.unique_token, "models", str(runner.t_env))
            save_path = save_path.replace('*', '_')
            os.makedirs(save_path, exist_ok=True)
            logger.console_logger.info(f"Saving models to {save_path}")

            # learner should handle saving/loading -- delegate actor save/load to mac,
            # use appropriate filenames to do critics, optimizer states
            local_learner.save_models(save_path)
            # context_learner.save_models(save_path)

        if (local_runner.t_env - last_log_T) / args.log_interval >= 1.0:
            local_logger.log_stat("episode", episode, local_runner.t_env)
            local_logger.print_recent_stats()
            last_log_T = local_runner.t_env

    local_learner.load_models(max_model_path)
    
    ## save best model
    save_path = os.path.join(
        args.local_results_path, args.unique_token, "models", str(runner.t_env)
    ) if os.getenv("AMLT_OUTPUT_DIR") is None else os.path.join(os.getenv("AMLT_OUTPUT_DIR"), "results", args.unique_token, "models", "best")
    save_path = save_path.replace('*', '_')
    os.makedirs(save_path, exist_ok=True)
    logger.console_logger.info("Saving best models to {}".format(save_path))
    local_learner.save_models(save_path)

    ## test and visualization
    n_test_runs = math.ceil(n_agents / eval4test_args.max_individual_envs)

    eval4test_return_list = []
    eval4train_return_list = []
    eval4test_balance_list = []
    eval4train_balance_list = []
    vis_save_path = os.path.join(
            args.local_results_path, args.unique_token, "vis"
        ) if os.getenv("AMLT_OUTPUT_DIR") is None else os.path.join(os.getenv("AMLT_OUTPUT_DIR"), "results", args.unique_token, "vis")
    vis_save_path = vis_save_path.replace('*', '_')
    os.makedirs(vis_save_path, exist_ok=True)

    for k in range(n_test_runs):
        start_agent = k * eval4test_args.max_individual_envs + 1
        agents_num = min(eval4test_args.max_individual_envs, n_agents-k*eval4test_args.max_individual_envs)
        C_trajectories = np.zeros(
            (agents_num, 3, 100 + 1), dtype=int
        )
        
        # for agent in range(1, 58+1):
        agent_ids = [i + start_agent for i in range(agents_num)]
        eval4test_runner.restart_ps(agent_ids)

        with torch.no_grad():
            eval4test_episode_batch = eval4test_runner.run(
                test_mode=True, C_trajectories=C_trajectories, 
                visualize=True, visual_outputs_path=vis_save_path
            )
        
        C_trajectories = np.zeros(
            (agents_num, 3, 100 + 1), dtype=int
        )
        agent_ids = [i + start_agent for i in range(agents_num)]
        eval4train_runner.restart_ps(agent_ids)
        with torch.no_grad():
            eval4train_episode_batch = eval4train_runner.run(
                test_mode=True, C_trajectories=C_trajectories
            )

        eval4test_return_sub = eval4test_episode_batch["reward"][:, :-1, 0].sum(1).cpu().numpy()#.mean()
        eval4train_return_sub = eval4train_episode_batch["reward"][:, :-1, 0].sum(1).cpu().numpy()#.mean()
        eval4test_balance_sub = eval4test_episode_batch["cur_balance"][:, -2, 0].cpu().numpy()
        eval4train_balance_sub = eval4train_episode_batch["cur_balance"][:, -2, 0].cpu().numpy()

        eval4test_return_list.append(eval4test_return_sub)
        eval4train_return_list.append(eval4train_return_sub)
        eval4test_balance_list.append(eval4test_balance_sub)
        eval4train_balance_list.append(eval4train_balance_sub)

    eval4test_return = np.concatenate(eval4test_return_list, axis=0)
    eval4train_return = np.concatenate(eval4train_return_list, axis=0)
    eval4test_balance = np.concatenate(eval4test_balance_list, axis=0)
    eval4train_balance = np.concatenate(eval4train_balance_list, axis=0)

    for agent in range(1, n_agents + 1):
        logger.console_logger.info(
            "sku{} | test_return:{:.2f} | test_balance:{:.2f} | train_return:{:.2f} | train_balance:{:.2f} |".format(
                agent, eval4test_return[agent - 1], eval4test_balance[agent - 1], 
                eval4train_return[agent - 1], eval4train_balance[agent - 1]
            )
        )

    logger.console_logger.info(
            "eval4test_return_sum: {:.2f}".format(eval4test_return.sum()),
    )
    logger.console_logger.info(
            "eval4train_return_sum: {:.2f}".format(eval4train_return.sum()),
    )
    logger.console_logger.info(
            "eval4test_balance_sum: {:.2f}".format(eval4test_balance.sum()),
    )
    logger.console_logger.info(
            "eval4train_balance_sum: {:.2f}".format(eval4train_balance.sum()),
    )
    logger.console_logger.info(
            "val_best_balance: {:.2f}".format(val_best_balance),
    )
    logger.console_logger.info(
            "test_best_balance: {:.2f}".format(test_best_balance),
    )
    logger.console_logger.info("Finished Evaluation")


    runner.close_env()
    logger.console_logger.info("Finished Training")


def args_sanity_check(config, _log):

    # set CUDA flags
    # config["use_cuda"] = True # Use cuda whenever possible!
    if config["use_cuda"] and not torch.cuda.is_available():
        config["use_cuda"] = False
        _log.warning(
            "CUDA flag use_cuda was switched OFF automatically because no CUDA devices are available!"
        )

    if config["test_nepisode"] < config["batch_size_run"]:
        config["test_nepisode"] = config["batch_size_run"]
    else:
        config["test_nepisode"] = (
            config["test_nepisode"] // config["batch_size_run"]
        ) * config["batch_size_run"]

    return config


def run_decoupled_training(
    episode_num, agent_ids, local_runner, local_infos, context_learner, learner, args,
):

    local_buffer = ReplayBuffer(
        local_infos["local_scheme"],
        local_infos["local_groups"],
        local_infos["buffer_size"],
        local_infos["episode_limit"] + 1,
        preprocess=local_infos["local_preprocess"],
        device="cpu"
        if local_infos["local_args"].buffer_cpu_only
        else local_infos["local_args"].device,
    )
    if args.use_zero_like_context:
        C_trajectories = np.zeros(
            (len(agent_ids), 3, args.env_args["time_limit"] + 1), dtype=int
        )
    else:
        C_trajectories = context_learner.generate_C_trajectories(
            agent_ids, prob=args.fake_dynamics_prob, aug_type=args.aug_type
        )
        C_trajectories = C_trajectories.cpu().numpy()
    local_runner.restart_ps(agent_ids)
    with torch.no_grad():
        episode_batch = local_runner.run(test_mode=False, C_trajectories=C_trajectories)
        local_buffer.insert_episode_batch(episode_batch)
    episode_sample = local_buffer[:]
    max_ep_t = episode_sample.max_t_filled()
    episode_sample = episode_sample[:, :max_ep_t]
    episode_sample.to(args.device)
    rets = learner.train(episode_sample, local_runner.t_env, episode_num)
    local_runner.clean_ps()

    return rets