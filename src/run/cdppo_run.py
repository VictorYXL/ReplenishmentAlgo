import copy
import datetime
import glob
import os
import re
import pprint
import threading
import time
from collections import defaultdict
from dataclasses import replace
from os.path import abspath, dirname
from types import SimpleNamespace as SN

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
from utils.timehelper import TimeStat, time_left, time_str


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
        "cur_balance": {"vshape": (1,), "group": "agents"},
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

    map_name = args.env_args["map_name"]
    n_agents = int(re.findall(r"n(.*)c", map_name)[0])
    max_capacity = int(re.findall(r"c(.*)d", map_name)[0])
    hist_len = int(re.findall(r"d(.*)s", map_name)[0])

    val_args = copy.deepcopy(args)
    val_map_name = f"n{n_agents}c{max_capacity}d{hist_len}s{300}l{100}"
    val_args.env_args["map_name"] = val_map_name
    val_runner = r_REGISTRY[args.runner](args=val_args, logger=logger)

    test_args = copy.deepcopy(args)
    test_map_name = f"n{n_agents}c{max_capacity}d{hist_len}s{400}l{100}"
    test_args.env_args["map_name"] = test_map_name
    test_runner = r_REGISTRY[args.runner](args=test_args, logger=logger)
    # import ipdb; ipdb.set_trace()

    # Give runner the scheme
    runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac)
    val_runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac)
    test_runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac)

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
    # local_args.use_individual_envs = True
    local_args.critic_input_seq_str = "o_la^i"
    local_args.learner = "local_ppo_learner"
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
    # print(f"local_env_info: {local_env_info}")

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
        "individual_rewards": {"vshape": (1,), "group": "agents"},
        "cur_balance": {"vshape": (1,), "group": "agents"},
        "sub_balance": {"vshape": (1,), "group": "agents"},
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

    # print("Actor Arch:", mac.agent)
    # if args.train_with_joint_data:
    #     print("Critic Arch:", learner.critic)
    # if args.n_decoupled_iterations > 0:
    #     print("Local Critic Arch:", local_learner.critic)

    if args.checkpoint_path != "":
        timesteps = []
        timestep_to_load = 0

        if not os.path.isdir(args.checkpoint_path):
            logger.console_logger.info(
                "Checkpoint directiory {} doesn't exist".format(args.checkpoint_path)
            )
            return

        # Go through all files in args.checkpoint_path
        for name in os.listdir(args.checkpoint_path):
            full_name = os.path.join(args.checkpoint_path, name)
            # Check if they are dirs the names of which are numbers
            if os.path.isdir(full_name) and name.isdigit():
                timesteps.append(int(name))

        if args.load_step == 0:
            # choose the max timestep
            timestep_to_load = max(timesteps)
        else:
            # choose the timestep closest to load_step
            timestep_to_load = min(timesteps, key=lambda x: abs(x - args.load_step))

        model_path = os.path.join(args.checkpoint_path, str(timestep_to_load))

        logger.console_logger.info("Loading model from {}".format(model_path))
        local_learner.load_models(model_path)
        context_learner.load_models(model_path)
        runner.t_env = timestep_to_load

    if args.evaluate:
        runner.log_train_stats_t = runner.t_env
        # if args.n_decoupled_iterations > 0:
        #     runner.mac.load_state(local_learner.mac)
        # evaluate_sequential(args, logger, runner)
        visual_outputs_path = os.path.join(
            args.local_results_path, args.unique_token, "visual_outputs"
        )
        visual_runner.run_visualize(visual_outputs_path, runner.t_env)
        logger.log_stat("episode", runner.t_env, runner.t_env)
        logger.print_recent_stats()
        logger.console_logger.info("Finished Evaluation")
        return

    # start training
    episode = 0
    local_episode = 0
    last_test_T = -args.test_interval - 1
    last_log_T = 0
    model_save_time = 0
    visual_time = 0
    max_avg_balance = -1
    test_max_avg_balance = -1
    max_model_path = None

    start_time = time.time()
    last_time = start_time

    logger.console_logger.info("Beginning training for {} timesteps".format(args.t_max))

    while runner.t_env <= args.t_max:  # for the_i in range(1):#

        # Run for a whole episode at a time
        # if args.n_decoupled_iterations > 0:
        #     runner.mac.load_state(local_learner.mac)
        # logger.console_logger.info(
        #     f"Collecting data on one {args.n_agents}-agents joint environment ......"
        # )
        with torch.no_grad():
            episode_batch = runner.run(test_mode=False)
            buffer.insert_episode_batch(episode_batch)
        episode_sample = buffer.sample(args.batch_size)
        # Truncate batch to only filled timesteps
        max_ep_t = episode_sample.max_t_filled()
        episode_sample = episode_sample[:, :max_ep_t]
        episode_sample.to(args.device)

        # local_logger.console_logger.info("Training ContextModel ......")
        # print(runner.get_C_trajectories())
        # print(np.isnan(runner.get_C_trajectories()).any())
        context_learner.train(runner.get_C_trajectories(), runner.t_env)

        if args.train_with_joint_data:
            # logger.console_logger.info("Training policy using joint data ")
            learner.train(episode_sample, runner.t_env, episode)
            # local_runner.mac.load_state(learner.mac)
            del episode_sample

        for _ in range(args.n_decoupled_iterations):
            if not args.select_ls_randomly:
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
                    args.n_agents,
                    size=[
                        args.max_individual_envs
                        if args.max_individual_envs >= 1
                        else max(1, int(args.n_agents * args.max_individual_envs))
                    ],
                    replace=False,
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
                wandb.log(rets, step=runner.t_env)

            local_episode += (
                args.max_individual_envs
                if args.max_individual_envs >= 1
                else max(1, int(args.n_agents * args.max_individual_envs))
            )

            episode += args.batch_size_run + joint_local_ratio * 1

        if (
            args.visualize
            and args.n_agents <= 100
            and ((runner.t_env - visual_time) / args.visualize_interval >= 1.0)
        ):
            visual_time = runner.t_env
            visual_outputs_path = os.path.join(
                args.local_results_path, args.unique_token, "visual_outputs"
            )
            logger.console_logger.info(
                f"Saving visualizations to {visual_outputs_path}/{runner.t_env}"
            )
            # if args.n_decoupled_iterations > 0:
            #     visual_runner.mac.load_state(local_learner.mac)
            visual_runner.run_visualize(visual_outputs_path, runner.t_env)
            if args.use_wandb:
                imgs = sorted(
                    glob.glob(
                        f"{visual_outputs_path}/{runner.t_env}/OuterSKUStoreUnit*.png"
                    ),
                    key=os.path.getmtime,
                )[::-1]
                wandb.log(
                    {
                        f"SKUBehaviour/sku_{i+1}": wandb.Image(imgs[i])
                        for i in range(args.n_agents)
                    },
                    step=runner.t_env,
                )

        # Execute test runs once in a while
        n_test_runs = max(1, args.test_nepisode // runner.batch_size)
        if (runner.t_env - last_test_T) / args.test_interval >= 1.0:
            # logger.console_logger.info(
            #     f"Evaluating model on the joint environment {n_test_runs} times ......."
            # )
            # if args.n_decoupled_iterations > 0:
            #     runner.mac.load_state(local_learner.mac)
            logger.console_logger.info(
                "t_env: {} / {}".format(runner.t_env, args.t_max)
            )
            logger.console_logger.info(
                "Estimated time left: {}. Time passed: {}".format(
                    time_left(last_time, last_test_T, runner.t_env, args.t_max),
                    time_str(time.time() - start_time),
                )
            )
            last_time = time.time()
            last_test_T = runner.t_env
            # for _ in range(n_test_runs):
            #     runner.run(test_mode=True)

            for _ in range(n_test_runs):
                val_runner.t_env = runner.t_env
                episode_batch = val_runner.run(test_mode=True, visual_outputs_path=None)
                cur_avg_balances = val_runner.get_overall_avg_balance()

                test_runner.t_env = runner.t_env
                test_episode_batch = test_runner.run(test_mode=True, visual_outputs_path=None)
                test_cur_avg_balances = test_runner.get_overall_avg_balance()

                if cur_avg_balances > max_avg_balance:
                    model_save_time = val_runner.t_env
                    save_path = os.path.join(
                        args.local_results_path, args.unique_token, "models", str(runner.t_env)
                    ) if os.getenv("AMLT_OUTPUT_DIR") is None else os.path.join(os.getenv("AMLT_OUTPUT_DIR"), "results", args.unique_token, "models", str(runner.t_env))
                    save_path = save_path.replace('*', '_')
                    max_model_path = save_path
                    max_avg_balance = cur_avg_balances
                    test_max_avg_balance = test_cur_avg_balances
                    os.makedirs(save_path, exist_ok=True)
                    logger.console_logger.info("Saving models to {}".format(save_path))
                    learner.save_models(save_path)

                logger.console_logger.info("val_cur_avg_balances : {}".format(cur_avg_balances))
                logger.console_logger.info("val_max_avg_balance : {}".format(max_avg_balance))
                logger.console_logger.info("test_cur_avg_balances : {}".format(test_cur_avg_balances))
                logger.console_logger.info("test_max_avg_balance : {}".format(test_max_avg_balance))

        if args.save_model and (
            # runner.t_env - model_save_time >= args.save_model_interval
            model_save_time == 0
        ):
            model_save_time = runner.t_env
            save_path = os.path.join(
                args.local_results_path, args.unique_token, "models", str(runner.t_env)
            ) if os.getenv("AMLT_OUTPUT_DIR") is None else os.path.join(os.getenv("AMLT_OUTPUT_DIR"), "results", args.unique_token, "models", str(runner.t_env))
            save_path = save_path.replace('*', '_')
            os.makedirs(save_path, exist_ok=True)
            logger.console_logger.info(f"Saving models to {save_path}")

            # learner should handle saving/loading -- delegate actor save/load to mac,
            # use appropriate filenames to do critics, optimizer states
            local_learner.save_models(save_path)
            context_learner.save_models(save_path)

        if (runner.t_env - last_log_T) >= args.log_interval:
            # local_logger.log_stat("episode", local_episode, local_runner.t_env)
            # local_logger.console_logger.info(
            #     f"Recent Stats | L_t_env: {local_runner.t_env:>10} | Episode: {local_episode:>8}"
            # )
            logger.log_stat("episode", episode, runner.t_env)
            logger.print_recent_stats()
            # local_logger.print_recent_stats()
            # else:
            # logger.log_stat("episode", episode, runner.t_env)
            # for key, ts in time_stats.items():
            #   logger.log_stat(f"{key}_time_mean", ts.mean or 0, runner.t_env)
            # logger.print_recent_stats()
            last_log_T = runner.t_env

    n_test_runs = max(1, args.test_nepisode // runner.batch_size)
    for _ in range(n_test_runs):
        learner.load_models(max_model_path)
        
        save_path = os.path.join(
            args.local_results_path, args.unique_token, "models", str(runner.t_env)
        ) if os.getenv("AMLT_OUTPUT_DIR") is None else os.path.join(os.getenv("AMLT_OUTPUT_DIR"), "results", args.unique_token, "models", "best")
        save_path = save_path.replace('*', '_')
        os.makedirs(save_path, exist_ok=True)
        logger.console_logger.info("Saving best models to {}".format(save_path))
        learner.save_models(save_path)

        test_runner.t_env = runner.t_env
        vis_save_path = os.path.join(
            args.local_results_path, args.unique_token, "vis"
        ) if os.getenv("AMLT_OUTPUT_DIR") is None else os.path.join(os.getenv("AMLT_OUTPUT_DIR"), "results", args.unique_token, "vis")
        os.makedirs(vis_save_path, exist_ok=True)
        vis_save_path = vis_save_path.replace('*', '_')
        test_runner.run(test_mode=True, visual_outputs_path=vis_save_path)
        test_avg_balances = test_runner.get_overall_avg_balance()
        logger.console_logger.info("test_avg_balances : {}".format(test_avg_balances))

    runner.close_env()
    val_runner.close_env()
    test_runner.close_env()
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