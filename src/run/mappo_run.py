import datetime
import glob
import os
import re
import threading
import time
import copy
from os.path import abspath, dirname
from types import SimpleNamespace as SN
import queue

import pandas as pd
import torch

import wandb
from components.episode_buffer import ReplayBuffer
from components.transforms import OneHot
from controllers import REGISTRY as mac_REGISTRY
from learners import REGISTRY as le_REGISTRY
from runners import REGISTRY as r_REGISTRY
from utils.logging import Logger
from utils.timehelper import time_left, time_str


def get_agent_own_state_size(env_args):
    sc_env = StarCraft2Env(**env_args)
    # qatten parameter setting (only use in qatten)
    return 4 + sc_env.shield_bits_ally + sc_env.unit_type_bits


def run(_run, _config, _log):

    # check args sanity
    _config = args_sanity_check(_config, _log)

    args = SN(**_config)
    args.device = "cuda" if args.use_cuda else "cpu"

    # setup loggers
    logger = Logger(_log)

    _log.info("Experiment Parameters:")
    tmp_config = {k: _config[k] for k in _config if k != "env_args"}
    tmp_config.update(
        {f"env_agrs.{k}": _config["env_args"][k] for k in _config["env_args"]}
    )
    '''
    print(
        pd.Series(tmp_config, name="HyperParameter Value")
        .transpose()
        .sort_index()
        .fillna("")
        .to_markdown()
    )
    '''

    # configure tensorboard logger
    ts = datetime.datetime.now().strftime("%m%dT%H%M")
    unique_token = f"{_config['name']}_{_config['env_args']['n_agents']}_{_config['env_args']['task_type']}_seed{_config['seed']}_{ts}"
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
    os._exit(os.EX_OK)


def evaluate_sequential(args, runner):

    for _ in range(args.test_nepisode):
        runner.run(test_mode=True)

    if args.save_replay:
        runner.save_replay()

    runner.close_env()


def run_sequential(args, logger):
    # Init runner so we can get env info
    runner = r_REGISTRY[args.runner](args=args, logger=logger)

    # Set up schemes and groups here
    env_info = runner.get_env_info()
    args.n_agents = env_info["n_agents"]
    args.n_actions = env_info["n_actions"]
    args.state_shape = env_info["state_shape"]
    args.accumulated_episodes = getattr(args, "accumulated_episodes", None)

    scheme = {
        "state": {"vshape": env_info["state_shape"]},
        "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
        "n_warehouses": {"vshape": env_info["obs_shape"], "group": "agents"},
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
        #TODO: 为什么还有一个个人的reward
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
            
    val_args = copy.deepcopy(args)
    val_args.env_args["mode"] = "validation"
    # 目前config中定义的runner全是parallel的
    val_runner = r_REGISTRY[args.runner](args=val_args, logger=logger)

    test_args = copy.deepcopy(args)
    test_args.env_args["vis_path"] = os.path.join(args.local_results_path, args.unique_token, "env_vis")
    test_args.env_args["mode"] = "test"
    test_runner = r_REGISTRY[args.runner](args=test_args, logger=logger)

    # Give runner the scheme
    runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac)
    val_runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac)
    test_runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac)

    if args.visualize:
        visual_runner = r_REGISTRY["episode"](args=args, logger=logger)
        visual_runner.setup(
            scheme=scheme, groups=groups, preprocess=preprocess, mac=mac
        )

    # Learner
    learner = le_REGISTRY[args.learner](mac, buffer.scheme, logger, args)

    if args.use_cuda:
        learner.cuda()
    if args.checkpoint_path != "":
        test_runner.mac.load_models(args.checkpoint_path)

        if args.evaluate or args.save_replay:
            vis_save_path = os.path.join(
                args.local_results_path, args.unique_token, "vis"
            ) if os.getenv("AMLT_OUTPUT_DIR") is None else os.path.join(os.getenv("AMLT_OUTPUT_DIR"), "results", args.unique_token, "vis")
            test_runner.run(test_mode=True, visual_outputs_path=vis_save_path)
            test_cur_avg_balance, test_cur_store_balance = test_runner.get_overall_avg_balance()
            logger.console_logger.info("test_cur_avg_balance : {}".format(test_cur_avg_balance))
            return

    # start training
    episode = 0
    last_test_T = -args.test_interval - 1
    last_log_T = 0
    model_save_time = 0
    visual_time = 0
    max_avg_balance = -1
    test_max_avg_balance = -1
    max_store_balance = [0,0,0]
    test_max_store_balance = [0,0,0]
    max_model_path = None

    start_time = time.time()
    last_time = start_time

    logger.console_logger.info("Beginning training for {} timesteps".format(args.t_max))
    # 停止条件为，最近10次测试中，最大波动不超过5%
    recent_val_balance_queue = queue.Queue(maxsize = 10)
    test_flag = False
    while runner.t_env <= args.t_max:

        # Run for a whole episode at a time

        with torch.no_grad():
            episode_batch = runner.run(test_mode=False)
            buffer.insert_episode_batch(episode_batch)

        if buffer.can_sample(args.batch_size):
            next_episode = episode + args.batch_size_run
            if (
                args.accumulated_episodes
                and next_episode % args.accumulated_episodes != 0
            ):
                continue

            episode_sample = buffer.sample(args.batch_size)

            # Truncate batch to only filled timesteps
            max_ep_t = episode_sample.max_t_filled()
            episode_sample = episode_sample[:, :max_ep_t]

            if episode_sample.device != args.device:
                episode_sample.to(args.device)

            learner.train(episode_sample, runner.t_env, episode)
            del episode_sample

        # Execute test runs once in a while
        # n_test_runs是什么意思？测试需要进行的episode数目，除以并行的episode数目
        n_test_runs = max(1, args.test_nepisode // runner.batch_size)
        if (runner.t_env - last_test_T) / args.test_interval >= 1.0:

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
            # 在n次测试上取平均值。因为是parallel的，所以例如，假如batch_size = 8,test_nepisode = 10, 那么n_test_runs = 1。
            for run_index in range(n_test_runs):
                val_runner.t_env = runner.t_env
                # 在验证集上跑，但是不进行可视化
                # 在测试模式上返回的episode_batch是不使用的，测试模式仅仅是为了获得统计数据
                episode_batch = val_runner.run(test_mode=True, visual_outputs_path=None)
                # TODO:若有问题，难道是get_overall_avg_balance就不对？应该就是统计的有问题！可能是网络的问题，也可能是统计计算的时候有问题，仔细看看
                cur_avg_balance, cur_store_balances = val_runner.get_overall_avg_balance()

                test_runner.t_env = runner.t_env
                test_episode_batch = test_runner.run(test_mode=True, visual_outputs_path=None)
                test_cur_avg_balance, test_cur_store_balance = test_runner.get_overall_avg_balance()

                if not recent_val_balance_queue.full():
                    recent_val_balance_queue.put(cur_avg_balance)
                else:
                    recent_val_balance_queue.get()
                    recent_val_balance_queue.put(cur_avg_balance)
                    test_flag = True
                os.makedirs(os.path.join(args.local_results_path, args.unique_token), exist_ok=True)
                log_path = os.path.join(args.local_results_path, args.unique_token, "log.txt")
                f = open(log_path, "a")
                f.write("Run: {}".format(run_index) + "\n")

                f.write("val_cur_avg_balance : {}\n".format(cur_avg_balance))
                f.write("val_max_avg_balance : {}\n".format(max_avg_balance))
                f.write("test_cur_avg_balance : {}\n".format(test_cur_avg_balance))
                f.write("test_max_avg_balance : {}\n".format(test_max_avg_balance))

                f.write("val_cur_store_balances : {}\n".format(cur_store_balances))
                f.write("val_max_store_balance : {}\n".format(max_store_balance))
                f.write("test_cur_store_balance : {}\n".format(test_cur_store_balance))
                f.write("test_max_store_balance : {}\n".format(test_max_store_balance))
                #TODO:在这里插入每个store的价格
                
                f.close()
                # 所以说这里存的最大值，都是在验证集上balance最大的时候存的，而非在测试集最大时存
                if cur_avg_balance > 0 and cur_avg_balance > max_avg_balance:
                    model_save_time = val_runner.t_env
                    save_path = os.path.join(
                        args.local_results_path, args.unique_token, "models", str(runner.t_env)
                    ) if os.getenv("AMLT_OUTPUT_DIR") is None else os.path.join(os.getenv("AMLT_OUTPUT_DIR"), "results", args.unique_token, "models", str(runner.t_env))
                    save_path = save_path.replace('*', '_')
                    max_model_path = save_path
                    max_avg_balance = cur_avg_balance
                    max_store_balance = cur_store_balances
                    test_max_avg_balance = test_cur_avg_balance
                    test_max_store_balance = test_cur_store_balance
                    os.makedirs(save_path, exist_ok=True)
                    logger.console_logger.info("Saving models to {}".format(save_path))
                    learner.save_models(save_path)

                logger.console_logger.info("val_cur_avg_balance : {}".format(cur_avg_balance))
                logger.console_logger.info("val_max_avg_balance : {}".format(max_avg_balance))
                logger.console_logger.info("test_cur_avg_balance : {}".format(test_cur_avg_balance))
                logger.console_logger.info("test_max_avg_balance : {}".format(test_max_avg_balance))

        # 每model_save_time保存一次
        if args.save_model and (
            runner.t_env - model_save_time >= args.save_model_interval
            or model_save_time == 0
        ):
            model_save_time = runner.t_env
            save_path = os.path.join(
                args.local_results_path, args.unique_token, "models", str(runner.t_env)
            ) if os.getenv("AMLT_OUTPUT_DIR") is None else os.path.join(os.getenv("AMLT_OUTPUT_DIR"), "results", args.unique_token, "models", str(runner.t_env))
            save_path = save_path.replace('*', '_')
            os.makedirs(save_path, exist_ok=True)
            logger.console_logger.info("Saving models to {}".format(save_path))

            # learner should handle saving/loading -- delegate actor save/load to mac,
            # use appropriate filenames to do critics, optimizer states
            learner.save_models(save_path)

        if args.visualize and (
            (runner.t_env - visual_time) / args.visualize_interval >= 1.0
        ):
            visual_time = runner.t_env
            visual_outputs_path = os.path.join(
                args.local_results_path, args.unique_token, "visual_outputs"
            )
            logger.console_logger.info(
                f"Saving visualizations to {visual_outputs_path}/{runner.t_env}"
            )

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

        episode += args.batch_size_run

        if (runner.t_env - last_log_T) >= args.log_interval:
            logger.log_stat("episode", episode, runner.t_env)
            logger.print_recent_stats()
            last_log_T = runner.t_env
        # 如果维护的最小值和最大值的差别小于他俩之和的5%，那就认为是收敛了
        if test_flag:
            test_flag = False
            max_balance = max(recent_val_balance_queue.queue)
            min_balance = min(recent_val_balance_queue.queue)
            mean_balance = sum(recent_val_balance_queue.queue)/len(recent_val_balance_queue.queue)
            positive_deviation_ratio = abs((max_balance-mean_balance)/mean_balance)
            negative_deviation_ratio = abs((min_balance-mean_balance)/mean_balance)
            print("max : {}, min : {}, mean : {}".format(max_balance, min_balance, mean_balance))
            print("positive_deviation_ratio : {}, negative_deviation_ratio : {}".format(positive_deviation_ratio,negative_deviation_ratio))
            if positive_deviation_ratio < 0.05 and negative_deviation_ratio < 0.05:
                break

            
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
        test_avg_balances, test_store_balances = test_runner.get_overall_avg_balance()
        logger.console_logger.info("test_avg_balances : {}".format(test_avg_balances))

    # close env
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