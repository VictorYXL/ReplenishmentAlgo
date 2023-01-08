# replenishment-marl-baselines

## Before using this repo
Please learn about basic usages about `epymarl` repo(you can read [Origin epymarl README](epymarl_readme.md) for more details)

## Installation
Run following commands to install dependences
```shell
git clone https://github.com/fmxFranky/replenishment-marl-baselines.git --recursive
cd replenishment-marl-baselines
conda env create -f conda_env.yml
conda activate replenishment
chmod +x ./modify_ray_rllib.sh
./modify_ray_rllib.sh
pip install -e ./replenishment-env/
```

## Usages
### Basic usage for running MAPPO on replenishment-env
`python src/main.py --config=mappo --env-config=replenishment --loglevel=INFO`
### Paralell environments for efficient data sampling
`python src/main.py --config=mappo --env-config=replenishment --loglevel=INFO with runner=parallel batch_size_run=10` 
### Use `wandb` for online logging
`python src/main.py --config=mappo --env-config=replenishment --loglevel=INFO with use_wandb=True wandb_project_name=<your_wandb_project_name>`
Please read [default.yaml](src/config/default.yaml) for more details about kwargs


When you start training in your console, you should see printouts that look like
```
[INFO 15:56:51] my_main t_env: 14600 / 10000000
[INFO 15:56:51] my_main Estimated time left: 1 hours, 59 minutes, 16 seconds. Time passed: 1 minutes, 2 seconds
[INFO 15:57:05] my_main Saving models to results/models/mappo_seed1234_replenishment_n5_0707T1555/14600
[INFO 15:57:05] my_main Recent Stats | t_env:      14600 | Episode:       10
advantage_mean:           -0.4732 agent_grad_norm:           2.1413 critic_grad_norm:        115.7179 critic_loss:               6.5013
ep_length_mean:          1460.0000 pg_loss:                   0.4344 pi_max:                    0.6306 profit_mean:             -6247622.0000
profit_std:              75310.6859 q_taken_mean:             -0.0956 return_mean:             -53.7872 return_std:                0.9117
target_mean:              -0.5688 td_error_abs:              2.0244 test_ep_length_mean:     365.0000 test_profit_mean:        -4576162.0000
test_profit_std:         114660.8223 test_return_mean:         -8.5616 test_return_std:           0.3420
[INFO 15:58:09] my_main Recent Stats | t_env:      29200 | Episode:       20
advantage_mean:           -0.4074 agent_grad_norm:           1.9695 critic_grad_norm:        100.5384 critic_loss:               6.6560
ep_length_mean:          1460.0000 pg_loss:                   0.3681 pi_max:                    0.6325 profit_mean:             -5564545.5000
profit_std:              84982.1190 q_taken_mean:             -0.1556 return_mean:             -45.6603 return_std:                1.0196
target_mean:              -0.5630 td_error_abs:              2.0449 test_ep_length_mean:     365.0000 test_profit_mean:        -4576162.0000
test_profit_std:         114660.8223 test_return_mean:         -8.5616 test_return_std:           0.3420
[INFO 15:59:13] my_main Recent Stats | t_env:      43800 | Episode:       30
advantage_mean:           -0.1561 agent_grad_norm:           2.0247 critic_grad_norm:         94.5232 critic_loss:               6.7414
ep_length_mean:          1460.0000 pg_loss:                   0.1160 pi_max:                    0.6363 profit_mean:             -4943861.0833
profit_std:              106837.5397 q_taken_mean:             -0.4005 return_mean:             -38.3445 return_std:                1.2622
target_mean:              -0.5565 td_error_abs:              2.0733 test_ep_length_mean:     365.0000 test_profit_mean:        -4576162.0000
test_profit_std:         114660.8223 test_return_mean:         -8.5616 test_return_std:           0.3420
```

## TODO List
- [x] Add [replenishment-env](https://github.com/fmxFranky/replenishment-env) as an internal submodule 
- [x] Add [Weights & Bias](https://wandb.ai/site) as online logger
- [x] Modify `_GymmaWrapper` for more infos:
  - [x] Logging total profit
  - [x] Distinguish training env and eval env in framework
- [ ] Run provided algos on `replenishment_env` and compare their performance
- [ ] Finetune hyper parameters
- [ ] Try empirical tricks to improve performcnce
- [ ] Modify shells in original `epymarl` repo(if necessary)


# replenishment-env
The replenishment environment, there are 2 versions of the env, the old_env models a whole pipeline of the supply chain while focusing on the last echelon (the retailers); the eff_env uses matrix operations to model the supply chain, thus the latter would be much more agile.

The old_env lie in [n5_local,n50_local,n100_local...], the number after n represents SKU number. To specify these old envs, change env_args.key in file './src/config/envs/replenishment.yaml' to old_env id, e.g. "replenishment_n5_local-v0"

The eff_env lie in efficient_env folder, backend logics is in 'efficient_replenishment_env.py', it loads config file from 'efficient_env/config' folder and demand data from 'efficient_env/data', the way to use the eff_env is to change env_args.key in file './src/config/envs/replenishment.yaml' to eff_env id, a.k.a "replenishment_efficient_env-v0"; map_name indicates the specific env setting, e.g. "n1000c8000d1" represents using 1000SKUs, and the storage is 8000, and we use 1 day history for training neural networks.

For eff_env, we now support SKU number 5, 10, 20, 50, 100, 200, 300, 500, 1000. The more SKU the env holds, the longer it takes to load the demand file. Different SKU number may result in different training or test interval, so the [test_interval,log_interval,runner_log_interval,learner_log_interval] variables in file './src/config/envs/replenishment.yaml' must be the same as [env_config["episod_duration"]] in file 'efficient_env/config/inventory_config_nxx.py'.

In file 'efficient_env/config/inventory_config_nxx.py':
[env_config["episod_duration"]] controls the training interval length,
[env_config["evaluation_len"]] controls the testing interval length.

If one wants to generate environments for different number of SKUs, or simply change the config or demand data, kindly refer to https://github.com/liugz18/m5_data_extractor, after generating the new config/data files, one needs to:
1. copy and paste the generated config or data, register it in file 'efficient_replenishment_env.py', mind the naming
2. call python read_mean.py to extract the demand mean in new data
3. copy and paste the demand mean in './src/learners/context_learner'
