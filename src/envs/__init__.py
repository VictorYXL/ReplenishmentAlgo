import os
import sys
from functools import partial

from .multiagentenv import MultiAgentEnv
from .replenishment import ReplenishmentEnv

try:
    gfootball = True
    from .gfootball import GoogleFootballEnv
except:
    gfootball = False


def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)


REGISTRY = {}

if sys.platform == "linux":
    os.environ.setdefault("SC2PATH", "~/StarCraftII")


REGISTRY["replenishment"] = partial(env_fn, env=ReplenishmentEnv)
