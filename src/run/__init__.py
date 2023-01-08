from .cdppo_run import run as cdppo_run
from .dop_run import run as dop_run
from .mappo_run import run as mappo_run
from .on_off_run import run as on_off_run
from .per_run import run as per_run
from .run import run as default_run
from .sarl_run import run as sarl_run
from .sarl4one_run import run as sarl4one_run
from .sarl_generalization import run as sarl_generalization_run
REGISTRY = {}
REGISTRY["default"] = default_run
REGISTRY["mappo_run"] = mappo_run
REGISTRY["cdppo_run"] = cdppo_run
REGISTRY["on_off"] = on_off_run
REGISTRY["dop_run"] = dop_run
REGISTRY["per_run"] = per_run
REGISTRY["sarl_run"] = sarl_run
REGISTRY["sarl4one_run"] = sarl4one_run
REGISTRY["sarl_generalization_run"] = sarl_generalization_run
