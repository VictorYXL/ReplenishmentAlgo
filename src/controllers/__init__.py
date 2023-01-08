from .basic_central_controller import CentralBasicMAC
from .basic_controller import BasicMAC
from .conv_controller import ConvMAC
from .dop_controller import DOPMAC
from .lica_controller import LICAMAC
from .mappo_controller import MAPPOMAC
from .mlp_controller import MLPMAC
from .n_controller import NMAC
from .ppo_controller import PPOMAC

REGISTRY = {}


REGISTRY["basic_mac"] = BasicMAC
REGISTRY["n_mac"] = NMAC
REGISTRY["mlp_mac"] = MLPMAC
REGISTRY["ppo_mac"] = PPOMAC
REGISTRY["mappo_mac"] = MAPPOMAC
REGISTRY["conv_mac"] = ConvMAC
REGISTRY["basic_central_mac"] = CentralBasicMAC
REGISTRY["lica_mac"] = LICAMAC
REGISTRY["dop_mac"] = DOPMAC
