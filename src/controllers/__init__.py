REGISTRY = {}

from .basic_controller import BasicMAC
from .mappo_controller import MAPPOMAC

REGISTRY["basic_mac"] = BasicMAC
REGISTRY["mappo_mac"] = MAPPOMAC