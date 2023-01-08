from .ac import ACCritic
from .central_critic import CentralVCritic
from .central_rnn_critic import CentralRNNVCritic
from .coma import COMACritic
from .deepset import DeepSet
from .fmac_critic import FMACCritic
from .lica import LICACritic
from .maddpg import MADDPGCritic
from .mappo_rnn_critic import MAPPORNNCritic
from .mlp import MLPCritic
from .offpg import OffPGCritic
from .mappo_rnn_critic_debug import MAPPORNNCriticDebug

REGISTRY = {}

REGISTRY["ac_critic"] = ACCritic
REGISTRY["cv_critic"] = CentralVCritic
REGISTRY["crnnv_critic"] = CentralRNNVCritic
REGISTRY["mappo_rnn_critic"] = MAPPORNNCritic
REGISTRY["coma_critic"] = COMACritic
REGISTRY["fmac_critic"] = FMACCritic
REGISTRY["lica_critic"] = LICACritic
REGISTRY["maddpg_critic"] = MADDPGCritic
REGISTRY["mlp_critic"] = MLPCritic
REGISTRY["offpg_critic"] = OffPGCritic
REGISTRY["deepset_critic"] = DeepSet
REGISTRY["mappo_rnn_critic_debug"] = MAPPORNNCriticDebug
