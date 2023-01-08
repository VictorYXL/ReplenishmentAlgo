from .atten_rnn_agent import ATTRNNAgent
from .central_rnn_agent import CentralRNNAgent
from .conv_agent import ConvAgent
from .ff_agent import FFAgent
from .mlp_agent import MLPAgent
from .n_rnn_agent import NRNNAgent
from .noisy_agents import NoisyRNNAgent
from .rnn_agent import RNNAgent
from .rnn_ppo_agent import RNNPPOAgent
from .mlp_mmoe_agent import MLPMMOEAgent
from .n_rnn_mmoe_agent import NRNNMMOEAgent
REGISTRY = {}


REGISTRY["rnn"] = RNNAgent
REGISTRY["n_rnn"] = NRNNAgent
REGISTRY["rnn_ppo"] = RNNPPOAgent
REGISTRY["conv_agent"] = ConvAgent
REGISTRY["ff"] = FFAgent
REGISTRY["central_rnn"] = CentralRNNAgent
REGISTRY["mlp"] = MLPAgent
REGISTRY["att_rnn"] = ATTRNNAgent
REGISTRY["noisy_rnn"] = NoisyRNNAgent
REGISTRY["mlp_mmoe"] = MLPMMOEAgent
REGISTRY["n_rnn_mmoe"] = NRNNMMOEAgent