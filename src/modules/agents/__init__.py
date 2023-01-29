REGISTRY = {}

from .rnn_agent import RNNAgent
from .n_rnn_agent import NRNNAgent

REGISTRY["rnn"] = RNNAgent
REGISTRY["n_rnn"] = NRNNAgent