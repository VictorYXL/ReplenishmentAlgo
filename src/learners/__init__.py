from .q_learner import QLearner
from .coma_learner import COMALearner
from .qtran_learner import QLearner as QTranLearner
from .local_ppo_learner import LocalPPOLearner
from .ppo_learner import PPOLearner

REGISTRY = {}

REGISTRY["q_learner"] = QLearner
REGISTRY["coma_learner"] = COMALearner
REGISTRY["qtran_learner"] = QTranLearner
REGISTRY["local_ppo_learner"] = LocalPPOLearner
REGISTRY["ppo_learner"] = PPOLearner
