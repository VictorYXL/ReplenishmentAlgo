from .cdppo_learner import CDPPOLearner
from .coma_learner import COMALearner
from .context_learner import ContextLearner
from .dmaq_qatten_learner import DMAQ_qattenLearner
from .fmac_learner import FMACLearner
from .lica_learner import LICALearner
from .local_ppo_learner import LocalPPOLearner
from .max_q_learner import MAXQLearner
from .nq_learner import NQLearner
from .offpg_learner import OffPGLearner
from .policy_gradient_v2 import PGLearner_v2
from .ppo_learner import PPOLearner
from .q_learner import QLearner
from .qtran_learner import QLearner as QTranLearner

REGISTRY = {}

REGISTRY["q_learner"] = QLearner
REGISTRY["coma_learner"] = COMALearner
REGISTRY["qtran_learner"] = QTranLearner
REGISTRY["ppo_learner"] = PPOLearner
REGISTRY["local_ppo_learner"] = LocalPPOLearner
REGISTRY["cdppo_learner"] = CDPPOLearner
REGISTRY["lica_learner"] = LICALearner
REGISTRY["nq_learner"] = NQLearner
REGISTRY["policy_gradient_v2"] = PGLearner_v2
REGISTRY["max_q_learner"] = MAXQLearner
REGISTRY["dmaq_qatten_learner"] = DMAQ_qattenLearner
REGISTRY["offpg_learner"] = OffPGLearner
REGISTRY["fmac_learner"] = FMACLearner
REGISTRY["context_learner"] = ContextLearner
