from .episode_runner import EpisodeRunner
from .local_parallel_runner import LocalParallelRunner
from .parallel_runner import ParallelRunner

REGISTRY = {}
REGISTRY["episode"] = EpisodeRunner
REGISTRY["parallel"] = ParallelRunner
REGISTRY["local_parallel"] = LocalParallelRunner
