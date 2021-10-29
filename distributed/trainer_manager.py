from distributed.remote.trainer import Trainer
from utility.utils import config_attr


class TrainerManager:
    def __init__(self, config, env_stats):
        config_attr(self, config)
        self._env_stats = env_stats

        self.trainers = {}

    def add_trainer(self, trainer_id, config, aid=None, sid=None, weights=None):
        self.trainers[trainer_id] = Trainer.as_remote().remote(self._env_stats)
        self.trainers[trainer_id].construct_trainer_from_config(
            config, aid, sid, weights)
