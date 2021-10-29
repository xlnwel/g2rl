import ray

from utility.utils import config_attr
from distributed.remote.actor import Actor


class ActorManager:
    def __init__(self, config, env_stats):
        config_attr(self, config, filter_dict=True)
        self._env_stats = env_stats

        self.actors = {}

    def add_remote_actor(self, actor_id, config, aid=None, sid=None, weights=None):
        self.actors[actor_id] = Actor.as_remote().remote(self._env_stats)
        self.actors[actor_id].construct_actor_from_config.remote(
            config, aid, sid, weights)

    def set_actor(self, actor, aid=None, sid=None):
        self.actors[aid].remote(actor, aid, sid)

    def set_actor_weights(self, weights, aid=None, sid=None):
        self.actors[aid].set_weights.remote(weights, aid, sid)

    def get_auxiliary_stats(self, aid=None, sid=None):
        return ray.get(self.actors[aid].get_auxiliary_stats.remote(aid, sid))
