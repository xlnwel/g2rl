from core.mixin import IdentifierConstructor
from utility import pkg
from distributed.remote.base import RayBase


class Actor(RayBase):
    def __init__(self, env_stats, name=None):
        self._env_stats = env_stats
        self._name = name
        self._idc = IdentifierConstructor()

        self.model_constructors = {}
        self.actor_constructors = {}

        self.actors = {}
        self.configs = {}
        # we defer all constructions to the run time

    def set_actor(self, actor, aid=None, sid=None):
        identifier = self._idc.get_identifier(aid, sid)
        self.actors[identifier] = actor

    def set_weights(self, weights, aid=None, sid=None):
        identifier = self._idc.get_identifier(aid, sid)
        self.actors[identifier].set_weights(weights, identifier=identifier)
    
    def get_weights(self, aid=None, sid=None):
        identifier = self._idc.get_identifier(aid, sid)
        self.actors[identifier].get_weights(identifier=identifier)
    
    def get_auxiliary_weights(self, aid=None, sid=None):
        identifier = self._idc.get_identifier(aid, sid)
        self.actors[identifier].get_auxiliary_weights(identifier=identifier)
    
    def construct_actor_from_config(self, config, aid=None, sid=None, weights=None):
        """ Constructor the actor from config """
        algo = config.algorithm
        self._setup_constructors(algo)
        actor = self._construct_actor(algo, config, self._env_stats)
        
        identifier = self._idc.get_identifier(aid, sid)
        self.actors[identifier] = actor
        self.configs[identifier] = config
        if weights is not None:
            self.actors[identifier].set_weights(weights, aid, sid)

    def _setup_constructors(self, algo):
        self.model_constructors[algo] = pkg.import_module(
            name='elements.model', algo=algo, place=-1).create_model
        self.actor_constructors[algo] = pkg.import_module(
            name='elements.actor', algo=algo, place=-1).create_actor

    def _construct_actor(self, algo, config, env_stats):
        model = self.model_constructors[algo](config.model, env_stats)
        actor = self.actor_constructors[algo](config.actor, model)

        return actor
