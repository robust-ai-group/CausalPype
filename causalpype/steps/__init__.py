from .load import LoadData
from .train import TrainModel
from .evaluate import Evaluate
from .base import Step
from .preprocess import AutoPreprocess
from .build_graph import BuildGraph

STEP_REGISTRY = {
    'load_data': LoadData,
    'train_model': TrainModel,
    'evaluate': Evaluate,
    'auto_preprocess': AutoPreprocess,
    'build_graph': BuildGraph
}

__all__ = ['LoadData', 'TrainModel', 'Evaluate', 'STEP_REGISTRY', 'Step', 
           'AutoPreprocess', 'BuildGraph']
