from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Dict, Any
from causalpype.config import PipelineConfig

@dataclass
class Step(ABC):
    
    @abstractmethod
    def execute(self, artifacts: Dict[str, Any], config: 'PipelineConfig') -> Dict[str, Any]:
        pass
