# Quick Start

This guide will walk you through your first CausalPype pipeline.

## Basic Pipeline

```python
from causalpype import Pipeline, PipelineConfig
import polars as pl
import networkx as nx

# 1. Prepare your data
data = pl.read_csv('data.csv')

# 2. Define causal graph
graph = nx.DiGraph([
    ('x1', 'y'),
    ('x2', 'y')
])

# 3. Configure pipeline
config = PipelineConfig(
    outcome='y',      # Target variable
    persist=True,     # Save results
    run_id='my_run'   # Unique identifier
)

# 4. Build pipeline
pipeline = Pipeline(config)
pipeline.artifacts['data'] = data
pipeline.artifacts['graph'] = graph

# 5. Add steps and run
results = (pipeline
    .train_model()
    .evaluate()
    .run())

# 6. Access results
model = results['model']
metrics = results['evaluation_metrics']
print(metrics)
```

## Using Custom Steps

```python
from dataclasses import dataclass
from causalpype.steps.base import Step

@dataclass
class MyStep(Step):
    data_key: str = 'data'

    def execute(self, artifacts, config):
        data = artifacts[self.data_key]
        # Your logic here
        return {'processed': data}

# Use in pipeline
pipeline.add_step(MyStep())
```

## Next Steps

- Learn about [Core Concepts](../guide/concepts.md)
- Explore [Built-in Steps](../guide/steps.md)
- See [Examples](../examples/basic.md)
