# CausalPype

A Python library for building modular causal inference pipelines.

## What is CausalPype?

CausalPype provides a simple, step-based framework for causal analysis workflows. Instead of writing scattered scripts, you define reusable pipeline steps that can be chained together.

## Key Features

- **Modular architecture** - Build pipelines from composable steps
- **Flexible configuration** - Define pipelines in Python or YAML
- **Extensible** - Easy to create custom steps for your needs
- **Causal models** - Built on DoWhy for structural causal modeling

## Quick Start

### Installation

```bash
uv pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ causalpype==0.0.1
```

### Basic Usage

```python
from causalpype import Pipeline, PipelineConfig
import polars as pl
import networkx as nx

# Configure pipeline
config = PipelineConfig(
    outcome='y',
    persist=True
)

# Load your data and graph
data = pl.read_csv('data.csv')
graph = nx.DiGraph([('x1', 'y'), ('x2', 'y')])

# Build and run pipeline
from causalpype.steps import TrainModel, Evaluate

pipeline = Pipeline(config)
pipeline.artifacts['data'] = data
pipeline.artifacts['graph'] = graph

results = (pipeline
    .add_step(TrainModel())
    .add_step(Evaluate())
    .run())

# Access results
model = results['model']
metrics = results['evaluation_metrics']
```

### Creating Custom Steps

```python
from dataclasses import dataclass
from causalpype.steps.base import Step

@dataclass
class MyCustomStep(Step):
    data_key: str = 'data'
    output_key: str = 'processed'

    def execute(self, artifacts, config):
        data = artifacts[self.data_key]

        # Your processing logic
        processed = data.with_columns([
            (pl.col('x') * 2).alias('x_doubled')
        ])

        return {self.output_key: processed}

# Use in pipeline
pipeline.add_step(MyCustomStep())
```

### YAML Configuration

```yaml
run_id: "my_analysis"
outcome: "outcome_variable"
persist: true

steps:
  - type: load_data
    params:
      path: "data.csv"

  - type: train_model
    params:
      quality: "better"

  - type: evaluate
```

```python
pipeline = Pipeline.from_yaml('config.yaml')
results = pipeline.run()
```

## Core Concepts

### Pipeline
Orchestrates step execution and manages data flow between steps.

### Steps
Individual processing units that transform data or train models. Each step:
- Takes inputs from the artifacts dict
- Performs a specific task
- Returns outputs to merge back into artifacts

### Configuration
Controls pipeline behavior, output settings, and step parameters.

## Current Status

⚠️ **Early Development** - API may change. Not recommended for production use yet.

## Future Directions

- Causal discovery algorithms
- Support for distributed/federated analysis
- Interactive visualization tools
- More built-in evaluation metrics
- Integration with additional causal libraries

## Examples

See the `examples/` directory for:
- Custom step implementations
- Complete workflow examples
- Notebook tutorials


## License

Apache License 2.0