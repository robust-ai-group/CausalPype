# Core Concepts

## Pipeline

A `Pipeline` orchestrates the execution of steps and manages data flow.

```python
pipeline = Pipeline(config)
results = pipeline.run()
```

The pipeline:
- Executes steps in order
- Passes data between steps via the `artifacts` dict
- Manages configuration and outputs

## Steps

Steps are the building blocks of pipelines. Each step:

1. **Takes inputs** from the artifacts dict
2. **Performs work** (load data, train model, etc.)
3. **Returns outputs** to merge back into artifacts

```python
@dataclass
class MyStep(Step):
    def execute(self, artifacts, config):
        # Get inputs
        data = artifacts['data']

        # Do work
        processed = process(data)

        # Return outputs
        return {'processed_data': processed}
```

## Artifacts

The artifacts dict stores all data flowing through the pipeline:

```python
{
    'data': DataFrame,
    'graph': DiGraph,
    'model': CausalModel,
    'evaluation_metrics': {...}
}
```

Steps read from and write to this shared dict.

## Configuration

`PipelineConfig` controls pipeline behavior:

```python
config = PipelineConfig(
    run_id='my_analysis',
    outcome='y',
    persist=True,
    output_dir='results'
)
```

The config is passed to every step for consistent behavior.

## Data Flow

```
Load Data → artifacts['data']
         ↓
Train Model (reads artifacts['data', 'graph'])
         ↓
    artifacts['model']
         ↓
Evaluate (reads artifacts['data', 'model'])
         ↓
    artifacts['metrics']
```

Each step adds to the artifacts, building up the complete analysis.
