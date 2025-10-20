# Built-in Steps

CausalPype includes several built-in steps for common causal inference tasks.

All steps are used with the `pipeline.add_step()` method:

```python
from causalpype.steps import LoadData, TrainModel, Evaluate

pipeline.add_step(LoadData(path='data.csv'))
pipeline.add_step(TrainModel())
pipeline.add_step(Evaluate())
```

---

## Available Steps

::: causalpype.steps
    options:
      show_root_heading: false
      show_source: false
      members: true
      filters:
        - "!^_"
        - "!^Step$"
        - "!^STEP_REGISTRY$"
      heading_level: 3

---

## Complete Example

```python
from causalpype import Pipeline, PipelineConfig
from causalpype.steps import LoadData, AutoPreprocess, BuildGraph, TrainModel, Evaluate

config = PipelineConfig(outcome='y', persist=True)
pipeline = Pipeline(config)

# Add steps
pipeline.add_step(LoadData(path='data.csv'))
pipeline.add_step(AutoPreprocess(null_threshold=0.8))
pipeline.add_step(BuildGraph(edge_list_path='graph.csv'))
pipeline.add_step(TrainModel())
pipeline.add_step(Evaluate())

# Run
results = pipeline.run()
```

---

## Creating Custom Steps

See the [Custom Steps Guide](custom-steps.md) for creating your own steps.
