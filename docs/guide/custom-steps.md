# Creating Custom Steps

## Basic Template

```python
from dataclasses import dataclass
from causalpype.steps.base import Step
from causalpype.config import PipelineConfig
from typing import Dict, Any

@dataclass
class MyCustomStep(Step):
    """Description of what this step does."""

    # Input/output keys
    data_key: str = 'data'
    output_key: str = 'result'

    # Parameters
    threshold: float = 0.5

    def execute(self, artifacts: Dict[str, Any], config: PipelineConfig) -> Dict[str, Any]:
        # Get inputs
        data = artifacts[self.data_key]

        # Your logic
        result = data.filter(pl.col('value') > self.threshold)

        # Return outputs
        return {self.output_key: result}
```

## Usage

```python
# Create step instance
step = MyCustomStep(
    data_key='data',
    threshold=0.8
)

# Add to pipeline
pipeline.add_step(step)
```

## Best Practices

1. **Use dataclasses** for clean parameter definition
2. **Type hint everything** for better IDE support
3. **Make keys configurable** so steps are reusable
4. **Return a dict** with all outputs
5. **Document your step** with a docstring

## Example: Feature Engineering

```python
@dataclass
class CreateInteractions(Step):
    """Create interaction features between variables."""

    data_key: str = 'data'
    output_key: str = 'engineered_data'
    interactions: List[tuple] = None

    def execute(self, artifacts, config):
        data = artifacts[self.data_key]

        new_features = []
        for col1, col2 in self.interactions or []:
            new_features.append(
                (pl.col(col1) * pl.col(col2)).alias(f'{col1}_x_{col2}')
            )

        data = data.with_columns(new_features)

        return {self.output_key: data}

# Use it
step = CreateInteractions(
    interactions=[('age', 'income'), ('x1', 'x2')]
)
pipeline.add_step(step)
```

See `examples/custom_steps/` for more examples.
