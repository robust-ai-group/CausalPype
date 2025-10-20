from dataclasses import dataclass
from causalpype.steps.base import Step
from typing import Optional, List
import networkx as nx
import polars as pl


@dataclass
class LoadData(Step):
    """Load data from CSV files with automatic type inference.

    Loads a CSV file into a Polars DataFrame, automatically infers data types,
    handles various null value representations, and attempts to convert string
    columns to numeric types when appropriate.

    Attributes:
        path: Path to the CSV file to load
        null_values: List of strings to treat as null values. Default includes
            common representations: '', ' ', 'NA', 'N/A', 'null', 'NULL', 'None', '?', '#DIV/0!'
        output_key: Key to store the loaded data in artifacts dict (default: 'data')

    Example:
        >>> from causalpype.steps import LoadData
        >>> step = LoadData(path='data.csv')
        >>> pipeline.add_step(step)
    """
    path: str
    null_values: Optional[List[str]] = None
    output_key: str = 'data'

    def execute(self, artifacts, config):
        null_values = self.null_values or ['', ' ', 'NA', 'N/A', 'null', 'NULL', 'None', '?', '#DIV/0!']
        
        data = pl.read_csv(self.path, infer_schema_length=10000, ignore_errors=True, null_values=self.null_values)

        # Convert empty strings to NaN for string columns
        data = data.with_columns(
            pl.when(pl.col(pl.Utf8).str.strip_chars() == "")
            .then(None)
            .otherwise(pl.col(pl.Utf8))
            .name.keep()
        )

        data = self._auto_convert_types(data)
        return {self.output_key: data}
    
    
    def _auto_convert_types(self, data: pl.DataFrame) -> pl.DataFrame:
          """Try to convert string columns to appropriate numeric types."""
          exprs = []

          for col in data.columns:
              if data[col].dtype == pl.Utf8:
                  # Try float conversion
                  try:
                      converted = data[col].cast(pl.Float64, strict=False)
                      non_null_ratio = converted.drop_nulls().len() / len(data)

                      if non_null_ratio > 0.8:  # If >80% successfully converted
                          print(f"    Converted '{col}': Utf8 -> Float64")
                          exprs.append(converted.alias(col))
                      else:
                          exprs.append(pl.col(col))
                  except:
                      exprs.append(pl.col(col))
              else:
                  exprs.append(pl.col(col))

          return data.select(exprs)