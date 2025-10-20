from dataclasses import dataclass
from causalpype.steps import Step
import polars as pl

@dataclass
class AutoPreprocess(Step):
    """Automatically preprocess data: handle nulls and ensure numeric types.

    Attributes:
        data_key: Key to retrieve input data from artifacts
        output_key: Key to store processed data in artifacts
        null_threshold: Drop columns with >threshold fraction nulls (0.5 = 50%)
        impute_numeric_strategy: Strategy for numeric imputation ('mean', 'median', 'zero')
        impute_categorical_strategy: Strategy for categorical imputation ('mode')
        encode_categorical: Whether to encode categorical variables as numeric
        drop_remaining_nuls: Drop rows with any remaining nulls after imputation
    """
    data_key: str = 'data'
    output_key: str = 'data'
    null_threshold: float = 0.5
    impute_numeric_strategy: str = 'mean'
    impute_categorical_strategy: str = 'mode'
    encode_categorical: bool = True
    drop_remaining_nuls: bool = True

    def execute(self, artifacts, config):
        data  = artifacts[self.data_key]
        print(f"    Input Shape: {data.shape}")
        print(f"    Initial Null Count: {data.null_count().sum_horizontal()[0]}")

        data = self._drop_high_null_columns(data)

        data = self._impute_columns(data)

        if self.encode_categorical:
            data = self._encode_categorical(data)
        
        if self.drop_remaining_nuls:
            data = self._drop_null_rows(data)
        
        data = self._ensure_numeric(data)

        print(f"    Output Shape: {data.shape}")
        print(f"    Final Null Count: {data.null_count().sum_horizontal()[0]}")
        print(f"    All Numeric: {all(dtype.is_numeric() for dtype in data.dtypes)}")

        return {self.output_key: data}
    
    def _drop_high_null_columns(self, data: pl.DataFrame) -> pl.DataFrame:
        null_percentages = data.null_count() / len(data)

        cols_to_keep = []
        cols_dropped = []

        for col in data.columns:
            null_pct = null_percentages[col][0]
            if null_pct > self.null_threshold:
                cols_dropped.append(col)
            else:
                cols_to_keep.append(col)
            
        
        if cols_dropped:
            print(f"    Dropped: {len(cols_dropped)} high-null columns: {cols_dropped}")
        
        return data.select(cols_to_keep)
    
    def _impute_columns(self, data: pl.DataFrame) -> pl.DataFrame:
        exprs = []

        for col in data.columns:
            dtype = data[col].dtype

            if dtype.is_numeric():
                if self.impute_numeric_strategy == 'mean':
                    exprs.append(pl.col(col).fill_null(pl.col(col).mean()))
                elif self.impute_numeric_strategy == 'median':
                    exprs.append(pl.col(col).fill_null(pl.col(col).median()))
                elif self.impute_numeric_strategy == 'zero':
                    exprs.append(pl.col(col).fill_null(0))
                else:
                    exprs.append(pl.col(col))
            
            else:
                if self.impute_categorical_strategy == 'mode':
                    mode_val = data[col].drop_nulls().mode().first()
                    exprs.append(pl.col(col).fill_null(mode_val))
                else:
                    exprs.append(pl.col(col))
        
        return data.with_columns(exprs)
    

    def _encode_categorical(self, data: pl.DataFrame) -> pl.DataFrame:
        """Label Encoding"""
        exprs = []

        for col in data.columns:
            dtype = data[col].dtype

            if dtype == pl.Utf8:
                try:
                    numeric_col = data[col].cast(pl.Float64, strict=False)

                    if numeric_col.drop_nulls().len() > 0:
                        print(f"    Converted string column '{col}' to numeric")
                        exprs.append(numeric_col.alias(col))
                        continue
                except:
                    pass


            if dtype == pl.Utf8 or dtype == pl.Categorical:
                unique_count = data[col].n_unique()
                total_count = len(data)

                if unique_count / total_count < 0.5:
                    print(f"    Encoding categorical column: '{col}'")
                    unique_vals = data[col].unique().drop_nulls().sort().to_list()

                    expr = pl.lit(None).cast(pl.Float64)
                    for idx, val in enumerate(unique_vals):
                        expr = pl.when(pl.col(col) == val).then(float(idx)).otherwise(expr)

                    exprs.append(expr.alias(col))

                else:
                    print(f"    Warning: Column '{col}' has {unique_count} unique_values ({unique_count/total_count:.1%}). Dropping.")
                    continue
            else:
                exprs.append(pl.col(col))
        
        return data.select(exprs)
    
    def _drop_null_rows(self, data: pl.DataFrame) -> pl.DataFrame:
        rows_before = len(data)
        data = data.drop_nulls()
        rows_after = len(data)

        if rows_before != rows_after:
            print(f"    Dropped {rows_before - rows_after} rows with nulls")
    
        return data
    
    def _ensure_numeric(self, data: pl.DataFrame) -> pl.DataFrame:
        exprs = []

        for col in data.columns:
            dtype = data[col].dtype

            if not dtype.is_numeric():
                try:
                    exprs.append(pl.col(col).cast(pl.Float64))
                    print(f"    Cast {col} to numeric")
                except:
                    print(f"    Warning: Could not convert {col} to numeric, keeping as-is")
                    exprs.append(pl.col(col))
            else:
                exprs.append(pl.col(col))
        
        return data.select(exprs)