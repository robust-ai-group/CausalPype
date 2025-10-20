import networkx as nx
import polars as pl
import numpy as np
from typing import Union, Dict, List
from dowhy import gcm
import pickle
import os
from dowhy.gcm.model_evaluation import CausalModelEvaluationResult



def do(x):
    "Constant intervention"
    return lambda a: x



# TODO: Fix type mismatches
class CausalModel:
    """A causal model for structural causal modeling."""

    def __init__(self, graph: nx.DiGraph, outcome: str):
        """Initialize causal model with graph and outcome variable."""
        self.graph = graph
        self.outcome = outcome
        self.model = gcm.InvertibleStructuralCausalModel(self.graph)

        self._fitted = False
        self.causal_mechanisms = None


    def assign_mechanisms(self, dataset: pl.DataFrame,
                          quality: Union[str, gcm.auto.AssignmentQuality] = "BETTER"):  # Added linear_scm option:
        
        quality = getattr(gcm.auto.AssignmentQuality, quality.upper())
        causal_mechanisms = gcm.auto.assign_causal_mechanisms(
            causal_model=self.model,
            based_on=dataset,
            quality=quality,
            override_models=True
        )

        self.causal_mechanisms = causal_mechanisms

        return causal_mechanisms

    def fit(self, data: pl.DataFrame):
        gcm.fit(causal_model=self.model, data=data)
        self._fitted = True
        return self

    def predict(self, X: pl.DataFrame) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Model must be fitted before prediction")

        target_node = self.model.graph.nodes[self.outcome]
        parent_cols = target_node['parents_during_fit']

        # Find which parent columns are available in the input data
        available_parents = [col for col in parent_cols if col in X.columns]

        if not available_parents:
            raise ValueError("None of the required parent columns are available in input data. "
                             f"Required columns: {parent_cols}")

        # Create a copy of input data
        X_copy = X.clone()

        # Log warning about missing columns
        missing = [col for col in parent_cols if col not in X.columns]
        if missing:
            print(f"Warning: Missing columns in input data: {missing}. "
                  f"Proceeding with available parent columns only: {available_parents}.")
            samples = self.draw_samples()
            for col in missing:
                X_copy[col] = samples[col].mean()

        X_copy = X_copy[parent_cols].to_numpy()

        # Use only available parent columns for prediction
        return target_node['causal_mechanism'].prediction_model.predict(X_copy).squeeze()

        # if not all(col in X.columns for col in parent_cols):
        #     missing = [col for col in parent_cols if col not in X.columns]
        #     raise ValueError(f"Missing columns in input data: {missing}")

        # return target_node['causal_mechanism'].prediction_model.predict(X[parent_cols]).squeeze()

    def draw_samples(self, num_samples: int = 1000) -> pl.DataFrame:
        if not self._fitted:
            raise RuntimeError("Model must be fitted before drawing samples")

        return gcm.draw_samples(causal_model=self.model, num_samples=num_samples)

    def evaluate(self, data: pl.DataFrame) -> CausalModelEvaluationResult:
        if not self._fitted:
            raise RuntimeError("Model must be fitted before evaluation")

        return gcm.evaluate_causal_model(
            causal_model=self.model,
            data=data.to_pandas(),
            evaluate_causal_mechanisms=True,
            compare_mechanism_baselines=True,
            evaluate_invertibility_assumptions=True,
            evaluate_overall_kl_divergence=True,
            evaluate_causal_structure=True
        )

    def features(self) -> List[str]:
        return list(self.model.graph.nodes)

    def save(self, filename: str):
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        with open(filename, 'wb') as f:
            # maybe use joblib.dump(self, filename) ??
            pickle.dump(self, f)
        print(f"Model saved to {filename}")

    # @classmethod
    # def load_from_file(cls, filename: str) -> 'CausalModel':
    #     """
    #     Load a causal model from a file

    #     Args:
    #         filename: Path to the model file

    #     Returns:
    #         CausalModel: Loaded model
    #     """
    #     with open(filename, 'rb') as f:
    #         model = pickle.load(f)
    #     print(f"Model loaded from {filename}")
    #     return model

    @classmethod
    def load_from_file(cls, filename: str) -> 'CausalSCM':
        try:
            # Load using joblib instead of pickle
            with open(filename, 'rb') as f:
                model = pickle.load(f)
            # model = joblib.load(filename)
            print(f"Model loaded from {filename}")
            return model
        except Exception as e:
            print(f"Error loading model from {filename}: {str(e)}")
            print("This might be due to incompatible scikit-learn versions.")
            print("Please ensure the model was saved with the same scikit-learn version.")
            raise

    
    def intervene(self, interventions: Dict[str, float], num_samples: int=1000) -> pl.DataFrame:
        """
        Simulate interventions on the causal model.
        An intervention forcibly sets variables to specific values, breaking
        their natural causal mechanisms.
        """
        if not self._fitted:
            raise RuntimeError("Model is not trained. Train first and then try to intervene.")
        
        invalid_vars = [var for var in interventions.keys() if var not in self.model.graph.nodes]
        if invalid_vars:
            raise ValueError(
                f"Intervention variables {invalid_vars} not found in causal graph."
            )


        intervention_functions = {var: do(value) for var, value in interventions.items()}

        inter = gcm.interventional_samples(
            self.model,
            intervention_functions,
            num_samples_to_draw=num_samples
        )

        return pl.DataFrame(inter)