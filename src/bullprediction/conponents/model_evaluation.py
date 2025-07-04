from pathlib import Path
from bullprediction.utils.common import save_json  
import numpy as np
from sklearn.metrics import mean_squared_log_error
from bullprediction.conponents.data_transformation import DataTransformation
from bullprediction.entity import DataTransformationConfig, ModelEvaluationConfig
import joblib  # for loading the model
import os
import json



class ModelEvaluator:
    def __init__(self, config:ModelEvaluationConfig, data_transformer:DataTransformation):
        self.config = config
        self.data_transformer = data_transformer
        self.save_path = config.save_path
        self.model = self._load_model(config.best_model_path)

    def _load_model(self, model_path):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        return joblib.load(model_path)

    def _rmsle(self, y_true, y_pred):
        """Compute Root Mean Squared Log Error (RMSLE)."""
        y_true = np.clip(y_true, a_min=0, a_max=None)
        y_pred = np.clip(y_pred, a_min=0, a_max=None)
        return np.sqrt(mean_squared_log_error(y_true, y_pred))
    
    def evaluate(self):
        (
            X_train,
            X_val,
            X_test,
            y_train,
            y_val,
            preprocessor_path
        ) = self.data_transformer.initiate_data_transformation_and_split()

        y_val_pred = self.model.predict(X_val)
        val_rmsle = self._rmsle(y_val, y_val_pred)

        results = {"validation_rmsle": val_rmsle}
        print(f"[ModelEvaluator] Validation RMSLE: {val_rmsle:.4f}")

        if self.save_path:
            save_json(Path(self.save_path), results)
            print(f"[ModelEvaluator] Results saved to {self.save_path}")

        return results