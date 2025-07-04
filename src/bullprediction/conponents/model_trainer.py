from bullprediction.entity import DataTransformationConfig
from bullprediction.conponents.data_transformation import DataTransformation
import os
import joblib
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    AdaBoostRegressor,
    GradientBoostingRegressor
)
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_log_error
import numpy as np
from bullprediction.utils.common import save_object
from bullprediction.entity import ModelTrainerConfig

class ModelTrainer:
    def __init__(self,config: ModelTrainerConfig, data_transformer:DataTransformation):
        self.config = config
        self.data_transformer = data_transformer

    def train(self):
        (
                input_feature_train_processed,
                input_feature_valid_processed,
                input_feature_test_processed,
                target_feature_train_data,
                target_feature_valid_data,
                preprocessor_path
        ) = self.data_transformer.initiate_data_transformation_and_split()
        

        models = {
        "Linear Regression": LinearRegression(),
        "Lasso": Lasso(random_state=42),
        "Ridge": Ridge(random_state=42),
        "K-Neighbors Regressor": KNeighborsRegressor(),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "Random Forest Regressor": RandomForestRegressor(random_state=42),
        "XGBRegressor": XGBRegressor(random_state=42), 
        "CatBoosting Regressor": CatBoostRegressor(verbose=False,random_state=42),
        "AdaBoost Regressor": AdaBoostRegressor(random_state=42),
        "Gradient Boosting Regressor" : GradientBoostingRegressor(random_state=42),
         }
    
        best_model = None
        best_model_name = ""
        best_rmsle = float("inf")
        scores = {}

        for name, model in models.items():
            model.fit(input_feature_train_processed, target_feature_train_data)
            y_pred = model.predict(input_feature_valid_processed)

        # Clip predictions and true values to avoid negative values or zeros (log issue)
            y_pred_clip = np.clip(y_pred, a_min=0, a_max=None)
            y_val_clip = np.clip(target_feature_valid_data, a_min=0, a_max=None)

            rmsle = np.sqrt(mean_squared_log_error(y_val_clip, y_pred_clip))
            scores[name] = rmsle

            if rmsle < best_rmsle:
                best_rmsle = rmsle
                best_model = model
                best_model_name = name

            print(f"[ModelTrainer] Best Model: {best_model_name} | RMSLE: {best_rmsle:.6f}")

        if self.config.model_save_path:
                save_object(self.config.model_save_path, best_model)
                print(f"Best model saved to: {self.config.model_save_path}")
        
        return {
            "best_model": best_model,
            "best_model_name": best_model_name,
            "best_rmsle": best_rmsle,
            "all_rmsle_scores": scores,
            "X_train": input_feature_train_processed,
            "y_train": target_feature_train_data,
            "X_val": input_feature_valid_processed,
            "y_val": target_feature_valid_data,
            "X_test": input_feature_test_processed,
            "preprocessor_path": preprocessor_path
           }