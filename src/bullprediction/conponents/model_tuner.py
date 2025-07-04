
from bullprediction.utils.common import save_object
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import mean_squared_log_error, make_scorer
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from bullprediction.entity import DataTransformationConfig, ModelTunerConfig
from bullprediction.conponents.data_transformation import DataTransformation


class ModelTuner:
    def __init__(self, config: ModelTunerConfig, data_transformer: DataTransformation):
        self.config = config
        self.data_transformer = data_transformer

    def _rmsle(self, y_true, y_pred):
        """Compute RMSLE after clipping to avoid log(0)."""
        y_true = np.clip(y_true, a_min=0, a_max=None)
        y_pred = np.clip(y_pred, a_min=0, a_max=None)
        return np.sqrt(mean_squared_log_error(y_true, y_pred))

    def tune(self):
        (
            X_train,
            X_val,
            X_test,
            y_train,
            y_val,
            preprocessor_path
        ) = self.data_transformer.initiate_data_transformation_and_split()

        print("[ModelTuner] Starting tuning for RandomForest")

        param_dist = self.config.param_dist.get("RandomForest", None)

        if not param_dist:
            raise ValueError("[ModelTuner] No param dist found for RandomForest in config.")

        rf_model = RandomForestRegressor(random_state=42, n_jobs=-1)

        scoring = make_scorer(self._rmsle, greater_is_better=False)
        tscv = TimeSeriesSplit(n_splits=self.config.cv_folds)

        random_search = RandomizedSearchCV(
            estimator=rf_model,
            param_distributions=param_dist,
            scoring=scoring,
            n_iter=10,
            cv=tscv,
            n_jobs=-1,
            verbose=2,
            random_state=42
        )

        random_search.fit(X_train, y_train)

        best_model = random_search.best_estimator_
        best_params = random_search.best_params_

        print(f"[ModelTuner] Best parameters: {best_params}")

        if self.config.tuner_save_path:
            save_object(self.config.tuner_save_path, best_model)
            print(f"[ModelTuner] Tuned Random Forest model saved to: {self.config.tuner_save_path}")


        return best_model, best_params





