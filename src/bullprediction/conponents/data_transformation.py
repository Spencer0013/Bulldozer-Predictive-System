import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
import logging
from bullprediction.utils.common import save_object
from bullprediction.entity import DataTransformationConfig


class DataTransformation:
    def __init__(self,config: DataTransformationConfig):
        self.config = config

    def build_preprocessor(self, df: pd.DataFrame):
        df = df.drop(columns=["SalePrice"], errors="ignore")

    # Identify numerical and categorical columns
        num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    # Define transformers
        num_transformer = SimpleImputer(strategy="median")
        cat_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
         ])

    # Build column transformer
        preprocessor = ColumnTransformer([
        ("num", num_transformer, num_cols),
        ("cat", cat_transformer, cat_cols)
    ])

        return preprocessor

    def initiate_data_transformation_and_split(self):
        train_df = pd.read_csv(self.config.train_path, low_memory=False)
        test_df = pd.read_csv(self.config.test_path,low_memory=False)

        valid_data = train_df[train_df.saleYear ==2012]
        train_data = train_df[train_df.saleYear!=2012]

        target_column_name = 'SalePrice'

        input_feature_train_data = train_data.drop(columns=[target_column_name])
        input_feature_valid_data = valid_data.drop(columns=[target_column_name])

        target_feature_train_data = train_data[target_column_name]
        target_feature_valid_data = valid_data[target_column_name]


        input_feature_test_df = test_df

        #preprocessing_obj = self.build_preprocessor(input_feature_train_data)

        logging.info("Applying preprocessing pipeline to train and test data.")

        preprocessing_obj = self.build_preprocessor(input_feature_train_data)
        input_feature_train_processed = preprocessing_obj.fit_transform(input_feature_train_data)
        input_feature_valid_processed = preprocessing_obj.transform(input_feature_valid_data)
    
        input_feature_test_processed = preprocessing_obj.transform(input_feature_test_df)

        save_object(
                file_path=self.config.preprocessor,
                obj=preprocessing_obj
            )
    
        return (input_feature_train_processed, input_feature_valid_processed, 
                input_feature_test_processed, target_feature_train_data,
                target_feature_valid_data, self.config.preprocessor)    

    