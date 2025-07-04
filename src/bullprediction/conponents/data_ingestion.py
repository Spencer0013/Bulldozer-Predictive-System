import pandas as pd
from bullprediction.entity import DataIngestionConfig


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config =config

    

    def read_data(self):
        self.df_train = pd.read_csv(self.config.source_train_path,low_memory=False,parse_dates=['saledate'])
        self.df_test = pd.read_csv(self.config.source_test_path,low_memory=False,parse_dates=['saledate'])

    def add_date_features(self, df:pd.DataFrame) -> pd.DataFrame:
        df = df.sort_values(by=['saledate'],ascending=True).copy()
        df["saleYear"] = df.saledate.dt.year
        df["saleMonth"] = df.saledate.dt.month
        df["saleDay"] = df.saledate.dt.day
        df["saleDayOfWeek"] = df.saledate.dt.dayofweek
        df["saleDayOfYear"] = df.saledate.dt.dayofyear
        df = df.drop("saledate", axis=1)
        return df
    
    def process_and_save(self):
        self.df_train = self.add_date_features(self.df_train)
        self.df_test = self.add_date_features(self.df_test)

        self.df_train.to_csv(self.config.train_path,index=False)
        self.df_test.to_csv(self.config.test_path, index=False)

    
    

