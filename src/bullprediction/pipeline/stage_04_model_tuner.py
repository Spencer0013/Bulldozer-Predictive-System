from bullprediction.config.configuration import ConfigurationManager
from bullprediction.conponents.model_tuner import ModelTuner
from bullprediction.conponents.data_transformation import DataTransformation


class ModelTunerPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_transformation_config = config.get_data_transformation()
        data_transformer = DataTransformation(config=data_transformation_config)
        model_tuner_config = config.get_model_tuner()
        model_tuner= ModelTuner(config=model_tuner_config, data_transformer=data_transformer)
        model_tuner.tune()