import sys
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.exception import MyException
from src.logger import logging


class TargetValueMapping:
    def __init__(self):
        self.yes = 0
        self.no = 1

    def _asdict(self):
        return self.__dict__

    def reverse_mapping(self):
        mapping_response = self._asdict()
        return dict(zip(mapping_response.values(), mapping_response.keys()))


class MyModel:
    def __init__(self, preprocessing_object: Pipeline, trained_model_object: object):
        self.preprocessing_object = preprocessing_object
        self.trained_model_object = trained_model_object

    def predict(self, dataframe: pd.DataFrame):
        """
        Apply preprocessing + model prediction
        """
        try:
            logging.info("Starting prediction process")

            if dataframe is None or len(dataframe) == 0:
                raise Exception("Empty input data received")

            # Step 1: preprocessing
            transformed_features = self.preprocessing_object.transform(dataframe)

            # Step 2: prediction
            logging.info("Running model prediction")
            predictions = self.trained_model_object.predict(transformed_features)

            return predictions

        except Exception as e:
            logging.error("Error in prediction pipeline", exc_info=True)
            raise MyException(e, sys) from e

    def __repr__(self):
        return f"{type(self.trained_model_object).__name__}()"

    def __str__(self):
        return f"{type(self.trained_model_object).__name__}()"