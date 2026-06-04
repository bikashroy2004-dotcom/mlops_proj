import json
import sys
import os
import pandas as pd

from pandas import DataFrame

from src.exception import MyException
from src.logger import logging
from src.utils.main_utils import read_yaml_file
from src.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from src.entity.config_entity import DataValidationConfig
from src.constants import SCHEMA_FILE_PATH


class DataValidation:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact,
                 data_validation_config: DataValidationConfig):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            self.schema = read_yaml_file(file_path=SCHEMA_FILE_PATH)
        except Exception as e:
            raise MyException(e, sys)

    # --------------------------
    # READ DATA
    # --------------------------
    @staticmethod
    def read_data(file_path) -> DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise MyException(e, sys)

    # --------------------------
    # CHECK REQUIRED COLUMNS
    # --------------------------
    def validate_required_columns(self, df: DataFrame, dataset_name: str) -> bool:
        try:
            required_cols = list(self.schema["columns"].keys())

            missing_cols = [col for col in required_cols if col not in df.columns]

            if missing_cols:
                logging.error(f"{dataset_name} missing columns: {missing_cols}")
                return False

            logging.info(f"All required columns present in {dataset_name}")
            return True

        except Exception as e:
            raise MyException(e, sys)

    # --------------------------
    # MAIN PIPELINE
    # --------------------------
    def initiate_data_validation(self) -> DataValidationArtifact:
        try:
            logging.info("Starting data validation")

            train_df = self.read_data(self.data_ingestion_artifact.trained_file_path)
            test_df = self.read_data(self.data_ingestion_artifact.test_file_path)

            error_msg = []

            # Validate train
            if not self.validate_required_columns(train_df, "Training Data"):
                error_msg.append("Training data missing required columns")

            # Validate test
            if not self.validate_required_columns(test_df, "Test Data"):
                error_msg.append("Test data missing required columns")

            validation_status = len(error_msg) == 0

            report = {
                "validation_status": validation_status,
                "message": " | ".join(error_msg)
            }

            # save report
            os.makedirs(os.path.dirname(self.data_validation_config.validation_report_file_path),
                        exist_ok=True)

            with open(self.data_validation_config.validation_report_file_path, "w") as f:
                json.dump(report, f, indent=4)

            artifact = DataValidationArtifact(
                validation_status=validation_status,
                message=" | ".join(error_msg),
                validation_report_file_path=self.data_validation_config.validation_report_file_path
            )

            logging.info(f"Data Validation Artifact: {artifact}")
            return artifact

        except Exception as e:
            raise MyException(e, sys) from e