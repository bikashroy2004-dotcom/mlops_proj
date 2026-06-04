import sys
import numpy as np
import pandas as pd

from imblearn.combine import SMOTEENN
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

from src.constants import TARGET_COLUMN, SCHEMA_FILE_PATH
from src.entity.config_entity import DataTransformationConfig
from src.entity.artifact_entity import (
    DataTransformationArtifact,
    DataIngestionArtifact,
    DataValidationArtifact
)
from src.exception import MyException
from src.logger import logging
from src.utils.main_utils import save_object, save_numpy_array_data, read_yaml_file


class DataTransformation:
    def __init__(
        self,
        data_ingestion_artifact: DataIngestionArtifact,
        data_transformation_config: DataTransformationConfig,
        data_validation_artifact: DataValidationArtifact
    ):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_transformation_config = data_transformation_config
            self.data_validation_artifact = data_validation_artifact
            self._schema_config = read_yaml_file(file_path=SCHEMA_FILE_PATH)

        except Exception as e:
            raise MyException(e, sys)

    # ---------------------------
    # READ DATA
    # ---------------------------
    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise MyException(e, sys)

    # ---------------------------
    # PREPROCESSOR
    # ---------------------------
    def get_data_transformer_object(self) -> ColumnTransformer:
        logging.info("Creating preprocessor object")

        try:
            numerical_features = self._schema_config["numerical_columns"]
            categorical_features = self._schema_config["categorical_columns"]

            numeric_pipeline = Pipeline(steps=[
                ("scaler", StandardScaler())
            ])

            # Fixed: Set sparse_output=False to prevent layout mismatch errors down the road
            categorical_pipeline = Pipeline(steps=[
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
            ])

            preprocessor = ColumnTransformer([
                ("num", numeric_pipeline, numerical_features),
                ("cat", categorical_pipeline, categorical_features)
            ])

            return preprocessor

        except Exception as e:
            raise MyException(e, sys)

    # ---------------------------
    # GENDER MAPPING
    # ---------------------------
    def _map_gender_column(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info("Mapping gender column")

        if "person_gender" in df.columns:
            df["person_gender"] = (
                df["person_gender"]
                .astype(str)
                .str.strip()
                .str.lower()
                .map({"female": 0, "male": 1})
            )

            df["person_gender"] = df["person_gender"].fillna(-1).astype(int)

        return df

    # ---------------------------
    # BOOL CONVERSION
    # ---------------------------
    def _convert_bool_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info("Converting boolean columns")

        for col in df.columns:
            if df[col].dtype == "bool":
                df[col] = df[col].astype(int)

        return df

    # ---------------------------
    # DROP COLUMNS
    # ---------------------------
    def _drop_id_column(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info("Dropping unwanted columns")

        drop_cols = self._schema_config.get("drop_columns", [])

        if isinstance(drop_cols, str):
            drop_cols = [drop_cols]

        for col in drop_cols:
            if col in df.columns:
                df = df.drop(columns=col)

        return df

    # ---------------------------
    # MAIN TRANSFORMATION
    # ---------------------------
    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:
            logging.info("Data Transformation Started")

            if not self.data_validation_artifact.validation_status:
                raise Exception(self.data_validation_artifact.message)

            train_df = self.read_data(self.data_ingestion_artifact.trained_file_path)
            test_df = self.read_data(self.data_ingestion_artifact.test_file_path)

            logging.info(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")

            input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN])
            target_feature_train_df = train_df[TARGET_COLUMN]

            input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN])
            target_feature_test_df = test_df[TARGET_COLUMN]

            # ---------------------------
            # TRANSFORMATIONS
            # ---------------------------
            input_feature_train_df = self._map_gender_column(input_feature_train_df)
            input_feature_train_df = self._convert_bool_columns(input_feature_train_df)
            input_feature_train_df = self._drop_id_column(input_feature_train_df)

            input_feature_test_df = self._map_gender_column(input_feature_test_df)
            input_feature_test_df = self._convert_bool_columns(input_feature_test_df)
            input_feature_test_df = self._drop_id_column(input_feature_test_df)

            # SAFETY CHECK
            if input_feature_train_df is None or input_feature_train_df.empty:
                raise Exception("Train input features are empty after preprocessing")

            if input_feature_test_df is None or input_feature_test_df.empty:
                raise Exception("Test input features are empty after preprocessing")

            logging.info("Custom preprocessing completed")

            # ---------------------------
            # TRANSFORMER
            # ---------------------------
            preprocessor = self.get_data_transformer_object()

            input_feature_train_arr = preprocessor.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor.transform(input_feature_test_df)

            # ---------------------------
            # SMOTEENN
            # ---------------------------
            logging.info("Applying SMOTEENN")

            smt = SMOTEENN(sampling_strategy="minority")

            input_feature_train_final, target_feature_train_final = smt.fit_resample(
                input_feature_train_arr,
                target_feature_train_df
            )

            # ---------------------------
            # FINAL ARRAYS
            # ---------------------------
            train_arr = np.c_[
                input_feature_train_final,
                np.array(target_feature_train_final)
            ]

            test_arr = np.c_[
                input_feature_test_arr,
                np.array(target_feature_test_df)
            ]

            # ---------------------------
            # SAVE
            # ---------------------------
            save_object(
                self.data_transformation_config.transformed_object_file_path,
                preprocessor
            )

            save_numpy_array_data(
                self.data_transformation_config.transformed_train_file_path,
                array=train_arr
            )

            save_numpy_array_data(
                self.data_transformation_config.transformed_test_file_path,
                array=test_arr
            )

            logging.info("Saved transformer + arrays successfully")

            return DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
            )

        except Exception as e:
            raise MyException(e, sys) from e