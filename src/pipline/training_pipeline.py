import sys

from src.exception import MyException
from src.logger import logging

from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

from src.entity.config_entity import (
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
    ModelTrainerConfig,
)

from src.entity.artifact_entity import (
    DataIngestionArtifact,
    DataValidationArtifact,
    DataTransformationArtifact,
    ModelTrainerArtifact
)


class TrainPipeline:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
        self.data_validation_config = DataValidationConfig()
        self.data_transformation_config = DataTransformationConfig()
        self.model_trainer_config = ModelTrainerConfig()

    # ---------------------------
    # 1. DATA INGESTION
    # ---------------------------
    def start_data_ingestion(self) -> DataIngestionArtifact:
        try:
            logging.info("🔵 Stage 1: Data Ingestion Started")

            data_ingestion = DataIngestion(
                data_ingestion_config=self.data_ingestion_config
            )

            artifact = data_ingestion.initiate_data_ingestion()

            if artifact is None:
                raise Exception("Data Ingestion failed: Artifact is None")

            logging.info("🟢 Data Ingestion Completed Successfully")
            return artifact

        except Exception as e:
            logging.error("🔴 Data Ingestion Failed")
            raise MyException(e, sys) from e

    # ---------------------------
    # 2. DATA VALIDATION
    # ---------------------------
    def start_data_validation(
        self,
        data_ingestion_artifact: DataIngestionArtifact
    ) -> DataValidationArtifact:
        try:
            logging.info("🔵 Stage 2: Data Validation Started")

            data_validation = DataValidation(
                data_ingestion_artifact=data_ingestion_artifact,
                data_validation_config=self.data_validation_config
            )

            artifact = data_validation.initiate_data_validation()

            if artifact is None:
                raise Exception("Data Validation failed: Artifact is None")

            if not artifact.validation_status:
                raise Exception(f"Validation Failed: {artifact.message}")

            logging.info("🟢 Data Validation Completed Successfully")
            return artifact

        except Exception as e:
            logging.error("🔴 Data Validation Failed")
            raise MyException(e, sys) from e

    # ---------------------------
    # 3. DATA TRANSFORMATION
    # ---------------------------
    def start_data_transformation(
        self,
        data_ingestion_artifact: DataIngestionArtifact,
        data_validation_artifact: DataValidationArtifact
    ) -> DataTransformationArtifact:
        try:
            logging.info("🔵 Stage 3: Data Transformation Started")

            data_transformation = DataTransformation(
                data_ingestion_artifact=data_ingestion_artifact,
                data_transformation_config=self.data_transformation_config,
                data_validation_artifact=data_validation_artifact
            )

            artifact = data_transformation.initiate_data_transformation()

            logging.info("🟢 Data Transformation Completed Successfully")
            return artifact

        except Exception as e:
            logging.error("🔴 Data Transformation Failed")
            raise MyException(e, sys) from e

    # ---------------------------
    # 4. MODEL TRAINER
    # ---------------------------
    def start_model_trainer(self, data_transformation_artifact: DataTransformationArtifact) -> ModelTrainerArtifact:
        """
        This method of TrainPipeline class is responsible for starting model training
        """
        try:
            logging.info("🔵 Stage 4: Model Training Started")
            model_trainer = ModelTrainer(
                data_transformation_artifact=data_transformation_artifact,
                model_trainer_config=self.model_trainer_config
            )
            model_trainer_artifact = model_trainer.initiate_model_trainer()
            logging.info("🟢 Model Training Completed Successfully")
            return model_trainer_artifact

        except Exception as e:
            logging.error("🔴 Model Training Failed")
            raise MyException(e, sys) from e

    # ---------------------------
    # 5. FULL PIPELINE
    # ---------------------------
    def run_pipeline(self) -> None:
        try:
            logging.info("🚀 Training Pipeline Started")

            ingestion_artifact = self.start_data_ingestion()

            validation_artifact = self.start_data_validation(
                data_ingestion_artifact=ingestion_artifact
            )

            transformation_artifact = self.start_data_transformation(
                data_ingestion_artifact=ingestion_artifact,
                data_validation_artifact=validation_artifact
            )
            
            
            model_trainer_artifact = self.start_model_trainer(
                data_transformation_artifact=transformation_artifact
            )
            
            logging.info("🎯 Pipeline Executed Successfully")

        except Exception as e:
            logging.error("❌ Pipeline Failed")
            raise MyException(e, sys) from e