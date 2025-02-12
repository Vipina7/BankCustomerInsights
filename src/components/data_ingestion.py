import os
import sys
from src.exception import CustomException
from src.logger import logging
from src.components.data_transformation import DataTransformation, DataTransformationConfig
from src.components.dimensionality_reduction import DimensionReductionConfig, DimensionReduction
from src.components.cluster import ClusterRetreiver
from src.components.model_trainer import ModelTrainerConfig, ModelTrainer
from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

@dataclass
class DataIngestionConfig:
    train_data_path:str = os.path.join('artifacts', 'train.csv')
    test_data_path:str = os.path.join('artifacts', 'test.csv')

class DataIngestion:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()

    def intiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            df = pd.read_csv('notebook/data/bank_marketing.csv')
            logging.info('Read the dataset as dataframe')

            os.makedirs(os.path.dirname(self.data_ingestion_config.train_data_path), exist_ok=True)
            logging.info('Artifacts folder created')

            df = df.drop(columns=['age','day', 'pdays', 'deposit'], axis=1)
            logging.info('Dropping the necessary columns')

            logging.info("Train test split initiated")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            train_set.to_csv(self.data_ingestion_config.train_data_path, index = False, header = True)
            test_set.to_csv(self.data_ingestion_config.test_data_path, index = False, header = True)
            logging.info('Ingestion of data is completed')

            return(
                self.data_ingestion_config.train_data_path,
                self.data_ingestion_config.test_data_path
            )

        except Exception as e:
            raise CustomException(e,sys)

if __name__ == "__main__":
    data_obj = DataIngestion()
    train_path, test_path = data_obj.intiate_data_ingestion()

    transform_obj = DataTransformation()
    train, test = transform_obj.initiate_data_transformation(train_path=train_path, test_path=test_path)

    pca_obj = DimensionReduction()
    pca_train_data, pca_test_data = pca_obj.get_pca_model(train_data=train, test_data=test)

    cluster_obj = ClusterRetreiver()
    res = cluster_obj.get_best_cluster(pca_data_train=pca_train_data, pca_data_test=pca_test_data)
    
    model_obj = ModelTrainer()
    model_obj.get_trained_model(scaled_data_train=pca_train_data,
                                scaled_data_test=pca_test_data,
                                train=train,
                                test=test)
    
