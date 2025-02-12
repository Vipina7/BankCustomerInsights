import os
import sys
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from dataclasses import dataclass

from sklearn.cluster import KMeans

@dataclass
class ModelTrainerConfig:
    trained_model_path:str = os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def get_trained_model(self, scaled_data_train, scaled_data_test, train, test):
        try:
            logging.info('Model training initiated')
            model = KMeans(n_clusters=2, random_state=42)
            train_labels = model.fit_predict(scaled_data_train)
            test_labels = model.predict(scaled_data_test)

            save_object(
                file_path= self.model_trainer_config.trained_model_path,
                obj=model
            )

            logging.info('Adding cluster labels to the train and test sets')
            train['cluster'] = train_labels
            test['cluster'] = test_labels

            train.to_csv('artifacts/labeled_train.csv')
            test.to_csv('artifacts/labeled_test.csv')

            return "Model has been saved and cluster labels are added to the original dataframe."

        except Exception as e:
            raise CustomException(e,sys)