import os
import sys
from src.logger import logging
from src.exception import CustomException
from src.utils import save_object
from dataclasses import dataclass

from sklearn.decomposition import PCA

@dataclass
class DimensionReductionConfig:
    pca_model_path:str = os.path.join('artifacts', 'pca.pkl')

class DimensionReduction:
    def __init__(self):
        self.dimension_reduction_config = DimensionReductionConfig()

    def get_pca_model(self, train_data, test_data):
        try:
            logging.info('Initiating dimensionality reduction model')
            pca = PCA(n_components=2, random_state=42)
            pca_train_data = pca.fit_transform(train_data)
            pca_test_data = pca.transform(test_data)
            logging.info('Dimension reduction complete')

            save_object(
                file_path= self.dimension_reduction_config.pca_model_path,
                obj = pca
            )
            logging.info('Dimension Reduction model saved successfully')
            
            return (
                pca_train_data,
                pca_test_data
                )
        
        except Exception as e:
            raise CustomException(e,sys)