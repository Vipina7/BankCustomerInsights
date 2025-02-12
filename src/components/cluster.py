import sys
import os
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, kmeans_cluster, agglo_cluster, db_cluster
from dataclasses import dataclass

import numpy as np

@dataclass
class ClusterRetreiver:
    def __init__(self):
        pass

    def get_best_cluster(self, pca_data_train, pca_data_test):
        try:
            k_range = range(2, 11)
            eps_values = np.arange(0.1, 2.0, 0.2)

            kmeans_scores = kmeans_cluster(k_range, pca_data_train)
            agglo_scores = agglo_cluster(k_range, pca_data_train)
            db_scores = db_cluster(eps_values, pca_data_train)

            kmeans_scores.to_csv('artifacts/kmeans_scores.csv', index=True)
            agglo_scores.to_csv('artifacts/agglo_scores.csv', index=True)
            db_scores.to_csv('artifacts/db_scores.csv', index=True)
            logging.info('Model silhouette scores saved successfully')

            return 'Silhouette scores secured'
        
        except Exception as e:
            raise CustomException(e, sys)