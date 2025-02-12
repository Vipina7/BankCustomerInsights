import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
import numpy as np
import pickle
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
        logging.info(f"Saved the model successfully")

    except Exception as e:
        raise CustomException(e, sys)
    
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def kmeans_cluster(ran:list, data):
    try:
        kmeans_silhouette = {}
        for k in ran:
            logging.info('Initiating the KMeans model')
            kmeans = KMeans(n_clusters=k, random_state=42)
            
            kmeans_labels = kmeans.fit_predict(data)
            logging.info('Prediction of KMeans labels succesful')

            kmeans_silhouette[k] = silhouette_score(data, kmeans_labels)

        return (
            pd.DataFrame.from_dict(kmeans_silhouette, orient='index', columns=['silhouette_score']))
    
    except Exception as e:
        raise CustomException(e,sys)
    
def agglo_cluster(ran:list, data):
    try:
        agglo_silhouette = {}
        for k in ran:
            logging.info('Initiating the Agglomerative cluster model')
            agglo = AgglomerativeClustering(n_clusters=k, affinity='cosine', linkage='average')
            
            agglo_labels = agglo.fit_predict(data)
            logging.info('Prediction of Agglomerative cluster labels succesful')

            agglo_silhouette[k] = silhouette_score(data, agglo_labels)
    
        return (
        pd.DataFrame.from_dict(agglo_silhouette, orient='index', columns=['silhouette_score']))

    except Exception as e:
        raise CustomException(e,sys)
    
def db_cluster(eps_values, data):
    try:
        db_silhouette = {}
        
        for eps in eps_values:
            logging.info('Initiating the DBSCAN model')
            db = DBSCAN(eps = eps, min_samples=5)
            db_labels = db.fit_predict(data)
            logging.info('Prediction of DBSCAN labels succesful')

            if len(set(db_labels)) > 1:
                db_silhouette[eps] = silhouette_score(data, db_labels)

        return (
            pd.DataFrame.from_dict(db_silhouette, orient='index', columns=['silhouette_score']))
    
    except Exception as e:
        raise CustomException(e,sys)