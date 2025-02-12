import os
import sys
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from dataclasses import dataclass

import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

class DataTransformationConfig:
    transformation_obj_path:str = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_preprocessor_object(self):
        try:
            cat_cols = ['job','marital','education','default','housing','loan','contact','poutcome','deposit']
            num_cols = ['balance', 'duration', 'campaign', 'previous']

            month_order = {'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 
               'jun': 6, 'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 
               'nov': 11, 'dec': 12}
            
            logging.info("Data preprocessing initiated")
            categorical_transformer = Pipeline(steps = [
                  ('encoder', OneHotEncoder())
                  ])

            numerical_transformer = Pipeline(
                  steps=[
                        ('scaler', StandardScaler())
                        ]
                    )
            ordinal_transformer = Pipeline(
                  steps=[
                        ('ordinal', OrdinalEncoder(categories=[list(month_order.keys())]))
                        ]
                    )
            
            preprocessor = ColumnTransformer([
                ('num', numerical_transformer, num_cols),
                ('cat', categorical_transformer, cat_cols),
                ('ord', ordinal_transformer, ['month'])
                ])
            logging.info('Preprocessor object obtained')

            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        try:
            logging.info("Importing train and test sets")
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('Obtaining preprocessor object')
            preprocessor_obj = self.get_preprocessor_object()

            logging.info('Apply preprocessor on the train and test sets')
            train = preprocessor_obj.fit_transform(train_df)
            test = preprocessor_obj.transform(test_df)
            logging.info("Data transformation complete")

            save_object(
                file_path=self.data_transformation_config.transformation_obj_path,
                obj=preprocessor_obj)
            
            return (
                train,
                test
            )
        
        except Exception as e:
            raise CustomException(e,sys)