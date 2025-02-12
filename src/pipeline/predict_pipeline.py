import sys
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            logging.info("Loading the model and preprocessor object")
            model_path = 'artifacts/model.pkl'
            preprocessor_path = 'artifacts/preprocessor.pkl'
            pca_path = 'artifacts/pca.pkl'

            significant_features = pd.read_csv('notebook\significant_features.csv')
            significant_features = significant_features['features']
            logging.info('Obtaining the significant features')

            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)

            logging.info("Preprocessing the input features")
            cat_cols = ['job','marital','education','default','housing','loan','contact','poutcome']
            num_cols = ['balance', 'duration', 'campaign', 'previous']

            scaled_features = preprocessor.transform(features)
            column_names = (num_cols + list(preprocessor.named_transformers_['cat'].named_steps['encoder'].get_feature_names_out(cat_cols))+['month'])
            scaled_df = pd.DataFrame(scaled_features, columns=column_names)
            scaled_df = scaled_df[significant_features]


            logging.info("Dimentionality reduction induced")
            pca = load_object(file_path=pca_path)
            reduced_features = pca.transform(scaled_df)
            
            logging.info("Making predictions successful")
            prediction = model.predict(reduced_features)
            return prediction
        
        except Exception as e:
            raise CustomException(e,sys)

class CustomData:
    def __init__(self, 
                 balance, 
                 duration, 
                 campaign, 
                 previous, 
                 job, 
                 marital, 
                 education, 
                 default, 
                 housing, 
                 loan, 
                 contact, 
                 poutcome, 
                 month):
        
        self.balance = balance
        self.duration = duration
        self.campaign = campaign
        self.previous = previous
        self.job = job
        self.marital = marital
        self.education = education
        self.default = default
        self.housing = housing
        self.loan = loan
        self.contact = contact
        self.poutcome = poutcome
        self.month = month

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "balance": [self.balance],
                "duration": [self.duration],
                "campaign": [self.campaign],
                "previous": [self.previous],
                "job": [self.job],
                "marital": [self.marital],
                "education": [self.education],
                "default": [self.default],
                "housing": [self.housing],
                "loan": [self.loan],
                "contact": [self.contact],
                "poutcome": [self.poutcome],
                "month": [self.month]
            }
            logging.info("Dataframe ready for prediction")

            return pd.DataFrame(custom_data_input_dict)
        
        except Exception as e:
            logging.error(f"Error occurred: {e}")
            raise Exception(e)
