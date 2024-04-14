import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
import sklearn
# sklearn.set_config(enable_metadata_routing=True)

@dataclass
class DataTransformationconfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'proprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationconfig()
    def get_data_transformer_object(self):
        '''
        This function can be utilized for different kind of data transformation
        '''

        try:
            numerical_columns = ['writing_score','reading_score']
            categorical_columns = [
                'gender',
                'race_ethnicity',
                'parental_level_of_education',
                'lunch',
                'test_preparation_course'
            ]

            num_pipeline=Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler(with_mean=False))
                ]
            )

            cat_pipeline = Pipeline(
                steps =[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('onehot_encoder', OneHotEncoder()), # we can set sparse=False or
                    ('scaler', StandardScaler(with_mean=False)) # we can set with_mean=False
                ]
            )

            logging.info('Numerical columns standard scaling completed')
            logging.info('Numerical columns encoding completed')

            preprocessor = ColumnTransformer(
                [
                    ('num_pipeline', num_pipeline, numerical_columns),
                    ('cat_pipeline', cat_pipeline, categorical_columns)
                ]
            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e, sys)
        
    
    def initiate_data_transformation(self, train_path, test_path):

        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info('Read train and test data completed')

            logging.info('Obtaining preprocessing object to preprocess the data')
            pp_obj = self.get_data_transformer_object()

            target_column_name = 'math_score'
            num_columns = ['writing_score', 'reading_score']

            input_feature_tr_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_tr_df = train_df[target_column_name]

            input_feature_test_df= test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info(f'Applying preprocessing object on training and test dataframe')
            
            input_feature_tr_arr = pp_obj.fit_transform(input_feature_tr_df) # we need to pass with_mean=False, otherwise it will give the error
            input_feature_test_arr = pp_obj.transform(input_feature_test_df)  # I think there should be input_feature_test_df

            tr_arr = np.c_[input_feature_tr_arr, np.array(target_feature_tr_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f'Saving preprocessing object')

            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = pp_obj

            )

            return(
                tr_arr, test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )


        except Exception as e:
            raise CustomException(e, sys)
            

