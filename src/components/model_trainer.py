import os
import sys
from dataclasses import dataclass

from sklearn.metrics import r2_score

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor


from model_params import params

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_filepath = os.path.join('artifacts', 'model.pkl')


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self, tr_arr, test_arr):
        try:
            logging.info('Split training and test input data')
            x_tr, y_tr, x_test, y_test = (  
                tr_arr[:, :-1],  # 2D arr indexing
                tr_arr[:, -1],
                test_arr[:, :-1], #removing the last column and keep rest 
                test_arr[:, -1] #Keeping the last column
            )
            
            models= {
                'Random Forest': RandomForestRegressor(),
                'Decision Tree': DecisionTreeRegressor(),
                'Gradient Boosting': GradientBoostingRegressor(),
                'Linear Regression': LinearRegression(),
                'XGBoost': XGBRegressor(),
                'CatBoost': CatBoostRegressor(),
                'AdaBoost': AdaBoostRegressor()
            }
        
            model_report=evaluate_models(x_tr, y_tr, x_test, y_test,
                                         models = models,  params=params)

            # to get the best model score form dict
            best_model_score = max(sorted(model_report.values()))

            # to get the name of best model fromd dict
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException('No best model, every model has r2_score below 0.6')
            
            logging.info(f'Best found model on both training and test dataset')

            save_object(
                file_path=self.model_trainer_config.trained_model_filepath,
                obj=best_model
            )

            predicted = best_model.predict(x_test)
            r2 = r2_score(y_test, predicted)

            return r2


        except Exception as e:
            raise CustomException(e, sys)












