import os
import sys

import numpy as np
import pandas as pd
import dill

from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    

def evaluate_models(x_tr, y_tr, x_test, y_test, models, params):
    try:
        report = {}
        
        for i in range(len(list(models))):
            model = list(models.values())[i]
            para = params[list(models.keys())[i]]
 
            gs = GridSearchCV(model, para, cv=3)
            gs.fit(x_tr, y_tr)

            model.set_params(**gs.best_params_)
            model.fit(x_tr, y_tr)

            y_test_pred = model.predict(x_test)
            test_model_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score
        
        return report

    except Exception as e:
        raise CustomException(e, sys)