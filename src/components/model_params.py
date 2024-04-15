
params = {
        'Decision Tree': {
            'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
           # 'spliter' : ['best', 'random'],
           # 'max_features' : ['sqrt', 'log2'],
        },
        'Random Forest': {
            # 'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
            # 'max_features': ['sqrt', 'log2', None],
            'n_estimators': [8,16,32,64,128,256]
        },
        'Gradient Boosting': {
            # 'loss': ['squared_error', 'huber', 'absolute_error', 'quantile'],
           'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
            # 'criterion': ['squared_error', 'friedman_mse']
            # 'max_features': ['sqrt', 'log2', 'auto']
            'n_estimators': [8,16,32,64,128,256]
        },
        'Linear Regression': {},
        'XGBoost': {
            'learning_rate': [0.1, 0.01, 0.05, 0.001],
            'n_estimators': [8,16,32,64,128,256]
        },
        'CatBoost': {
            'depth': [6,8,10],
            'learning_rate': [0.1, 0.01, 0.05],
            'iterations': [30,50, 100]
        },
        'AdaBoost':{
            'learning_rate': [0.1, 0.01, 0.05, 0.001],
            # 'loss' : ['linear', 'square', 'exponential']
            'n_estimators': [8,16,32,64,128,256]
        }
    }