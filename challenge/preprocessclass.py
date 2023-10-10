## importar libreria a usar

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import sys
import re
import json
import re
from collections import Counter
import xgboost as xgb
from xgboost import plot_importance
from datetime import datetime


class pre_process():
    def __init__(self):
        self.dummies = None
    
    def get_features(self, data, target_column):
        if target_column is None:
            columns = ['OPERA', 'MES', 'TIPOVUELO', 'SIGLADES', 'DIANOM']
            present_columns = [col for col in columns if col in data.columns]
            missing_columns = [col for col in columns if col not in data.columns]
            for col in missing_columns:
                data[col] = pd.Series(dtype='object')
            data = data[present_columns]
            return data
        else:
            data = self.get_delay(data)
            columns = ['OPERA', 'MES', 'TIPOVUELO', 'SIGLADES', 'DIANOM', target_column]
            present_columns = [col for col in columns if col in data.columns]
            missing_columns = [col for col in columns if col not in data.columns]
            for col in missing_columns:
                data[col] = pd.Series(dtype='object')
            data = data[present_columns]
            return data
        
    def get_delay(self, data, threshold_in_minutes = 15):
        def get_min_diff(data):
            fecha_o = datetime.strptime(data['Fecha-O'], '%Y-%m-%d %H:%M:%S')
            fecha_i = datetime.strptime(data['Fecha-I'], '%Y-%m-%d %H:%M:%S')
            min_diff = ((fecha_o - fecha_i).total_seconds())/60
            return min_diff
        
        data['min_diff'] = data.apply(get_min_diff, axis = 1)
        data['delay'] = np.where(data['min_diff'] > threshold_in_minutes, 1, 0)
        data.drop('min_diff', axis=1, inplace=True)
        return data

    def get_dummies(self, features, cat_features, Trained=None):
        # # Generate dummy variables based on the trained features
        if Trained is None:
            for _feature in cat_features:
                dummies = pd.get_dummies(features[_feature], prefix=_feature, drop_first=True)
                features = pd.concat([features, dummies], axis=1)
                features.drop(_feature, axis=1, inplace=True)
            self.dummies = features.columns
            return features
        else:
            for _feature in cat_features:
                try:
                    dummies = pd.get_dummies(features[_feature], prefix=_feature, drop_first=True)
                    features = pd.concat([features, dummies], axis=1)
                    features.drop(_feature, axis=1, inplace=True)
                except:
                    pass
            return features.reindex(columns=self.dummies, fill_value=0)
    
    def get_scale(self, target):
        n_y0 = len(target[target == 0])
        n_y1 = len(target[target == 1])
        scale = n_y0/n_y1
        return scale
    
    def get_best_features(self, features, target, n_features = 10):
        xgb_model = xgb.XGBClassifier(random_state=1, learning_rate=0.01)
        xgb_model.fit(features, target)

        importances = xgb_model.feature_importances_
        indices = np.argsort(importances)[::-1]
        top_X_features = [features.columns[i] for i in indices[:n_features]]
        
        return top_X_features