## importar libreria a usar

import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import re
import json
import re
from collections import Counter
import unidecode
import unicodedata
import emoji
from sqlalchemy import create_engine
import xgboost as xgb
from xgboost import plot_importance


class pre_process():
    def __init__(self):
        self.dummies = None
    
    def get_features(self, data, target_column):
        if target_column is None:
            data = data[['OPERA', 'MES', 'TIPOVUELO', 'SIGLADES', 'DIANOM']]
            return data
        else:
            data = data[['OPERA', 'MES', 'TIPOVUELO', 'SIGLADES', 'DIANOM', target_column]]
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
                dummies = pd.get_dummies(features[_feature], prefix=_feature, drop_first=True)
                features = pd.concat([features, dummies], axis=1)
                features.drop(_feature, axis=1, inplace=True)
            features = features.reindex(columns=self.dummies, fill_value=0)
            return features

        
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