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


class pre_process():
    def get_features(self, data, target_column):
        if target_column is None:
            data = data[['OPERA', 'MES', 'TIPOVUELO', 'SIGLADES', 'DIANOM']]
            return data
        else:
            data = data[['OPERA', 'MES', 'TIPOVUELO', 'SIGLADES', 'DIANOM', 'delay']]
            return data
                
    def get_dummies(self, data, cat_features):
        for feature in cat_features:
            dummies = pd.get_dummies(data[feature], prefix=feature, drop_first=True)
            data = pd.concat([data, dummies], axis=1)
            data.drop(feature, axis=1, inplace=True)
        return data
    
    def get_scale(self, target):
        n_y0 = len(target[target == 0])
        n_y1 = len(target[target == 1])
        scale = n_y0/n_y1
        return scale