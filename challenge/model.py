import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix, classification_report
import xgboost as xgb
from xgboost import plot_importance

from preprocessclass import pre_process

from typing import Tuple, Union, List

class DelayModel:

    def __init__(
        self
    ):
        self._model = None # Model should be saved in this attribute.

    def preprocess(
        self,
        data: pd.DataFrame,
        target_column: str = None
    ) -> Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
        """
        Prepare raw data for training or predict.

        Args:
            data (pd.DataFrame): raw data.
            target_column (str, optional): if set, the target is returned.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: features and target.
            or
            pd.DataFrame: features.
        """
        
        ### Using the class pre_process to get the features ready for training or prediction
        preprocess = pre_process()
        print(data)
        data = preprocess.get_features(data, target_column)

        cat_features = ['OPERA', 'SIGLADES', 'DIANOM', 'TIPOVUELO', 'MES']
        data = preprocess.get_dummies(data, cat_features)

        top_10_features = ['TIPOVUELO_I',
                            'OPERA_Copa Air',
                            'MES_12',
                            'OPERA_Air Canada',
                            'OPERA_Qantas Airways',
                            'MES_7',
                            'OPERA_Gol Trans',
                            'OPERA_American Airlines',
                            'OPERA_Aeromexico',
                            'OPERA_Delta Air']
        features = data[top_10_features]

        if target_column is not None:
            target = data[target_column]
            return features, target
        else:
            return features

    def fit(
        self,
        features: pd.DataFrame,
        target: pd.DataFrame
    ) -> None:
        """
        Fit model with preprocessed data.

        Args:
            features (pd.DataFrame): preprocessed data.
            target (pd.DataFrame): target.
        """
        preprocess = pre_process()
        
        scale = preprocess.get_scale(target)


        features['delay'] = target
        features = shuffle(features)

        target = features['delay']
        features.drop('delay', axis=1, inplace=True)

        self._model = xgb.XGBClassifier(learning_rate=0.01, scale_pos_weight = scale)
        self._model.fit(features, target) 
        return

    def predict(
        self,
        features: pd.DataFrame
    ) -> List[int]:
        """
        Predict delays for new flights.

        Args:
            features (pd.DataFrame): preprocessed data.
        
        Returns:
            (List[int]): predicted targets.
        """
        if self._model is None:
            return print('Model not trained yet. Please run fit method first.')
        else:
            predictions = self._model.predict(features)
            return predictions