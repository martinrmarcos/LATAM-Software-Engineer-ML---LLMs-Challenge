import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
        self._preprocess = pre_process()
        self.top_10_features = None

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
        

        data = self._preprocess.get_features(data, target_column)

        cat_features = ['OPERA', 'SIGLADES', 'DIANOM', 'TIPOVUELO', 'MES']

        if target_column is not None:
            target = data[target_column]
            features = data.drop(target_column, axis=1)
            if self._model is None:
                features = self._preprocess.get_dummies(features, cat_features)
            else:
                features = self._preprocess.get_dummies(features, cat_features, Trained = 'y')
            return pd.DataFrame(features).reset_index(drop=True), pd.DataFrame(target).reset_index(drop=True)
        else:
            features = data
            if self._model is None:
                features = self._preprocess.get_dummies(features, cat_features)
            else:
                features = self._preprocess.get_dummies(features, cat_features, Trained = 'y')
            return pd.DataFrame(features).reset_index(drop=True)


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
        target = target.iloc[:, 0]



        scale = self._preprocess.get_scale(target)


        self.top_10_features = self._preprocess.get_best_features(features, target, 10)

        self._model = xgb.XGBClassifier(learning_rate=0.01, scale_pos_weight = scale)
        self._model.fit(features[self.top_10_features], target) 
        
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
        #features = self.preprocess(features)
        if self._model is None:
            return print('Model not trained yet. Please run fit method first.')
        else:
            return self._model.predict(features[self.top_10_features]).tolist()