import unittest
import pandas as pd
import os
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from challenge.model import DelayModel

class TestModel(unittest.TestCase):

    FEATURES_COLS = [
        "OPERA_Latin American Wings", 
        "MES_7",
        "MES_10",
        "OPERA_Grupo LATAM",
        "MES_12",
        "TIPOVUELO_I",
        "MES_4",
        "MES_11",
        "OPERA_Sky Airline",
        "OPERA_Copa Air"
    ]

    TARGET_COL = [
        "delay"
    ]


    def setUp(self) -> None:
        super().setUp()
        self.model = DelayModel()
        parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.data = pd.read_csv(filepath_or_buffer=parent_dir+"/data/data.csv")
        

    def test_model_preprocess_for_training(
        self
    ):
        features, target = self.model.preprocess(
            data=self.data,
            target_column="delay"
        )

        _, features_validation, _, target_validation = train_test_split(features, target, test_size = 0.33, random_state = 42)

        self.model.fit(
            features=features,
            target=target
        )
        #### Modified this to integrate the get_best_features function
        self.FEATURES_COLS = self.model.top_10_features
        assert isinstance(features, pd.DataFrame)
        assert features[self.model.top_10_features].shape[1] == len(self.FEATURES_COLS)
        assert set(features[self.model.top_10_features].columns) == set(self.FEATURES_COLS)

        assert isinstance(target, pd.DataFrame)
        assert target.shape[1] == len(self.TARGET_COL)
        assert set(target.columns) == set(self.TARGET_COL)


    def test_model_preprocess_for_serving(
        self
    ):
        features, target = self.model.preprocess(
            data=self.data,
            target_column="delay"
        )

        _, features_validation, _, target_validation = train_test_split(features, target, test_size = 0.33, random_state = 42)

        self.model.fit(
            features=features,
            target=target
        )
        #### Modified this to integrate the get_best_features function
        self.FEATURES_COLS = self.model.top_10_features
        assert isinstance(features, pd.DataFrame)
        assert features[self.model.top_10_features].shape[1] == len(self.FEATURES_COLS)
        assert set(features[self.model.top_10_features].columns) == set(self.FEATURES_COLS)


    def test_model_fit(
        self
    ):
        features, target = self.model.preprocess(
            data=self.data,
            target_column="delay"
        )

        _, features_validation, _, target_validation = train_test_split(features, target, test_size = 0.33, random_state = 42)

        self.model.fit(
            features=features,
            target=target
        )

        predicted_target = self.model.predict(
            features_validation
        )

        report = classification_report(target_validation, predicted_target, output_dict=True)
        
        assert report["0"]["recall"] < 0.60
        assert report["0"]["f1-score"] < 0.70
        assert report["1"]["recall"] > 0.60
        assert report["1"]["f1-score"] > 0.30


    def test_model_predict(
        self
    ):
        features, target = self.model.preprocess(
            data=self.data,
            target_column="delay"
        )
        _, features_validation, _, target_validation = train_test_split(features, target, test_size = 0.33, random_state = 42)

        self.model.fit(
            features=features,
            target=target
        )

        predicted_targets = self.model.predict(
            features=features_validation
        )

        assert isinstance(predicted_targets, list)
        assert len(predicted_targets) == features_validation.shape[0]
        assert all(isinstance(predicted_target, int) for predicted_target in predicted_targets)