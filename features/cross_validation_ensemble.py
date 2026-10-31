import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.base import clone
from features.data_preprocessor import DataPreprocessor
from .seller_folds import seller_folds


class CrossValidationEnsemble:
    def __init__(self, base_model, preprocessor: DataPreprocessor, features, n_splits=5):
        self.base_model = base_model
        self.base_preprocessor = preprocessor
        self.features = features
        self.n_splits = n_splits
        self.models_ = []
        self.preprocessors_ = []

    def fit(self, train_df: pd.DataFrame):
        self.models_ = []
        self.preprocessors_ = []
        oof_preds = np.zeros(len(train_df))

        for fold, (fit_idx, val_idx) in enumerate(seller_folds(train_df['seller_id'], n_splits=self.n_splits)):
            print(f"--- Fold {fold + 1} ---")

            df_fit = train_df.iloc[fit_idx].copy()
            df_val = train_df.iloc[val_idx].copy()

            dp = clone(self.base_preprocessor)
            df_fit = dp.fit_transform(df_fit)
            df_val = dp.transform(df_val)

            X_fit = df_fit[self.features]
            y_fit = df_fit['target']
            X_val = df_val[self.features]
            y_val = df_val['target']

            clf = clone(self.base_model)
            clf.fit(X_fit, y_fit)

            val_probs = clf.predict_proba(X_val)[:, 1]
            oof_preds[val_idx] = val_probs

            fold_auc = roc_auc_score(y_val, val_probs)
            fold_ap = average_precision_score(y_val, val_probs)
            print(f"ROC-AUC: {fold_auc:.4f} | PR-AUC: {fold_ap:.4f}")

            self.models_.append(clf)
            self.preprocessors_.append(dp)

        total_auc = roc_auc_score(train_df['target'], oof_preds)
        total_ap = average_precision_score(train_df['target'], oof_preds)
        print("\n" + "=" * 30)
        print(f"OOF ROC-AUC : {total_auc:.4f}")
        print(f"OOF PR-AUC  : {total_ap:.4f}")
        print("=" * 30)

    def predict_proba(self, test_df: pd.DataFrame) -> np.ndarray:
        test_preds_list = []

        for dp, clf in zip(self.preprocessors_, self.models_):
            test_transformed = dp.transform(test_df)
            X_test = test_transformed[self.features]

            probs = clf.predict_proba(X_test)[:, 1]
            test_preds_list.append(probs)

        return np.mean(test_preds_list, axis=0)