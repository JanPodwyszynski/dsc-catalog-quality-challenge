import numpy as np
from sklearn.metrics import classification_report
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import clone

class CategoryProbabilityExtractor:
    def __init__(self, base_model, n_splits=5):
        self.base_model = base_model
        self.n_splits = n_splits
        self.models = []
        self.vectorizers = []
        self.seller_to_model_idx = {}

    @staticmethod
    def __create_text_column(dataset):
        dataset['text'] = dataset['brand'].fillna('').str.lower() + ' ' + dataset['description'].fillna('').str.lower() + ' ' + dataset['title'].fillna('').str.lower()


    def fit(self, dataset: pd.DataFrame):
        gkf = GroupKFold(n_splits=self.n_splits)

        dataset = dataset.copy()
        self.__create_text_column(dataset)

        y_true, y_pred = [], []

        for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(dataset, groups=dataset['seller_id'])):

            val_sellers = dataset.iloc[val_idx]['seller_id'].unique()
            for seller in val_sellers:
                self.seller_to_model_idx[seller] = fold_idx

            train_fold = dataset.iloc[train_idx]
            train_clean = train_fold[(train_fold['target'] == 0) & (train_fold['declared_category'].notna())]

            vec = TfidfVectorizer(max_features=10000, min_df=3)
            X_train_vec = vec.fit_transform(train_clean['text'])

            y_train = train_clean['declared_category']

            model = clone(self.base_model)
            model.fit(X_train_vec, y_train)

            self.vectorizers.append(vec)
            self.models.append(model)

            val_fold = dataset.iloc[val_idx]
            val_clean = val_fold[(val_fold['target'] == 0) & (val_fold['declared_category'].notna())]

            if not val_clean.empty:
                X_val_vec = vec.transform(val_clean['text'])
                fold_preds = model.predict(X_val_vec)

                y_true.extend(val_clean['declared_category'].tolist())
                y_pred.extend(fold_preds.tolist())

        print(classification_report(y_true, y_pred))

    def transform(self, dataset: pd.DataFrame) -> pd.DataFrame:
        dataset = dataset.copy()
        self.__create_text_column(dataset)

        dataset['declared_cat_prob'] = -1.0
        dataset['predicted_category'] = None

        model_to_indices = {i: [] for i in range(self.n_splits)}
        test_indices = []

        for idx, seller in enumerate(dataset['seller_id']):
            if seller in self.seller_to_model_idx:
                model_idx = self.seller_to_model_idx[seller]
                model_to_indices[model_idx].append(idx)
            else:
                test_indices.append(idx)

        for model_idx, indices in model_to_indices.items():
            if not indices:
                continue

            sub = dataset.iloc[indices]
            vec = self.vectorizers[model_idx]
            model = self.models[model_idx]

            X_vec = vec.transform(sub['text'])
            preds_proba = model.predict_proba(X_vec)
            classes = list(model.classes_)

            for local_i, global_i in enumerate(indices):
                dataset.iat[global_i, dataset.columns.get_loc('predicted_category')] = classes[np.argmax(preds_proba[local_i])]

                cat = sub.iloc[local_i]['declared_category']
                if pd.notna(cat) and cat in classes:
                    dataset.iat[global_i, dataset.columns.get_loc('declared_cat_prob')] = preds_proba[local_i, classes.index(cat)]

        if test_indices:
            sub = dataset.iloc[test_indices]
            reference_classes = list(self.models[0].classes_)

            preds_list = []
            for vec, model in zip(self.vectorizers, self.models):
                X_vec = vec.transform(sub['text'])
                preds_proba = model.predict_proba(X_vec)

                classes = list(model.classes_)
                aligned_proba = np.zeros((len(sub), len(reference_classes)))

                for c_idx, c in enumerate(reference_classes):
                    if c in classes:
                        aligned_proba[:, c_idx] = preds_proba[:, classes.index(c)]

                preds_list.append(aligned_proba)

            avg_preds_proba = np.mean(preds_list, axis=0)
            best_cat = np.array(reference_classes)[np.argmax(avg_preds_proba, axis=1)]

            for local_i, global_i in enumerate(test_indices):
                dataset.iat[global_i, dataset.columns.get_loc('predicted_category')] = best_cat[local_i]

                cat = sub.iloc[local_i]['declared_category']
                if pd.notna(cat) and cat in reference_classes:
                    dataset.iat[global_i, dataset.columns.get_loc('declared_cat_prob')] = avg_preds_proba[
                        local_i, reference_classes.index(cat)]

        dataset = dataset.drop(columns=['text'])

        return dataset