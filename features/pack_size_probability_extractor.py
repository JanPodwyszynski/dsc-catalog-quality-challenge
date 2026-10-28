import numpy as np
import pandas as pd
import re
from sklearn.metrics import classification_report
from sklearn.model_selection import GroupKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import clone

class PackSizeProbabilityExtractor:
    def __init__(self, base_model, n_splits=5):
        self.base_model = base_model
        self.n_splits = n_splits
        self.models = []
        self.vectorizers = []
        self.seller_to_model_idx = {}

    @staticmethod
    def __clean_free_accessories(text):
        if not isinstance(text, str):
            return ""
        return re.sub(r'[^.]*free accessory[^.]*\.?', '', text, flags=re.IGNORECASE)

    @staticmethod
    def __create_text_column(dataset):
        desc_clean = dataset['description'].apply(PackSizeProbabilityExtractor.__clean_free_accessories)
        dataset['text'] = dataset['title'].fillna('').str.lower() + ' ' + desc_clean.str.lower()

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
            train_clean = train_fold[(train_fold['target'] == 0) & (train_fold['pack_size'].notna())]

            vec = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), min_df=3, max_features=10000)
            X_train_vec = vec.fit_transform(train_clean['text'])

            y_train = train_clean['pack_size']

            model = clone(self.base_model)
            model.fit(X_train_vec, y_train)

            self.vectorizers.append(vec)
            self.models.append(model)

            val_fold = dataset.iloc[val_idx]
            val_clean = val_fold[(val_fold['target'] == 0) & (val_fold['pack_size'].notna())]

            if not val_clean.empty:
                X_val_vec = vec.transform(val_clean['text'])
                fold_preds = model.predict(X_val_vec)

                y_true.extend(val_clean['pack_size'].tolist())
                y_pred.extend(fold_preds.tolist())

        print(classification_report(y_true, y_pred))

    def transform(self, dataset: pd.DataFrame) -> pd.DataFrame:
        dataset = dataset.copy()
        self.__create_text_column(dataset)

        dataset['declared_pack_size_prob'] = -1.0
        dataset['predicted_pack_size'] = np.nan

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
                dataset.iat[global_i, dataset.columns.get_loc('predicted_pack_size')] = classes[np.argmax(preds_proba[local_i])]

                pack = sub.iloc[local_i]['pack_size']
                if pd.notna(pack) and pack in classes:
                    dataset.iat[global_i, dataset.columns.get_loc('declared_pack_size_prob')] = preds_proba[local_i, classes.index(pack)]

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
            best_pack = np.array(reference_classes)[np.argmax(avg_preds_proba, axis=1)]

            for local_i, global_i in enumerate(test_indices):
                dataset.iat[global_i, dataset.columns.get_loc('predicted_pack_size')] = best_pack[local_i]

                pack = sub.iloc[local_i]['pack_size']
                if pd.notna(pack) and pack in reference_classes:
                    dataset.iat[global_i, dataset.columns.get_loc('declared_pack_size_prob')] = avg_preds_proba[
                        local_i, reference_classes.index(pack)]

        dataset = dataset.drop(columns=['text'])

        return dataset