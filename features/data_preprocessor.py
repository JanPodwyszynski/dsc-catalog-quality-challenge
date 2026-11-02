import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class DataPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.price_stats_ = None
        self.weight_stats_ = None

    # obliczamy statystyki dla zbioru train, aby zbiór test był tak samo liczony
    def fit(self, X: pd.DataFrame):
        x = X.copy()

        effective_pack = x['predicted_pack_size'].fillna(x['pack_size']).fillna(1.0)
        effective_pack = effective_pack.clip(lower=1.0)

        log_unit_price = np.log1p((x['price_pln'] / effective_pack).clip(lower=0))
        log_unit_weight = np.log1p((x['shipping_weight_kg'] / effective_pack).clip(lower=0))
        cat_col = x['predicted_category'].fillna('missing')

        stats_df = pd.DataFrame({
            'cat': cat_col,
            'log_unit_price': log_unit_price,
            'log_unit_weight': log_unit_weight
        })

        # Zapamiętanie średniej i odchylenia standardowego TYLKO ze zbioru treningowego
        self.price_stats_ = stats_df.groupby('cat')['log_unit_price'].agg(['mean', 'std'])
        self.weight_stats_ = stats_df.groupby('cat')['log_unit_weight'].agg(['mean', 'std'])

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        x = X.copy()

        effective_pack = x['predicted_pack_size'].fillna(x['pack_size']).fillna(1.0)
        effective_pack = effective_pack.clip(lower=1.0)

        x['title'] = x['title'].fillna('missing').astype(str)
        x['description'] = x['description'].fillna('missing').astype(str)

        # Ceny i wagi jednostkowe
        x['unit_price'] = x['price_pln'] / effective_pack
        x['unit_weight'] = x['shipping_weight_kg'] / effective_pack
        x['price_per_weight'] = x['unit_price'] / x['unit_weight']

        x['log_unit_price'] = np.log1p(x['unit_price'].clip(lower=0))
        x['log_unit_weight'] = np.log1p(x['unit_weight'].clip(lower=0))
        x['log_price'] = np.log1p(x['price_pln'].clip(lower=0))
        x['log_weight'] = np.log1p(x['shipping_weight_kg'].clip(lower=0))

        # Flagi niezgodności
        x['category_mismatch'] = (
                (x['declared_category'] != x['predicted_category'])
                & x['declared_category'].notna()
        ).astype(int)

        x['pack_size_mismatch'] = (
                (x['pack_size'] != x['predicted_pack_size']) & x['pack_size'].notna()
        ).astype(int)

        x['cat_prob_delta'] = 1.0 - x['declared_cat_prob'].clip(lower=0.0)
        x['pack_prob_delta'] = 1.0 - x['declared_pack_size_prob'].clip(lower=0.0)

        # Długości tekstu i brakujące wartości
        x['title_len'] = x['title'].fillna('').astype(str).str.len()
        x['desc_len'] = x['description'].fillna('').astype(str).str.len()
        x['has_missing_price'] = x['price_pln'].isna().astype(int)
        x['has_missing_weight'] = x['shipping_weight_kg'].isna().astype(int)

        # Z-score cen i wag per kategoria (na podstawie statystyk z metody fit)
        cat_col = x['predicted_category'].fillna('missing')

        # Mapowanie i zabezpieczenie (fillna) na wypadek nowej kategorii w zbiorze testowym
        mean_p = cat_col.map(self.price_stats_['mean']).fillna(self.price_stats_['mean'].mean())
        std_p = cat_col.map(self.price_stats_['std']).fillna(self.price_stats_['std'].mean())

        mean_w = cat_col.map(self.weight_stats_['mean']).fillna(self.weight_stats_['mean'].mean())
        std_w = cat_col.map(self.weight_stats_['std']).fillna(self.weight_stats_['std'].mean())

        x['z_score_price'] = (x['log_unit_price'] - mean_p) / (std_p + 1e-5)
        x['z_score_price_abs'] = x['z_score_price'].abs()

        x['z_score_weight'] = (x['log_unit_weight'] - mean_w) / (std_w + 1e-5)
        x['z_score_weight_abs'] = x['z_score_weight'].abs()

        return x