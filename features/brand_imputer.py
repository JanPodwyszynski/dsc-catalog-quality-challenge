import re
import pandas as pd
from nltk.metrics import edit_distance

class BrandImputer:
    def __init__(self, brands, max_distance=1):
        self.brands = [b.lower() for b in brands]
        self.max_distance = max_distance

    def _extract_brand(self, text):
        if pd.isna(text) or text == '':
            return None

        clean_text = re.sub(r'[^a-z0-9]', ' ', str(text).lower())
        tokens = clean_text.split()

        best_brand = None
        min_dist = self.max_distance + 1

        for token in tokens:
            for brand in self.brands:
                if abs(len(token) - len(brand)) >= min_dist:
                    continue

                dist = edit_distance(token, brand)
                if dist < min_dist:
                    min_dist = dist
                    best_brand = brand

                    if min_dist == 0:
                        return best_brand.capitalize()

        if best_brand is not None and min_dist <= self.max_distance:
            return best_brand.capitalize()

        return None

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        text_source = X['title'].fillna('') + ' ' + X['description'].fillna('')
        missing_mask = X['brand'].isna()

        imputed_brands = text_source[missing_mask].apply(self._extract_brand)
        X.loc[missing_mask, 'brand'] = imputed_brands
        return X