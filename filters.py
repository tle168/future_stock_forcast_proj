import pandas as pd

def advanced_filter(df, max_volume=None, max_price=None, min_value=None):
    df = df.copy()
    df['GT'] = df['Volume'] * df['Close']

    if max_volume is not None:
        df = df[df['Volume'] <= max_volume]
    if max_price is not None:
        df = df[df['Close'] <= max_price]
    if min_value is not None:
        df = df[df['GT'] >= min_value]

    return df

def filter_by_favorites(df, favorite_codes):
    return df[df['Code'].isin(favorite_codes)]
