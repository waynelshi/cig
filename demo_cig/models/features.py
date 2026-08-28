import pandas as pd


def process_features(df: pd.DataFrame) -> pd.DataFrame:
    df["ad_exposure"] = df["user_tenure"] * 0.45
    df["discount_pct"] = df["user_tenure"] * 0.05 + df["marketing_tier"] * 0.1
    return df
