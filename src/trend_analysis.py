def compute_trends(df):
    trend = df.groupby(["YearMonth", "cluster"]).size().unstack(fill_value=0)
    return trend
