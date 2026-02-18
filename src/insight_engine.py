def cluster_impact(df):
    return df.groupby("cluster")["Score"].mean().sort_values()


def top_clusters(df, n=10):
    return df["cluster"].value_counts().head(n)


def worst_clusters(df, n=5):
    return df.groupby("cluster")["Score"].mean().sort_values().head(n)
