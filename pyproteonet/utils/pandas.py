import pandas as pd

def matrix_to_multiindex(df: pd.DataFrame)->pd.DataFrame:
    df = df.stack(dropna=False)
    df.index.set_names(["id", "sample"], inplace=True)
    df = df.swaplevel()
    return df