import pandas as pd
import polars as pl
from tqdm import tqdm
import os

FACTOR_DATA_DIR = "../../data/raw/研究策略组20251024/"


df = pd.read_pickle(os.path.join(FACTOR_DATA_DIR, "all_factors_kline.pkl"))
total_len = len(df)
lf = pl.from_pandas(df).lazy()

for file_name in tqdm(os.listdir(FACTOR_DATA_DIR)):
    if not file_name.startswith("factor_"):
        continue

    factor_series = pd.read_pickle(os.path.join(FACTOR_DATA_DIR, file_name))
    factor_series = factor_series.rename(f"factor_{factor_series.name}")
    factor_lf = pl.DataFrame(pl.from_pandas(factor_series)).lazy()

    lf = pl.concat([lf, factor_lf], how="horizontal")

lf.with_columns(
    pl.col("交易日期").cast(pl.Date)
)

lf.sink_parquet("../../data/all_factors_kline.parquet", mkdir=True, compression="zstd", compression_level=5, )


df = pl.read_parquet("../../data/all_factors_kline.parquet")

print(df)
