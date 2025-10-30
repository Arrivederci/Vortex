import pandas as pd
import polars as pl
from tqdm import tqdm
import os

FACTOR_DATA_DIR = "../../data/raw/研究策略组20251024/"


def cal_fuquan_price(df, fuquan_type="后复权", method=None):
    """
    用于计算复权价格

    参数:
    df (DataFrame): 必须包含的字段：收盘价，前收盘价，开盘价，最高价，最低价
    fuquan_type (str, optional): 复权类型，可选值为 '前复权' 或 '后复权'，默认为 '后复权'
    method (str, optional): 额外计算复权价格的方法，如 '开盘'，默认为 None

    返回:
    DataFrame: 最终输出的df中，新增字段：收盘价_复权，开盘价_复权，最高价_复权，最低价_复权
    """

    # 计算复权因子
    fq_factor = df["复权因子"]

    # 计算前复权或后复权收盘价
    if fuquan_type == "后复权":  # 如果使用后复权方法
        fq_close = fq_factor * (df.iloc[0]["收盘价"] / fq_factor.iloc[0])
    elif fuquan_type == "前复权":  # 如果使用前复权方法
        fq_close = fq_factor * (df.iloc[-1]["收盘价"] / fq_factor.iloc[-1])
    else:  # 如果给的复权方法非上述两种标准方法会报错
        raise ValueError(f"计算复权价时，出现未知的复权类型：{fuquan_type}")

    # 计算其他价格的复权值
    fq_open = df["开盘价"] / df["收盘价"] * fq_close
    fq_high = df["最高价"] / df["收盘价"] * fq_close
    fq_low = df["最低价"] / df["收盘价"] * fq_close

    # 一次性赋值，提高计算效率
    df = df.assign(
        复权因子=fq_factor, 收盘价_复权=fq_close, 开盘价_复权=fq_open, 最高价_复权=fq_high, 最低价_复权=fq_low
    )

    # 如果指定了额外的方法，计算该方法的复权价格
    if method and method != "开盘":
        df[f"{method}_复权"] = df[method] / df["收盘价"] * fq_close

    # 删除中间变量复权因子
    # df.drop(columns=['复权因子'], inplace=True)

    return df


df = pd.read_pickle(os.path.join(FACTOR_DATA_DIR, "all_factors_kline.pkl"))
df = cal_fuquan_price(df)

total_len = len(df)
lf = pl.from_pandas(df).lazy()

factor_list = ["市值", "涨跌幅Std_20", "近期涨跌幅_20", "Rsj_20", "ROE_单季", "EP_单季",
               "归母净利润同比增速_60", "换手率_20", "是否破N日前高_20", "当前回撤_20", "非流动性因子", "N日内有涨停_30"]
factor_list = [f"factor_{name}.pkl" for name in factor_list]

for file_name in tqdm(os.listdir(FACTOR_DATA_DIR)):
    if not file_name.startswith("factor_"):
        continue
    elif file_name not in factor_list:
        continue

    factor_series = pd.read_pickle(os.path.join(FACTOR_DATA_DIR, file_name))
    factor_series = factor_series.rename(f"factor_{factor_series.name}")
    factor_lf = pl.DataFrame(pl.from_pandas(factor_series)).lazy()

    lf = pl.concat([lf, factor_lf], how="horizontal")

lf.with_columns(
    pl.col("交易日期").cast(pl.Date)
)

lf.sink_parquet("../../data/selected_factors_kline3.parquet", mkdir=True, compression="zstd", compression_level=5, )


df = pl.read_parquet("../../data/selected_factors_kline3.parquet")

print(df.columns)
print(df)
