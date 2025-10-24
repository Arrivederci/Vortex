# Vortex 项目说明

## 项目简介
Vortex 是一个面向量化投资策略研究的端到端机器学习框架，通过统一的编排器将数据准备、特征工程、模型训练、超参搜索以及绩效分析串联起来，帮助研究人员快速验证想法并产出可追溯的分析报告。

## 核心功能总览
- **配置驱动的工作流**：`ConfigLoader` 负责加载 YAML 配置，并在缺少 PyYAML 时退化为内置解析器，保证流程可用性。
- **数据合并与目标构建**：`DataLoader` 可读取 Parquet 格式的因子数据与 CSV 行情数据，自动对齐多层索引并按照配置计算前瞻收益目标。
- **时间序列切分**：`DatasetManager` 支持带禁区的 Walk-Forward 滚动切分与去污染的时间序列交叉验证，适合回测场景。
- **特征选择流水线**：通过配置化的特征选择器注册表，组合 Lasso、PCA、Rank IC 筛选、树模型重要性等步骤。
- **模型训练与持久化**：`ModelTrainer` 将特征预处理器与模型组装在一起，并提供统一的保存/加载接口；内置 ElasticNet、随机森林、LightGBM、MLP 等模型。
- **嵌套调参**：在 Walk-Forward 外层循环下，支持基于去污染时间序列交叉验证的超参搜索，评估指标可选 MSE 或 Rank IC。
- **绩效分析与可视化**：`PerformanceAnalyzer` 会输出 Rank IC、R²、分位组合收益、长短组合指标等，同时生成 JSON 指标文件与 Plotly HTML 报表。

## 目录结构
```
vortex/
  config/            # 配置文件加载
  data/              # 数据读取与目标收益计算
  dataset/           # Walk-Forward 切分与时间序列交叉验证
  features/          # 特征选择器基类与注册体系
  metrics/           # 调参与评估使用的损失函数
  models/            # 预处理器、算法实现、模型训练器
  performance/       # 绩效指标与报表生成
  orchestrator.py    # 串联所有组件的主流程
requirements.txt     # 依赖列表
```

## 快速开始
### 1. 安装依赖
```bash
pip install -r requirements.txt
```
若需要启用 LightGBM 模型或树模型重要性计算中的 LightGBM 选项，请额外安装 `lightgbm`。

### 2. 准备数据
- **因子文件**：Parquet 格式，至少包含日期列（默认 `交易日期`）、资产列（默认 `股票代码`）以及多个特征列。
- **行情文件**：目录下的 CSV 文件集合，需包含与因子表一致的日期和资产列，以及开盘/收盘等价格列供目标收益计算使用。

### 3. 编写配置
以下示例摘自项目测试用例，可作为最小可运行配置：
```yaml
data_sources:
  factor_path: /path/to/factors.parquet
  ohlc_path: /path/to/ohlc_dir
  target_return:
    period: 5
    type: forward_return

dataset_manager:
  walk_forward:
    start_date: "2020-01-01"
    end_date: "2020-03-31"
    train_window_type: "fixed"
    train_length: 15
    test_length: 5
    step_length: 5
    embargo_length: 1
  inner_cv:
    enable: true
    n_splits: 3
    embargo_length: 1
    metric: mse

feature_selector:
  pipeline:
    - method: rank_ic_filter
      params:
        top_k: 20

model:
  preprocessor:
    method: rank
    params: {}
  name: random_forest
  params:
    n_estimators: 200
    random_state: 42
  tuning:
    enable: true
    candidates:
      - {n_estimators: 200}
      - {n_estimators: 400}
  persistence:
    enable: true
    save_path: ./artifacts/models
    filename_template: "rf_{train_end_date}"

performance_analyzer:
  quantiles: 5
  risk_free_rate: 0.02
  group_by_columns: []
```

### 4. 运行工作流
```python
from vortex.orchestrator import run_workflow

metrics = run_workflow("./config.yaml", output_dir="./artifacts")
print(metrics)
```
成功运行后，`output_dir` 中会生成：
- `results.json`：聚合后的指标字典；
- `report.html`：包含 Rank IC 曲线与分位数组合累计收益的交互式报表；
- （可选）`models/`：当配置了持久化后，会在此目录保存模型与预处理器快照。

## 配置项详解
### data_sources
- `factor_path`：Parquet 因子文件路径。
- `ohlc_path`：行情 CSV 文件所在目录。
- `target_return`：控制目标收益计算，支持以下字段：
  - `period`：持有期长度；
  - `type`：目前支持 `forward_return`；
  - `entry_price_column`/`exit_price_column`：自定义开/平仓价格列；
  - `entry_shift`/`exit_shift`：控制买入、卖出相对当前日期的偏移。

### dataset_manager
- `walk_forward`：定义滚动训练窗口。
  - `train_window_type`：`expanding` 或 `fixed`；
  - `train_length`/`test_length`/`step_length`：训练、测试窗口长度及滑动步长；
  - `embargo_length`：训练与测试之间的禁区长度。
- `inner_cv`（可选）：启用嵌套时间序列交叉验证时的配置。
  - `metric` 可选 `mse` 或 `rank_ic`，指标统一遵循“越小越好”的比较准则。

### feature_selector
- `pipeline`：数组形式的步骤列表，每个步骤包含 `method` 与 `params`。
- 内置选择器：
  - `lasso`：基于 L1 稀疏系数筛选特征；
  - `pca`：主成分降维；
  - `rank_ic_filter`：按秩相关系数筛选；
  - `tree_based_importance`：基于树模型重要性。

### model
- `preprocessor`：特征预处理器，支持 `none`（直通）、`rank`（分位排名）、`standard_scaler`（标准化）。
- `name`：模型名称，内置 `elasticnet`、`random_forest`、`lightgbm`、`mlp`。
- `params`：传递给具体模型的参数。
- `tuning`（可选）：
  - `enable`：是否开启调参；
  - `candidates`：候选参数字典列表。
- `persistence`（可选）：控制模型与预处理器的保存目录与文件名模板。

### performance_analyzer
- `quantiles`：构建的分位数组合数量；
- `risk_free_rate`：用于计算超额收益与夏普比率；
- `group_by_columns`：可指定额外的分组列以生成分组指标（保留扩展能力）。

## 高级用法
### 自定义特征选择器
实现 `FeatureSelectorBase` 子类，并使用 `@register_selector("your_name")` 装饰器注册。随后即可在配置的 `pipeline` 中引用。

### 自定义模型
继承 `BaseModel`，实现 `fit` 与 `predict`，并通过 `@register_model("your_model")` 装饰器注册。配置中的 `model.name` 即可指向新的实现。

### 模型持久化与回放
- 当 `persistence.enable` 为 `true` 时，`ModelTrainer` 会将模型保存为 `.joblib` 文件，同时保存预处理器，便于复现。
- 可使用 `ModelTrainer.load_model(model_path, preprocessor_path)` 载入历史模型，配合同样的特征流水线进行预测。

## 测试
项目使用 `pytest` 提供单元测试覆盖核心流程：
```bash
pytest
```

## 常见问题
- **缺少 PyYAML**：`ConfigLoader` 会自动回退到简易解析器，建议在生产环境安装 `pyyaml` 以完整支持复杂语法。
- **缺少 LightGBM**：仅在选择 LightGBM 模型或树模型重要性并指定 `model_name=lightgbm` 时需要安装，对其他场景可选。

