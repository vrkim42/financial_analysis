# 🔍 财务造假检测数据预处理系统

![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)
![Pandas](https://img.shields.io/badge/pandas-1.4+-red.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-green.svg)

## 项目概述

本项目是一个用于财务造假检测的数据预处理系统，专为审计和金融风控场景设计。系统提供从原始财务数据加载到特征工程的全流程处理，输出高质量的特征数据供后续建模使用。

## ✨ 主要功能

### 数据处理流程

- **数据加载与整合**
  - 多源数据融合（行业、财务数据）
  - 灵活的行业筛选机制
  - 自动构建中文字段含义映射

- **探索性数据分析（EDA）**
  - 目标变量（造假标记）分布可视化
  - 数值型变量多维度分析（直方图+箱线图）
  - 缺失值模式识别与可视化
  - 特征相关性热力图分析

- **数据预处理**
  - 智能缺失值处理策略
  - 异常值检测与平滑处理
  - 多种数据变换方法（分箱、对数变换等）

- **特征工程**
  - 自动构建财务比率特征
  - PCA降维（可配置方差比例）
  - 基于统计显著性的特征选择

- **结果导出**
  - 多格式数据导出（Excel、CSV）
  - 统计报告自动生成
  - 数据字典持久化

## 环境要求

### Python依赖
```
numpy>=1.22.0
pandas>=1.4.0
matplotlib>=3.5.0
seaborn>=0.11.0
scikit-learn>=1.0.0
openpyxl>=3.0.0
joblib>=1.1.0
missingno>=0.5.0  # 可选，用于缺失值可视化
```

安装所有依赖：
```bash
pip install numpy pandas matplotlib seaborn scikit-learn openpyxl joblib missingno
```

### 数据文件要求
项目根目录下需要`data`文件夹包含：
1. `data1.csv` - 行业类别数据
2. `data2.csv` - 财务数据
3. `data3.xlsx` - 字段含义说明

## 使用说明
### 配置系统
可通过初始化时传入配置字典调整系统行为：
```python
config = {
    'missing_value_threshold': 70,  # 缺失值删除阈值(%)
    'pca_n_components': 0.95,       # PCA解释方差比例
    'feature_selection_k': 10,      # 特征选择数量
    'plot_style': 'seaborn',        # 图表风格
    'bin_smooth_method': 'mean',    # 分箱平滑方法
    'industry_filter': '制造业'      # 行业筛选条件
}

preprocessor = FinancialDataPreprocessor(data_paths, config)
### 运行项目
```python
python main.py
```

### 处理流程
1. **数据加载**：自动加载并合并数据源
2. **EDA分析**：生成可视化图表到`plots`目录
3. **数据预处理**：清洗、转换数据
4. **特征工程**：降维和特征选择
5. **结果导出**：保存处理结果到`output`目录

### 目录结构
```
项目根目录/
├── main.py                # 主程序
├── data/                  # 数据目录（必须）
│   ├── data1.csv          # 行业数据
│   ├── data2.csv          # 财务数据
│   └── data3.xlsx         # 字段说明
├── plots/                 # 自动生成-可视化图表
└── output/                # 自动生成-处理结果
```

## 输出文件说明
### 输出文件概览
财务数据处理流程结束后，会生成`plots`（可视化图表）和`output`（处理结果）两个目录，以下通过表格形式说明各文件内容与用途。

### plots 目录（可视化图表）
|文件名称|说明|
| ---- | ---- |
|{字段名}_dist.png|单个字段的分布图，包含直方图（展示数据分布形态）和箱线图（呈现数据离散程度、异常值情况 ）|
|numeric_distributions.png|所有数值字段分布图合集，可快速概览多个数值型字段的数据分布趋势|
|missing_values_matrix.png|缺失值矩阵图，直观展示数据中缺失值的位置分布（需安装`missingno`库才可正常生成、查看 ）|
|{分类字段名}_category_dist.png|分类字段的值分布柱状图，若分类值过多，仅展示前10个主要分类（针对分类变量分析生成 ）|
|{分类字段名}_vs_flag.png|分类字段与财务造假标记（`FLAG`）的关联关系图，以堆积柱状图呈现不同分类下造假标记的分布（若数据含`FLAG`字段 ）|
|correlation_matrix.png|特征相关性矩阵热力图，展示数值型字段间的线性相关关系|
|strong_correlations.csv|存储强相关特征对（相关系数绝对值≥0.7 ），包含“特征1”“特征2”“相关系数”信息，辅助筛选关键关联特征|

### output 目录（处理结果）
|文件名称|内容|格式|
| ---- | ---- | ---- |
|processed_data.xlsx|处理结果数据汇总，含两个工作表|Excel|
|├─ 原始数据 sheet|加载并清洗（去重、基础合并等 ）后的原始财务数据，保留数据初始状态| - |
|└─ 处理后数据 sheet|经缺失值处理、异常值处理、噪声处理、数据泛化、数值规约、特征工程后的标准化数据，可用于后续分析建模| - |
|fields_dict.pkl|字段含义字典，键为字段名，值为元组`(含义, 单位)`，方便程序读取字段解释|Pickle|
|processed_data.csv|与`processed_data.xlsx`中“处理后数据”工作表内容一致，以CSV格式存储，适配简单数据读取场景|CSV|
|data_profile_report.html|基于`pandas_profiling`生成的财务数据分析报告（若安装该库 ），涵盖数据概述、字段分布、缺失值、相关性等内容，浏览器打开可查看|HTML| 

## 控制台输出示例

```
数据加载完成，形状: (5000, 120)
财务造假分布:
0    4500
1     500
Name: FLAG, dtype: int64

缺失值统计表:
            字段名  缺失值数量  缺失率(%)      数据类型
0  TICKER_SYMBOL        0     0.00       int64
1             AP      120     2.40     float64
...

删除了 15 个高缺失率字段
填充了 42 个字段的缺失值

主成分方差解释比例: [0.25, 0.18, 0.12,...]
累计方差解释比例: 0.95
PCA降维结果形状: (5000, 8)

筛选出的重要特征: ['AP', 'ADVANCE_RECEIPTS', 'TAXES_PAYABLE', ...]

结果已导出到 ./output 目录
处理流程完成
```

## 注意事项

1. 确保数据文件使用UTF-8编码，如有中文乱码可尝试`encoding='gbk'`
2. 首次运行会自动创建`plots`和`output`目录
3. 缺失值可视化需要安装`missingno`库
4. 项目默认筛选制造业数据，如需修改请调整`load_and_merge_data`方法

## ⚙️ 高级配置选项

### 主要参数设置
| 参数 | 说明 | 默认值 | 位置 |
|------|------|--------|------|
| `missing_value_threshold` | 缺失率超过此值的字段将被删除(%) | 70 | 配置字典 |
| `pca_n_components` | PCA保留方差比例 | 0.95 | 配置字典 |
| `feature_selection_k` | 特征选择保留数量 | 10 | 配置字典 |
| `plot_style` | 图表风格 | 'seaborn' | 配置字典 |
| `bin_smooth_method` | 分箱平滑方法('mean'/'median') | 'mean' | 配置字典 |
| `industry_filter` | 行业筛选条件 | '制造业' | 配置字典 |

---

*更新日期: 2025年6月7日*
