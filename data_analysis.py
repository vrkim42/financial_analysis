import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings, time, datetime, joblib
warnings.filterwarnings('ignore')

# 1.获取数据
df_category = pd.read_csv('data/data1.csv', encoding='utf-8', on_bad_lines='skip') # 行业类别
df_all = pd.read_csv('data/data2.csv', encoding='utf-8', on_bad_lines='skip') # 财务数据
df_fields = pd.read_excel('data/data3.xlsx') # 字段含义

# 方便后续查看字段含义
fields = dict(zip(df_fields['字段名'], zip(df_fields['含义'], df_fields['单位'])))
print(fields['TICKER_SYMBOL'])

# #fields # In a script, this would print a large dictionary. Commenting out.
# print(fields)

#数据合并
df_category.columns = ['TICKER_SYMBOL', '所属行业']
df_all = pd.merge(df_all, df_category, how='left', on=['TICKER_SYMBOL'])
df_all.insert(1, 'INDUSTRY', df_all.pop('所属行业'))
fields['INDUSTRY'] = ('所属行业', np.nan)

print(df_all.head())
print(df_all.shape)
print(df_all['FLAG'].value_counts())
# print(df_all['FLAG'].value_counts()) # Redundant display in notebook

print(df_all.dtypes)


# 2.数据探索

## 数据分布可视化
print(df_all['INDUSTRY'])

df = df_all[df_all['INDUSTRY'] == '制造业']
# 数据去重
df.drop_duplicates(inplace=True)
# df = df.drop_duplicates() # Redundant

#drop ,reset_index
df = df.drop_duplicates().reset_index(drop=True) # Added reset_index for good practice

print(df.shape)

plt.rcParams['font.sans-serif'] = ['fangsong'] # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False # 解决负号显示异常

numeric_cols_test = ['AP', 'ADVANCE_RECEIPTS', 'SOLD_FOR_REPUR_FA', 'PAYROLL_PAYABLE', 'TAXES_PAYABLE',
                     'INT_PAYABLE', 'DIV_PAYABLE', 'OTH_PAYABLE']

# The following loop generates plots. In a script, these will be displayed sequentially.
# If saving is needed, plt.savefig() would be used.
for col in numeric_cols_test:
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    sns.histplot(df[col], kde=True)
    plt.title(f'Histogram of {fields[col]}')
    plt.subplot(1, 2, 2)
    sns.boxplot(x=df[col])
    plt.title(f'Boxplot of {fields[col]}')
    plt.savefig(f'plots/{col}_hist_box.png')


# 数值型变量分布可视化
def plot_numeric_distribution(df_plot, numeric_cols_plot, n_cols=4):
    n_rows = (len(numeric_cols_plot) + n_cols - 1) // n_cols
    plt.figure(figsize=(20, 5 * n_rows))
    
    for i, col in enumerate(numeric_cols_plot):
        plt.subplot(n_rows, n_cols, i + 1)
        sns.histplot(df_plot[col], kde=True, bins=30)
        plt.title(f'{fields[col]} 分布', fontsize=12)
        plt.xlabel('')
    
    plt.tight_layout()
    plt.savefig(f'plots/{col}_hist_box.png')

# 选择部分数值型变量进行可视化
plot_numeric_distribution(df, numeric_cols_test)


print(df.describe())

# 数据探索：不同字段分布的可视化绘图
# 数值型字段的直方图和箱线图
numeric = df.describe()
numeric_cols = numeric.columns
print(numeric_cols)

# 分类型字段的柱状图
nominal = df_all.describe(include=['O']) # Note: Using df_all as in notebook, might intend df
nominal_cols = nominal.columns
print(df.describe(include='O')) # In notebook, df.describe(include='O') was executed

str_col = df.describe(include='O') # Corrected to use df as per subsequent cells
str_cols = str_col.columns
print(str_cols)


# The following loop generates plots.
for col in nominal_cols: # Note: nominal_cols is from df_all, but plotting uses df
    if col in df.columns: # Added check to prevent error if col not in df
        plt.figure(figsize=(6, 4))
        sns.countplot(x=col, data=df)
        plt.title(f'Bar plot of {fields.get(col, (col,None))[0]}') # Used .get for fields
        plt.xticks(rotation=45)
        plt.savefig(f'plots/{col}_hist_box.png')
    else:
        print(f"Column {col} not in filtered DataFrame df, skipping countplot.")


## 缺失值分析

# 缺失值统计
print(df.isnull().sum())

# !pip install missingno # This should be done in the environment, not in the script
import missingno
missingno.matrix(df, figsize=(15, 10))
plt.savefig(f'plots/{col}_hist_box.png')

# 缺失值统计
# 制作缺失值统计表格
def miss_data_count(data_in):
    data_null_out = pd.DataFrame(data_in.isnull().sum(), columns=['缺失值数量'])
    data_null_out['缺失率'] = 0.0
    for i in range(data_null_out.shape[0]):
        data_null_out.iloc[i, data_null_out.columns.get_loc('缺失率')] = round((data_null_out.iloc[i, data_null_out.columns.get_loc('缺失值数量')] / data_in.shape[0]) * 100, 2)

    data_null_out['数据类型'] = data_in.dtypes
    data_null_out.reset_index(inplace=True)
    data_null_out.columns = ['字段名', '缺失值数量', '缺失率', '数据类型']
    return data_null_out

data_null = miss_data_count(df)
print(data_null)

# 3.数据预处理
## 缺失值处理

# # 1. 人工填写空缺值（示例：手动将字段 'FIELD_NAME' 的缺失值填充为 'VALUE'）
# # df['FIELD_NAME'] = df['FIELD_NAME'].fillna('VALUE') # This was a placeholder

# # 2. 全局变量填充（示例：将所有缺失值填充为 0）
# df_global_fill = df.fillna(0) # Creates a new DataFrame, not used later in executed path

# # 3. 平均值填充（示例：对数值型字段进行平均值填充）
# numeric_cols_fill_mean = df.select_dtypes(include=[np.number]).columns
# df_mean_fill = df.copy()
# for col in numeric_cols_fill_mean:
#     mean_value = df[col].mean()
#     df_mean_fill[col] = df_mean_fill[col].fillna(mean_value) # Creates a new DataFrame

# # 4. 同类样本均值填充（示例：按 'INDUSTRY' 分组，对数值型字段进行同类样本均值填充）
# df_group_fill = df.copy()
# for col in numeric_cols_fill_mean: # Assuming numeric_cols_fill_mean is still relevant
#     if col != 'INDUSTRY': # Avoid trying to fill the group_col itself if it's numeric
#         df_group_fill[col] = df_group_fill.groupby('INDUSTRY')[col].transform(lambda x: x.fillna(x.mean())) # Creates new DF

# # 5. 预测填充（示例：使用线性回归对缺失值进行预测填充）
# # from sklearn.linear_model import LinearRegression
# #
# # df_pred_fill = df.copy()
# # for col in numeric_cols: # Assuming numeric_cols is still relevant
# #     if df_pred_fill[col].isnull().sum() > 0:
# #         # This part needs careful feature engineering for X_train and X_test
# #         # The original raw cell had a conceptual outline but not directly executable code
# #         # for a generic case without specifying features for regression.
# #         print(f"Skipping predictive fill for {col} due to complexity in script conversion.")
# #         pass

# # 6. 删除缺失值
# # #df_dropna = df.dropna()
# # # 删除缺失值占比大于70%的列
# # def drop_missing(df_to_drop, threshold=0.7):
# # # 删除缺失率超过threshold的列
# #     missing_ratio = df_to_drop.isnull().sum() / len(df_to_drop)
# #     cols_to_drop_func = missing_ratio[missing_ratio > threshold].index
# #     df_dropped = df_to_drop.drop(columns=cols_to_drop_func)
# # # 删除仍有缺失值的行
# # #df_dropped = df_dropped.dropna()
# # return df_dropped
# #
# # # 示例：删除缺失率超过70%的列和仍有缺失值的行
# # df0_dropped = drop_missing(df, 0.7) # This was a raw cell, not executed in the main flow on 'df'

# Actual executed missing value handling:
dropna_cols = []
fill0_cols = []

# data_null was created from the original df
for index, row in data_null.iterrows():
    if row['缺失率'] >= 70:
        dropna_cols.append(row['字段名'])
    # FLAG is the target, handle its NaN separately if needed, or it might be in fill0_cols
    elif row['字段名'] != 'FLAG' and row['缺失率'] > 0 : # Ensure FLAG is not in fill0_cols for simple fill(0)
        fill0_cols.append(row['字段名'])
    elif row['字段名'] == 'FLAG' and row['缺失率'] > 0: # Handle FLAG separately if needed
        print(f"Target variable 'FLAG' has {row['缺失值数量']} missing values. Consider imputation or removal.")
        # For now, let's assume FLAG missing values rows will be dropped or handled later
        # The notebook code puts FLAG into fill0_cols if < 70% missing
        if row['缺失率'] < 70:
             fill0_cols.append(row['字段名'])


df.drop(columns=dropna_cols, inplace=True)
df[fill0_cols] = df[fill0_cols].fillna(value=0) # axis=1 is for dropping, not fillna here

# Handle remaining NaNs in FLAG if not already handled by fill0_cols
if 'FLAG' in df.columns and df['FLAG'].isnull().sum() > 0:
    print(f"FLAG still has {df['FLAG'].isnull().sum()} NaNs. Dropping these rows for now.")
    df.dropna(subset=['FLAG'], inplace=True)

print(df.head())
print(df.shape)

## 噪声数据处理

# Create df0 from the cleaned df for transformations
df0 = df.copy()

# 分箱平滑
def bin_smooth(df_in, col, n_bins=5, method='mean'):
    df_out = df_in.copy()
    # Ensure the column is numeric and has no NaNs for pd.cut
    if pd.api.types.is_numeric_dtype(df_out[col]) and df_out[col].notnull().all():
        try:
            df_out[f'{col}_bin'] = pd.cut(df_out[col], bins=n_bins)
            
            if method == 'mean':
                smooth_values = df_out.groupby(f'{col}_bin')[col].transform('mean')
            elif method == 'median':
                smooth_values = df_out.groupby(f'{col}_bin')[col].transform('median')
            elif method == 'boundary': # This method doesn't make sense for filling
                print(f"Boundary method for bin_smooth on {col} is illustrative, not for filling.")
                return df_out # Return original df for this case or handle differently
            else:
                print(f"Unknown smoothing method: {method}")
                return df_out

            df_out[f'{col}_smooth'] = smooth_values
            df_out.drop(columns=[f'{col}_bin'], inplace=True) # Drop the temporary bin column
        except Exception as e:
            print(f"Error during bin_smooth for {col}: {e}")
    else:
        print(f"Column {col} is not suitable for bin_smooth (not numeric or contains NaNs).")
    return df_out


# 示例：对营业税金及附加进行分箱平滑
print(pd.cut(df0['BIZ_TAX_SURCHG'], bins=5)) # Show bins as in notebook
df0 = bin_smooth(df0, 'BIZ_TAX_SURCHG', n_bins=5, method='mean')
# data0_null = miss_data_count(df0) # For inspection, not strictly needed for flow
# print(data0_null[data0_null['字段名']=='TFA_TURNOVER']) # This field might have been dropped

# # 回归平滑
# # from sklearn.linear_model import LinearRegression
# # def regression_smooth(df, target_col, feature_cols):
# # # This was a raw cell and requires specific feature_cols which are not defined for this generic case.
# # print("Skipping regression_smooth example as it was in a raw cell and needs specific setup.")
# # # df0 = regression_smooth(df0, 'ACCOUNTS_RECEIVABLE', ['TOTAL_ASSETS', 'TOTAL_LIABILITIES'])

## 数据变换
# 移动平均光滑
def moving_average_smooth(df_in, col, window=3):
    df_out = df_in.copy()
    if col in df_out.columns and pd.api.types.is_numeric_dtype(df_out[col]):
        df_out[f'{col}_smooth'] = df_out[col].rolling(window=window, min_periods=1).mean()
    else:
        print(f"Column {col} not found or not numeric for moving_average_smooth.")
    return df_out

# 示例：对所有者权益进行移动平均光滑
df0 = moving_average_smooth(df0, 'T_SH_EQUITY', window=3)
print(df0.head())

# 数据泛化（将数值转换为类别）
def generalize_data(df_in, col, bins, labels):
    df_out = df_in.copy()
    if col in df_out.columns and pd.api.types.is_numeric_dtype(df_out[col]):
        df_out[f'{col}_generalized'] = pd.cut(df_out[col], bins=bins, labels=labels, right=False) # Added right=False for consistency if needed
    else:
        print(f"Column {col} not found or not numeric for generalize_data.")
    return df_out

# 示例：将净利润泛化为小、中、大三类
bins = [-float('inf'), 1e8, 1e9, float('inf')] # Adjusted bins to include negative profits if any
labels = ['小型企业', '中型企业', '大型企业']
df0 = generalize_data(df0, 'N_INCOME', bins, labels)
print(df0.head())


# 数值规约（对数变换）
def reduce_numerical(df_in, cols_reduce):
    df_out = df_in.copy()
    for col in cols_reduce:
        if col in df_out.columns and pd.api.types.is_numeric_dtype(df_out[col]):
            # Ensure non-negative values for log1p, or handle negatives appropriately
            if (df_out[col] < 0).any():
                print(f"Warning: Column {col} contains negative values. Log transform might not be appropriate or needs adjustment.")
                # Example: shift before log, if meaningful
                # df_out[f'log_{col}'] = np.log1p(df_out[col] - df_out[col].min())
                df_out[f'log_{col}'] = np.log1p(df_out[col].apply(lambda x: x if x >= 0 else 0)) # Simple handling
            else:
                df_out[f'log_{col}'] = np.log1p(df_out[col])
        else:
            print(f"Column {col} not found or not numeric for reduce_numerical.")
    return df_out

# 示例：对营业外收入进行对数变换
cols_to_reduce = ['NOPERATE_INCOME']
df0 = reduce_numerical(df0, cols_to_reduce)
print(df0[['NOPERATE_INCOME','log_NOPERATE_INCOME']])

# PCA降维
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def pca_reduction(df_in, cols_pca, n_components=0.95):
    df_out = df_in.copy()
    # Ensure all columns for PCA are numeric and have no NaNs
    numeric_pca_cols = [col for col in cols_pca if col in df_out.columns and pd.api.types.is_numeric_dtype(df_out[col])]
    
    if not numeric_pca_cols:
        print("No suitable numeric columns found for PCA.")
        return df_out, None

    X = df_out[numeric_pca_cols].fillna(0) # Simple NaN fill for PCA, consider more robust methods

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    
    pca_cols_names = [f'PC{i+1}' for i in range(X_pca.shape[1])]
    df_pca_out = pd.DataFrame(X_pca, columns=pca_cols_names, index=df_out.index)
    
    print(f'解释方差比例: {pca.explained_variance_ratio_}')
    print(f'累计解释方差比例: {sum(pca.explained_variance_ratio_)}')
    
    return df_pca_out

# 示例：对财务指标进行PCA降维
# Make sure df0.describe().columns only contains numeric columns or handle non-numeric ones
cols_for_pca = [col for col in df0.columns if pd.api.types.is_numeric_dtype(df0[col]) and col != 'FLAG']
df1_pca = pca_reduction(df0, cols_for_pca, n_components=0.95)
#1. **整数**：直接指定保留的主成分数量。例如，n_components=2表示保留前两个主成分。
#2. **浮点数（0到1之间）**：表示保留的方差比例。例如，n_components=0.95意味着保留95%的方差所需的主成分数量。
print(df0.shape, df1_pca.shape if df1_pca is not None else "PCA not performed")
print(df0.head())


## 特征选择
from sklearn.feature_selection import SelectKBest, f_classif

# 过滤式特征选择
def filter_feature_selection(X_fs, y_fs, k=10):
    # Ensure y_fs has no NaNs
    valid_indices = y_fs.notnull()
    X_fs_clean = X_fs[valid_indices].fillna(0) # Fill NaNs in X, or use a more sophisticated method
    y_fs_clean = y_fs[valid_indices]

    if X_fs_clean.empty or y_fs_clean.empty:
        print("Not enough data for feature selection after cleaning NaNs.")
        return None, []

    selector = SelectKBest(score_func=f_classif, k=min(k, X_fs_clean.shape[1])) # k cannot be > n_features
    try:
        X_new_fs = selector.fit_transform(X_fs_clean, y_fs_clean)
        selected_features_fs = X_fs_clean.columns[selector.get_support()]
        print(f'Selected features: {list(selected_features_fs)}')
        return X_new_fs, selected_features_fs
    except Exception as e:
        print(f"Error during feature selection: {e}")
        return None, []


# 示例：使用ANOVA F-value选择前10个特征
# Define drop_cols before using it
drop_cols=['TICKER_SYMBOL','INDUSTRY','ACCOUTING_STANDARDS','N_INCOME_generalized','REPORT_TYPE','CURRENCY_CD',
           'BIZ_TAX_SURCHG_bin', 'BIZ_TAX_SURCHG_smooth', 'T_SH_EQUITY_smooth', 'log_NOPERATE_INCOME'] # Added bin/smooth/log columns
# Also ensure that all columns in X are numeric and FLAG (target) is handled
cols_for_X = [col for col in df0.columns if col not in (['FLAG'] + drop_cols) and pd.api.types.is_numeric_dtype(df0[col])]

if 'FLAG' in df0.columns and df0['FLAG'].notnull().sum() > 0:
    X = df0[cols_for_X]
    y = df0['FLAG']
    X_new, selected_features = filter_feature_selection(X, y, k=10)
else:
    print("Target variable 'FLAG' is missing or all NaNs. Skipping feature selection.")

print("Script finished.")

# 导出数据到Excel文件
import os

# 确保output目录存在
output_dir = './output'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 导出原始数据
df.to_excel(f'{output_dir}/data.xlsx', sheet_name='原始数据', index=False)

# 如果需要，导出处理后的数据和PCA结果到不同Sheet
with pd.ExcelWriter(f'{output_dir}/data.xlsx', mode='a', engine='openpyxl') as writer:
    df0.to_excel(writer, sheet_name='处理后数据', index=False)
    if isinstance(df1_pca, pd.DataFrame) and not df1_pca.empty:
        df1_pca.to_excel(writer, sheet_name='PCA结果', index=False)
    
    # 导出缺失值统计
    data_null.to_excel(writer, sheet_name='缺失值统计', index=False)

print(f"数据已成功导出到 {output_dir}/data.xlsx")