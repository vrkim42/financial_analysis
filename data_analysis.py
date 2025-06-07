import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
import joblib
import argparse
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler

# 配置
warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['fangsong']  # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示异常问题

class FinancialDataPreprocessor:
    def __init__(self, data_paths, config=None):
        """
        初始化财务数据预处理器
        参数:
        data_paths (dict): 数据路径字典
        config (dict): 配置参数字典，包含预处理参数和阈值设置
        """
        self.data_paths = data_paths
        self.fields = {}  # 字段含义字典
        self.df = None  # 主数据集
        self.processed_df = None  # 处理后的数据
        
        # 默认配置
        self.config = {
            'missing_value_threshold': 70,  # 缺失值删除阈值(%)
            'pca_n_components': 0.95,       # PCA解释方差比例
            'feature_selection_k': 10,      # 特征选择数量
            'plot_style': 'seaborn',        # 图表风格
            'bin_smooth_method': 'mean',    # 分箱平滑方法
            'industry_filter': '制造业'      # 行业筛选条件
        }
        
        # 更新自定义配置
        if config:
            self.config.update(config)
            
        # 设置绘图风格
        plt.style.use(self.config['plot_style'])
        
    def load_and_merge_data(self):
        """加载并合并数据"""
        # 1. 加载数据
        df_category = pd.read_csv(self.data_paths['category'], encoding='utf-8', on_bad_lines='skip')
        df_financial = pd.read_csv(self.data_paths['financial'], encoding='utf-8', on_bad_lines='skip')
        df_fields = pd.read_excel(self.data_paths['fields'])
        
        # 创建字段含义字典
        self.fields = dict(zip(df_fields['字段名'], zip(df_fields['含义'], df_fields['单位'])))
        
        # 2. 数据合并
        df_category.columns = ['TICKER_SYMBOL', '所属行业']
        self.df = pd.merge(df_financial, df_category, how='left', on=['TICKER_SYMBOL'])
        self.df.insert(1, 'INDUSTRY', self.df.pop('所属行业'))
        self.fields['INDUSTRY'] = ('所属行业', np.nan)
        
        # 3. 筛选制造业数据
        self.df = self.df[self.df['INDUSTRY'] == '制造业'].drop_duplicates().reset_index(drop=True)
        return self.df
    
    def exploratory_data_analysis(self, output_dir='plots'):
        """
        执行探索性数据分析(EDA)
        
        参数:
        output_dir (str): 输出图表目录
        """
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 打印基本信息
        print("数据形状:", self.df.shape)
        print("财务造假分布:\n", self.df['FLAG'].value_counts())
        print("财务造假占比:", self.df['FLAG'].value_counts(normalize=True))
        
        # 数据类型分布
        dtypes_count = self.df.dtypes.value_counts()
        print("数据类型分布:\n", dtypes_count)
        
        # 选择部分数值型字段进行可视化
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [col for col in numeric_cols if col != 'FLAG' and self.df[col].nunique() > 5]
        
        # 如果数值列太多，只选择前10个
        if len(numeric_cols) > 10:
            numeric_cols = numeric_cols[:10]
        
        # 生成每个字段的直方图和箱线图
        for col in numeric_cols:
            self._plot_distribution(col, output_dir)
        
        # 数值型变量分布可视化
        self._plot_numeric_distributions(numeric_cols, output_dir)
        
        # 缺失值分析
        self._analyze_missing_values(output_dir)
        
        # 相关性分析
        self._plot_correlation_matrix(output_dir)
        
        # 分类变量分析
        categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
        if categorical_cols:
            self._analyze_categorical_features(categorical_cols, output_dir)
        
    def _handle_outliers(self, df):
        """处理异常值"""
        print("处理异常值...")
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        # 排除FLAG和ID类字段
        numeric_cols = [col for col in numeric_cols if col not in ['FLAG', 'TICKER_SYMBOL']]
        
        for col in numeric_cols:
            if df[col].count() < 10:  # 数据太少，跳过
                continue
                
            # 计算四分位数和IQR
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            # 定义异常值边界
            lower_bound = Q1 - 3 * IQR
            upper_bound = Q3 + 3 * IQR
            
            # 计算异常值数量
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col].count()
            if outliers > 0:
                print(f"  字段 {col} 找到 {outliers} 个异常值")
                
                # 创建一个新的列存储原始值
                df[f'{col}_original'] = df[col].copy()
                
                # 截断异常值
                df.loc[df[col] < lower_bound, col] = lower_bound
                df.loc[df[col] > upper_bound, col] = upper_bound
        
        return df
    
    def preprocess_data(self):
        """执行数据预处理"""
        if self.df is None:
            raise ValueError("请先加载数据")
            
        print("开始数据预处理...")
        df_processed = self.df.copy()
        
        # 1. 缺失值处理
        df_processed = self._handle_missing_values(df_processed)
        
        # 2. 异常值检测与处理
        df_processed = self._handle_outliers(df_processed)
        
        # 3. 噪声处理
        smooth_cols = ['BIZ_TAX_SURCHG', 'T_SH_EQUITY', 'PROFIT_TOTAL', 'OPERATE_PROFIT']
        for col in smooth_cols:
            if col in df_processed.columns:
                if df_processed[col].nunique() > 5:  # 确保有足够的唯一值进行分箱
                    df_processed = self._bin_smooth(df_processed, col, n_bins=5, 
                                                   method=self.config['bin_smooth_method'])
                df_processed = self._moving_average_smooth(df_processed, col, window=3)
        
        # 4. 数据泛化
        generalize_configs = [
            {'col': 'N_INCOME', 'bins': [-float('inf'), 1e8, 1e9, float('inf')], 
             'labels': ['小型企业', '中型企业', '大型企业']},
            {'col': 'TREV', 'bins': [-float('inf'), 5e8, 2e9, float('inf')], 
             'labels': ['低营收', '中等营收', '高营收']}
        ]
        
        for config in generalize_configs:
            if config['col'] in df_processed.columns:
                df_processed = self._generalize_data(df_processed, **config)
        
        # 5. 数值规约
        log_transform_cols = ['NOPERATE_INCOME', 'PROFIT_TOTAL', 'OPERATE_PROFIT', 'T_ASSETS']
        log_transform_cols = [col for col in log_transform_cols if col in df_processed.columns]
        df_processed = self._reduce_numerical(df_processed, log_transform_cols)
        
        self.processed_df = df_processed
        print("数据预处理完成")
        return df_processed
    
    def feature_engineering(self):
        """执行特征工程"""
        if self.processed_df is None:
            raise ValueError("请先预处理数据")
        
        print("开始特征工程...")
        df_fe = self.processed_df.copy()
        
        # 1. 创建一些比率特征
        if all(col in df_fe.columns for col in ['T_ASSETS', 'T_LIABILITY']):
            df_fe['资产负债率'] = df_fe['T_LIABILITY'] / df_fe['T_ASSETS']
            
        if all(col in df_fe.columns for col in ['PROFIT_TOTAL', 'T_ASSETS']):
            df_fe['资产收益率'] = df_fe['PROFIT_TOTAL'] / df_fe['T_ASSETS']
            
        if all(col in df_fe.columns for col in ['PROFIT_TOTAL', 'T_SH_EQUITY']):
            df_fe['净资产收益率'] = df_fe['PROFIT_TOTAL'] / df_fe['T_SH_EQUITY']
            
        # 2. 特征交互
        if all(col in df_fe.columns for col in ['T_ASSETS', 'T_LIABILITY', 'T_SH_EQUITY']):
            df_fe['财务杠杆'] = df_fe['T_ASSETS'] / df_fe['T_SH_EQUITY']
        
        # 3. 数据降维
        numeric_cols = [col for col in df_fe.columns 
                        if pd.api.types.is_numeric_dtype(df_fe[col]) 
                        and col != 'FLAG']
        pca_result = self._pca_reduction(df_fe, numeric_cols, self.config['pca_n_components'])
        
        # 4. 特征选择
        drop_cols = ['TICKER_SYMBOL', 'INDUSTRY', 'ACCOUTING_STANDARDS', 'N_INCOME_generalized', 
                    'REPORT_TYPE', 'CURRENCY_CD'] + [col for col in df_fe.columns if '_bin' in col or '_smooth' in col or '_original' in col]
        
        feature_cols = [col for col in df_fe.columns 
                       if col not in (['FLAG'] + drop_cols) 
                       and pd.api.types.is_numeric_dtype(df_fe[col])]
        
        if 'FLAG' in df_fe.columns and df_fe['FLAG'].notnull().sum() > 0:
            X = df_fe[feature_cols]
            y = df_fe['FLAG']
            selected_features = self._filter_feature_selection(X, y, self.config['feature_selection_k'])
            
            # 保存特征工程后的数据
            self.processed_df = df_fe
            
            return pca_result, selected_features
        
        # 保存特征工程后的数据
        self.processed_df = df_fe
        
        return pca_result, None
    
    def export_results(self, output_dir='output'):
        """导出处理结果"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 导出到Excel
        with pd.ExcelWriter(f'{output_dir}/processed_data.xlsx') as writer:
            if self.df is not None:
                self.df.to_excel(writer, sheet_name='原始数据', index=False)
            
            if self.processed_df is not None:
                self.processed_df.to_excel(writer, sheet_name='处理后数据', index=False)
                
                # 导出数据描述性统计
                stats_df = self.processed_df.describe(include='all').T
                stats_df['非空值数'] = self.processed_df.count()
                stats_df['缺失值数'] = self.processed_df.shape[0] - stats_df['非空值数']
                stats_df['缺失率'] = stats_df['缺失值数'] / self.processed_df.shape[0]
                stats_df['数据类型'] = self.processed_df.dtypes
                stats_df.to_excel(writer, sheet_name='数据统计', index=True)
        
        # 保存字段含义字典
        joblib.dump(self.fields, f'{output_dir}/fields_dict.pkl')
        
        # 保存处理后数据的CSV格式
        self.processed_df.to_csv(f'{output_dir}/processed_data.csv', index=False)
        
        # 生成HTML分析报告
        try:
            from pandas_profiling import ProfileReport
            profile = ProfileReport(self.processed_df, title="财务数据分析报告", minimal=True)
            profile.to_file(f"{output_dir}/data_profile_report.html")
            print(f"生成了数据分析HTML报告")
        except ImportError:
            print("未安装pandas_profiling，跳过生成HTML报告")
        
        print(f"结果已导出到 {output_dir} 目录")
    
    #辅助工具函数
    def _plot_distribution(self, col, output_dir):
        """绘制单个字段的分布图"""
        plt.figure(figsize=(8, 4))
        plt.subplot(1, 2, 1)
        sns.histplot(self.df[col], kde=True)
        plt.title(f'{self.fields[col][0]}直方图')
        
        plt.subplot(1, 2, 2)
        sns.boxplot(x=self.df[col])
        plt.title(f'{self.fields[col][0]}箱线图')
        
        plt.savefig(f'{output_dir}/{col}_dist.png')
        plt.close()
    
    def _plot_numeric_distributions(self, cols, output_dir, n_cols=4):
        """绘制多个数值变量的分布"""
        n_rows = (len(cols) + n_cols - 1) // n_cols
        plt.figure(figsize=(20, 5 * n_rows))
        
        for i, col in enumerate(cols):
            plt.subplot(n_rows, n_cols, i + 1)
            sns.histplot(self.df[col], kde=True, bins=30)
            plt.title(f'{self.fields[col][0]}分布', fontsize=12)
            plt.xlabel('')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/numeric_distributions.png')
        plt.close()
    
    def _analyze_categorical_features(self, cols, output_dir):
        """分析分类特征"""
        if not cols:
            return
            
        print("分析分类特征...")
        
        # 创建一个图表来显示每个分类特征的值分布
        for col in cols:
            if col in self.df.columns:
                value_counts = self.df[col].value_counts()
                
                # 如果唯一值太多，只展示前10个
                if len(value_counts) > 10:
                    value_counts = value_counts.iloc[:10]
                
                plt.figure(figsize=(10, 6))
                sns.barplot(x=value_counts.index, y=value_counts.values)
                plt.title(f'{col}分类分布')
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                plt.savefig(f'{output_dir}/{col}_category_dist.png')
                plt.close()
                
                # 如果存在FLAG列，分析分类特征与FLAG的关系
                if 'FLAG' in self.df.columns:
                    try:
                        plt.figure(figsize=(12, 6))
                        crosstab = pd.crosstab(self.df[col], self.df['FLAG'], normalize='index')
                        crosstab.plot(kind='bar', stacked=True)
                        plt.title(f'{col}与造假标记的关系')
                        plt.legend(title='FLAG')
                        plt.tight_layout()
                        plt.savefig(f'{output_dir}/{col}_vs_flag.png')
                        plt.close()
                    except Exception as e:
                        print(f"分析{col}与FLAG关系时出错: {e}")
    
    def _plot_correlation_matrix(self, output_dir):
        """绘制相关性矩阵"""
        # 选择数值型列
        numeric_df = self.df.select_dtypes(include=[np.number])
        # 填充缺失值，否则相关性计算会有问题
        numeric_df = numeric_df.fillna(0)
        
        # 如果列太多，选择方差最大的前20个
        if numeric_df.shape[1] > 20:
            variances = numeric_df.var().sort_values(ascending=False)
            top_cols = variances.index[:20]
            numeric_df = numeric_df[top_cols]
        
        # 计算相关性矩阵
        corr = numeric_df.corr()
        
        # 绘制热力图
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr, dtype=bool))
        cmap = sns.diverging_palette(230, 20, as_cmap=True)
        
        sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
                    square=True, linewidths=.5, annot=False, fmt='.2f')
        
        plt.title('特征相关性矩阵', fontsize=15)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/correlation_matrix.png', dpi=300)
        plt.close()
        
        # 保存强相关特征
        strong_corr = pd.DataFrame()
        for col in corr.columns:
            strong_corr_features = corr.index[(corr[col] > 0.7) | (corr[col] < -0.7)]
            strong_corr_features = strong_corr_features[strong_corr_features != col]
            if not strong_corr_features.empty:
                for feature in strong_corr_features:
                    strong_corr = pd.concat([strong_corr, pd.DataFrame({
                        '特征1': [col], 
                        '特征2': [feature], 
                        '相关系数': [corr.loc[feature, col]]
                    })])
        
        if not strong_corr.empty:
            strong_corr = strong_corr.sort_values('相关系数', ascending=False)
            strong_corr.to_csv(f'{output_dir}/strong_correlations.csv', index=False)

    def _analyze_missing_values(self, output_dir):
        """分析并可视化缺失值"""
        # 缺失值统计
        data_null = self._miss_data_count(self.df)
        print("缺失值统计表:\n", data_null)
        
        # 缺失值可视化
        try:
            import missingno as msno
            msno.matrix(self.df, figsize=(15, 10))
            plt.savefig(f'{output_dir}/missing_values_matrix.png')
            plt.close()
        except ImportError:
            print("未安装missingno库，跳过缺失值可视化")
    
    def _miss_data_count(self, data):
        """生成缺失值统计表"""
        null_count = data.isnull().sum().to_frame('缺失值数量')
        null_count['缺失率(%)'] = (null_count['缺失值数量'] / len(data) * 100).round(2)
        null_count['数据类型'] = data.dtypes
        return null_count.reset_index().rename(columns={'index': '字段名'})
    
    def _handle_missing_values(self, df):
        """处理缺失值"""
        data_null = self._miss_data_count(df)
        
        # 确定处理策略
        drop_cols = data_null[data_null['缺失率(%)'] >= 70]['字段名'].tolist()
        fill0_cols = data_null[(data_null['缺失率(%)'] > 0) & (data_null['缺失率(%)'] < 70)]['字段名'].tolist()
        
        # 执行处理
        df = df.drop(columns=drop_cols)
        df[fill0_cols] = df[fill0_cols].fillna(0)
        
        print(f"删除了 {len(drop_cols)} 个高缺失率字段")
        print(f"填充了 {len(fill0_cols)} 个字段的缺失值")
        return df
    
    def _bin_smooth(self, df, col, n_bins=5, method='mean'):
        """分箱平滑处理"""
        if pd.api.types.is_numeric_dtype(df[col]) and df[col].notnull().all():
            try:
                df[f'{col}_bin'] = pd.cut(df[col], bins=n_bins)
                
                if method == 'mean':
                    smooth_values = df.groupby(f'{col}_bin')[col].transform('mean')
                elif method == 'median':
                    smooth_values = df.groupby(f'{col}_bin')[col].transform('median')
                else:
                    return df
                
                df[f'{col}_smooth'] = smooth_values
            except Exception as e:
                print(f"分箱平滑处理出错: {e}")
        return df
    
    def _moving_average_smooth(self, df, col, window=3):
        """移动平均光滑处理"""
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            df[f'{col}_smooth'] = df[col].rolling(window=window, min_periods=1).mean()
        return df
    
    def _generalize_data(self, df, col, bins, labels):
        """数据泛化处理"""
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            df[f'{col}_generalized'] = pd.cut(df[col], bins=bins, labels=labels, right=False)
        return df
    
    def _reduce_numerical(self, df, cols):
        """数值规约处理"""
        for col in cols:
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                # 处理负值问题
                if (df[col] < 0).any():
                    df[f'log_{col}'] = np.log1p(df[col].apply(lambda x: x if x >= 0 else 0))
                else:
                    df[f'log_{col}'] = np.log1p(df[col])
        return df
    
    def _pca_reduction(self, df, cols, n_components=0.95):
        """PCA降维"""
        numeric_cols = [col for col in cols if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]
        
        if not numeric_cols:
            print("没有找到适合PCA的数值型字段")
            return None
        
        X = df[numeric_cols].fillna(0)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X_scaled)
        
        pca_cols = [f'PC{i+1}' for i in range(X_pca.shape[1])]
        df_pca = pd.DataFrame(X_pca, columns=pca_cols, index=df.index)
        
        print(f'主成分方差解释比例: {pca.explained_variance_ratio_}')
        print(f'累计方差解释比例: {sum(pca.explained_variance_ratio_)}')
        
        return df_pca
    
    def _filter_feature_selection(self, X, y, k=10):
        """过滤式特征选择"""
        valid_indices = y.notnull()
        X_clean = X[valid_indices].fillna(0)
        y_clean = y[valid_indices]
        
        if X_clean.empty or y_clean.empty:
            print("清洗后数据不足，无法进行特征选择")
            return []
        
        selector = SelectKBest(score_func=f_classif, k=min(k, X_clean.shape[1]))
        try:
            selector.fit(X_clean, y_clean)
            selected_features = X.columns[selector.get_support()]
            print(f'筛选出的重要特征: {list(selected_features)}')
            return selected_features
        except Exception as e:
            print(f"特征选择出错: {e}")
            return []
    
# 主函数
def main():
    # 默认配置
    config = {
        'missing_value_threshold': 70,
        'pca_n_components': 0.95,
        'feature_selection_k': 10,
        'plot_style': 'seaborn-v0_8-whitegrid',
        'bin_smooth_method': 'mean',
        'industry_filter': '制造业'
    }
    
    # 命令行参数解析
    import argparse
    parser = argparse.ArgumentParser(description='财务数据分析工具')
    parser.add_argument('--data_dir', type=str, default='data', help='数据文件目录')
    parser.add_argument('--output_dir', type=str, default='output', help='输出目录')
    parser.add_argument('--plots_dir', type=str, default='plots', help='图表输出目录')
    args = parser.parse_args()
    
    # 配置数据路径
    data_paths = {
        'category': f'{args.data_dir}/data1.csv',
        'financial': f'{args.data_dir}/data2.csv',
        'fields': f'{args.data_dir}/data3.xlsx'
    }
    
    # 创建预处理器实例
    preprocessor = FinancialDataPreprocessor(data_paths, config)
    
    try:
        # 1. 加载并合并数据
        raw_data = preprocessor.load_and_merge_data()
        print("数据加载完成，形状:", raw_data.shape)
        
        # 2. 探索性数据分析
        preprocessor.exploratory_data_analysis(args.plots_dir)
        
        # 3. 数据预处理
        processed_data = preprocessor.preprocess_data()
        print("数据预处理完成，形状:", processed_data.shape)
        
        # 4. 特征工程
        pca_result, selected_features = preprocessor.feature_engineering()
        if pca_result is not None:
            print("PCA降维结果形状:", pca_result.shape)
        if selected_features is not None and len(selected_features) > 0:
            print("筛选的特征:")
            for i, feature in enumerate(selected_features, 1):
                print(f"  {i}. {feature}")
        
        # 5. 导出结果
        preprocessor.export_results(args.output_dir)
        print("处理流程完成")
        
    except Exception as e:
        print(f"处理过程中出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
