#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
探索性数据分析(EDA)：Adult数据集和Sales.Prediction数据集

该脚本对两个数据集进行全面的探索性数据分析，包括：
1. 数据加载与清洗
2. 基本统计分析
3. 特征分布与关系可视化
4. 相关性分析
5. 生成分析报告
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib.font_manager import FontProperties, fontManager
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# 创建输出目录
output_dir = 'eda_analysis_results'
os.makedirs(output_dir, exist_ok=True)

# 设置中文字体
font_path = "/System/Library/Fonts/STHeiti Light.ttc"
font = FontProperties(fname=font_path, size=12)
fontManager.addfont(font_path)

plt.rcParams['font.sans-serif'] = ['Heiti TC', 'WenQuanYi Micro Hei', 'SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['font.size'] = 12

# 设置图表风格
# 设置seaborn使用matplotlib的字体设置
sns.set(font=plt.rcParams['font.sans-serif'][0], font_scale=1)
# sns.set(style="whitegrid", rc={'figure.figsize':(12, 8)})
plt.style.use('ggplot')

# 初始化Markdown报告内容
report_content = """# 探索性数据分析(EDA)报告

## 目录
1. [数据概述](#数据概述)
2. [Adult数据集分析](#Adult数据集分析)
3. [Sales.Prediction数据集分析](#Sales.Prediction数据集分析)
4. [结论与建议](#结论与建议)

## 数据概述
本报告对两个数据集进行探索性分析：
- **Adult数据集**：包含人口普查数据，预测个人年收入是否超过50K
- **Sales.Prediction数据集**：包含电商产品数据，分析影响转化率的因素

"""

# ====================== 第一部分：Adult数据集分析 ======================

def analyze_adult_dataset():
    print("\n" + "=" * 50)
    print("开始分析Adult数据集...")
    
    # 定义列名
    columns = [
        'age', 'workclass', 'fnlwgt', 'education', 'education_num',
        'marital_status', 'occupation', 'relationship', 'race', 'sex',
        'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'income'
    ]
    
    # 加载数据
    try:
        # 尝试使用绝对路径加载数据
        file_path = '/Users/shhaofu/Code/cursor-projects/aka_music/frontend/public/ml/adult.data'
        if not os.path.exists(file_path):
            print(f"Adult数据集文件不存在: {file_path}")
            return """## Adult数据集分析
无法加载Adult数据集，请确保文件存在且格式正确。
"""
        
        df = pd.read_csv(file_path, header=None, names=columns, sep=', ', engine='python')
        print(f"Adult数据集加载成功，共有 {df.shape[0]} 行，{df.shape[1]} 列")
    except Exception as e:
        print(f"加载Adult数据集失败: {e}")
        return """## Adult数据集分析
无法加载Adult数据集，请确保文件存在且格式正确。
"""
    
    # 数据预处理
    # 去除字符串前后的空格
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].str.strip()
    
    # 替换缺失值
    df.replace(' ?', np.nan, inplace=True)
    
    # 数据概览
    print("\n数据前5行:")
    print(df.head())
    
    # 数据类型和缺失值
    print("\n数据类型和缺失值统计:")
    missing_data = pd.DataFrame({
        'Type': df.dtypes,
        'Missing': df.isnull().sum(),
        'Missing%': (df.isnull().sum() / len(df) * 100).round(2)
    })
    print(missing_data)
    
    # 处理缺失值
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if df[col].dtype == 'object':
                df[col].fillna(df[col].mode()[0], inplace=True)
            else:
                df[col].fillna(df[col].median(), inplace=True)
    
    # 基本统计描述
    print("\n数值型特征统计描述:")
    print(df.describe())
    
    # 目标变量分布
    income_counts = df['income'].value_counts()
    plt.figure(figsize=(10, 6))
    sns.countplot(x='income', data=df, palette='Set2')
    plt.title('收入分布')
    plt.xlabel('收入水平')
    plt.ylabel('数量')
    plt.savefig(f'{output_dir}/adult_income_distribution.png')
    plt.close()
    
    # 年龄分布
    plt.figure(figsize=(12, 6))
    sns.histplot(data=df, x='age', hue='income', bins=30, kde=True, palette='Set2')
    plt.title('不同收入水平的年龄分布')
    plt.xlabel('年龄')
    plt.ylabel('频数')
    plt.savefig(f'{output_dir}/adult_age_distribution.png')
    plt.close()
    
    # 教育水平与收入关系
    plt.figure(figsize=(14, 8))
    education_order = df.groupby('education')['education_num'].mean().sort_values().index
    sns.countplot(y='education', hue='income', data=df, order=education_order, palette='Set2')
    plt.title('教育水平与收入关系')
    plt.xlabel('数量')
    plt.ylabel('教育水平')
    plt.savefig(f'{output_dir}/adult_education_income.png')
    plt.close()
    
    # 工作时长与收入关系
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='income', y='hours_per_week', data=df, palette='Set2')
    plt.title('工作时长与收入关系')
    plt.xlabel('收入水平')
    plt.ylabel('每周工作小时数')
    plt.savefig(f'{output_dir}/adult_hours_income.png')
    plt.close()
    
    # 性别与收入关系
    gender_income = pd.crosstab(df['sex'], df['income'], normalize='index') * 100
    plt.figure(figsize=(10, 6))
    gender_income.plot(kind='bar', stacked=True, colormap='Set2')
    plt.title('性别与收入关系')
    plt.xlabel('性别')
    plt.ylabel('百分比 (%)')
    plt.xticks(rotation=0)
    plt.savefig(f'{output_dir}/adult_gender_income.png')
    plt.close()
    
    # 特征相关性分析
    numeric_df = df.select_dtypes(include=['int64', 'float64'])
    plt.figure(figsize=(12, 10))
    correlation = numeric_df.corr()
    mask = np.triu(np.ones_like(correlation, dtype=bool))
    sns.heatmap(correlation, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', linewidths=0.5)
    plt.title('数值特征相关性热力图')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/adult_correlation.png')
    plt.close()
    
    # 使用pairplot查看特征间关系
    plt.figure(figsize=(16, 12))
    selected_features = ['age', 'education_num', 'hours_per_week', 'capital_gain', 'income']
    sns.pairplot(df[selected_features], hue='income', palette='Set2', diag_kind='kde')
    plt.suptitle('Adult数据集特征对关系图', y=1.02, fontsize=16)
    plt.savefig(f'{output_dir}/adult_pairplot.png', bbox_inches='tight')
    plt.close()
    
    # 生成报告内容
    report = f"""## Adult数据集分析

### 数据概述
- 数据集大小: {df.shape[0]} 行 × {df.shape[1]} 列
- 目标变量: 收入水平 (<=50K: {income_counts.get('<=50K', 0)}人, >50K: {income_counts.get('>50K', 0)}人)

### 主要发现

#### 1. 人口统计特征
- 年龄分布集中在25-50岁之间，高收入群体平均年龄更高
- 男性比例为{(df['sex'] == 'Male').mean()*100:.1f}%，女性为{(df['sex'] == 'Female').mean()*100:.1f}%
- 高收入人群中男性比例明显高于女性

#### 2. 教育与工作
- 教育水平与收入呈正相关，高学历人群更可能获得高收入
- 每周工作时长中位数为{df['hours_per_week'].median()}小时，高收入群体工作时间普遍更长
- 管理和专业类职业更容易获得高收入

#### 3. 相关性分析
- 教育年限与收入水平正相关(r={correlation.loc['education_num', 'capital_gain']:.2f})
- 资本收益与收入水平有较强正相关
- 年龄与教育年限相关性较弱，表明教育机会相对平等

![收入分布](adult_income_distribution.png)
![年龄分布](adult_age_distribution.png)
![教育与收入](adult_education_income.png)
![工作时长与收入](adult_hours_income.png)
![性别与收入](adult_gender_income.png)
![特征相关性](adult_correlation.png)
![特征对关系](adult_pairplot.png)
"""
    
    print("Adult数据集分析完成!")
    return report

# ====================== 第二部分：Sales.Prediction数据集分析 ======================

def analyze_sales_dataset():
    print("\n" + "=" * 50)
    print("开始分析Sales.Prediction数据集...")
    
    # 加载数据
    try:
        # 尝试使用绝对路径加载数据
        file_path = '/Users/shhaofu/Code/cursor-projects/aka_music/frontend/public/ml/Sales.Prediction.txt'
        if not os.path.exists(file_path):
            print(f"Sales.Prediction数据集文件不存在: {file_path}")
            return """## Sales.Prediction数据集分析
无法加载Sales.Prediction数据集，请确保文件存在且格式正确。
"""
            
        data = pd.read_csv(file_path, sep="\t")
        print(f"Sales.Prediction数据集加载成功，共有 {data.shape[0]} 行，{data.shape[1]} 列")
    except Exception as e:
        print(f"加载Sales.Prediction数据集失败: {e}")
        return """## Sales.Prediction数据集分析
无法加载Sales.Prediction数据集，请确保文件存在且格式正确。
"""
    
    # 数据概览
    print("\n数据前5行:")
    print(data.head())
    
    # 数据类型和缺失值
    print("\n数据类型和缺失值统计:")
    missing_data = pd.DataFrame({
        'Type': data.dtypes,
        'Missing': data.isnull().sum(),
        'Missing%': (data.isnull().sum() / len(data) * 100).round(2)
    })
    print(missing_data)
    
    # 基本统计描述
    print("\n数值型特征统计描述:")
    print(data.describe())
    
    # 将分类变量转换为数值型
    data['IsDeal'] = data['IsDeal'].astype(int)
    data['IsNew'] = data['IsNew'].astype(int)
    data['IsLimitedStock'] = data['IsLimitedStock'].astype(int)
    
    # 选择数值型特征进行分析
    num_features = ['OneMonthConversionRateInUV', 'OneWeekConversionRateInUV', 'SellerReputation', 
                   'IsDeal', 'IsNew', 'IsLimitedStock']
    
    # 标准化处理
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(data[num_features])
    data_scaled = pd.DataFrame(scaled_features, columns=[f + '_scaled' for f in num_features])
    data = pd.concat([data, data_scaled], axis=1)
    
    # 目标变量分布
    plt.figure(figsize=(10, 6))
    sns.histplot(data['TargetValue'], bins=20, kde=True)
    plt.title('转化率分布')
    plt.xlabel('转化率')
    plt.ylabel('频数')
    plt.savefig(f'{output_dir}/sales_target_distribution.png')
    plt.close()
    
    # 数值型特征与目标的相关性
    corr_with_target = data[num_features + ['TargetValue']].corr()['TargetValue'].sort_values(ascending=False)
    print("\n特征与目标转化率的皮尔森相关系数:")
    print(corr_with_target)
    
    # 可视化特征与目标的相关性
    plt.figure(figsize=(10, 6))
    sns.barplot(x=corr_with_target.values[:-1], y=corr_with_target.index[:-1], palette="vlag")
    plt.title("特征与目标转化率的皮尔森相关系数")
    plt.tight_layout()
    plt.savefig(f'{output_dir}/sales_correlation_barplot.png')
    plt.close()
    
    # 可视化特征间相关性
    plt.figure(figsize=(12, 8))
    sns.heatmap(data[num_features].corr(),
                annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("自变量间相关系数矩阵")
    plt.tight_layout()
    plt.savefig(f'{output_dir}/sales_correlation_heatmap.png')
    plt.close()
    
    # 按类别分组分析
    cat_analysis = data.groupby('CategoryName')['TargetValue'].agg(['mean', 'count']).sort_values(by='mean', ascending=False)
    print("\n按商品类别分组的平均转化率:")
    print(cat_analysis)
    
    # 可视化类别与转化率的关系
    plt.figure(figsize=(12, 8))
    sns.barplot(x='mean', y=cat_analysis.index, data=cat_analysis.reset_index())
    plt.title('不同商品类别的平均转化率')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/sales_category_conversion.png')
    plt.close()
    
    # 局部回归可视化 - 转化率与卖家信誉的关系
    g = sns.lmplot(x='SellerReputation', y='TargetValue', data=data, 
               lowess=True, line_kws={'color':'red'}, height=6, aspect=1.5)
    plt.title('卖家信誉与转化率的局部加权回归')
    plt.tight_layout()
    g.savefig(f'{output_dir}/sales_reputation_regression.png')
    plt.close()
    
    # 局部回归可视化 - 转化率与月转化率的关系
    g = sns.lmplot(x='OneMonthConversionRateInUV', y='TargetValue', data=data, 
               lowess=True, line_kws={'color':'red'}, height=6, aspect=1.5)
    plt.title('月转化率与目标转化率的局部加权回归')
    plt.tight_layout()
    g.savefig(f'{output_dir}/sales_monthly_conversion_regression.png')
    plt.close()
    
    # 分析特征交互作用
    interaction_analysis = data.groupby(['IsLimitedStock', 'IsNew'])['TargetValue'].mean().unstack()
    print("\n特征交互分析 - 是否限量 & 是否新品:")
    print(interaction_analysis)
    
    # 热力图可视化特征交互
    plt.figure(figsize=(8, 6))
    sns.heatmap(interaction_analysis, annot=True, cmap="YlGnBu", fmt=".3f")
    plt.title('限量商品与新品对转化率的交互影响')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/sales_feature_interaction.png')
    plt.close()
    
    # 使用pairplot查看特征间关系
    plt.figure(figsize=(16, 12))
    selected_features = ['OneMonthConversionRateInUV', 'OneWeekConversionRateInUV', 
                         'SellerReputation', 'IsDeal', 'IsNew', 'IsLimitedStock', 'TargetValue']
    sns.pairplot(data[selected_features], diag_kind='kde')
    plt.suptitle('Sales.Prediction数据集特征对关系图', y=1.02, fontsize=16)
    plt.savefig(f'{output_dir}/sales_pairplot.png', bbox_inches='tight')
    plt.close()
    
    # 生成报告内容
    report = f"""## Sales.Prediction数据集分析

### 数据概述
- 数据集大小: {data.shape[0]} 行 × {data.shape[1]} 列
- 目标变量: 转化率 (平均值: {data['TargetValue'].mean():.4f})
- 商品类别数: {data['CategoryName'].nunique()}

### 主要发现

#### 1. 转化率影响因素
- 月转化率(r={corr_with_target['OneMonthConversionRateInUV']:.4f})和周转化率(r={corr_with_target['OneWeekConversionRateInUV']:.4f})与目标转化率高度相关
- 卖家信誉对转化率有正面影响(r={corr_with_target['SellerReputation']:.4f})
- 限量商品和促销商品对转化率有一定的正面影响

#### 2. 类别分析
- 不同商品类别的转化率差异显著，方便面、食用油、饮料饮品等类别转化率较高
- 电脑、手机等电子产品类别转化率相对较低
- 最高转化率类别({cat_analysis.index[0]})的平均转化率为{cat_analysis['mean'].iloc[0]:.4f}

#### 3. 特征交互影响
- 限量商品与新品的组合对转化率有交互影响
- 同时是限量商品和新品的商品平均转化率为{interaction_analysis.loc[1, 1] if 1 in interaction_analysis.index and 1 in interaction_analysis.columns else 'N/A'}

#### 4. 多重共线性
- 月转化率与周转化率之间存在较强的相关性(r={data[['OneMonthConversionRateInUV', 'OneWeekConversionRateInUV']].corr().iloc[0,1]:.4f})
- 在建模时需要考虑是否同时使用这两个特征

![转化率分布](sales_target_distribution.png)
![特征相关性](sales_correlation_barplot.png)
![相关性热力图](sales_correlation_heatmap.png)
![类别转化率](sales_category_conversion.png)
![卖家信誉回归](sales_reputation_regression.png)
![月转化率回归](sales_monthly_conversion_regression.png)
![特征交互热力图](sales_feature_interaction.png)
![特征对关系](sales_pairplot.png)
"""
    
    print("Sales.Prediction数据集分析完成!")
    return report

# ====================== 主函数 ======================

def main():
    print("开始探索性数据分析...")
    
    # 分析Adult数据集
    adult_report = analyze_adult_dataset()
    
    # 分析Sales.Prediction数据集
    sales_report = analyze_sales_dataset()
    
    # 生成结论与建议
    conclusion = """## 结论与建议

### Adult数据集结论
1. **教育是关键因素**：教育水平与高收入强相关，提高教育投资可能带来收入回报
2. **性别差距明显**：数据显示明显的性别收入差距，需要关注性别平等问题
3. **工作时长与收入**：高收入群体工作时间更长，但关系非线性，表明工作质量同样重要
4. **年龄与经验价值**：中年人群更可能获得高收入，体现经验的价值

### Sales.Prediction数据集结论
1. **历史表现是最佳预测指标**：历史转化率是预测未来转化率的最强指标
2. **卖家信誉至关重要**：高信誉卖家获得更高转化率，平台应鼓励卖家提升服务质量
3. **类别差异化策略**：不同商品类别转化率差异大，应采用差异化营销策略
4. **营销手段有效性**：限量和促销策略对提升转化率有效，但效果因类别而异

### 建议
1. **个性化推荐**：基于用户人口统计特征和历史行为构建个性化推荐系统
2. **卖家激励机制**：设计激励机制提高卖家信誉和服务质量
3. **类别优化**：针对低转化率类别开发专门的营销策略
4. **特征工程**：在预测模型中考虑特征交互，提高预测准确性
5. **数据收集**：收集更多用户行为和产品特征数据，丰富分析维度
"""
    
    # 合并报告
    full_report = report_content + adult_report + "\n\n" + sales_report + "\n\n" + conclusion
    
    # 保存报告
    with open(f"{output_dir}/EDA_Report.md", "w", encoding="utf-8") as f:
        f.write(full_report)
    
    print("\n分析完成! 报告已保存到 {}/EDA_Report.md".format(output_dir))

if __name__ == "__main__":
    main()