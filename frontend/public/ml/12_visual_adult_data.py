#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
探索性数据分析：Adult数据集可视化

该脚本对Adult数据集进行探索性数据分析和可视化，
分析人口统计学特征与收入之间的关系。
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
from matplotlib.font_manager import FontProperties, fontManager

# 设置中文字体支持
# macOS 系统字体路径
font_paths = [
    "/System/Library/Fonts/STHeiti Light.ttc",
    "/System/Library/Fonts/PingFang.ttc",
    "/System/Library/Fonts/Supplemental/Songti.ttc"
]

# 添加字体到 Matplotlib
for font_path in font_paths:
    try:
        font = FontProperties(fname=font_path, size=12)
        fontManager.addfont(font_path)
        print(f"成功加载字体: {font_path}")
    except Exception as e:
        print(f"加载字体失败 {font_path}: {e}")

# 设置matplotlib参数
plt.rcParams['font.sans-serif'] = ['PingFang TC', 'Songti TC', 'Heiti TC', 'WenQuanYi Micro Hei', 'SimHei', 'Microsoft YaHei']  # 按优先级设置中文字体
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号
plt.rcParams['font.family'] = 'sans-serif'  # 使用无衬线字体族
plt.rcParams['axes.titlesize'] = 14  # 标题字体大小
plt.rcParams['font.size'] = 12  # 全局字体大小

# 设置seaborn使用matplotlib的字体设置
sns.set(font=plt.rcParams['font.sans-serif'][0], font_scale=1)

# 设置图表风格
sns.set(style="whitegrid", rc={'figure.figsize':(12, 8)})
plt.style.use('ggplot')

# 确保seaborn图表中的中文显示正常
sns.set_context("notebook", font_scale=1.2)
# 设置seaborn图表标题和标签的字体
mpl.rcParams['axes.labelsize'] = 14
mpl.rcParams['axes.titlesize'] = 16
mpl.rcParams['xtick.labelsize'] = 12
mpl.rcParams['ytick.labelsize'] = 12

# 修复seaborn中文显示问题的函数
def fix_seaborn_chinese(fig=None):
    """修复seaborn图表中的中文显示问题"""
    if fig is None:
        fig = plt.gcf()
    for ax in fig.axes:
        # 设置标题和标签字体
        if ax.get_title():
            ax.set_title(ax.get_title(), fontproperties=font)
        if ax.get_xlabel():
            ax.set_xlabel(ax.get_xlabel(), fontproperties=font)
        if ax.get_ylabel():
            ax.set_ylabel(ax.get_ylabel(), fontproperties=font)
        # 设置刻度标签字体
        for label in ax.get_xticklabels():
            label.set_fontproperties(font)
        for label in ax.get_yticklabels():
            label.set_fontproperties(font)
        # 设置图例字体
        if ax.get_legend() is not None:
            for text in ax.get_legend().get_texts():
                text.set_fontproperties(font)

# 定义列名
columns = [
    'age', 'workclass', 'fnlwgt', 'education', 'education_num',
    'marital_status', 'occupation', 'relationship', 'race', 'sex',
    'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'income'
]

# 加载数据
df = pd.read_csv('adult.data', header=None, names=columns, sep=', ', engine='python')

# 数据预处理
def preprocess_data(df):
    # 去除字符串前后的空格
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].str.strip()
    
    # 将收入转换为二元变量
    df['income_binary'] = df['income'].apply(lambda x: 1 if x == '>50K' else 0)
    
    return df

df = preprocess_data(df)

# 基本数据探索
print("数据集形状:", df.shape)
print("\n数据集前5行:")
print(df.head())
print("\n数据集信息:")
print(df.info())
print("\n数据集统计描述:")
print(df.describe())

# 检查缺失值
print("\n缺失值统计:")
print(df.isnull().sum())

# 检查'?'值（在这个数据集中通常表示缺失）
print("\n'?'值统计:")
for col in df.select_dtypes(include='object').columns:
    missing_count = (df[col] == '?').sum()
    if missing_count > 0:
        print(f"{col}: {missing_count} '?'值")

# 设置图表标题和标签的字体
def set_plot_labels(ax, title=None, xlabel=None, ylabel=None):
    """设置图表的标题和坐标轴标签，确保中文正常显示"""
    if title:
        ax.set_title(title, fontproperties=font)
    if xlabel:
        ax.set_xlabel(xlabel, fontproperties=font)
    if ylabel:
        ax.set_ylabel(ylabel, fontproperties=font)
    # 设置刻度标签字体
    for label in ax.get_xticklabels():
        label.set_fontproperties(font)
    for label in ax.get_yticklabels():
        label.set_fontproperties(font)

# 保存图表的函数
def save_fig(fig, filename, dpi=100):
    # 在保存前修复中文显示问题
    fix_seaborn_chinese(fig)
    fig.savefig(filename, dpi=dpi, bbox_inches='tight')
    plt.close(fig)

# 1. 单变量分析

# 1.1 数值变量分布
num_features = ['age', 'fnlwgt', 'education_num', 'capital_gain', 'capital_loss', 'hours_per_week']

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for i, feature in enumerate(num_features):
    sns.histplot(df[feature], kde=True, ax=axes[i])
    axes[i].set_title(f'{feature}分布')
    axes[i].set_xlabel(feature)
    axes[i].set_ylabel('频数')

plt.tight_layout()
save_fig(fig, 'numerical_distributions.png')

# 1.2 分类变量分布
cat_features = ['workclass', 'education', 'marital_status', 'occupation', 'relationship', 'race', 'sex', 'native_country']

# 选择前4个分类变量进行可视化
selected_cat_features = cat_features[:4]
fig, axes = plt.subplots(2, 2, figsize=(18, 12))
axes = axes.flatten()

for i, feature in enumerate(selected_cat_features):
    # 获取前10个最频繁的类别
    value_counts = df[feature].value_counts()
    if len(value_counts) > 10:
        value_counts = value_counts.head(10)
    
    sns.barplot(x=value_counts.index, y=value_counts.values, ax=axes[i])
    axes[i].set_title(f'{feature}分布 (前10类)')
    axes[i].set_xlabel(feature)
    axes[i].set_ylabel('频数')
    axes[i].tick_params(axis='x', rotation=45)

plt.tight_layout()
save_fig(fig, 'categorical_distributions_1.png')

# 选择后4个分类变量进行可视化
selected_cat_features = cat_features[4:]
fig, axes = plt.subplots(2, 2, figsize=(18, 12))
axes = axes.flatten()

for i, feature in enumerate(selected_cat_features):
    # 获取前10个最频繁的类别
    value_counts = df[feature].value_counts()
    if len(value_counts) > 10:
        value_counts = value_counts.head(10)
    
    sns.barplot(x=value_counts.index, y=value_counts.values, ax=axes[i])
    axes[i].set_title(f'{feature}分布 (前10类)')
    axes[i].set_xlabel(feature)
    axes[i].set_ylabel('频数')
    axes[i].tick_params(axis='x', rotation=45)

plt.tight_layout()
save_fig(fig, 'categorical_distributions_2.png')

# 1.3 收入分布
fig, ax = plt.subplots(figsize=(10, 6))
sns.countplot(x='income', data=df, ax=ax)
ax.set_title('收入分布')
ax.set_xlabel('收入')
ax.set_ylabel('频数')
plt.tight_layout()
save_fig(fig, 'income_distribution.png')

# 2. 双变量分析

# 2.1 数值变量与收入的关系
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for i, feature in enumerate(num_features):
    sns.boxplot(x='income', y=feature, data=df, ax=axes[i])
    axes[i].set_title(f'{feature}与收入的关系')
    axes[i].set_xlabel('收入')
    axes[i].set_ylabel(feature)

plt.tight_layout()
save_fig(fig, 'numerical_vs_income.png')

# 2.2 分类变量与收入的关系
# 选择前4个分类变量
selected_cat_features = cat_features[:4]
fig, axes = plt.subplots(2, 2, figsize=(18, 12))
axes = axes.flatten()

for i, feature in enumerate(selected_cat_features):
    # 获取前5个最频繁的类别
    top_categories = df[feature].value_counts().head(5).index
    # 筛选数据
    subset = df[df[feature].isin(top_categories)]
    
    # 计算每个类别中>50K的比例
    income_prop = pd.crosstab(subset[feature], subset['income'], normalize='index')
    
    sns.barplot(x=income_prop.index, y=income_prop['>50K'], ax=axes[i])
    axes[i].set_title(f'{feature}与高收入(>50K)比例的关系 (前5类)')
    axes[i].set_xlabel(feature)
    axes[i].set_ylabel('高收入比例')
    axes[i].tick_params(axis='x', rotation=45)

plt.tight_layout()
save_fig(fig, 'categorical_vs_income_1.png')

# 选择后4个分类变量
selected_cat_features = cat_features[4:]
fig, axes = plt.subplots(2, 2, figsize=(18, 12))
axes = axes.flatten()

for i, feature in enumerate(selected_cat_features):
    # 获取前5个最频繁的类别
    top_categories = df[feature].value_counts().head(5).index
    # 筛选数据
    subset = df[df[feature].isin(top_categories)]
    
    # 计算每个类别中>50K的比例
    income_prop = pd.crosstab(subset[feature], subset['income'], normalize='index')
    
    sns.barplot(x=income_prop.index, y=income_prop['>50K'], ax=axes[i])
    axes[i].set_title(f'{feature}与高收入(>50K)比例的关系 (前5类)')
    axes[i].set_xlabel(feature)
    axes[i].set_ylabel('高收入比例')
    axes[i].tick_params(axis='x', rotation=45)

plt.tight_layout()
save_fig(fig, 'categorical_vs_income_2.png')

# 3. 多变量分析

# 3.1 数值变量之间的相关性
num_df = df[num_features + ['income_binary']]
fig, ax = plt.subplots(figsize=(12, 10))
corr = num_df.corr()
# 设置热力图中的注释字体
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', ax=ax,
           annot_kws={'fontsize':10, 'fontfamily':plt.rcParams['font.sans-serif'][0]})
# 使用辅助函数设置标题和标签
set_plot_labels(ax, 
               title='数值变量相关性热力图', 
               xlabel='特征', 
               ylabel='特征')
plt.tight_layout()
save_fig(fig, 'correlation_heatmap.png')

# 3.2 教育水平、工作时间与收入的关系
fig, ax = plt.subplots(figsize=(12, 8))
sns.boxplot(x='education', y='hours_per_week', hue='income', data=df, ax=ax, 
            order=sorted(df['education'].unique(), key=lambda x: df[df['education']==x]['education_num'].iloc[0]))
ax.set_title('教育水平、工作时间与收入的关系')
ax.set_xlabel('教育水平')
ax.set_ylabel('每周工作时间')
ax.tick_params(axis='x', rotation=45)
plt.tight_layout()
save_fig(fig, 'education_hours_income.png')

# 3.3 年龄、教育水平与收入的关系
fig, ax = plt.subplots(figsize=(12, 8))
sns.violinplot(x='education', y='age', hue='income', data=df, ax=ax, split=True,
               order=sorted(df['education'].unique(), key=lambda x: df[df['education']==x]['education_num'].iloc[0]))
ax.set_title('年龄、教育水平与收入的关系')
ax.set_xlabel('教育水平')
ax.set_ylabel('年龄')
ax.tick_params(axis='x', rotation=45)
plt.tight_layout()
save_fig(fig, 'age_education_income.png')

# 3.4 性别、职业与收入的关系
# 获取前6个最频繁的职业
top_occupations = df['occupation'].value_counts().head(6).index
# 筛选数据
subset = df[df['occupation'].isin(top_occupations)]

fig, ax = plt.subplots(figsize=(14, 8))
sns.countplot(x='occupation', hue='income', data=subset, ax=ax)
ax.set_title('职业与收入的关系 (前6个职业)')
ax.set_xlabel('职业')
ax.set_ylabel('频数')
ax.tick_params(axis='x', rotation=45)
plt.tight_layout()
save_fig(fig, 'occupation_income.png')

# 3.5 性别、收入差距分析
fig, ax = plt.subplots(figsize=(12, 8))
# 计算每个性别中>50K的比例
income_by_gender = pd.crosstab(df['sex'], df['income'], normalize='index')
sns.barplot(x=income_by_gender.index, y=income_by_gender['>50K'], ax=ax)
ax.set_title('性别与高收入(>50K)比例的关系')
ax.set_xlabel('性别')
ax.set_ylabel('高收入比例')
plt.tight_layout()
save_fig(fig, 'gender_income_gap.png')

# 3.6 婚姻状况、性别与收入的关系
# 获取前4个最频繁的婚姻状况
top_marital = df['marital_status'].value_counts().head(4).index
# 筛选数据
subset = df[df['marital_status'].isin(top_marital)]

fig, ax = plt.subplots(figsize=(14, 8))
sns.countplot(x='marital_status', hue='income', data=subset, ax=ax)
ax.set_title('婚姻状况与收入的关系 (前4个婚姻状况)')
ax.set_xlabel('婚姻状况')
ax.set_ylabel('频数')
ax.tick_params(axis='x', rotation=45)
plt.tight_layout()
save_fig(fig, 'marital_income.png')

# 4. 特征工程与高级分析

# 4.1 创建年龄组
df['age_group'] = pd.cut(df['age'], bins=[0, 25, 35, 45, 55, 65, 100], 
                         labels=['<25', '25-35', '35-45', '45-55', '55-65', '>65'])

# 分析年龄组与收入的关系
fig, ax = plt.subplots(figsize=(12, 6))
income_by_age = pd.crosstab(df['age_group'], df['income'], normalize='index')
sns.barplot(x=income_by_age.index, y=income_by_age['>50K'], ax=ax)
ax.set_title('年龄组与高收入(>50K)比例的关系')
ax.set_xlabel('年龄组')
ax.set_ylabel('高收入比例')
plt.tight_layout()
save_fig(fig, 'age_group_income.png')

# 4.2 工作时间分组
df['work_hours_group'] = pd.cut(df['hours_per_week'], bins=[0, 20, 40, 60, 100], 
                               labels=['<20', '20-40', '40-60', '>60'])

# 分析工作时间与收入的关系
fig, ax = plt.subplots(figsize=(12, 6))
income_by_hours = pd.crosstab(df['work_hours_group'], df['income'], normalize='index')
sns.barplot(x=income_by_hours.index, y=income_by_hours['>50K'], ax=ax)
ax.set_title('工作时间与高收入(>50K)比例的关系')
ax.set_xlabel('每周工作时间')
ax.set_ylabel('高收入比例')
plt.tight_layout()
save_fig(fig, 'work_hours_income.png')

# 4.3 教育与性别的交互作用
fig, ax = plt.subplots(figsize=(14, 8))
# 获取前6个最频繁的教育水平
top_education = df['education'].value_counts().head(6).index
# 筛选数据
subset = df[df['education'].isin(top_education)]

# 计算每个教育水平和性别组合中>50K的比例
income_by_edu_sex = pd.crosstab([subset['education'], subset['sex']], subset['income'], normalize='index')
income_by_edu_sex = income_by_edu_sex.reset_index()

# 重塑数据以便于绘图
income_by_edu_sex_pivot = income_by_edu_sex.pivot(index='education', columns='sex', values='>50K')

# 绘制堆叠条形图
income_by_edu_sex_pivot.plot(kind='bar', ax=ax)
ax.set_title('教育水平、性别与高收入(>50K)比例的关系')
ax.set_xlabel('教育水平')
ax.set_ylabel('高收入比例')
ax.tick_params(axis='x', rotation=45)
plt.tight_layout()
save_fig(fig, 'education_gender_income.png')

# 4.4 资本收益与损失分析
# 创建一个新特征：净资本收益
df['net_capital'] = df['capital_gain'] - df['capital_loss']

# 分析净资本收益与收入的关系
fig, ax = plt.subplots(figsize=(12, 6))
# 只选择有净资本收益或损失的样本
subset = df[(df['net_capital'] != 0)]
# 对净资本收益进行对数变换以便于可视化
subset['log_net_capital'] = np.log1p(np.abs(subset['net_capital'])) * np.sign(subset['net_capital'])

sns.boxplot(x='income', y='log_net_capital', data=subset, ax=ax)
ax.set_title('净资本收益与收入的关系 (对数尺度)')
ax.set_xlabel('收入')
ax.set_ylabel('净资本收益 (对数尺度)')
plt.tight_layout()
save_fig(fig, 'net_capital_income.png')

# 5. 结论与发现
print("\n数据分析结论:")
print("1. 年龄、教育水平、婚姻状况与收入有显著相关性")
print("2. 性别差异明显，男性高收入比例高于女性")
print("3. 工作时间与收入正相关，但超过一定时间后相关性减弱")
print("4. 职业类型对收入有重要影响，管理和专业类职业高收入比例更高")
print("5. 资本收益与高收入强相关，表明投资收入是高收入人群的重要来源")

print("\n分析完成，所有图表已保存。")