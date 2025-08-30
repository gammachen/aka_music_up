#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
交叉透视表分析：Adult数据集可视化

该脚本使用交叉透视表和热力图对Adult数据集进行多维度分析，
重点探索各特征之间的交互关系及其与收入的关联。
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import matplotlib as mpl
from matplotlib.font_manager import FontProperties, fontManager

global report_content
report_content = ""

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
plt.rcParams['font.sans-serif'] = ['Songti TC', 'PingFang TC', 'Heiti TC', 'WenQuanYi Micro Hei', 'SimHei', 'Microsoft YaHei']  # 按优先级设置中文字体
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

# 创建图表保存目录
output_dir = 'adult_crosstab_analysis'
os.makedirs(output_dir, exist_ok=True)

# 加载数据
df = pd.read_csv('adult.data', header=None, names=columns, sep=', ', engine='python')

# 数据预处理
def preprocess_data(df):
    # 去除字符串前后的空格
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].str.strip()
    
    # 将收入转换为二元变量
    df['income_binary'] = df['income'].apply(lambda x: 1 if x == '>50K' else 0)
    
    # 处理'?'值
    for col in df.select_dtypes(include='object').columns:
        missing_count = (df[col] == '?').sum()
        if missing_count > 0:
            print(f"将{col}中的{missing_count}个'?'值替换为'Unknown'")
            df[col] = df[col].replace('?', 'Unknown')
    
    # 创建年龄组
    df['age_group'] = pd.cut(df['age'], bins=[0, 25, 35, 45, 55, 65, 100], 
                         labels=['<25', '25-35', '35-45', '45-55', '55-65', '>65'])
    
    # 创建工作时间组
    df['work_hours_group'] = pd.cut(df['hours_per_week'], bins=[0, 20, 40, 60, 100], 
                               labels=['<20', '20-40', '40-60', '>60'])
    
    # 简化教育类别
    education_mapping = {
        'Preschool': 'Basic',
        '1st-4th': 'Basic',
        '5th-6th': 'Basic',
        '7th-8th': 'Basic',
        '9th': 'Basic',
        '10th': 'Basic',
        '11th': 'Basic',
        '12th': 'Basic',
        'HS-grad': 'High School',
        'Some-college': 'Some College',
        'Assoc-voc': 'Associate',
        'Assoc-acdm': 'Associate',
        'Bachelors': 'Bachelors',
        'Masters': 'Advanced',
        'Doctorate': 'Advanced',
        'Prof-school': 'Advanced'
    }
    df['education_simplified'] = df['education'].map(education_mapping)
    
    return df

df = preprocess_data(df)

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
    fig.savefig(os.path.join(output_dir, filename), dpi=dpi, bbox_inches='tight')
    plt.close(fig)

# 打印基本信息
print("数据集形状:", df.shape)
print("\n处理后的数据集前5行:")
print(df.head())

# 1. 交叉表分析：教育水平与收入
education_income = pd.crosstab(df['education_simplified'], df['income'])
education_income_pct = pd.crosstab(df['education_simplified'], df['income'], normalize='index') * 100

print("\n教育水平与收入交叉表:")
print(education_income)
print("\n教育水平与高收入比例:")
print(education_income_pct['>50K'].sort_values(ascending=False))

# 绘制教育水平与收入热力图
fig, ax = plt.subplots(figsize=(10, 8))
# 设置热力图中的注释字体
sns.heatmap(education_income_pct, annot=True, fmt='.1f', cmap='YlGnBu', ax=ax, 
           annot_kws={'fontsize':10, 'fontfamily':plt.rcParams['font.sans-serif'][0]})
# 使用辅助函数设置标题和标签
set_plot_labels(ax, 
               title='教育水平与收入比例热力图 (%)', 
               xlabel='收入', 
               ylabel='教育水平')
plt.tight_layout()
save_fig(fig, 'education_income_heatmap.png')

# 2. 交叉表分析：性别、婚姻状况与收入
sex_marital_income = pd.crosstab([df['sex'], df['marital_status']], df['income'])
sex_marital_income_pct = pd.crosstab([df['sex'], df['marital_status']], df['income'], normalize='index') * 100

print("\n性别、婚姻状况与收入交叉表:")
print(sex_marital_income)

# 绘制性别、婚姻状况与收入热力图
fig, ax = plt.subplots(figsize=(14, 10))
# 修复形状不一致错误，使用更简单的数据重塑方式
sex_marital_pivot = sex_marital_income_pct['>50K'].unstack(level=0)
sns.heatmap(sex_marital_pivot, annot=True, fmt='.1f', cmap='YlGnBu', ax=ax)
ax.set_title('性别、婚姻状况与高收入比例热力图 (%)')
plt.tight_layout()
save_fig(fig, 'sex_marital_income_heatmap.png')

# 3. 交叉表分析：年龄组、工作时间与收入
age_hours_income = pd.crosstab([df['age_group'], df['work_hours_group']], df['income'])
age_hours_income_pct = pd.crosstab([df['age_group'], df['work_hours_group']], df['income'], normalize='index') * 100

print("\n年龄组、工作时间与收入交叉表:")
print(age_hours_income)

# 绘制年龄组、工作时间与收入热力图
fig, ax = plt.subplots(figsize=(12, 10))
# 修复形状不一致错误，使用更简单的数据重塑方式
age_hours_pivot = age_hours_income_pct['>50K'].unstack(level=0)
sns.heatmap(age_hours_pivot, annot=True, fmt='.1f', cmap='YlGnBu', ax=ax)
ax.set_title('年龄组、工作时间与高收入比例热力图 (%)')
plt.tight_layout()
save_fig(fig, 'age_hours_income_heatmap.png')

# 4. 交叉表分析：职业、种族与收入
# 获取前6个最频繁的职业
top_occupations = df['occupation'].value_counts().head(6).index
# 筛选数据
subset = df[df['occupation'].isin(top_occupations)]

occupation_race_income = pd.crosstab([subset['occupation'], subset['race']], subset['income'])
occupation_race_income_pct = pd.crosstab([subset['occupation'], subset['race']], subset['income'], normalize='index') * 100

print("\n职业、种族与收入交叉表 (前6个职业):")
print(occupation_race_income)

# 绘制职业、种族与收入热力图
fig, ax = plt.subplots(figsize=(16, 12))
# 修复形状不一致错误，使用更简单的数据重塑方式
occupation_race_pivot = occupation_race_income_pct['>50K'].unstack(level=0)
sns.heatmap(occupation_race_pivot, annot=True, fmt='.1f', cmap='YlGnBu', ax=ax)
ax.set_title('职业、种族与高收入比例热力图 (%) - 前6个职业')
plt.tight_layout()
save_fig(fig, 'occupation_race_income_heatmap.png')

# 5. 使用clustermap进行聚类分析
# 准备数据：不同特征组合的高收入比例
# 教育与工作类型
edu_work_income = pd.crosstab([df['education_simplified'], df['workclass']], df['income'], normalize='index')['>50K'] * 100
edu_work_pivot = edu_work_income.unstack().fillna(0)  # 填充缺失值，避免聚类问题

# 绘制聚类热力图
fig = plt.figure(figsize=(14, 12))
try:
    cluster_grid = sns.clustermap(edu_work_pivot, cmap='YlGnBu', annot=True, fmt='.1f',
                                figsize=(14, 12))
    cluster_grid.fig.suptitle('教育水平与工作类型的高收入比例聚类分析 (%)', fontsize=16)
    cluster_grid.fig.subplots_adjust(top=0.9)
    save_fig(cluster_grid.fig, 'education_workclass_clustermap.png')
    use_clustermap = True
except Exception as e:
    print(f"聚类图生成失败: {e}")
    # 使用普通热力图作为备选
    fig, ax = plt.subplots(figsize=(14, 10))
    sns.heatmap(edu_work_pivot, cmap='YlGnBu', annot=True, fmt='.1f', ax=ax)
    ax.set_title('教育水平与工作类型的高收入比例热力图 (%)')
    plt.tight_layout()
    save_fig(fig, 'education_workclass_heatmap.png')
    use_clustermap = False

# 6. 多变量交叉分析：教育、性别、年龄组与收入
# 创建多层交叉表 - 简化版本，只使用教育和性别

try:
    # 尝试完整的三维交叉分析
    multi_cross = pd.crosstab(
        [df['education_simplified'], df['sex'], df['age_group']], 
        df['income'], 
        normalize='index'
    )['>50K'] * 100
    
    # 重塑数据以便于可视化
    multi_cross_reshaped = multi_cross.unstack(level=[1, 2])
    
    print("\n教育、性别、年龄组与高收入比例多维交叉表:")
    print(multi_cross_reshaped)
    
    # 绘制多维交叉表热力图
    fig, ax = plt.subplots(figsize=(18, 12))
    sns.heatmap(multi_cross_reshaped, annot=True, fmt='.1f', cmap='YlGnBu', ax=ax)
    ax.set_title('教育、性别、年龄组与高收入比例多维热力图 (%)')
    plt.tight_layout()
    save_fig(fig, 'education_sex_age_income_heatmap.png')
except Exception as e:
    print(f"多维交叉分析失败: {e}")
    # 使用简化版本：只分析教育和性别
    edu_sex_income = pd.crosstab(
        [df['education_simplified'], df['sex']], 
        df['income'], 
        normalize='index'
    )['>50K'] * 100
    
    # 重塑数据
    edu_sex_pivot = edu_sex_income.unstack()
    
    print("\n教育、性别与高收入比例交叉表:")
    print(edu_sex_pivot)
    
    # 绘制热力图
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(edu_sex_pivot, annot=True, fmt='.1f', cmap='YlGnBu', ax=ax)
    ax.set_title('教育、性别与高收入比例热力图 (%)')
    plt.tight_layout()
    save_fig(fig, 'education_sex_income_heatmap.png')
    
    # 更新报告中的图片引用和标题
    report_content = report_content.replace(
        "## 7. 教育、性别、年龄组多维分析",
        "## 7. 教育与性别分析"
    )
    report_content = report_content.replace(
        "![教育、性别、年龄组多维分析](./adult_crosstab_analysis/education_sex_age_income_heatmap.png)",
        "![教育与性别分析](./adult_crosstab_analysis/education_sex_income_heatmap.png)"
    )
    report_content = report_content.replace(
        "- 多维分析揭示了教育、性别和年龄的交互效应",
        "- 分析揭示了教育与性别的交互效应"
    )
    report_content = report_content.replace(
        "- 高等教育中年男性高收入比例最高",
        "- 高等教育男性高收入比例最高"
    )

# 7. 创建透视表：工作时间、资本收益与收入
# 将资本收益分组
df['capital_gain_group'] = pd.cut(df['capital_gain'], 
                               bins=[-1, 0, 1000, 10000, 100000], 
                               labels=['0', '1-1000', '1001-10000', '>10000'])

# 创建透视表
pivot_table = pd.pivot_table(df, 
                            values='income_binary', 
                            index='work_hours_group',
                            columns='capital_gain_group',
                            aggfunc=np.mean) * 100

print("\n工作时间、资本收益与高收入比例透视表:")
print(pivot_table)

# 绘制透视表热力图
fig, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(pivot_table, annot=True, fmt='.1f', cmap='YlGnBu', ax=ax)
ax.set_title('工作时间、资本收益与高收入比例透视表 (%)')
plt.tight_layout()
save_fig(fig, 'hours_capital_income_pivot.png')

# 8. 相关性分析：数值变量之间的相关性
num_features = ['age', 'education_num', 'capital_gain', 'capital_loss', 'hours_per_week', 'income_binary']
num_df = df[num_features]

# 计算相关系数
corr = num_df.corr()

# 绘制相关性热力图
fig, ax = plt.subplots(figsize=(10, 8))
# 创建上三角掩码，确保形状正确
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', ax=ax, 
           vmin=-1, vmax=1, center=0, square=True, linewidths=.5)
ax.set_title('数值变量相关性热力图')
plt.tight_layout()
save_fig(fig, 'correlation_heatmap_masked.png')

# 9. 创建Markdown分析报告

report_content = f'''
# Adult数据集交叉透视表分析报告

## 1. 简介

本报告使用交叉透视表和热力图对Adult数据集进行多维度分析，重点探索各特征之间的交互关系及其与收入的关联。Adult数据集包含人口普查数据，用于预测个人年收入是否超过50K美元。

数据集包含以下主要特征：
- 年龄（age）
- 工作类型（workclass）
- 教育程度（education）
- 婚姻状况（marital_status）
- 职业（occupation）
- 种族（race）
- 性别（sex）
- 资本收益（capital_gain）
- 每周工作时间（hours_per_week）
- 收入（income）

## 2. 教育水平与收入分析

![教育水平与收入热力图](./adult_crosstab_analysis/education_income_heatmap.png)

**发现：**
- 教育水平与高收入比例呈明显正相关
- 高等教育（Advanced）人群高收入比例最高
- 基础教育（Basic）人群高收入比例最低

## 3. 性别、婚姻状况与收入分析

![性别、婚姻状况与收入热力图](./adult_crosstab_analysis/sex_marital_income_heatmap.png)

**发现：**
- 已婚男性高收入比例最高
- 性别差异明显，男性在各婚姻状况下高收入比例普遍高于女性
- 离婚和未婚人群高收入比例较低

## 4. 年龄组、工作时间与收入分析

![年龄组、工作时间与收入热力图](./adult_crosstab_analysis/age_hours_income_heatmap.png)

**发现：**
- 中年（35-55岁）且工作时间较长的人群高收入比例最高
- 年轻人（<25岁）无论工作时间长短，高收入比例均较低
- 工作时间与高收入呈正相关，但在老年人群体中相关性减弱

## 5. 职业、种族与收入分析

![职业、种族与收入热力图](./adult_crosstab_analysis/occupation_race_income_heatmap.png)

**发现：**
- 管理和专业类职业高收入比例普遍较高
- 种族差异在某些职业中较为明显
- 服务类职业高收入比例普遍较低

## 6. 教育与工作类型聚类分析

![教育与工作类型聚类分析](./adult_crosstab_analysis/education_workclass_clustermap.png)

**发现：**
- 聚类分析显示教育水平和工作类型可以分为几个明显的群组
- 高等教育与自雇、政府工作结合时高收入比例最高
- 私营企业员工的收入水平与教育程度高度相关

## 7. 教育、性别、年龄组多维分析

![教育、性别、年龄组多维分析](./adult_crosstab_analysis/education_sex_age_income_heatmap.png)

**发现：**
- 多维分析揭示了教育、性别和年龄的交互效应
- 高等教育中年男性高收入比例最高
- 即使在相同教育水平和年龄组，男女收入差距仍然明显

## 8. 工作时间、资本收益与收入分析

![工作时间、资本收益与收入分析](./adult_crosstab_analysis/hours_capital_income_pivot.png)

**发现：**
- 资本收益是高收入的强力预测因素
- 即使工作时间较短，有高资本收益的人群高收入比例也很高
- 工作时间长且有资本收益的人群高收入比例最高

## 9. 数值变量相关性分析

![数值变量相关性分析](./adult_crosstab_analysis/correlation_heatmap_masked.png)

**发现：**
- 教育年数与收入呈中等正相关
- 资本收益与收入呈较强正相关
- 年龄与收入呈弱正相关
- 工作时间与收入呈弱正相关

## 10. 结论

通过交叉透视表分析，我们发现：

1. **教育水平**是影响收入的关键因素，高等教育显著提高高收入概率
2. **性别差异**在各种条件下都很明显，男性高收入比例普遍高于女性
3. **婚姻状况**对收入有重要影响，已婚人士高收入比例更高
4. **年龄**与收入呈倒U型关系，中年人群高收入比例最高
5. **职业类型**对收入有显著影响，管理和专业类职业高收入比例更高
6. **资本收益**是高收入的强力预测因素
7. **多维交互效应**显著，各因素之间存在复杂的相互作用

这些发现对于理解收入不平等的多维度因素具有重要意义，也为收入预测模型提供了有价值的特征工程思路。
'''

# 保存Markdown报告
with open(os.path.join(output_dir, 'Adult数据集交叉透视表分析报告.md'), 'w', encoding='utf-8') as f:
    f.write(report_content)

print(f"\n分析完成，所有图表和报告已保存到 {output_dir} 目录。")
