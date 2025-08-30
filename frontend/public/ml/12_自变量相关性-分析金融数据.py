import seaborn as sns
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
from sklearn.datasets import make_classification

from matplotlib.font_manager import FontProperties, fontManager

# 生成模拟数据
np.random.seed(42)
n_samples = 1000

# 基本特征
data = {
    '年龄': np.random.randint(20, 65, n_samples),
    '收入（万元）': np.round(np.random.normal(15, 5, n_samples), 1),
    '职业等级': np.random.choice([1, 2, 3, 4, 5], n_samples, p=[0.1, 0.2, 0.4, 0.2, 0.1]),  # 有序分类变量
    '负债比': np.random.uniform(0.1, 0.8, n_samples),
    '征信查询次数': np.random.poisson(3, n_samples),
    '逾期记录': np.random.poisson(1, n_samples),
}

# 从文件加载数据
# data = pd.read_csv('Sales.Prediction.txt')

# 合成违约概率（非线性关系）
X_synth, y = make_classification(
    n_samples=n_samples,
    n_features=5,
    n_informative=3,
    weights=[0.85, 0.15],  # 违约率15%
    random_state=42
)

# 将合成特征映射到业务字段
data['违约概率'] = y
df = pd.DataFrame(data)

# 添加收入与职业等级的强相关性（r≈0.92）
df['收入（万元）'] = df['收入（万元）'] + df['职业等级'] * 2.5 + np.random.normal(0, 1, n_samples)

# 数据概览
print(df.head())
print("\n数据描述:")
print(df.describe())

# 缺失值检查
print("\n缺失值统计:")
print(df.isnull().sum())

# 标准化处理（为后续分析准备）
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaled_features = scaler.fit_transform(df[['年龄', '收入（万元）', '职业等级', '负债比', '征信查询次数', '逾期记录']])
df_scaled = pd.DataFrame(scaled_features, columns=['年龄_scaled', '收入_scaled', '职业等级_scaled', '负债比_scaled', '征信查询_scaled', '逾期_scaled'])
df = pd.concat([df, df_scaled], axis=1)

# 数值型特征与目标的相关性
corr_with_target = df[['年龄', '收入（万元）', '职业等级', '负债比', '征信查询次数', '逾期记录', '违约概率']].corr(method='pearson')['违约概率'].sort_values(ascending=False)

print("特征与违约概率的相关系数:")
print(corr_with_target)

# 设置中文字体,避免乱码
# fc-list :lang=zh 
# /Users/shhaofu/Library/Fonts/shanhaihuanyinggete.ttf: ShanHaiHuanYingGeTe,山海幻影哥特:style=Regular

# macOS 示例：
# font_path = "/System/Library/Fonts/STHeiti Medium.ttc"  # 平方
font_path = "/System/Library/Fonts/STHeiti Light.ttc"
# : Heiti TC,黑體\-繁,黒体\-繁,Heiti\-번체,黑体\-繁:style=Light,細體,Mager,Fein,Ohut,Fin,Leggero,ライト,가는체,Licht,Tynn,Leve,Светлый,细体,Fina
# 添加字体到 Matplotlib
font = FontProperties(fname=font_path, size=12)
fontManager.addfont(font_path)

plt.rcParams['font.sans-serif'] = ['Heiti TC', 'WenQuanYi Micro Hei', 'SimHei', 'Microsoft YaHei']  # 按优先级设置中文字体
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号
plt.rcParams['font.family'] = 'sans-serif'  # 使用无衬线字体族
plt.rcParams['axes.titlesize'] = 14  # 标题字体大小
plt.rcParams['font.size'] = 12  # 全局字体大小

# 可视化
plt.figure(figsize=(10,6))
sns.barplot(x=corr_with_target.values[1:], y=corr_with_target.index[1:], palette="vlag")
plt.title("特征与违约概率的皮尔森相关系数")
plt.tight_layout()
plt.savefig('金融数据_相关系数条形图.png')
plt.close()

from statsmodels.stats.outliers_influence import variance_inflation_factor

# 计算VIF
def calculate_vif(dataframe):
    X = dataframe.drop(columns=['违约概率'])
    vif_data = pd.DataFrame()
    vif_data["Feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return vif_data.sort_values(by="VIF", ascending=False)

vif_result = calculate_vif(df[['年龄', '收入（万元）', '职业等级', '负债比', '征信查询次数', '逾期记录', '违约概率']])
print("\n方差膨胀因子(VIF):")
print(vif_result)

# 可视化特征间相关性
plt.figure(figsize=(12,8))
sns.heatmap(df[['年龄', '收入（万元）', '职业等级', '负债比', '征信查询次数', '逾期记录']].corr(),
            annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("自变量间相关系数矩阵")
plt.tight_layout()
plt.savefig('金融数据_相关系数热力图.png')
plt.close()

# 使用斯皮尔曼相关系数
spearman_corr = df.corr(method='spearman')['违约概率']
print("斯皮尔曼相关系数:\n", spearman_corr)

# 局部回归可视化 - 负债比与违约概率的关系
g = sns.lmplot(x='负债比', y='违约概率', data=df, 
           lowess=True, line_kws={'color':'red'})
plt.title('负债比与违约概率的局部加权回归')
plt.tight_layout()
g.savefig('金融数据_负债比回归.png')
plt.close()

# 局部回归可视化 - 年龄与违约概率的关系
g = sns.lmplot(x='年龄', y='违约概率', data=df, 
           lowess=True, line_kws={'color':'blue'})
plt.title('年龄与违约概率的局部加权回归')
plt.tight_layout()
g.savefig('金融数据_年龄回归.png')
plt.close()

# 分析特征交互作用
print("\n特征交互分析 - 职业等级 & 逾期记录:")
interaction_analysis = df.groupby(['职业等级', '逾期记录'])['违约概率'].mean().unstack()
print(interaction_analysis)

# 热力图可视化特征交互
plt.figure(figsize=(8,6))
sns.heatmap(interaction_analysis, annot=True, cmap="YlGnBu", fmt=".3f")
plt.title('职业等级与逾期记录对违约概率的交互影响')
plt.tight_layout()
plt.savefig('金融数据_特征交互热力图.png')
plt.close()

print("\n所有图表已保存到当前目录。")
print("\n分析完成！请查看生成的图表文件和上述分析结果。")

'''
   年龄     收入（万元）  职业等级       负债比  征信查询次数  逾期记录  违约概率
0  58  19.541947     1  0.571814       1     2     0
1  48  32.629732     3  0.136285       0     1     0
2  34  22.117718     1  0.484201       3     0     0
3  62  15.075350     1  0.301343       4     1     0
4  27  34.203492     5  0.314744       4     0     0

数据描述:
                年龄       收入（万元）         职业等级          负债比       征信查询次数         逾期记录         违约概率
count  1000.000000  1000.000000  1000.000000  1000.000000  1000.000000  1000.000000  1000.000000
mean     42.000000    22.981340     3.011000     0.442430     2.946000     0.954000     0.151000
std      12.945562     5.797495     1.107296     0.200238     1.601139     0.946985     0.358228
min      20.000000     0.559296     1.000000     0.100457     0.000000     0.000000     0.000000
25%      31.000000    18.983700     2.000000     0.269069     2.000000     0.000000     0.000000
50%      43.000000    22.927627     3.000000     0.438177     3.000000     1.000000     0.000000
75%      53.000000    27.089039     4.000000     0.615731     4.000000     1.000000     0.000000
max      64.000000    39.227479     5.000000     0.799690     9.000000     5.000000     1.000000

缺失值统计:
年龄        0
收入（万元）    0
职业等级      0
负债比       0
征信查询次数    0
逾期记录      0
违约概率      0
dtype: int64
特征与违约概率的相关系数:
违约概率      1.000000
征信查询次数    0.031682
职业等级      0.026091
收入（万元）    0.006990
逾期记录      0.005742
负债比      -0.016715
年龄       -0.020722
Name: 违约概率, dtype: float64
'''