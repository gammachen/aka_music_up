import seaborn as sns
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
from sklearn.datasets import make_classification

from matplotlib.font_manager import FontProperties, fontManager

# 从文件加载数据
data = pd.read_csv('Sales.Prediction.txt', sep='\t')
print("数据加载成功，共有 {} 行，{} 列".format(data.shape[0], data.shape[1]))
print("数据列名：", data.columns.tolist())


# 使用电商数据集进行分析
df = data

# 查看数据类型
print("\n数据类型:")
print(df.dtypes)

# 数据概览
print(df.head())
print("\n数据描述:")
print(df.describe())

# 缺失值检查
print("\n缺失值统计:")
print(df.isnull().sum())

# 将分类变量转换为数值型
df['IsDeal'] = df['IsDeal'].astype(int)
df['IsNew'] = df['IsNew'].astype(int)
df['IsLimitedStock'] = df['IsLimitedStock'].astype(int)

# 选择数值型特征进行分析
num_features = ['OneMonthConversionRateInUV', 'OneWeekConversionRateInUV', 'SellerReputation', 
               'IsDeal', 'IsNew', 'IsLimitedStock']

# 标准化处理（为后续分析准备）
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaled_features = scaler.fit_transform(df[num_features])
df_scaled = pd.DataFrame(scaled_features, columns=[f + '_scaled' for f in num_features])
df = pd.concat([df, df_scaled], axis=1)

# 数值型特征与目标的相关性
corr_with_target = df[num_features + ['TargetValue']].corr(method='pearson')['TargetValue'].sort_values(ascending=False)

print("\n特征与目标转化率的皮尔森相关系数:")
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

# 可视化特征与目标的相关性
plt.figure(figsize=(10,6))
sns.barplot(x=corr_with_target.values[:-1], y=corr_with_target.index[:-1], palette="vlag")
plt.title("特征与目标转化率的皮尔森相关系数")
plt.tight_layout()
plt.savefig('相关系数条形图.png')
plt.close()

from statsmodels.stats.outliers_influence import variance_inflation_factor

# 计算VIF
def calculate_vif(dataframe, features):
    X = dataframe[features]
    vif_data = pd.DataFrame()
    vif_data["Feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return vif_data.sort_values(by="VIF", ascending=False)

vif_result = calculate_vif(df, num_features)
print("\n方差膨胀因子(VIF):")
print(vif_result)

# 可视化特征间相关性
plt.figure(figsize=(12,8))
sns.heatmap(df[num_features].corr(),
            annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("自变量间相关系数矩阵")
plt.tight_layout()
plt.savefig('相关系数热力图.png')
plt.close()

# 使用斯皮尔曼相关系数
spearman_corr = df[num_features + ['TargetValue']].corr(method='spearman')['TargetValue']
print("\n斯皮尔曼相关系数:\n", spearman_corr)

# 按类别分组分析
print("\n按商品类别分组的平均转化率:")
cat_analysis = df.groupby('CategoryName')['TargetValue'].agg(['mean', 'count']).sort_values(by='mean', ascending=False)
print(cat_analysis)

# 可视化类别与转化率的关系
plt.figure(figsize=(12,6))
sns.barplot(x='mean', y=cat_analysis.index, data=cat_analysis.reset_index())
plt.title('不同商品类别的平均转化率')
plt.tight_layout()
plt.savefig('类别转化率.png')
plt.close()

# 局部回归可视化 - 转化率与卖家信誉的关系
g = sns.lmplot(x='SellerReputation', y='TargetValue', data=df, 
           lowess=True, line_kws={'color':'red'})
plt.title('卖家信誉与转化率的局部加权回归')
plt.tight_layout()
g.savefig('卖家信誉回归.png')
plt.close()

# 局部回归可视化 - 转化率与月转化率的关系
g = sns.lmplot(x='OneMonthConversionRateInUV', y='TargetValue', data=df, 
           lowess=True, line_kws={'color':'red'})
plt.title('月转化率与目标转化率的局部加权回归')
plt.tight_layout()
g.savefig('月转化率回归.png')
plt.close()

# 分析特征交互作用
print("\n特征交互分析 - 是否限量 & 是否新品:")
interaction_analysis = df.groupby(['IsLimitedStock', 'IsNew'])['TargetValue'].mean().unstack()
print(interaction_analysis)

# 热力图可视化特征交互
plt.figure(figsize=(8,6))
sns.heatmap(interaction_analysis, annot=True, cmap="YlGnBu", fmt=".3f")
plt.title('限量商品与新品对转化率的交互影响')
plt.tight_layout()
plt.savefig('特征交互热力图.png')
plt.close()

print("\n所有图表已保存到当前目录。")
print("\n分析完成！请查看生成的图表文件和上述分析结果。")

'''
# 电商数据自变量相关性分析结论：

## 1. 相关性分析结果
- 月转化率(OneMonthConversionRateInUV)与目标转化率(TargetValue)有最强的正相关性，这表明历史转化表现是预测未来转化的重要指标
- 周转化率(OneWeekConversionRateInUV)也与目标转化率有较强的正相关性，但略低于月转化率
- 卖家信誉(SellerReputation)对转化率有正面影响，信誉越高，转化率越高
- 限量商品(IsLimitedStock)和促销商品(IsDeal)对转化率有一定的正面影响
- 新品(IsNew)与转化率的相关性相对较弱

## 2. 类别分析
- 不同商品类别的转化率差异显著，如方便面、食用油、饮料饮品等类别转化率较高
- 电脑、手机等电子产品类别转化率相对较低

## 3. 特征交互影响
- 限量商品与新品的组合对转化率有交互影响
- 卖家信誉与转化率呈现非线性关系，信誉达到一定程度后，对转化率的边际提升减小

## 4. 多重共线性检测
- 月转化率与周转化率之间存在较强的多重共线性，VIF值较高
- 在建模时需要考虑是否同时使用这两个特征

## 5. 建议
- 重点关注历史转化率指标和卖家信誉
- 针对不同商品类别制定差异化的营销策略
- 适当利用限量和促销手段提高转化率
- 在预测模型中考虑特征间的交互作用
'''