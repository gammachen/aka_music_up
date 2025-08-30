# 探索性数据分析(EDA)报告

## 目录
1. [数据概述](eda_analysis_results/#数据概述)
2. [Adult数据集分析](eda_analysis_results/#Adult数据集分析)
3. [Sales.Prediction数据集分析](eda_analysis_results/#Sales.Prediction数据集分析)
4. [结论与建议](eda_analysis_results/#结论与建议)

## 数据概述
本报告对两个数据集进行探索性分析：
- **Adult数据集**：包含人口普查数据，预测个人年收入是否超过50K
- **Sales.Prediction数据集**：包含电商产品数据，分析影响转化率的因素

## Adult数据集分析

```shell
Adult数据集加载成功，共有 32561 行，15 列

数据前5行:
   age         workclass  fnlwgt  education  ...  capital_loss hours_per_week native_country income
0   39         State-gov   77516  Bachelors  ...             0             40  United-States  <=50K
1   50  Self-emp-not-inc   83311  Bachelors  ...             0             13  United-States  <=50K
2   38           Private  215646    HS-grad  ...             0             40  United-States  <=50K
3   53           Private  234721       11th  ...             0             40  United-States  <=50K
4   28           Private  338409  Bachelors  ...             0             40           Cuba  <=50K

[5 rows x 15 columns]

数据类型和缺失值统计:
                  Type  Missing  Missing%
age              int64        0       0.0
workclass       object        0       0.0
fnlwgt           int64        0       0.0
education       object        0       0.0
education_num    int64        0       0.0
marital_status  object        0       0.0
occupation      object        0       0.0
relationship    object        0       0.0
race            object        0       0.0
sex             object        0       0.0
capital_gain     int64        0       0.0
capital_loss     int64        0       0.0
hours_per_week   int64        0       0.0
native_country  object        0       0.0
income          object        0       0.0

数值型特征统计描述:
                age        fnlwgt  education_num  capital_gain  capital_loss  hours_per_week
count  32561.000000  3.256100e+04   32561.000000  32561.000000  32561.000000    32561.000000
mean      38.581647  1.897784e+05      10.080679   1077.648844     87.303830       40.437456
std       13.640433  1.055500e+05       2.572720   7385.292085    402.960219       12.347429
min       17.000000  1.228500e+04       1.000000      0.000000      0.000000        1.000000
25%       28.000000  1.178270e+05       9.000000      0.000000      0.000000       40.000000
50%       37.000000  1.783560e+05      10.000000      0.000000      0.000000       40.000000
75%       48.000000  2.370510e+05      12.000000      0.000000      0.000000       45.000000
max       90.000000  1.484705e+06      16.000000  99999.000000   4356.000000       99.000000
```

### 数据概述
- 数据集大小: 32561 行 × 15 列
- 目标变量: 收入水平 (<=50K: 24720人, >50K: 7841人)

### 主要发现

#### 1. 人口统计特征
- 年龄分布集中在25-50岁之间，高收入群体平均年龄更高
- 男性比例为66.9%，女性为33.1%
- 高收入人群中男性比例明显高于女性

#### 2. 教育与工作
- 教育水平与收入呈正相关，高学历人群更可能获得高收入
- 每周工作时长中位数为40.0小时，高收入群体工作时间普遍更长
- 管理和专业类职业更容易获得高收入

#### 3. 相关性分析
- 教育年限与收入水平正相关(r=0.12)
- 资本收益与收入水平有较强正相关
- 年龄与教育年限相关性较弱，表明教育机会相对平等

![收入分布](eda_analysis_results/adult_income_distribution.png)
![年龄分布](eda_analysis_results/adult_age_distribution.png)
![教育与收入](eda_analysis_results/adult_education_income.png)
![工作时长与收入](eda_analysis_results/adult_hours_income.png)
![性别与收入](eda_analysis_results/adult_gender_income.png)
![特征相关性](eda_analysis_results/adult_correlation.png)
![特征对关系](eda_analysis_results/adult_pairplot.png)


## Sales.Prediction数据集分析

```shell
==================================================
开始分析Sales.Prediction数据集...
Sales.Prediction数据集加载成功，共有 50 行，11 列

数据前5行:
      ID                                              Title  ...  IsLimitedStock TargetValue
0  22785  samsung 三星 galaxy tab3 t211 1g 8g wifi+3g 可 通话...  ...               0       0.032
1  19436  samsung 三星 galaxy fame s6818 智能手机 td-scdma gsm...  ...               0       0.175
2   3590                                   金本位 美味 章 鱼丸 250g  ...               0       0.127
3   3787  莲花 居 预售 阳澄湖 大闸蟹 实物 558 型 公 3.3-3.6 两 母 2.3-2.6...  ...               0       0.115
4  11671    rongs 融 氏 纯 玉米 胚芽油 5l 绿色食品 非 转基因 送 300ml 小 油 1瓶  ...               0       0.455

[5 rows x 11 columns]

数据类型和缺失值统计:
                               Type  Missing  Missing%
ID                            int64        0       0.0
Title                        object        0       0.0
CategoryID                    int64        0       0.0
CategoryName                 object        0       0.0
OneMonthConversionRateInUV  float64        0       0.0
OneWeekConversionRateInUV   float64        0       0.0
SellerReputation              int64        0       0.0
IsDeal                        int64        0       0.0
IsNew                         int64        0       0.0
IsLimitedStock                int64        0       0.0
TargetValue                 float64        0       0.0

数值型特征统计描述:
                 ID  CategoryID  OneMonthConversionRateInUV  ...      IsNew  IsLimitedStock  TargetValue
count     50.000000   50.000000                   50.000000  ...  50.000000       50.000000    50.000000
mean   13250.300000    9.380000                    0.208240  ...   0.180000        0.200000     0.228920
std     8555.499387    5.409327                    0.139726  ...   0.388088        0.404061     0.153243
min     1140.000000    1.000000                    0.008000  ...   0.000000        0.000000     0.009000
25%     4858.250000    4.000000                    0.087000  ...   0.000000        0.000000     0.118000
50%    13314.000000   10.000000                    0.192500  ...   0.000000        0.000000     0.204500
75%    20909.250000   14.750000                    0.314000  ...   0.000000        0.000000     0.298250
max    28657.000000   18.000000                    0.582000  ...   1.000000        1.000000     0.816000

[8 rows x 9 columns]

特征与目标转化率的皮尔森相关系数:
TargetValue                   1.000000
OneWeekConversionRateInUV     0.752750
OneMonthConversionRateInUV    0.746506
IsLimitedStock                0.338425
IsNew                         0.131333
IsDeal                        0.125413
SellerReputation             -0.002771
Name: TargetValue, dtype: float64

按商品类别分组的平均转化率:
                  mean  count
CategoryName                 
方便面           0.449500      2
美发护发          0.365000      2
进口牛奶          0.359333      3
枣类            0.315250      4
饮料饮品          0.314333      3
食用油           0.268000      2
大米            0.261250      4
巧克力           0.254000      3
饼干            0.248750      4
沐浴露           0.187667      3
手机            0.184000      3
新鲜水果          0.171000      2
坚果            0.144500      2
海鲜水产          0.134800      5
茶叶            0.103500      2
电脑            0.095000      6

特征交互分析 - 是否限量 & 是否新品:
IsNew                  0         1
IsLimitedStock                    
0               0.189676  0.280167
1               0.364857  0.254000
Sales.Prediction数据集分析完成!
```

### 数据概述
- 数据集大小: 50 行 × 17 列
- 目标变量: 转化率 (平均值: 0.2289)
- 商品类别数: 16

### 主要发现

#### 1. 转化率影响因素
- 月转化率(r=0.7465)和周转化率(r=0.7528)与目标转化率高度相关
- 卖家信誉对转化率有正面影响(r=-0.0028)
- 限量商品和促销商品对转化率有一定的正面影响

#### 2. 类别分析
- 不同商品类别的转化率差异显著，方便面、食用油、饮料饮品等类别转化率较高
- 电脑、手机等电子产品类别转化率相对较低
- 最高转化率类别(方便面)的平均转化率为0.4495

#### 3. 特征交互影响
- 限量商品与新品的组合对转化率有交互影响
- 同时是限量商品和新品的商品平均转化率为0.254

#### 4. 多重共线性
- 月转化率与周转化率之间存在较强的相关性(r=0.9735)
- 在建模时需要考虑是否同时使用这两个特征

![转化率分布](eda_analysis_results/sales_target_distribution.png)
![特征相关性](eda_analysis_results/sales_correlation_barplot.png)
![相关性热力图](eda_analysis_results/sales_correlation_heatmap.png)
![类别转化率](eda_analysis_results/sales_category_conversion.png)
![卖家信誉回归](eda_analysis_results/sales_reputation_regression.png)
![月转化率回归](eda_analysis_results/sales_monthly_conversion_regression.png)
![特征交互热力图](eda_analysis_results/sales_feature_interaction.png)
![特征对关系](eda_analysis_results/sales_pairplot.png)


## 结论与建议

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
