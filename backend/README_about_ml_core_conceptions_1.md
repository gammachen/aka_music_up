以下是机器学习核心概念的详细解释，结合实例与代码说明：

---

### **1. 样本 (Sample)**
**定义**：单个数据实例或观测值，是机器学习的基本处理单元  
**示例**：  
- 图像识别：一张28x28像素的手写数字图片  
- 房价预测：某个房屋的{面积, 房龄, 位置}记录  

**代码示例**：
```python
# MNIST数据集中的单个样本
from keras.datasets import mnist
(train_images, train_labels), _ = mnist.load_data()
sample_image = train_images[0]  # 获取第一个样本（手写数字图像）
```

---

### **2. 特征 (Feature)**
**定义**：描述样本属性的变量，是模型的输入维度  
**关键区分**：  
- **原始特征**：直接观测值（如像素值）  
- **衍生特征**：人工构造的特征（如像素平均值）  

**示例**：  
- 鸢尾花分类：花瓣长度、萼片宽度  
- 垃圾邮件检测：关键词出现频率  

**代码展示**：
```python
# 鸢尾花数据集特征
from sklearn.datasets import load_iris
iris = load_iris()
print("特征名称:", iris.feature_names)  # ['sepal length (cm)', 'sepal width (cm)', ...]
```

---

### **3. 特征向量 (Feature Vector)**
**定义**：将样本所有特征数值化后的向量表示  
**数学形式**：x = [x₁, x₂, ..., xₙ] ∈ ℝⁿ  
**示例**：  
- 房价样本：[120.5（面积）, 5（房龄）, 34.7（纬度）]  
- 用户画像：[25（年龄）, 1（性别男）, 0.87（活跃度）]  

**代码实现**：
```python
import numpy as np
# 构造特征向量
house_features = np.array([120.5, 5, 34.7])  # 形状(3,)
```

---

### **4. 标签 (Label)**
**定义**：监督学习中样本的预期输出值  
**类型**：  
- 分类任务：离散值（如"猫"/"狗"）  
- 回归任务：连续值（如房价9.8万元）  

**示例**：  
- 医疗诊断：CT影像 → ["良性", "恶性"]  
- 股票预测：历史数据 → 明日涨跌幅  

**代码示例**：
```python
# 分类标签与独热编码
from keras.utils import to_categorical
y = [0, 1, 2]  # 原始标签（三类）
y_onehot = to_categorical(y)  # [[1,0,0], [0,1,0], [0,0,1]]
```

---

### **5. 数据集 (Dataset)**
**定义**：样本与标签的集合，通常分为三个子集  
**标准划分**：  
| 子集类型 | 用途                  | 典型比例 |  
|----------|-----------------------|----------|  
| 训练集   | 模型参数学习          | 60-70%   |  
| 验证集   | 超参数调优/早停       | 15-20%   |  
| 测试集   | 最终性能评估          | 15-20%   |  

**代码实现划分**：
```python
from sklearn.model_selection import train_test_split
X, y = load_data()  # 假设已加载数据

# 首次分割：训练集+临时集
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3)

# 二次分割：验证集+测试集
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5)
```

---

### **6. 特征工程 (Feature Engineering)**
**定义**：通过转换/组合原始特征提升模型性能的过程  
**常用技术**：  
- 归一化：MinMaxScaler  
- 离散化：将年龄分段为[儿童, 青年, 中年]  
- 交叉特征：面积 × 单价  

**代码示例**：
```python
from sklearn.preprocessing import PolynomialFeatures

# 创建交互特征（面积×房龄）
X = [[120, 5], [90, 3]]
poly = PolynomialFeatures(interaction_only=True)
X_poly = poly.fit_transform(X)  # 包含[1, 120, 5, 120*5]等特征
```

---

### **7. 过拟合与泛化 (Overfitting & Generalization)**
**关键概念对比**：  
| 现象      | 表现                          | 解决方案                |  
|-----------|-------------------------------|-------------------------|  
| 过拟合    | 训练集准确率高，测试集差      | 正则化/Dropout/早停     |  
| 欠拟合    | 训练集和测试集表现均差        | 增加模型复杂度/特征工程 |  

**可视化示例**：  
- 过拟合：复杂曲线完美拟合所有训练点但震荡剧烈  
- 良好拟合：平滑曲线既拟合趋势又保持稳定  

---

### **代码综合示例**
```python
# 完整数据预处理流程示例
import pandas as pd
from sklearn.preprocessing import StandardScaler

# 加载数据集
data = pd.read_csv('housing.csv')
X = data[['area', 'age']]  # 特征矩阵
y = data['price']          # 标签向量

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 数据集划分
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42)
```

---

### **核心要点总结**
1. **样本是原子单位**，特征向量是其数学表示  
2. **标签决定学习目标类型**（分类/回归）  
3. **数据集划分防止信息泄露**：测试集应全程不可见  
4. **特征工程质量决定模型上限**：好的特征比复杂模型更重要  

建议在PPT中用「房屋数据集」作为贯穿案例，保持示例一致性，帮助听众建立系统认知。代码部分可选择性展示关键步骤，避免过多技术细节冲淡概念理解。

