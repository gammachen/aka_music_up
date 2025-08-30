# 资金管理系统架构与设计文档

## 项目概述

这是一个基于 Node.js 和 Vue.js 的资金管理系统，允许用户创建账户、管理交易记录、查看资金统计数据等。系统采用前后端分离架构，后端使用 Express 框架提供 API 服务，前端使用 Vue 3 构建用户界面。

## 系统架构

### 整体架构

```
资金管理系统
├── 前端 (Vue 3 + Element Plus)
│   ├── 用户界面
│   ├── API 请求模块
│   └── 状态管理 (Pinia)
└── 后端 (Express + MongoDB)
    ├── API 路由
    ├── 控制器
    ├── 数据模型
    └── 中间件
```

### 部署架构

系统采用 Docker 容器化部署，包含以下服务：
- 前端服务：Vue 应用
- 后端服务：Express 应用
- 数据库服务：MongoDB
- 反向代理：Nginx

## 数据模型设计

### 用户模型 (User)

用户模型存储系统用户信息，包括邮箱、密码等。

### 账户模型 (Account)

```javascript
{
  user: ObjectId,       // 关联用户ID
  name: String,         // 账户名称
  balance: Number,      // 当前余额
  currency: String,     // 货币类型，默认CNY
  createdAt: Date,      // 创建时间
  updatedAt: Date       // 更新时间
}
```

### 交易模型 (Transaction)

```javascript
{
  user: ObjectId,           // 关联用户ID
  accountId: ObjectId,      // 关联账户ID
  type: String,             // 交易类型：income(收入)、expense(支出)、transfer(转账)
  amount: Number,           // 交易金额
  description: String,      // 交易描述
  targetAccountId: ObjectId,// 目标账户ID（转账时使用）
  date: Date,               // 交易日期
  createdAt: Date,          // 创建时间
  updatedAt: Date           // 更新时间
}
```

## 模块设计

### 后端模块

1. **用户管理模块**
   - 用户注册
   - 用户登录
   - 令牌刷新
   - 用户信息管理

2. **账户管理模块**
   - 创建账户
   - 查询账户列表
   - 修改账户名称
   - 删除账户

3. **交易管理模块**
   - 添加交易记录
   - 查询交易记录
   - 交易统计

4. **统计分析模块**
   - 用户资金统计
   - 余额趋势分析
   - 账户余额排行

### 前端模块

1. **登录注册模块**
   - 用户登录
   - 用户注册

2. **账户管理模块**
   - 账户列表
   - 账户创建
   - 账户编辑
   - 账户删除

3. **交易管理模块**
   - 交易记录列表
   - 添加交易记录
   - 交易记录筛选

4. **统计分析模块**
   - 资金概览
   - 余额趋势图
   - 账户排行榜

## 核心流程

### 用户注册流程

1. 用户输入邮箱、密码等信息
2. 前端发送注册请求到后端
3. 后端验证邮箱是否已存在
4. 创建用户记录，加密存储密码
5. 返回注册成功信息

### 用户登录流程

1. 用户输入邮箱和密码
2. 前端发送登录请求到后端
3. 后端验证用户凭证
4. 生成 JWT 令牌
5. 返回用户信息和令牌

### 账户创建流程

1. 用户输入账户名称
2. 前端发送创建请求到后端
3. 后端验证账户名称是否重复
4. 创建账户记录
5. 返回账户信息

### 交易记录添加流程

1. 用户选择账户、交易类型、金额等信息
2. 前端发送添加交易请求到后端
3. 后端验证交易信息
4. 根据交易类型更新账户余额
   - 收入：增加账户余额
   - 支出：减少账户余额
   - 转账：减少源账户余额，增加目标账户余额
5. 创建交易记录
6. 返回更新后的账户信息

## API 接口设计

### 用户相关接口

- `POST /api/users/register` - 用户注册
- `POST /api/users/login` - 用户登录
- `POST /api/users/refreshToken` - 刷新令牌

### 账户相关接口

- `POST /api/accounts` - 创建账户
- `GET /api/accounts` - 获取账户列表
- `PUT /api/accounts/:accountId/name` - 修改账户名称
- `DELETE /api/accounts/:accountId` - 删除账户

### 交易相关接口

- `POST /api/accounts/:accountId/transaction` - 添加交易记录
- `GET /api/accounts/transactions` - 获取交易记录列表

### 统计相关接口

- `GET /api/accounts/stats` - 获取用户统计数据
- `GET /api/accounts/balance-trend` - 获取余额趋势数据
- `GET /api/accounts/balance-ranking` - 获取账户余额排行

## 安全设计

1. **身份验证**
   - 使用 JWT 进行用户身份验证
   - 令牌过期机制
   - 令牌刷新机制

2. **数据安全**
   - 密码加密存储
   - 请求参数验证
   - 防止跨站请求伪造

3. **权限控制**
   - 用户只能访问自己的数据
   - API 访问权限控制

## 技术栈

### 前端技术栈

- Vue 3 - 前端框架
- Vite - 构建工具
- Element Plus - UI 组件库
- Axios - HTTP 客户端
- Pinia - 状态管理
- Vue Router - 路由管理

### 后端技术栈

- Node.js - 运行环境
- Express - Web 框架
- MongoDB - 数据库
- Mongoose - ODM 工具
- JWT - 身份验证
- PM2 - 进程管理

### 部署技术

- Docker - 容器化
- Docker Compose - 多容器管理
- Nginx - 反向代理

## 序列图

### 用户登录序列图

```
用户 -> 前端: 输入登录信息
前端 -> 后端: 发送登录请求
后端 -> 数据库: 查询用户信息
数据库 -> 后端: 返回用户数据
后端 -> 后端: 验证密码
后端 -> 后端: 生成JWT令牌
后端 -> 前端: 返回用户信息和令牌
前端 -> 前端: 存储令牌
前端 -> 用户: 显示登录成功
```

### 添加交易记录序列图

```
用户 -> 前端: 填写交易信息
前端 -> 后端: 发送添加交易请求
后端 -> 数据库: 查询账户信息
数据库 -> 后端: 返回账户数据
后端 -> 后端: 验证交易信息
后端 -> 数据库: 更新账户余额
后端 -> 数据库: 创建交易记录
数据库 -> 后端: 返回操作结果
后端 -> 前端: 返回成功信息
前端 -> 用户: 显示交易添加成功
```

# 资金管理系统架构与设计文档（续）

## 模块设计（续）

### 前端模块

1. **登录注册模块**
   - 用户登录
   - 用户注册

2. **账户管理模块**
   - 账户列表
   - 账户创建
   - 账户编辑
   - 账户删除

3. **交易管理模块**
   - 交易记录列表
   - 添加交易记录
   - 交易记录筛选

4. **统计分析模块**
   - 资金概览
   - 余额趋势图
   - 账户排行榜

## API 接口设计

### 用户相关接口

- `POST /api/users/register` - 用户注册
- `POST /api/users/login` - 用户登录
- `POST /api/users/refreshToken` - 刷新令牌

### 账户相关接口

- `POST /api/accounts` - 创建账户
- `GET /api/accounts` - 获取账户列表
- `PUT /api/accounts/:accountId/name` - 修改账户名称
- `DELETE /api/accounts/:accountId` - 删除账户

### 交易相关接口

- `POST /api/accounts/:accountId/transaction` - 添加交易记录
- `GET /api/accounts/transactions` - 获取交易记录列表

### 统计相关接口

- `GET /api/accounts/stats` - 获取用户统计数据
- `GET /api/accounts/balance-trend` - 获取余额趋势数据
- `GET /api/accounts/balance-ranking` - 获取账户余额排行

## 数据流程

### 账户管理流程

1. 用户创建账户
   - 前端发送请求到 `/api/accounts`
   - 后端验证账户名称是否重复
   - 创建账户记录，初始余额为0
   - 返回账户信息

2. 查询账户列表
   - 前端发送请求到 `/api/accounts`
   - 后端查询当前用户的所有账户
   - 返回账户列表和分页信息

3. 修改账户名称
   - 前端发送请求到 `/api/accounts/:accountId/name`
   - 后端更新账户名称
   - 返回更新后的账户信息

4. 删除账户
   - 前端发送请求到 `/api/accounts/:accountId`
   - 后端删除账户及相关交易记录
   - 返回删除成功信息

### 交易管理流程

1. 添加交易记录
   - 前端发送请求到 `/api/accounts/:accountId/transaction`
   - 后端根据交易类型处理：
     - 收入：增加账户余额
     - 支出：减少账户余额
     - 转账：减少源账户余额，增加目标账户余额
   - 创建交易记录
   - 返回更新后的账户信息

2. 查询交易记录
   - 前端发送请求到 `/api/accounts/transactions`
   - 后端查询符合条件的交易记录
   - 返回交易记录列表和分页信息

## 技术实现细节

### 数据模型实现

系统使用 Mongoose 定义数据模型，例如交易模型：

```javascript
const TransactionSchema = new mongoose.Schema(
  {
    user: {
      type: mongoose.Schema.Types.ObjectId,
      ref: "User",
      required: true,
    },
    accountId: {
      type: mongoose.Schema.Types.ObjectId,
      required: true,
      ref: "Account",
    },
    type: {
      type: String,
      enum: ["income", "expense", "transfer"],
      required: true,
    },
    amount: {
      type: Number,
      required: true,
    },
    description: {
      type: String,
      maxlength: 256,
    },
    targetAccountId: {
      type: mongoose.Schema.Types.ObjectId,
      ref: "Account",
    },
    date: {
      type: Date,
      default: Date.now,
    },
  },
  {
    timestamps: true,
    toJSON: { virtuals: true },
    toObject: { virtuals: true },
  }
);
```

### 虚拟字段

使用 Mongoose 的虚拟字段功能关联账户信息：

```javascript
TransactionSchema.virtual("accountInfo", {
  ref: "Account",
  localField: "accountId",
  foreignField: "_id",
  justOne: true,
});

TransactionSchema.virtual("targetAccountInfo", {
  ref: "Account",
  localField: "targetAccountId",
  foreignField: "_id",
  justOne: true,
});
```

### 前端组件实现

前端使用 Vue 3 组合式 API 和 Element Plus 组件库实现用户界面，例如交易记录组件：

```vue
<template>
  <el-dialog
    v-model="dialogVisible"
    title="添加交易记录"
    width="600px"
    :close-on-click-modal="false"
    @close="onCloseDialog"
    @opened="onOpenedDialog"
  >
    <el-form
      :model="accountForm"
      :rules="accountRules"
      @submit.prevent
      label-position="top"
      ref="refAccountForm"
    >
      <!-- 表单内容 -->
    </el-form>
    <template #footer>
      <span class="dialog-footer">
        <el-button @click="dialogVisible = false">取消</el-button>
        <el-button type="primary" @click="onCreateTransactionOK">确认</el-button>
      </span>
    </template>
  </el-dialog>
</template>
```

### API 请求实现

前端使用 Axios 发送 API 请求：

```javascript
// 添加交易记录
export function postAccountTransactions(accountId, data) {
  return request({
    method: 'post',
    url: `/api/accounts/${accountId}/transaction`,
    data
  });
}

// 查询交易记录
export function getAccountTransactions(params) {
  return request({
    method: 'get',
    url: '/api/accounts/transactions',
    params
  });
}
```

### 后端控制器实现

后端使用 Express 控制器处理 API 请求：

```javascript
// 为指定账户添加交易记录
exports.addAccountTransaction = async (req, res) => {
  try {
    const accountId = req.params.accountId;
    const { type, amount, targetAccountId, date } = req.body;

    // 验证账户和交易信息
    // 根据交易类型更新账户余额
    // 创建交易记录
    // 返回更新后的账户信息
  } catch (error) {
    res.status(500).json({ error: "服务器错误" });
  }
};
```

## 系统优化

### 性能优化

1. **数据库索引**
   - 为常用查询字段创建索引
   - 使用复合索引优化多字段查询

2. **分页查询**
   - 实现分页机制减少数据传输量
   - 支持排序和筛选功能

3. **缓存策略**
   - 使用前端缓存减少请求次数
   - 实现数据预加载提升用户体验

### 安全优化

1. **输入验证**
   - 前后端同时进行输入验证
   - 防止 SQL 注入和 XSS 攻击

2. **权限控制**
   - 实现细粒度的权限控制
   - 确保用户只能访问自己的数据

3. **日志记录**
   - 记录关键操作日志
   - 实现异常监控和报警机制

## 扩展性设计

系统设计考虑了未来的扩展需求：

1. **多币种支持**
   - 账户模型已包含货币类型字段
   - 可扩展支持多币种账户和汇率转换

2. **预算管理**
   - 可添加预算模型关联账户
   - 实现预算跟踪和超支提醒

3. **报表功能**
   - 可扩展更丰富的统计分析功能
   - 支持自定义报表和导出功能

## 总结

本资金管理系统采用现代化的前后端分离架构，实现了账户管理、交易记录、统计分析等核心功能。系统设计注重数据安全、用户体验和扩展性，为用户提供了便捷的个人财务管理工具。

通过合理的模块划分和数据模型设计，系统具有良好的可维护性和可扩展性，能够满足用户日常的资金管理需求，并可根据需求进一步扩展功能。