# AKA Music 音乐站点 - 完整技术方案

## 项目概述

AKA Music 是一个现代化的全栈音乐内容聚合平台，集成了音乐播放、海报设计、内容管理、用户系统、支付系统等核心功能。项目采用前后端分离架构，支持高并发、高可用部署。

## 🏗️ 系统架构

### 整体架构图

```mermaid
graph TB
    subgraph 前端层
        A[Vue3 + TypeScript] --> B[Vite构建工具]
        B --> C[Ant Design Vue + Arco Design]
        C --> D[Pinia状态管理]
        D --> E[Vue Router路由]
    end
    
    subgraph API网关
        F[Nginx反向代理] --> G[HTTPS证书]
        G --> H[负载均衡]
    end
    
    subgraph 后端层
        I[Flask框架] --> J[RESTful API]
        J --> K[JWT认证]
        K --> L[Socket.IO实时通信]
    end
    
    subgraph 数据层
        M[PostgreSQL] --> N[Redis缓存]
        N --> O[文件存储系统]
        O --> P[CDN加速]
    end
    
    subgraph 监控运维
        Q[SkyWalking APM] --> R[Prometheus监控]
        R --> S[Grafana可视化]
    end
    
    A --> F
    F --> I
    I --> M
    L --> N
    S --> Q
```

## 📁 项目结构详解

### 根目录结构
```
aka_music/
├── frontend/                 # 前端项目目录
│   ├── src/
│   │   ├── api/             # API接口封装
│   │   ├── assets/          # 静态资源
│   │   ├── components/      # 通用组件
│   │   ├── views/          # 页面组件
│   │   ├── store/          # 状态管理
│   │   ├── router/         # 路由配置
│   │   └── utils/          # 工具函数
│   ├── public/             # 公共资源
│   ├── package.json        # 前端依赖
│   └── vite.config.ts      # Vite配置
├── backend/                 # 后端项目目录
│   ├── app/
│   │   ├── models/         # 数据模型
│   │   ├── routes/         # 路由控制器
│   │   ├── services/       # 业务逻辑
│   │   ├── utils/          # 工具函数
│   │   └── scripts/        # 数据脚本
│   ├── static/             # 静态文件
│   ├── tests/              # 测试用例
│   └── requirements.txt    # Python依赖
├── docker-compose.yml      # 容器编排
└── README.md              # 项目文档
```

## 🚀 技术栈详情

### 前端技术栈

#### 核心框架
- **Vue 3.4+** - 渐进式JavaScript框架
- **TypeScript 5.0+** - 类型安全的JavaScript超集
- **Vite 6.0+** - 下一代前端构建工具

#### UI框架与组件库
- **Ant Design Vue 4.2+** - 企业级UI设计体系
- **Arco Design Vue 2.57+** - 字节跳动开源设计系统
- **Element Plus 2.9+** - 基于Vue 3的组件库
- **UnoCSS** - 原子化CSS引擎

#### 状态管理
- **Pinia 2.3+** - Vue官方状态管理库
- **Pinia Plugin Persistedstate** - 状态持久化插件

#### 路由与导航
- **Vue Router 4.2+** - Vue官方路由管理器

#### 网络请求
- **Axios 1.8+** - 基于Promise的HTTP客户端
- **Socket.IO Client 4.8+** - 实时双向通信

#### 富文本编辑器
- **TinyMCE 6.8+** - 功能丰富的富文本编辑器
- **WangEditor 5.1+** - 轻量级富文本编辑器

#### 图形与可视化
- **LeaferJS 1.5+** - 高性能2D图形渲染引擎
- **Fabric.js** - 强大的Canvas交互库
- **ECharts** - 数据可视化图表库

#### 多媒体处理
- **Video.js 8.21+** - HTML5视频播放器
- **Hls.js** - HTTP Live Streaming支持

#### 构建与开发工具
- **Unplugin Auto Import** - 自动按需导入
- **Unplugin Vue Components** - 自动组件注册
- **Vite Plugin SVG Icons** - SVG图标管理

### 后端技术栈

#### Web框架
- **Flask 3.1+** - 轻量级Python Web框架
- **Flask-RESTful 0.3+** - RESTful API扩展
- **Flask-SocketIO 5.3+** - WebSocket实时通信

#### 数据库与缓存
- **PostgreSQL 14+** - 关系型数据库
- **pgvector** - PostgreSQL向量扩展
- **Redis 7+** - 内存数据结构存储
- **SQLAlchemy 2.0+** - Python SQL工具包

#### 认证与安全
- **PyJWT 2.3+** - JSON Web Token实现
- **Flask-JWT-Extended** - JWT认证扩展
- **bcrypt 4.2+** - 密码哈希库
- **passlib 1.7+** - 密码哈希框架

#### 文件处理
- **Pillow** - 图像处理库
- **OpenCV 4.8+** - 计算机视觉库
- **face_recognition** - 人脸识别库
- **rembg** - 背景移除工具

#### 机器学习与AI
- **PyTorch 2.0+** - 深度学习框架
- **Transformers** - 预训练模型库
- **EasyOCR 1.7+** - OCR文字识别
- **Yolov8** - 目标检测模型

#### 支付集成
- **Alipay SDK** - 支付宝支付集成
- **RSA加密** - 数字签名验证

#### 监控与日志
- **Apache SkyWalking** - APM应用性能监控
- **OpenTelemetry** - 可观测性框架
- **Prometheus + Grafana** - 监控告警系统

## 🔧 核心功能模块

### 1. 用户认证系统

#### 认证流程图
```mermaid
sequenceDiagram
    participant U as 用户
    participant F as 前端
    participant B as 后端
    participant DB as 数据库
    participant JWT as JWT服务

    U->>F: 输入用户名密码
    F->>B: POST /api/auth/login
    B->>DB: 验证用户信息
    DB-->>B: 用户数据
    B->>JWT: 生成JWT令牌
    JWT-->>B: 返回token
    B-->>F: 返回{token, user}
    F->>F: 存储token到localStorage
    F->>B: 后续请求携带Authorization头
    B->>JWT: 验证token有效性
    JWT-->>B: 验证结果
```

#### 核心代码实现

**前端认证服务**:
```typescript
// src/api/auth.ts
import request from '@/utils/request'

export interface LoginData {
  username: string
  password: string
}

export interface UserInfo {
  id: number
  username: string
  email: string
  avatar: string
  gold: number
  vip_level: number
}

export const authApi = {
  login: (data: LoginData) =>
    request.post('/api/auth/login', data),
    
  register: (data: LoginData & { email: string }) =>
    request.post('/api/auth/register', data),
    
  getUserInfo: () =>
    request.get<UserInfo>('/api/auth/me'),
    
  refreshToken: () =>
    request.post('/api/auth/refresh')
}
```

**后端认证实现**:
```python
# backend/app/routes/auth.py
from flask import Blueprint, request, jsonify
from flask_jwt_extended import create_access_token, jwt_required, get_jwt_identity
from werkzeug.security import check_password_hash, generate_password_hash
from app.models.user import User

auth_bp = Blueprint('auth', __name__)

@auth_bp.route('/api/auth/login', methods=['POST'])
def login():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')
    
    user = User.query.filter_by(username=username).first()
    
    if not user or not check_password_hash(user.password, password):
        return jsonify({'error': 'Invalid credentials'}), 401
    
    access_token = create_access_token(identity=user.id)
    
    return jsonify({
        'access_token': access_token,
        'user': {
            'id': user.id,
            'username': user.username,
            'email': user.email,
            'avatar': user.avatar,
            'gold': user.gold,
            'vip_level': user.vip_level
        }
    })

@auth_bp.route('/api/auth/register', methods=['POST'])
def register():
    data = request.get_json()
    
    # 检查用户名是否已存在
    if User.query.filter_by(username=data['username']).first():
        return jsonify({'error': 'Username already exists'}), 400
    
    # 创建新用户
    user = User(
        username=data['username'],
        email=data['email'],
        password=generate_password_hash(data['password']),
        gold=100  # 初始赠送100金币
    )
    
    db.session.add(user)
    db.session.commit()
    
    access_token = create_access_token(identity=user.id)
    
    return jsonify({
        'access_token': access_token,
        'user': {
            'id': user.id,
            'username': user.username,
            'email': user.email,
            'gold': user.gold,
            'vip_level': 0
        }
    }), 201
```

### 2. 音乐内容管理

#### 音乐数据结构
```typescript
// src/types/music.ts
export interface Music {
  id: number
  title: string
  artist: string
  album: string
  cover: string
  duration: number
  file_url: string
  category_id: number
  tags: string[]
  play_count: number
  like_count: number
  is_liked: boolean
  created_at: string
}

export interface MusicCategory {
  id: number
  name: string
  description: string
  cover: string
  music_count: number
  sort_order: number
}
```

#### 音乐播放器组件
```vue
<!-- src/components/MusicPlayer.vue -->
<template>
  <div class="music-player" :class="{ 'is-playing': isPlaying }">
    <div class="player-controls">
      <button @click="togglePlay" class="play-btn">
        <Icon :icon="isPlaying ? 'pause' : 'play'" />
      </button>
      
      <div class="progress-bar">
        <div class="progress" :style="{ width: progress + '%' }"></div>
        <input 
          type="range" 
          v-model="currentTime" 
          :max="duration"
          @input="seek"
        />
      </div>
      
      <div class="time-display">
        {{ formatTime(currentTime) }} / {{ formatTime(duration) }}
      </div>
      
      <div class="volume-control">
        <Icon icon="volume" />
        <input 
          type="range" 
          v-model="volume"
          min="0" 
          max="1" 
          step="0.1"
        />
      </div>
    </div>
    
    <audio 
      ref="audioRef"
      :src="currentMusic?.file_url"
      @timeupdate="updateTime"
      @loadedmetadata="onLoadedMetadata"
      @ended="onEnded"
    ></audio>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, watch, onMounted } from 'vue'
import { useMusicStore } from '@/store/music'

const musicStore = useMusicStore()
const audioRef = ref<HTMLAudioElement>()

const isPlaying = computed(() => musicStore.isPlaying)
const currentMusic = computed(() => musicStore.currentMusic)
const volume = computed({
  get: () => musicStore.volume,
  set: (val) => musicStore.setVolume(val)
})

const currentTime = ref(0)
const duration = ref(0)

const progress = computed(() => 
  duration.value ? (currentTime.value / duration.value) * 100 : 0
)

const togglePlay = () => {
  if (isPlaying.value) {
    audioRef.value?.pause()
  } else {
    audioRef.value?.play()
  }
  musicStore.togglePlay()
}

const updateTime = () => {
  currentTime.value = audioRef.value?.currentTime || 0
}

const onLoadedMetadata = () => {
  duration.value = audioRef.value?.duration || 0
}

const seek = () => {
  if (audioRef.value) {
    audioRef.value.currentTime = currentTime.value
  }
}

const formatTime = (seconds: number): string => {
  const mins = Math.floor(seconds / 60)
  const secs = Math.floor(seconds % 60)
  return `${mins}:${secs.toString().padStart(2, '0')}`
}

onMounted(() => {
  if (audioRef.value) {
    audioRef.value.volume = volume.value
  }
})
</script>
```

### 3. 海报设计系统

#### 设计器架构
```mermaid
graph LR
    A[用户界面] --> B[画布管理]
    B --> C[元素系统]
    C --> D[图层管理]
    C --> E[样式系统]
    C --> F[交互控制]
    
    D --> G[文本元素]
    D --> H[图片元素]
    D --> I[形状元素]
    D --> J[背景元素]
    
    E --> K[颜色系统]
    E --> L[字体系统]
    E --> M[动画系统]
    
    F --> N[拖拽控制]
    F --> O[缩放旋转]
    F --> P[对齐分布]
```

#### 核心设计器组件
```vue
<!-- src/views/poster/PosterLanding.vue -->
<template>
  <div class="poster-designer">
    <!-- 顶部工具栏 -->
    <div class="toolbar">
      <div class="tool-group">
        <button @click="addText" class="tool-btn">
          <Icon icon="text" /> 添加文字
        </button>
        <button @click="addImage" class="tool-btn">
          <Icon icon="image" /> 添加图片
        </button>
        <button @click="addShape" class="tool-btn">
          <Icon icon="shape" /> 添加形状
        </button>
      </div>
      
      <div class="tool-group">
        <button @click="undo" :disabled="!canUndo">
          <Icon icon="undo" /> 撤销
        </button>
        <button @click="redo" :disabled="!canRedo">
          <Icon icon="redo" /> 重做
        </button>
      </div>
      
      <div class="tool-group">
        <button @click="save" class="save-btn">
          <Icon icon="save" /> 保存
        </button>
        <button @click="exportImage" class="export-btn">
          <Icon icon="download" /> 导出
        </button>
      </div>
    </div>
    
    <!-- 左侧元素面板 -->
    <div class="sidebar">
      <div class="panel">
        <h3>元素列表</h3>
        <draggable 
          v-model="elements" 
          item-key="id"
          @change="onElementReorder"
        >
          <template #item="{ element, index }">
            <div 
              class="element-item"
              :class="{ active: selectedElement?.id === element.id }"
              @click="selectElement(element)"
            >
              <span>{{ element.type }} - {{ element.name }}</span>
              <button @click.stop="deleteElement(element)">
                <Icon icon="delete" />
              </button>
            </div>
          </template>
        </draggable>
      </div>
    </div>
    
    <!-- 中间画布区域 -->
    <div class="canvas-container">
      <div 
        class="canvas"
        :style="canvasStyle"
        @drop="onDrop"
        @dragover.prevent
      >
        <canvas-element
          v-for="element in elements"
          :key="element.id"
          :element="element"
          :is-selected="selectedElement?.id === element.id"
          @select="selectElement"
          @update="updateElement"
        />
      </div>
    </div>
    
    <!-- 右侧属性面板 -->
    <div class="properties-panel">
      <element-properties
        v-if="selectedElement"
        :element="selectedElement"
        @update="updateElement"
      />
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted } from 'vue'
import { usePosterStore } from '@/store/poster'
import Draggable from 'vuedraggable'

const posterStore = usePosterStore()

const elements = computed(() => posterStore.elements)
const selectedElement = computed(() => posterStore.selectedElement)
const canUndo = computed(() => posterStore.canUndo)
const canRedo = computed(() => posterStore.canRedo)

const canvasStyle = computed(() => ({
  width: posterStore.canvas.width + 'px',
  height: posterStore.canvas.height + 'px',
  backgroundColor: posterStore.canvas.backgroundColor
}))

const addText = () => {
  posterStore.addElement({
    type: 'text',
    content: '双击编辑文字',
    x: 100,
    y: 100,
    fontSize: 24,
    color: '#000000'
  })
}

const addImage = () => {
  // 打开图片选择器
  const input = document.createElement('input')
  input.type = 'file'
  input.accept = 'image/*'
  input.onchange = (e) => {
    const file = (e.target as HTMLInputElement).files?.[0]
    if (file) {
      const reader = new FileReader()
      reader.onload = (e) => {
        posterStore.addElement({
          type: 'image',
          src: e.target?.result as string,
          x: 50,
          y: 50,
          width: 200,
          height: 200
        })
      }
      reader.readAsDataURL(file)
    }
  }
  input.click()
}

const selectElement = (element: any) => {
  posterStore.selectElement(element)
}

const updateElement = (updates: any) => {
  if (selectedElement.value) {
    posterStore.updateElement(selectedElement.value.id, updates)
  }
}

const deleteElement = (element: any) => {
  posterStore.deleteElement(element.id)
}

const save = () => {
  posterStore.save()
}

const exportImage = () => {
  posterStore.exportAsImage()
}
</script>
```

### 4. 支付系统

#### 支付流程图
```mermaid
sequenceDiagram
    participant U as 用户
    participant F as 前端
    participant B as 后端
    participant P as 支付网关
    participant DB as 数据库

    U->>F: 选择充值金额
    F->>B: POST /api/payment/create
    B->>DB: 创建订单
    DB-->>B: 订单ID
    B->>P: 调用支付宝API
    P-->>B: 返回支付参数
    B-->>F: 返回支付信息
    F->>U: 跳转到支付页面
    U->>P: 完成支付
    P->>B: 异步通知支付结果
    B->>DB: 更新订单状态
    B->>DB: 更新用户金币
    B-->>P: 返回成功响应
    B-->>F: WebSocket通知前端
```

#### 支付服务实现
```python
# backend/app/services/payment.py
import os
import json
import time
from datetime import datetime
from alipay.aop.api.AlipayClientConfig import AlipayClientConfig
from alipay.aop.api.DefaultAlipayClient import DefaultAlipayClient
from alipay.aop.api.domain.AlipayTradePagePayModel import AlipayTradePagePayModel
from alipay.aop.api.request.AlipayTradePagePayRequest import AlipayTradePagePayRequest
from alipay.aop.api.request.AlipayTradeQueryRequest import AlipayTradeQueryRequest
from app.models.payment import PaymentOrder
from app.models.user import User
from app import db

class PaymentService:
    def __init__(self):
        self.alipay_client = self._init_alipay_client()
    
    def _init_alipay_client(self):
        """初始化支付宝客户端"""
        alipay_client_config = AlipayClientConfig()
        alipay_client_config.server_url = 'https://openapi.alipay.com/gateway.do'
        alipay_client_config.app_id = os.getenv('ALIPAY_APP_ID')
        alipay_client_config.app_private_key = os.getenv('ALIPAY_PRIVATE_KEY')
        alipay_client_config.alipay_public_key = os.getenv('ALIPAY_PUBLIC_KEY')
        return DefaultAlipayClient(alipay_client_config)
    
    def create_order(self, user_id: int, amount: float, product_name: str) -> dict:
        """创建支付订单"""
        # 生成订单号
        order_no = f"AKA{int(time.time())}{user_id}"
        
        # 创建订单记录
        order = PaymentOrder(
            order_no=order_no,
            user_id=user_id,
            amount=amount,
            product_name=product_name,
            status='pending',
            created_at=datetime.utcnow()
        )
        db.session.add(order)
        db.session.commit()
        
        # 构建支付请求
        model = AlipayTradePagePayModel()
        model.out_trade_no = order_no
        model.total_amount = str(amount)
        model.subject = product_name
        model.product_code = "FAST_INSTANT_TRADE_PAY"
        
        request = AlipayTradePagePayRequest(biz_model=model)
        request.return_url = os.getenv('ALIPAY_RETURN_URL')
        request.notify_url = os.getenv('ALIPAY_NOTIFY_URL')
        
        # 获取支付表单
        form = self.alipay_client.page_execute(request)
        
        return {
            'order_no': order_no,
            'form': form,
            'amount': amount,
            'product_name': product_name
        }
    
    def verify_payment(self, data: dict) -> bool:
        """验证支付结果"""
        try:
            # 查询订单状态
            request = AlipayTradeQueryRequest()
            request.biz_content = json.dumps({
                'out_trade_no': data['out_trade_no']
            })
            
            response = self.alipay_client.execute(request)
            response_data = json.loads(response)
            
            if response_data.get('alipay_trade_query_response', {}).get('trade_status') == 'TRADE_SUCCESS':
                return self._update_order_status(data['out_trade_no'])
            
            return False
        except Exception as e:
            print(f"支付验证失败: {e}")
            return False
    
    def _update_order_status(self, order_no: str) -> bool:
        """更新订单状态并增加用户金币"""
        order = PaymentOrder.query.filter_by(order_no=order_no).first()
        if not order or order.status == 'completed':
            return False
        
        # 更新订单状态
        order.status = 'completed'
        order.completed_at = datetime.utcnow()
        
        # 增加用户金币
        user = User.query.get(order.user_id)
        if user:
            user.gold += int(order.amount * 100)  # 1元=100金币
        
        db.session.commit()
        return True
```

## 📊 数据库设计

### 核心数据模型

#### 用户表结构
```sql
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    avatar VARCHAR(255),
    gold INTEGER DEFAULT 100,
    vip_level INTEGER DEFAULT 0,
    vip_expires_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_users_username ON users(username);
CREATE INDEX idx_users_email ON users(email);
```

#### 音乐内容表
```sql
CREATE TABLE music_content (
    id SERIAL PRIMARY KEY,
    title VARCHAR(255) NOT NULL,
    artist VARCHAR(255),
    album VARCHAR(255),
    cover_url VARCHAR(500),
    file_url VARCHAR(500) NOT NULL,
    duration INTEGER,
    category_id INTEGER REFERENCES categories(id),
    play_count INTEGER DEFAULT 0,
    like_count INTEGER DEFAULT 0,
    tags TEXT[],
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_music_category ON music_content(category_id);
CREATE INDEX idx_music_created ON music_content(created_at DESC);
```

#### 支付订单表
```sql
CREATE TABLE payment_orders (
    id SERIAL PRIMARY KEY,
    order_no VARCHAR(64) UNIQUE NOT NULL,
    user_id INTEGER REFERENCES users(id),
    amount DECIMAL(10,2) NOT NULL,
    product_name VARCHAR(255),
    status VARCHAR(20) DEFAULT 'pending',
    payment_method VARCHAR(20),
    transaction_id VARCHAR(64),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP,
    notified_at TIMESTAMP
);

CREATE INDEX idx_orders_user ON payment_orders(user_id);
CREATE INDEX idx_orders_status ON payment_orders(status);
CREATE INDEX idx_orders_created ON payment_orders(created_at DESC);
```

## 🛠️ 开发环境配置

### 前端环境配置

#### 1. 环境准备
```bash
# 安装Node.js 18+
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt-get install -y nodejs

# 验证安装
node --version  # v18.x.x
npm --version   # 9.x.x
```

#### 2. 项目初始化
```bash
# 克隆项目
git clone https://github.com/your-username/aka-music.git
cd aka-music/frontend

# 安装依赖
npm install

# 启动开发服务器
npm run dev
```

#### 3. 环境变量配置
```bash
# 创建环境配置文件
cp .env.example .env.local

# 编辑环境变量
VITE_API_URL=https://localhost:5000
VITE_SOCKET_URL=https://localhost:5000
VITE_CDN_URL=https://cdn.example.com
```

### 后端环境配置

#### 1. Python环境
```bash
# 安装Python 3.8+
sudo apt update
sudo apt install python3.8 python3.8-venv python3.8-dev

# 创建虚拟环境
python3.8 -m venv venv
source venv/bin/activate

# 升级pip
pip install --upgrade pip
```

#### 2. 数据库配置
```bash
# 安装PostgreSQL
sudo apt install postgresql postgresql-contrib

# 创建数据库
sudo -u postgres psql
CREATE DATABASE aka_music;
CREATE USER aka_user WITH PASSWORD 'your_password';
GRANT ALL PRIVILEGES ON DATABASE aka_music TO aka_user;
\q

# 安装Redis
sudo apt install redis-server
sudo systemctl start redis-server
```

#### 3. 项目初始化
```bash
# 进入后端目录
cd aka-music/backend

# 安装依赖
pip install -r requirements.txt

# 数据库迁移
flask db init
flask db migrate -m "Initial migration"
flask db upgrade

# 导入初始数据
python scripts/init_db.py
python scripts/init_category_data.py

# 启动开发服务器
flask run --debug
```

## 🚀 部署指南

### Docker容器化部署

#### 1. Docker Compose配置
```yaml
# docker-compose.yml
version: '3.8'

services:
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - frontend
      - backend

  frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile
    environment:
      - VITE_API_URL=https://api.example.com
    depends_on:
      - backend

  backend:
    build:
      context: ./backend
      dockerfile: Dockerfile
    environment:
      - FLASK_ENV=production
      - DATABASE_URL=postgresql://aka_user:password@postgres:5432/aka_music
      - REDIS_URL=redis://redis:6379/0
    depends_on:
      - postgres
      - redis

  postgres:
    image: postgres:14-alpine
    environment:
      - POSTGRES_DB=aka_music
      - POSTGRES_USER=aka_user
      - POSTGRES_PASSWORD=password
    volumes:
      - postgres_data:/var/lib/postgresql/data

  redis:
    image: redis:7-alpine
    volumes:
      - redis_data:/data

volumes:
  postgres_data:
  redis_data:
```

#### 2. 生产环境部署脚本
```bash
#!/bin/bash
# deploy.sh

# 设置环境变量
export ENV=production
export DOMAIN=your-domain.com

# 拉取最新代码
git pull origin main

# 构建前端
npm run build

# 构建Docker镜像
docker-compose build

# 启动服务
docker-compose up -d

# 检查服务状态
docker-compose ps

# 查看日志
docker-compose logs -f
```

### 监控与运维

#### 1. 应用性能监控
```yaml
# docker-compose.monitoring.yml
version: '3.8'

services:
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana_data:/var/lib/grafana

  skywalking-oap:
    image: apache/skywalking-oap-server:latest
    ports:
      - "11800:11800"
      - "12800:12800"

  skywalking-ui:
    image: apache/skywalking-ui:latest
    ports:
      - "8080:8080"
    environment:
      - SW_OAP_ADDRESS=http://skywalking-oap:12800

volumes:
  grafana_data:
```

## 📱 移动端适配

### 响应式设计策略

#### 1. 断点配置
```typescript
// src/config/responsive.ts
export const breakpoints = {
  xs: 480,   // 手机竖屏
  sm: 768,   // 手机横屏/小平板
  md: 992,   // 平板
  lg: 1200,  // 小屏桌面
  xl: 1600   // 大屏桌面
}

export const responsive = {
  mobile: `@media (max-width: ${breakpoints.sm}px)`,
  tablet: `@media (min-width: ${breakpoints.sm + 1}px) and (max-width: ${breakpoints.md}px)`,
  desktop: `@media (min-width: ${breakpoints.md + 1}px)`
}
```

#### 2. 移动端组件优化
```vue
<!-- src/components/MobileMusicPlayer.vue -->
<template>
  <div class="mobile-player" :class="{ expanded: isExpanded }">
    <!-- 迷你播放器 -->
    <div class="mini-player" @click="expand">
      <img :src="currentMusic?.cover" class="mini-cover" />
      <div class="mini-info">
        <div class="title">{{ currentMusic?.title }}</div>
        <div class="artist">{{ currentMusic?.artist }}</div>
      </div>
      <button @click.stop="togglePlay" class="mini-play">
        <Icon :name="isPlaying ? 'pause' : 'play'" />
      </button>
    </div>
    
    <!-- 全屏播放器 -->
    <div class="full-player" v-if="isExpanded">
      <div class="player-header">
        <button @click="collapse">
          <Icon name="chevron-down" />
        </button>
        <div class="now-playing">正在播放</div>
        <button>
          <Icon name="more-horizontal" />
        </button>
      </div>
      
      <div class="cover-container">
        <img 
          :src="currentMusic?.cover" 
          class="cover-image"
          :class="{ rotating: isPlaying }"
        />
      </div>
      
      <div class="player-controls">
        <div class="progress-section">
          <input 
            type="range"
            v-model="currentTime"
            :max="duration"
            @input="seek"
            class="progress-slider"
          />
          <div class="time-display">
            <span>{{ formatTime(currentTime) }}</span>
            <span>{{ formatTime(duration) }}</span>
          </div>
        </div>
        
        <div class="control-buttons">
          <button @click="previous">
            <Icon name="skip-back" />
          </button>
          <button @click="togglePlay" class="play-button">
            <Icon :name="isPlaying ? 'pause-circle' : 'play-circle'" />
          </button>
          <button @click="next">
            <Icon name="skip-forward" />
          </button>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
@media (max-width: 768px) {
  .mobile-player {
    position: fixed;
    bottom: 0;
    left: 0;
    right: 0;
    z-index: 1000;
  }
  
  .mini-player {
    display: flex;
    align-items: center;
    padding: 8px 16px;
    background: #fff;
    border-top: 1px solid #eee;
  }
  
  .full-player {
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: linear-gradient(to bottom, #667eea 0%, #764ba2 100%);
    color: white;
  }
  
  .cover-image.rotating {
    animation: rotate 20s linear infinite;
  }
  
  @keyframes rotate {
    from { transform: rotate(0deg); }
    to { transform: rotate(360deg); }
  }
}
</style>
```

## 🔒 安全最佳实践

### 1. 认证安全
```typescript
// src/utils/auth.ts
import CryptoJS from 'crypto-js'

class AuthSecurity {
  private static readonly SECRET_KEY = import.meta.env.VITE_AUTH_SECRET
  
  static encryptToken(token: string): string {
    return CryptoJS.AES.encrypt(token, this.SECRET_KEY).toString()
  }
  
  static decryptToken(encryptedToken: string): string {
    const bytes = CryptoJS.AES.decrypt(encryptedToken, this.SECRET_KEY)
    return bytes.toString(CryptoJS.enc.Utf8)
  }
  
  static isTokenExpired(token: string): boolean {
    try {
      const payload = JSON.parse(atob(token.split('.')[1]))
      return Date.now() >= payload.exp * 1000
    } catch {
      return true
    }
  }
}
```

### 2. 输入验证
```python
# backend/app/utils/validation.py
from marshmallow import Schema, fields, validate

class UserRegistrationSchema(Schema):
    username = fields.Str(
        required=True,
        validate=[
            validate.Length(min=3, max=50),
            validate.Regexp(r'^[a-zA-Z0-9_]+$', error='用户名只能包含字母、数字和下划线')
        ]
    )
    email = fields.Email(required=True)
    password = fields.Str(
        required=True,
        validate=validate.Length(min=8, max=128)
    )

class PaymentSchema(Schema):
    amount = fields.Decimal(
        required=True,
        validate=validate.Range(min=0.01, max=10000)
    )
    product_id = fields.Int(required=True)
```

### 3. 文件上传安全
```python
# backend/app/utils/file_security.py
import os
import magic
from werkzeug.utils import secure_filename

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'mp3', 'wav'}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

def validate_file(file):
    """验证上传文件的安全性"""
    
    # 检查文件扩展名
    filename = secure_filename(file.filename)
    if '.' not in filename:
        return False, "文件名无效"
    
    ext = filename.rsplit('.', 1)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        return False, f"不支持的文件类型: {ext}"
    
    # 检查文件大小
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)
    
    if file_size > MAX_FILE_SIZE:
        return False, "文件过大"
    
    # 检查文件MIME类型
    file_content = file.read(2048)
    file.seek(0)
    
    mime = magic.from_buffer(file_content, mime=True)
    allowed_mimes = {
        'image/png', 'image/jpeg', 'image/gif',
        'audio/mpeg', 'audio/wav'
    }
    
    if mime not in allowed_mimes:
        return False, "文件类型不匹配"
    
    return True, filename
```

## 📈 性能优化

### 1. 前端性能优化

#### 代码分割与懒加载
```typescript
// src/router/index.ts
const routes = [
  {
    path: '/music',
    component: () => import('@/views/music/Mulist.vue'),
    children: [
      {
        path: ':category_id',
        component: () => import('@/views/music/Mugrid.vue'),
        props: true
      }
    ]
  },
  {
    path: '/poster',
    component: () => import('@/views/poster/PosterLanding.vue'),
    meta: { preload: true }
  }
]
```

#### 图片优化
```vue
<!-- src/components/OptimizedImage.vue -->
<template>
  <img
    :src="src"
    :srcset="srcset"
    :sizes="sizes"
    :alt="alt"
    loading="lazy"
    @load="onLoad"
    @error="onError"
  />
</template>

<script setup lang="ts">
const props = defineProps<{
  src: string
  alt: string
  width: number
  height: number
}>()

const srcset = computed(() => {
  return `
    ${props.src}?w=320 320w,
    ${props.src}?w=640 640w,
    ${props.src}?w=1024 1024w
  `
})

const sizes = computed(() => {
  return '(max-width: 320px) 320px, (max-width: 640px) 640px, 1024px'
})
</script>
```

### 2. 后端性能优化

#### 数据库查询优化
```python
# backend/app/models/music.py
class MusicContent(db.Model):
    __tablename__ = 'music_content'
    
    # 添加索引
    __table_args__ = (
        db.Index('idx_music_category_created', 'category_id', 'created_at'),
        db.Index('idx_music_active_play', 'is_active', 'play_count'),
    )
    
    @classmethod
    def get_by_category(cls, category_id, page=1, per_page=20):
        """优化分页查询"""
        return cls.query.filter_by(
            category_id=category_id,
            is_active=True
        ).order_by(
            cls.created_at.desc()
        ).paginate(
            page=page,
            per_page=per_page,
            error_out=False
        )
    
    @classmethod
    def get_popular(cls, limit=10):
        """获取热门音乐"""
        return cls.query.filter_by(
            is_active=True
        ).order_by(
            cls.play_count.desc()
        ).limit(limit).all()
```

#### 缓存策略
```python
# backend/app/utils/cache.py
import redis
import json
from functools import wraps

redis_client = redis.Redis(host='localhost', port=6379, db=0)

def cache_result(expiration=3600):
    """缓存装饰器"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 生成缓存键
            cache_key = f"{func.__name__}:{str(args)}:{str(kwargs)}"
            
            # 尝试从缓存获取
            cached = redis_client.get(cache_key)
            if cached:
                return json.loads(cached)
            
            # 执行函数并缓存结果
            result = func(*args, **kwargs)
            redis_client.setex(
                cache_key, 
                expiration, 
                json.dumps(result, default=str)
            )
            
            return result
        return wrapper
    return decorator

# 使用示例
@cache_result(expiration=1800)
def get_music_list(category_id, page=1):
    return MusicContent.get_by_category(category_id, page)
```

## 🧪 测试策略

### 1. 前端测试

#### 单元测试示例
```typescript
// tests/unit/components/MusicPlayer.spec.ts
import { describe, it, expect, vi } from 'vitest'
import { mount } from '@vue/test-utils'
import MusicPlayer from '@/components/MusicPlayer.vue'

vi.mock('@/store/music', () => ({
  useMusicStore: () => ({
    currentMusic: {
      title: '测试歌曲',
      artist: '测试艺术家',
      file_url: '/test.mp3'
    },
    isPlaying: false,
    volume: 0.8,
    togglePlay: vi.fn()
  })
}))

describe('MusicPlayer', () => {
  it('renders correctly', () => {
    const wrapper = mount(MusicPlayer)
    expect(wrapper.find('.player-controls')).toBeTruthy()
  })
  
  it('toggles play state', async () => {
    const wrapper = mount(MusicPlayer)
    const playBtn = wrapper.find('.play-btn')
    
    await playBtn.trigger('click')
    expect(useMusicStore().togglePlay).toHaveBeenCalled()
  })
})
```

### 2. 后端测试

#### API测试
```python
# tests/test_music_api.py
import pytest
from app import create_app
from app.models.music import MusicContent

@pytest.fixture
def client():
    app = create_app('testing')
    with app.test_client() as client:
        yield client

@pytest.fixture
def auth_headers(client):
    # 创建测试用户并获取token
    response = client.post('/api/auth/register', json={
        'username': 'testuser',
        'email': 'test@example.com',
        'password': 'testpass123'
    })
    token = response.json['access_token']
    return {'Authorization': f'Bearer {token}'}

def test_get_music_list(client, auth_headers):
    # 创建测试数据
    music = MusicContent(
        title='测试歌曲',
        artist='测试艺术家',
        file_url='/test.mp3',
        category_id=1
    )
    db.session.add(music)
    db.session.commit()
    
    # 测试API
    response = client.get('/api/music?category_id=1', headers=auth_headers)
    assert response.status_code == 200
    assert len(response.json['items']) == 1
    assert response.json['items'][0]['title'] == '测试歌曲'

def test_music_playback(client, auth_headers):
    # 测试播放计数增加
    music = MusicContent.query.first()
    initial_count = music.play_count
    
    response = client.post(
        f'/api/music/{music.id}/play',
        headers=auth_headers
    )
    
    assert response.status_code == 200
    assert MusicContent.query.get(music.id).play_count == initial_count + 1
```

## 📚 API文档

### RESTful API规范

#### 音乐内容API
```
GET /api/music
获取音乐列表

参数:
- category_id: 分类ID (可选)
- page: 页码 (默认: 1)
- per_page: 每页数量 (默认: 20)
- sort: 排序方式 (created_at, play_count, like_count)

响应:
{
  "items": [...],
  "total": 100,
  "page": 1,
  "per_page": 20,
  "pages": 5
}

POST /api/music/{id}/play
记录播放次数

PUT /api/music/{id}/like
点赞/取消点赞

GET /api/music/{id}/recommendations
获取推荐音乐
```

#### 支付API
```
POST /api/payment/create
创建支付订单

请求体:
{
  "amount": 9.99,
  "product_name": "100金币",
  "payment_method": "alipay"
}

POST /api/payment/notify
支付结果通知 (支付宝回调)

GET /api/payment/orders
获取用户订单列表
```

## 🔄 持续集成/部署

### GitHub Actions工作流
```yaml
# .github/workflows/deploy.yml
name: Deploy to Production

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    
    services:
      postgres:
        image: postgres:14
        env:
          POSTGRES_PASSWORD: postgres
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
      
      redis:
        image: redis:7
        options: >-
          --health-cmd "redis-cli ping"
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Setup Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.8'
    
    - name: Setup Node.js
      uses: actions/setup-node@v3
      with:
        node-version: '18'
        cache: 'npm'
        cache-dependency-path: frontend/package-lock.json
    
    - name: Install dependencies
      run: |
        cd backend
        pip install -r requirements.txt
        cd ../frontend
        npm ci
    
    - name: Run tests
      run: |
        cd backend
        pytest tests/ -v
        cd ../frontend
        npm run test:unit
    
    - name: Build frontend
      run: |
        cd frontend
        npm run build
    
    - name: Deploy to production
      if: github.ref == 'refs/heads/main'
      run: |
        # 部署脚本
        ssh user@server 'cd /app && ./deploy.sh'
```

## 🎯 未来发展规划

### 功能路线图

#### Phase 1: 核心功能完善 (当前)
- ✅ 用户认证系统
- ✅ 音乐播放功能
- ✅ 海报设计器
- ✅ 支付系统
- ✅ 移动端适配

#### Phase 2: 社交功能增强
- 🔄 用户关注系统
- 🔄 音乐评论功能
- 🔄 分享与协作
- 🔄 用户生成内容

#### Phase 3: AI能力集成
- 📋 智能推荐算法
- 📋 音频AI分析
- 📋 自动海报生成
- 📋 内容审核系统

#### Phase 4: 商业化扩展
- 📋 会员等级系统
- 📋 广告投放平台
- 📋 数据分析仪表板
- 📋 多语言国际化

### 技术债务管理

#### 代码质量工具
```bash
# 前端代码检查
npm run lint
npm run type-check
npm run test:unit

# 后端代码检查
flake8 backend/
mypy backend/
black backend/
```

#### 性能监控指标
- 页面加载时间 < 2s
- API响应时间 < 200ms
- 错误率 < 0.1%
- 移动端FPS > 30

## 📞 支持与联系

### 开发团队
- **项目负责人**: [Your Name]
- **前端开发**: [Frontend Team]
- **后端开发**: [Backend Team]
- **UI/UX设计**: [Design Team]

### 技术支持
- **文档更新**: [项目Wiki](https://github.com/your-username/aka-music/wiki)
- **问题反馈**: [GitHub Issues](https://github.com/your-username/aka-music/issues)
- **功能请求**: [GitHub Discussions](https://github.com/your-username/aka-music/discussions)

---

**最后更新**: 2024年3月
**版本**: v1.0.0