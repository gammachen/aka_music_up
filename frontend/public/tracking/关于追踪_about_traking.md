## 安装配置MMTracking
```bash
# 创建conda环境（建议使用Mamba加速）
conda create -n mmtracking python=3.8 -y
conda activate mmtracking

# 安装PyTorch（需CUDA 11.1驱动支持）
# 根据显卡架构选择合适版本（Turing架构推荐1.9+）
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 \
  -f https://download.pytorch.org/whl/torch_stable.html

# 验证CUDA可用性（应输出Compute Capability版本）
python -c "import torch; \
  print(f'CUDA可用: {torch.cuda.is_available()}, 版本:{torch.version.cuda}')"

# 安装MMLab系列框架（注意版本兼容性）
pip install mmcv-full==1.4.7 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html
pip install mmdet==2.25.1  # 目标检测基础库
pip install mmtrack==0.14.0  # 多目标跟踪核心库

# 完整验证安装（应输出0.14.0）
python -c "from mmtrack import __version__; print(__version__)"
```

### 框架配置说明
1. **环境配置**：
   - 建议使用Python 3.8 + PyTorch 1.9组合以获得最佳兼容性
   - 对于Ampere架构显卡（RTX 30/40系列），需升级PyTorch到1.12+并对应调整CUDA版本

2. **MMCV编译选项**：
   ```bash
   # 启用CUDA算子优化（需安装Ninja）
   MMCV_WITH_OPS=1 pip install mmcv-full==1.4.7
   ```

3. **分布式训练配置**：
   ```python
   # configs/_base_/default_runtime.py
   checkpoint_config = dict(interval=5)  # 每5个epoch保存一次
   log_config = dict(
       interval=50,
       hooks=[
           dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook')
       ])
   ```

## 视频目标检测、视频示例分割
### YOLOv5集成方案

### YOLOv8-Face人脸跟踪示例
```python
from mmtrack.apis import init_model, inference_mot

# 加载预训练人脸检测模型
model = init_model(
    config='configs/mot/bytetrack/yolov8_face_bytetrack.py',
    checkpoint='checkpoints/yolov8_face_crowdhuman.pt')

# 视频推理流程（支持人脸关键点检测）
results = inference_mot(
    model,
    video_path='demo/face_demo.mp4',
    show_result=True,
    out_dir='output/',
    detector_kwargs={
        'with_landmarks': True,  # 启用关键点检测
        'score_thr': 0.5
    })
```
![人脸跟踪效果](//i0.hdslb.com/bfs/article/87c3a9d4c6e6b5a9f8e7d6c5b4a3f2e1d0g9h8.png)
```python
from mmtrack.apis import inference_mot

# 加载预训练模型
model = init_model(
    config='configs/mot/deepsort/yolov5_deepsort_sort.py',
    checkpoint='checkpoints/yolov5_mot20.pt')

# 视频推理流程
results = inference_mot(
    model,
    video_path='demo/demo.mp4',
    show_result=True,
    out_dir='output/',
    # 多目标跟踪参数调优
    tracker_params=dict(
        motion=dict(
            max_age=30,          # 目标最大存活帧数
            n_init=5,            # 确认跟踪所需连续检测次数
            std_weight=1.2       # 卡尔曼滤波器噪声权重
        ),
        appearance=dict(
            metric='cosine',     # 特征相似度度量方式
            threshold=0.2,       # 关联匹配阈值
            budget=100           # 外观特征缓存数量
        )
    ),
    # 检测器参数优化
    detector_kwargs=dict(
        score_thr=0.4,          # 检测置信度阈值
        nms_thr=0.5,            # NMS阈值
        max_per_img=100         # 每帧最大检测目标数
    ))
```
![视频检测效果](//i0.hdslb.com/bfs/article/0465d3b7f245ffddfa60f5e42a7d5e8c9b8b2f12.png)

## 单目标跟踪
### MOSSE算法实现
```python
import cv2
from mmtrack.apis import init_model, inference_sot

# 初始化MOSSE跟踪器
model = init_model(
    config='configs/sot/mosse/mosse_r50_20e_otb.py',
    checkpoint='checkpoints/mosse_otb100.pth')

# 视频首帧处理
def get_first_frame_bbox(video_path):
    cap = cv2.VideoCapture(video_path)
    success, frame = cap.read()
    if success:
        # 手动标注或检测器获取ROI
        bbox = cv2.selectROI('Select Object', frame)
        cv2.destroyAllWindows()
        return frame, bbox
    return None, None

# 跟踪循环
frame, init_bbox = get_first_frame_bbox('demo/billiards.mp4')
if init_bbox:
    result = inference_sot(model, frame, init_bbox, frame_idx=0)
    # 持续跟踪后续帧
    tracker_params = dict(
        update_interval=5,          # 动态ROI更新间隔（帧数）
        confidence_thr=0.6,         # 跟踪置信度阈值
        recovery_attempts=3         # 失败重检测次数
    )
    while True:
        success, frame = cap.read()
        if not success: break
        
        # 动态调整ROI（当置信度低于阈值时触发重检测）
        if result['track_score'] < tracker_params['confidence_thr']:
            roi = cv2.selectROI('Recovery Detection', frame)
            cv2.destroyAllWindows()
            result = inference_sot(model, frame, roi, frame_idx=0)
            
        result = inference_sot(
            model,
            frame,
            prev_result=result,
            # 运动模型更新参数
            motion_params=dict(
                adaptive=True,       # 启用自适应模型更新
                learning_rate=0.02   # 模型更新速率
            )
        )
        
        # 每N帧执行模型重初始化
        if result['frame_idx'] % tracker_params['update_interval'] == 0:
            model.reinit_tracker(
                new_appearance=result['features'][-1],
                update_motion=True
            )
```

## 获取视频第一帧的单目标检测框

## 趣味Demo：花式台球杆法研究
```python
import numpy as np
from mmtrack.apis import inference_mot
import matplotlib.pyplot as plt

# 轨迹预测可视化
def visualize_cue_effect(trajectory):
    """
    可视化台球杆法效果
    :param trajectory: 历史轨迹数据 (N,2)
    """
    # 物理参数设置
    FRICTION = 0.98  # 台呢摩擦系数
    BALL_RADIUS = 2.85  # 台球半径(cm)
    
    # 轨迹预测（未来30帧）
    predicted = [trajectory[-1]]
    velocity = trajectory[-1] - trajectory[-2]
    for _ in range(30):
        velocity *= FRICTION
        predicted.append(predicted[-1] + velocity)
    predicted = np.array(predicted)

    # 可视化设置
    plt.figure(figsize=(12, 6))
    plt.title('台球运动轨迹预测', fontsize=14)
    plt.scatter(trajectory[:,0], trajectory[:,1], c='b', label='实际轨迹')
    plt.plot(predicted[:,0], predicted[:,1], 'r--', lw=2, label='预测轨迹')
    plt.quiver(predicted[::3,0], predicted[::3,1], 
               velocity[::3,0], velocity[::3,1], 
               scale=50, color='g', label='速度矢量')
    
    # 添加台球桌标记
    plt.plot([0,200], [0,0], 'k-', lw=3)  # 底边
    plt.plot([0,200], [100,100], 'k-', lw=3)  # 顶边
    plt.xlim(-10, 210)
    plt.ylim(-10, 110)
    
    plt.legend()
    plt.savefig('output/cue_effect_prediction.png', dpi=300)
    plt.close()

# 应用示例
results = inference_mot(model, video_path='demo/billiards.mp4')
trajectory = np.array([res['track_bbox'][:2] for res in results])
visualize_cue_effect(trajectory)
```

## 多目标跟踪
### ByteTrack算法配置
```python
from mmtrack.apis import init_model, inference_mot

# 初始化ByteTrack模型
model = init_model(
    config='configs/mot/bytetrack/yolox_x_8x8_300e_coco.py',
    checkpoint='checkpoints/bytetrack_mot20.pth'
)

# 视频推理参数配置
results = inference_mot(
    model,
    video_path='demo/shopping_mall.mp4',
    tracker=dict(
        type='ByteTrackTracker',
        obj_score_thrs=dict(high=0.5, low=0.3),
        motion=dict(type='KalmanFilter', std_weight=[1.0, 1.0]),
        reid=dict(
            type='BaseReID',
            backbone=dict(
                type='ResNet',
                depth=50,
                num_stages=4,
                out_indices=(3,)),
            pretrained='checkpoints/reid_res50.pth'
        )
    ),
    detector=dict(
        score_thr=0.4,
        nms=dict(type='nms', iou_threshold=0.7)
    )
)
```

## 人流量计数与足迹跟踪
### 热力图生成与轨迹回放
```python
import cv2
import numpy as np
from collections import deque

# 人流量热力图生成
class HeatmapGenerator:
    def __init__(self, img_size=(1920, 1080), decay=0.99):
        self.heatmap = np.zeros(img_size[::-1], dtype=np.float32)
        self.decay = decay
        
    def update(self, detections):
        self.heatmap *= self.decay
        for det in detections:
            x, y = int(det[0]), int(det[1])
            cv2.circle(self.heatmap, (x,y), 20, 1.0, -1)
        
    def visualize(self):
        norm_heatmap = cv2.normalize(self.heatmap, None, 0, 255, cv2.NORM_MINMAX)
        return cv2.applyColorMap(norm_heatmap.astype(np.uint8), cv2.COLORMAP_JET)

# 足迹跟踪可视化
class TrailVisualizer:
    def __init__(self, maxlen=50):
        self.trails = {}
        
    def update(self, track_ids, positions):
        for tid, pos in zip(track_ids, positions):
            if tid not in self.trails:
                self.trails[tid] = deque(maxlen=50)
            self.trails[tid].append(pos)
        
    def draw(self, frame):
        for tid, trail in self.trails.items():
            for i in range(1, len(trail)):
                cv2.line(frame, tuple(map(int, trail[i-1])), 
                        tuple(map(int, trail[i])), (0,255,0), 2)
        return frame

# 使用示例
heatmap_gen = HeatmapGenerator()
trail_viz = TrailVisualizer()

cap = cv2.VideoCapture('demo/shopping_mall.mp4')
while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    
    # 获取检测结果
    results = inference_mot(model, frame)
    
    # 更新可视化
    heatmap_gen.update(results['detections'])
    trail_viz.update(results['track_ids'], results['positions'])
    
    # 叠加显示
    heatmap = heatmap_gen.visualize()
    blended = cv2.addWeighted(frame, 0.7, heatmap, 0.3, 0)
    final_frame = trail_viz.draw(blended)
    
    cv2.imshow('Analytics', final_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
```

## 人流量计数与足迹跟踪、图标可视化
### 基于WoodScape的可视化方案
```python
import matplotlib.pyplot as plt
from mmtrack.apis import visualize_results

# 鱼眼镜头校正
from mmcv.ops import DistortionCorrection

"""
WoodScape数据集畸变校正配置要点：
1. calib_file: 标定文件包含相机内参和畸变系数
   - fx/fy: 焦距（像素单位）
   - cx/cy: 光心坐标
   - k1-k4: 径向畸变系数
   - p1-p2: 切向畸变系数
2. distortion_model: 支持fisheye/radtan两种模型
3. img_size: 输入图像分辨率（宽, 高）
"""

distortion_corrector = DistortionCorrection(
    calib_file='calib/woodscape.yml',  # 标定文件路径
    distortion_model='fisheye',        # 鱼眼镜头模型
    img_size=(1920, 1080),             # 输入图像尺寸
    correction_mode='perspective'     # 输出透视投影
)

# 可视化前处理
img = cv2.imread('data/WoodScape/fisheye/0001.jpg')
corrected_img = distortion_corrector(img)

# 生成热力图时应用校正
results = visualize_results(
    model,
    img=corrected_img,
    result_path='output/0001_corrected.jpg',
    heatmap_kwargs={
        'cmap': 'jet',
        'alpha': 0.5,
        'thickness': 2
    })

# 时间窗口聚合逻辑（每5分钟）
time_window = 300  # 单位：秒
frame_rate = 30     # 视频帧率
window_frames = time_window * frame_rate

# 初始化流量计数器
counter = {
    'timestamps': [],
    'counts': [],
    'heatmaps': []
}

# 实时聚合处理
for frame_idx, heatmap in enumerate(results['heatmap_sequence']):
    # 按时间窗口聚合
    if frame_idx % window_frames == 0:
        # 生成当前窗口热力图
        window_heatmap = np.mean(counter['heatmaps'][-window_frames:], axis=0)
        # 可视化存储
        plt.figure(figsize=(18,10))
        plt.title(f'人流量热力图 {time_window//60}分钟聚合 (窗口{frame_idx//window_frames})',
                 fontsize=14)
        plt.imshow(window_heatmap, cmap='jet', alpha=0.6,
                  extent=[0, corrected_img.shape[1], 0, corrected_img.shape[0]])
        plt.colorbar(label='人流量密度')
        plt.savefig(f'output/mall_heatmap_{frame_idx//window_frames}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    # 更新计数器
    counter['timestamps'].append(frame_idx / frame_rate)
    counter['counts'].append(np.sum(heatmap))
    counter['heatmaps'].append(heatmap)

# 生成热力图
plt.figure(figsize=(12,8))
plt.imshow(results['heatmap'])
plt.savefig('output/heatmap.png')
```
![人流量热力图](//i0.hdslb.com/bfs/article/4e0b3b9c9c7d5d8a9b0c8b7a6b5c4d3e2f1g0h.png)

## 在数据集上评估模型性能

## 训练DeepSORT多目标跟踪算法
### 多数据集配置

#### PathTrack基础配置
```yaml
# configs/mot/deepsort/deepsort_pathtrack.py
data = dict(
    train=dict(
        type='PathTrackDataset',
        ann_file='data/PathTrack/annotations/train.json',
        img_prefix='data/PathTrack/train/',
        ...
    )
)
```

#### CIHP人体解析增强配置
```yaml
train=dict(
    type='CocoDataset',
    ann_file='data/CIHP/annotations/instance_train.json',
    img_prefix='data/CIHP/Train/',
    # 人体解析增强流程
    pipeline=[
        dict(
            type='LoadHumanParsingAnnotations',
            with_parsing=True,
            parsing_dir='data/CIHP/Train/Parsing',
            # 解析图处理参数
            parsing_threshold=0.5,      # 语义分割置信度阈值
            use_contour=True,           # 启用轮廓提取
            contour_dilation=3          # 轮廓扩张像素数
        ),
        dict(
            type='PackDetInputs',
            meta_keys=('img_id', 'img_path', 'parsing_map', 'contour_mask'),
            # 添加解析特征到检测头
            parser_feat_dim=256
        )
    ],
    # 用于DeepSORT的特征增强
    feature_fusion=dict(
        enable=True,
        parsing_weight=0.3,  # 解析特征权重
        appearance_weight=0.7
    )
)
```
```yaml
# configs/mot/deepsort/deepsort_pathtrack.py
data = dict(
    train=dict(
        type='PathTrackDataset',
        ann_file='data/PathTrack/annotations/train.json',
        img_prefix='data/PathTrack/train/',
        detector=dict(
            type='FasterRCNN',
            backbone=dict(
                type='ResNet',
                depth=50,
                num_stages=4,
                out_indices=(3, )))
    )
)
```
### 训练命令
```bash
./tools/dist_train.sh configs/mot/deepsort/deepsort_pathtrack.py 8 --validate
```