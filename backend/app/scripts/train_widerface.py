import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from model.u2net import U2NET  # 确保已导入 U^2-Net 模型定义

# ---------------------------
# 1. 数据集路径配置
# ---------------------------
widerface_root = "/Users/shhaofu/.cache/kagglehub/datasets/iamprateek/wider-face-a-face-detection-dataset/versions/1"  # 数据集根目录
train_image_dir = os.path.join(widerface_root, "WIDER_train/images")
train_annot_file = os.path.join(widerface_root, "wider_face_annotations/wider_face_split/wider_face_train_bbx_gt.txt")
val_image_dir = os.path.join(widerface_root, "WIDER_val/images")
val_annot_file = os.path.join(widerface_root, "wider_face_annotations/wider_face_split/wider_face_val_bbx_gt.txt")

# ---------------------------
# 2. 自定义数据集类
# ---------------------------
class WiderFaceDataset(Dataset):
    def __init__(self, image_dir, annot_file, img_size=512, is_train=True):
        self.image_dir = image_dir
        self.img_size = img_size
        self.annotations = self._parse_annotations(annot_file)
        
        # 数据增强
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]) if is_train else transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def _parse_annotations(self, annot_file):
        """解析 WIDER FACE 标注文件"""
        annotations = []
        with open(annot_file, 'r') as f:
            lines = f.readlines()
            idx = 0
            while idx < len(lines):
                img_name = lines[idx].strip()
                idx += 1
                num_faces = int(lines[idx].strip())
                idx += 1
                bboxes = []
                for _ in range(num_faces):
                    bbox = list(map(int, lines[idx].strip().split()[:4]))
                    bboxes.append(bbox)  # [x1, y1, w, h]
                    idx += 1
                annotations.append((img_name, bboxes))
        return annotations

    def _generate_mask(self, img_shape, bboxes):
        """根据边界框生成二值掩膜"""
        mask = np.zeros((img_shape[0], img_shape[1]), dtype=np.float32)
        for (x1, y1, w, h) in bboxes:
            x2 = min(x1 + w, img_shape[1])
            y2 = min(y1 + h, img_shape[0])
            mask[y1:y2, x1:x2] = 1.0
        return mask

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_name, bboxes = self.annotations[idx]
        img_path = os.path.join(self.image_dir, img_name)
        
        # 加载图像并调整尺寸
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        image = cv2.resize(image, (self.img_size, self.img_size))
        
        # 生成掩膜并调整尺寸
        mask = self._generate_mask((h, w), bboxes)
        mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
        mask = mask[np.newaxis, ...]  # 添加通道维度 (1, H, W)

        # 应用数据增强
        image = self.transform(image)
        return image, torch.from_numpy(mask)

# ---------------------------
# 3. 数据加载器
# ---------------------------
batch_size = 8
num_workers = 4

train_dataset = WiderFaceDataset(train_image_dir, train_annot_file, is_train=True)
val_dataset = WiderFaceDataset(val_image_dir, val_annot_file, is_train=False)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

# ---------------------------
# 4. 模型初始化
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = U2NET(in_ch=3, out_ch=1).to(device)

# 加载预训练权重
pretrained_weights = torch.load("u2net.pth", map_location=device)
model.load_state_dict(pretrained_weights)

# ---------------------------
# 5. 损失函数与优化器
# ---------------------------
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

# ---------------------------
# 6. 训练循环
# ---------------------------
best_val_loss = float('inf')
num_epochs = 10 # 100

for epoch in range(num_epochs):
    # 训练阶段
    model.train()
    train_loss = 0.0
    for images, masks in train_loader:
        images = images.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item() * images.size(0)
    
    train_loss /= len(train_loader.dataset)

    # 验证阶段
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            val_loss += loss.item() * images.size(0)
    
    val_loss /= len(val_loader.dataset)
    scheduler.step(val_loss)

    # 保存最佳模型
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), f"u2net_face_best.pth")
    
    # 打印日志
    print(f"Epoch [{epoch+1}/{num_epochs}] | "
          f"Train Loss: {train_loss:.4f} | "
          f"Val Loss: {val_loss:.4f} | "
          f"LR: {optimizer.param_groups[0]['lr']:.2e}")

# 保存最终模型
torch.save(model.state_dict(), f"u2net_face_final.pth")

import cv2
import torch
from model import U2NET

def detect_faces(image_path, model_path="u2net_face_best.pth", img_size=512):
    # 加载模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = U2NET(in_ch=3, out_ch=1).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 预处理图像
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = image.shape[:2]
    img_input = cv2.resize(image, (img_size, img_size))
    img_tensor = transforms.ToTensor()(img_input).unsqueeze(0).to(device)
    img_tensor = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img_tensor)

    # 推理
    with torch.no_grad():
        mask = model(img_tensor).squeeze().cpu().numpy()
    
    # 后处理
    mask = (mask > 0.5).astype(np.uint8) * 255
    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    
    # 提取边界框
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bboxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        bboxes.append((x, y, x+w, y+h))  # 返回 (x1, y1, x2, y2)
    
    return bboxes


image_path = "/Users/shhaofu/Code/cursor-projects/aka_music/backend/app/static/beauty/写真/artistic portrait_1_0.jpg"
bboxes = detect_faces(image_path)
print(bboxes)