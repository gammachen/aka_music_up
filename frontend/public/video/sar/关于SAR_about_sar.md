
## 概念

在视频开发中，**SAR（Sample Aspect Ratio，样本宽高比）** 是处理像素宽高比的核心参数，直接影响视频的最终显示效果。以下是针对视频开发者对 **SAR 的详细应用说明**，结合 FFmpeg 的典型场景和操作指南：

---

### **1. SAR 的核心概念**
- **定义**：  
  **SAR** 表示单个像素的宽度与高度的比例（如 `1:1` 表示正方形像素）。  
  其与 **PAR（Pixel Aspect Ratio，图像宽高比）** 和 **DAR（Display Aspect Ratio，显示宽高比）** 的关系为：  
  \[
  \text{DAR} = \text{SAR} \times \text{PAR}
  \]
  - **PAR** = 视频存储分辨率宽高比（如 `1920x1080` 的 PAR 为 `16:9`）。  
  - **DAR** = 最终屏幕显示宽高比（如 `16:9` 或 `4:3`）。

- **常见场景**：  
  - 非正方形像素的视频源（如 DV 视频的 SAR 为 `10:11` 或 `40:33`）。  
  - 不同设备采集的视频需要统一显示比例（如监控摄像头、手机录像）。

---

### **2. SAR 的应用场景**
#### **(1) 修复视频显示变形**
- **问题**：视频在播放时被拉伸或压缩（如圆形变椭圆）。  
- **原因**：SAR 设置错误，导致 DAR 计算异常。  
- **解决**：通过调整 SAR 值，使 DAR 符合预期比例。

#### **(2) 视频格式转换**
- **需求**：将非正方形像素视频（如 PAL DV）转换为正方形像素格式（如 MP4）。  
- **操作**：在转码时重置 SAR 为 `1:1`，并调整分辨率以保持 DAR 不变。

#### **(3) 多视频源合成**
- **挑战**：合并不同 SAR 的视频时显示比例不一致。  
- **处理**：统一所有视频流的 SAR 或动态调整输出比例。

---

### **3. 使用 FFmpeg 操作 SAR**
#### **(1) 查看视频的 SAR 信息**
```bash
ffmpeg -i input.mp4
```
在输出中查找 `Stream #0:0` 的 `sar` 值：
```
Stream #0:0: Video: h264 (sar 40:33, ...)
```

#### **(2) 强制设置 SAR**
使用 `setsar` 滤镜直接指定 SAR 值：
```bash
ffmpeg -i input.mp4 -vf "setsar=1:1" output.mp4
```
- **效果**：强制像素宽高比为 `1:1`（正方形像素）。

#### **(3) 通过 DAR 反向推导 SAR**
若已知目标 DAR（如 `16:9`）和分辨率（如 `1280x720`，PAR=16:9），则 SAR 应为 `1:1`。  
若需要保持原 DAR 但修改分辨率，例如将 `720x480`（PAR=3:2）转换为 `1280x720`：
```bash
# 计算 SAR 以保持 DAR=16:9
ffmpeg -i input.mp4 -vf "scale=1280:720,setsar=1:1" output.mp4
```

#### **(4) 修复非正方形像素视频**
将 SAR 为 `40:33` 的 DV 视频转换为正方形像素：
```bash
ffmpeg -i input.dv -vf "setsar=1:1" -c:v libx264 output.mp4
```

#### **(5) 批量处理文件夹内视频**
```bash
for f in *.mov; do
  ffmpeg -i "$f" -vf "setsar=1:1" "output_${f%.*}.mp4"
done
```

---

### **4. 高级技巧**
#### **(1) 动态调整 SAR**
在复杂处理链中结合 `scale` 和 `setsar`：
```bash
ffmpeg -i input.mp4 -vf "scale=1920:1080, setsar=1:1" output.mp4
```

#### **(2) 与 `setdar` 结合使用**
直接指定目标 DAR，FFmpeg 自动计算 SAR：
```bash
ffmpeg -i input.mp4 -vf "setdar=16:9" output.mp4
```

#### **(3) 处理旋转视频（含元数据）**
部分视频包含旋转元数据（如手机竖屏视频），需先纠正旋转再调整 SAR：
```bash
ffmpeg -i input.mp4 -vf "transpose=1,setsar=1:1" output.mp4
```

---

### **5. 常见问题与调试**
#### **(1) 播放器不识别 SAR**
- **现象**：修改 SAR 后播放器仍显示异常。  
- **解决**：确保同时设置容器的显示比例参数：  
  ```bash
  ffmpeg -i input.mp4 -vf "setsar=1:1" -aspect 16:9 output.mp4
  ```

#### **(2) 分辨率与 SAR 冲突**
- **公式验证**：  
  若输入分辨率为 `720x480`（PAR=3:2），设置 `setsar=10:11` 后：  
  \[
  \text{DAR} = \text{SAR} \times \text{PAR} = (10/11) \times (3/2) = 15/11 \approx 1.36:1
  \]

#### **(3) 检查处理结果**
使用 `ffprobe` 验证输出视频的 SAR 和 DAR：
```bash
ffprobe -v error -select_streams v:0 -show_entries stream=sample_aspect_ratio,display_aspect_ratio output.mp4
```

---

### **6. 总结**
- **关键公式**：DAR = SAR × PAR，确保三者逻辑一致。  
- **FFmpeg 核心操作**：`setsar`、`setdar`、`-aspect`。  
- **典型场景**：修复变形、转码兼容、多源合成。  

通过合理调整 SAR，开发者可精准控制视频显示效果，避免比例失真问题。

## Info

```shell
指定setsar（sar滤镜）进行转码，查看sar的不同值

(base) shhaofu@shhaofudeMacBook-Pro mv % ffmpeg -i sun_yan_zhi_mv.mp4 -vf "setsar=1:3" sun_yan_zhi_mv_sar_1_3.mp4

(base) shhaofu@shhaofudeMacBook-Pro mv % ffmpeg -i sun_yan_zhi_mv.mp4 | grep 'Stream'
ffmpeg version 7.1.1 Copyright (c) 2000-2025 the FFmpeg developers

  Stream #0:0[0x1](und): Video: h264 (High) (avc1 / 0x31637661), yuv420p(tv, bt709, progressive), 1920x1080 [SAR 1:1 DAR 16:9], 1734 kb/s, 30 fps, 30 tbr, 16k tbn (default)
      Metadata:
        handler_name    : VideoHandler
        vendor_id       : [0][0][0][0]
  
(base) shhaofu@shhaofudeMacBook-Pro mv % ffmpeg -i sun_yan_zhi_mv_sar_1_3.mp4 | grep 'Stream'
ffmpeg version 7.1.1 Copyright (c) 2000-2025 the FFmpeg developers

  Stream #0:0[0x1](und): Video: h264 (High) (avc1 / 0x31637661), yuv420p(tv, bt709, progressive), 1920x1080 [SAR 1:1 DAR 16:9], 2586 kb/s, 30 fps, 30 tbr, 15360 tbn (default)
      Metadata:
        handler_name    : VideoHandler
        vendor_id       : [0][0][0][0]
        encoder         : Lavc61.19.101 libx264
  Stream #0:1[0x2](und): Audio: aac (LC) (mp4a / 0x6134706D), 48000 Hz, stereo, fltp, 128 kb/s (default)
      Metadata:
        handler_name    : SoundHandler
        vendor_id       : [0][0][0][0]

1920x1080 [SAR 1:1 DAR 16:9] 貌似都没有变化

```

若原始视频的 **显示宽高比（DAR）为 16:9**，且 **SAR（像素宽高比）为 1:1**（正方形像素），想要通过调整 SAR 值让画面产生 **更明显的视觉变化**（拉伸或压缩），可按照以下分步方案操作：

---

### **一、基本原理**
- **DAR = SAR × PAR**（PAR = 分辨率宽高比 = 宽度/高度）
- **原始状态**：  
  - 若 SAR=1:1，则 DAR = PAR = 16:9（例如分辨率为 1920x1080）。
  - 此时画面无变形，显示正常。

- **调整目标**：  
  通过修改 SAR 值，使 DAR ≠ 16:9，从而产生明显的拉伸或压缩效果。

---

### **二、操作步骤与效果对比**

#### **1. 横向拉伸（SAR > 1:1）**
- **示例 SAR 值**：`2:1`（像素宽度是高度的 2 倍）
- **计算 DAR**：  
  \[
  \text{DAR} = \text{SAR} \times \text{PAR} = \frac{2}{1} \times \frac{16}{9} = \frac{32}{9} \approx 3.56:1
  \]
- **视觉效果**：  
  画面被横向拉伸，圆形变椭圆，人物变宽，类似“宽银幕电影”但更极端。
- **FFmpeg 命令**：
  ```bash
  ffmpeg -i input.mp4 -vf "setsar=2/1" -c:v libx264 output_wide.mp4
  ```

#### **2. 纵向拉伸（SAR < 1:1）**
- **示例 SAR 值**：`1:2`（像素高度是宽度的 2 倍）
- **计算 DAR**：  
  \[
  \text{DAR} = \frac{1}{2} \times \frac{16}{9} = \frac{8}{9} \approx 0.89:1
  \]
- **视觉效果**：  
  画面被纵向拉伸，圆形变高椭圆，人物变瘦，类似“瘦身滤镜”效果。
- **FFmpeg 命令**：
  ```bash
  ffmpeg -i input.mp4 -vf "setsar=1/2" -c:v libx264 output_tall.mp4
  ```

#### **3. 极端变形（测试用）**
- **示例 SAR 值**：`4:1` 或 `1:4`  
- **效果**：  
  - `4:1` → DAR=64:9 ≈ 7.11:1（严重横向拉伸，几乎无法辨认内容）。  
  - `1:4` → DAR=4:9 ≈ 0.44:1（严重纵向拉伸，画面压缩成细条）。  
- **命令示例**：
  ```bash
  ffmpeg -i input.mp4 -vf "setsar=4/1" -c:v libx264 output_extreme_wide.mp4
  ```

---

### **三、验证方法**
#### **1. 元数据检查**
使用 `ffprobe` 确认 SAR 和 DAR 的修改是否生效：
```bash
ffprobe -v error -select_streams v:0 \
  -show_entries stream=sample_aspect_ratio,display_aspect_ratio \
  output_wide.mp4
```
输出示例：
```
sample_aspect_ratio=2:1
display_aspect_ratio=32:9
```

#### **2. 视觉验证**
- **使用测试图案**：  
  生成一个包含圆形和正方形的测试视频，更容易观察变形：
  ```bash
  ffmpeg -f lavfi -i testsrc=duration=5:size=1920x1080:rate=30 \
    -vf "drawgrid=width=100:height=100:color=white,drawcircle=200:200:50:50:color=red" \
    test_input.mp4
  ```
- **播放对比**：  
  在支持 SAR 的播放器（如 **VLC**）中播放修改后的视频，观察图形变形。

---

### **四、注意事项**
1. **播放器兼容性**：  
   部分播放器可能忽略 SAR，直接按分辨率显示。此时需强制指定容器宽高比：
   ```bash
   ffmpeg -i input.mp4 -vf "setsar=2/1" -aspect 32:9 output_wide.mp4
   ```

2. **分辨率适配**：  
   若需保持 DAR=16:9 但修改 SAR，需调整分辨率：
   ```bash
   # 计算新分辨率（假设原分辨率 1920x1080，SAR=2/1）
   # 新宽度 = 原宽度 × (新 SAR / 原 SAR) = 1920 × (2/1) = 3840
   ffmpeg -i input.mp4 -vf "scale=3840:1080,setsar=1/1" -aspect 16:9 output_scaled.mp4
   ```

---

### **五、总结**
- **明显变化关键**：选择 SAR 值与 1:1 差异越大（如 `2:1` 或 `1:2`），视觉变形越显著。
- **操作核心**：通过 `setsar` 滤镜调整 SAR，结合 `-aspect` 强制容器比例。
- **验证工具**：`ffprobe` + 视觉对比测试视频。

通过上述方法，你可以轻松实现视频的极端比例变形，适用于特效制作、兼容性测试或故障模拟等场景。

```shell

ffmpeg -i sun_yan_zhi_mv.mp4 -vf "setsar=2/1" sun_yan_zhi_mv_sar_sar_2_1.mp4

ffmpeg -i sun_yan_zhi_mv.mp4 -vf "setsar=1/2" sun_yan_zhi_mv_sar_sar_1_2.mp4

(base) shhaofu@shhaofudeMacBook-Pro mv % ffprobe -v error -select_streams v:0 -show_entries stream=sample_aspect_ratio,display_aspect_ratio sun_yan_zhi_mv_sar_sar_2_1.mp4

[STREAM]
sample_aspect_ratio=2:1
display_aspect_ratio=32:9
[/STREAM]
(base) shhaofu@shhaofudeMacBook-Pro mv % ffprobe -v error -select_streams v:0 -show_entries stream=sample_aspect_ratio,display_aspect_ratio sun_yan_zhi_mv_sar_sar_1_2.mp4

[STREAM]
sample_aspect_ratio=1:2
display_aspect_ratio=8:9
[/STREAM]
(base) shhaofu@shhaofudeMacBook-Pro mv % ffmpeg -i sun_yan_zhi_mv.mp4 -vf "setsar=1:10, setdar=16:9" sun_yan_zhi_mv_sar_sar_1_10_dar_16_9.mp4
(base) shhaofu@shhaofudeMacBook-Pro mv % ffprobe -v error -select_streams v:0 -show_entries stream=sample_aspect_ratio,display_aspect_ratio sun_yan_zhi_mv.mp4
[STREAM]
sample_aspect_ratio=1:1
display_aspect_ratio=16:9
[/STREAM]

```




