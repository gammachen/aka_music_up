
# Termux-API 完整使用手册

## 📥 安装准备
### 应用下载
- **官方渠道**：
  - [Google Play 版](https://play.google.com/store/apps/details?id=com.termux.api)
  - [F-Droid 版](https://f-droid.org/packages/com.termux.api/)
- **⚠️ 重要警告**：
  ```text
  请勿混用 Google Play 和 F-Droid 的 Termux 主程序与插件

  需要在F-Droid中安装两者，否则后续会出现各种问题，到时候Termux-API无法使用（费劲）
  ```

### 权限配置
> *"Termux-api 可直接操作手机底层，需在系统设置中开启所有权限"*

### 终端安装
```bash
pkg install termux-api
```

---

## 🔋 硬件控制
### 电池状态
```bash
termux-battery-status
```
▶️ 输出示例：
```json
~/storage/downloads $ termux-battery-status
{
  "present": true,
  "technology": "Li-ion",
  "health": "GOOD",
  "plugged": "PLUGGED_USB",
  "status": "CHARGING",
  "temperature": 35.5,
  "voltage": 3000,
  "current": -1179,
  "current_average": null,
  "percentage": 60,
  "level": 60,
  "scale": 100,
  "charge_counter": 2928000,
  "energy": null
}
~/storage/downloads $ termux-battery-status
{
  "present": true,
  "technology": "Li-ion",
  "health": "GOOD",
  "plugged": "PLUGGED_USB",
  "status": "CHARGING",
  "temperature": 35.5,
  "voltage": 4000,
  "current": -1354,
  "current_average": null,
  "percentage": 60,
  "level": 60,
  "scale": 100,
  "charge_counter": 2928000,
  "energy": null
}
```

### 相机系统
1. 获取相机信息：
   ```bash
   termux-camera-info
   ```
2. 拍照（作者吐槽版）：
   ```bash
   termux-camera-photo -c 0 guoguang.jpg
   # -c 0=后置摄像头 | 1=前置
   ```

---

## 📞 通讯与传感器
### 通讯录读取
```bash
termux-contact-list
```

### 红外控制
```bash
termux-infrared-frequencies  # 查看频率
termux-infrared-transmit -f 20,50,20,30  # 发射信号
```

### 蜂窝网络信息
```bash
termux-telephony-cellinfo    # 无线电信息
termux-telephony-deviceinfo  # 运营商信息
```
▶️ 中国联通示例：
```json
~/storage/downloads $ termux-telephony-deviceinfo
{
  "data_enabled": "false",
  "data_activity": "none",
  "data_state": "disconnected",
  "device_id": null,
  "device_software_version": "17",
  "phone_count": 2,
  "phone_type": "gsm",
  "network_operator": "46001",
  "network_operator_name": "中国联通",
  "network_country_iso": "cn",
  "network_type": "nr",
  "network_roaming": false,
  "sim_country_iso": "cn",
  "sim_operator": "46001",
  "sim_operator_name": "中国联通",
  "sim_serial_number": null,
  "sim_subscriber_id": null,
  "sim_state": "ready"
}
```

---

## 📶 网络功能
### WiFi 管理
```bash
termux-wifi-connectioninfo  # 当前连接
termux-wifi-scaninfo        # 扫描网络（Android 10+可能失效）
termux-wifi-enable true     # 开关WiFi
```

```bash
[
  {
    "bssid": "08:aa:89:7d:6c:fc",
    "frequency_mhz": 2452,
    "rssi": -38,
    "ssid": "NullPSF",
    "timestamp": 8149016362,
    "channel_bandwidth_mhz": "40",
    "center_frequency_mhz": 2462,
    "capabilities": "[WPA-PSK-CCMP][WPA2-PSK-CCMP][RSN-PSK-CCMP][ESS][WPS]"
  },
  {
    "bssid": "08:aa:89:7d:6c:fd",
    "frequency_mhz": 5745,
    "rssi": -50,
    "ssid": "NullPSF-5G2",
    "timestamp": 8149016342,
    "channel_bandwidth_mhz": "80",
    "center_frequency_mhz": 5775,
    "capabilities": "[WPA-PSK-CCMP][WPA2-PSK-CCMP][RSN-PSK-CCMP][ESS][WPS]"
  },
  {
    "bssid": "c8:bf:4c:01:7c:d7",
    "frequency_mhz": 2412,
    "rssi": -55,
    "ssid": "NullP24",
    "timestamp": 8149016312,
    "channel_bandwidth_mhz": "40",
    "center_frequency_mhz": 2422,
    "capabilities": "[WPA-PSK-TKIP+CCMP][WPA2-PSK-TKIP+CCMP][RSN-PSK-TKIP+CCMP][ESS][WPS]"
  },
  {
    "bssid": "c8:bf:4c:01:7c:d8",
    "frequency_mhz": 5745,
    "rssi": -69,
    "ssid": "NullP24_5G",
    "timestamp": 8149016354,
    "channel_bandwidth_mhz": "80",
    "center_frequency_mhz": 5775,
    "capabilities": "[WPA-PSK-TKIP+CCMP][WPA2-PSK-TKIP+CCMP][RSN-PSK-TKIP+CCMP][ESS][WPS]"
  },
  {
    "bssid": "86:2a:fd:75:ae:06",
    "frequency_mhz": 2437,
    "rssi": -68,
    "ssid": "DIRECT-06-HP M329dw LJ",
    "timestamp": 8149016379,
    "channel_bandwidth_mhz": "20",
    "capabilities": "[WPA2-PSK-CCMP][RSN-PSK-CCMP][ESS][WPS]"
  },
  {
    "bssid": "74:6f:88:b2:04:8f",
    "frequency_mhz": 2422,
    "rssi": -72,
    "ssid": "杭州融裕机电",
    "timestamp": 8149016372,
    "channel_bandwidth_mhz": "20",
    "capabilities": "[WPA-PSK-CCMP][WPA2-PSK-CCMP][RSN-PSK-CCMP][ESS][WPS]"
  }
]
```

### 下载器调用
```bash
termux-download -d '测试' -t 'QQ.apk' 'https://example.com/qq.apk'
```

---

## 🎤 媒体功能
### 语音合成(TTS)
```bash
termux-tts-speak -e "com.xiaomi.mibrain.speech" "大家转载要标明出处"
# -e 指定引擎（可用 termux-tts-engines 查询）
```

### 音频录制
```bash
termux-microphone-record -d  # 默认参数录制
termux-microphone-record -i  # 查看录制状态
```

### 视频播放
```bash
termux-media-player play hacker.mp4  # 播放
termux-media-player info            # 查看进度
```

---

## 📍 定位服务
```bash
termux-location -p network  # 使用网络定位
```
▶️ 返回结果（作者打码版）：
```json
{
  "latitude":xx.xxxx,
  "longitude":xx.xxxx,
  "provider":"network"
}
```

---

## ✨ 系统交互
### 通知中心
```bash
termux-notification -t "标题" -c "内容" \
  --led-color FF0000 --vibrate 500,1000
```

### 闪光灯控制
```bash
termux-torch on  # 开启
```

### 壁纸更换
```bash
termux-wallpaper -u 'https://bing.com/image.jpg'
# -u 使用网络图片 | -f 使用本地文件
```

---

## 🛠️ 高级功能
### USB 设备操作
1. 安装依赖：
   ```bash
   pkg install termux-api libusb clang
   ```
2. 设备列表：
   ```bash
   termux-usb -l
   # 输出示例：["/dev/bus/usb/001/002"]
   ```
3. 权限请求：
   ```bash
   termux-usb -r /dev/bus/usb/001/002
   ```
4. C语言操作示例（作者SSD秀）：
   ```c
   #include <libusb-1.0/libusb.h>
   // ...完整代码见原文档...
   ```
   编译执行：
   ```bash
   gcc usbtest.c -lusb-1.0 -o usbtest
   termux-usb -e ./usbtest /dev/bus/usb/001/002
   ```
   > *"暴露了我的512GB海康威视SSD，溜了溜了～"*

---

## 🖥️ 对话框交互
### 确认对话框
```bash
termux-dialog confirm -i "提示文本" -t "标题"
```

### 单选按钮
```bash
termux-dialog radio -v "小哥哥,小姐姐" -t "性别选择"
```

### 日期选择
```bash
termux-dialog date -d 'yyyy-MM-dd' -t "生日"
```

▶️ 完整参数支持：
```text
confirm    - 确认框
checkbox   - 多选框
counter    - 数字选择
date       - 日期选择
radio      - 单选按钮
sheet      - 底部菜单
spinner    - 下拉选择
text       - 文本输入
time       - 时间选择
```

---
