以下是不同平台和场景下控制台指令录制的教程，结合知识库中的信息整理而成：

---

### **一、游戏控制台指令录制（以CSGO为例）**
#### **适用场景**：录制游戏中的控制台指令（如视角、参数设置等）。
#### **步骤**：
1. **启动游戏并打开控制台**：
   - 按 `~` 键打开游戏控制台。
   - 确保已启用开发者控制台（通过 `Options > Keyboard > Enable Developer Console`）。

2. **录制指令**：
   - 输入 `record 文件名` 开始录制（例如 `record mydemo`）。
   - 游戏会自动录制你的操作和控制台输入，生成 `.dem` 文件（路径：`Steam/steamapps/common/CSGO/csgo/csgo_demos/`）。

3. **停止录制**：
   - 按 `~` 打开控制台，输入 `stop` 停止录制。

4. **播放录制内容**：
   - 输入 `playdemo 文件名`（例如 `playdemo mydemo`）回放录制内容。
   - 回放时可用 `Shift + F2` 打开 demo 控制台，按 `X` 启用 X光模式，按 `Alt` 查看投掷物轨迹。

---

### **二、Linux 命令行指令录制**
#### **适用场景**：录制终端操作（命令输入和输出）。
#### **方法 1：使用 `script` 命令**
1. **开始录制**：
   ```bash
   script -t 2> timing.txt -a output.txt
   ```
   - `-t`：记录时间戳到 `timing.txt`。
   - `-a`：追加模式，避免覆盖文件。
   - `output.txt`：保存命令和输出的文件。

2. **停止录制**：
   - 按 `Ctrl+D` 或输入 `exit`。

3. **回放**：
   ```bash
   scriptreplay timing.txt output.txt
   ```

#### **方法 2：使用 `asciinema`**
1. **安装**：
   ```bash
   sudo apt install asciinema  # Debian/Ubuntu
   ```

2. **开始录制**：
   ```bash
   asciinema rec myrecording.cast
   ```

3. **停止录制**：
   - 按 `Ctrl+D`。

4. **回放**：
   ```bash
   asciinema play myrecording.cast
   ```

#### **方法 3：使用 `ttyrec`**
1. **安装**：
   ```bash
   sudo apt install ttyrec  # Debian/Ubuntu
   ```

2. **开始录制**：
   ```bash
   ttyrec myrecording.tty
   ```

3. **停止录制**：
   - 按 `Ctrl+D` 或关闭终端。

4. **回放**：
   ```bash
   ttyplay myrecording.tty
   ```

---

### **三、macOS 命令行指令录制**
#### **适用场景**：录制终端操作（命令输入和输出）或键盘按键。
#### **方法 1：使用 `script` 命令**
1. **开始录制**：
   ```bash
   script -a output.txt
   ```

2. **停止录制**：
   - 按 `Ctrl+D` 或输入 `exit`。

#### **方法 2：使用 `macos-key-cast`（录制键盘按键）**
1. **安装**：
   ```bash
   npm install -g macos-key-cast
   ```

2. **开始录制**：
   ```bash
   key-cast --bounds '{"bounds":[[0,0],[1920,1080]]}'  # 自定义录制区域
   ```

3. **停止录制**：
   - 按 `Ctrl+C`。

---

### **四、Windows 命令行指令录制**
#### **适用场景**：录制命令行操作（如 PowerShell 或 CMD）。
#### **方法 1：使用 `nircmd` 工具**
1. **下载并安装**：
   - 访问 [NirCmd 官网](http://www.nirsoft.net/utils/nircmd.html) 下载并解压到系统路径（如 `C:\Windows`）。

2. **创建批处理脚本（`record.bat`）**：
   ```batch
   @echo off
   FOR /L %%i IN (1,1,100000) DO (
       nircmd savescreenshot %%i.png  # 每次截图保存为图片
       ping -n 1 127.0.0.1 -w 500 > nul  # 间隔500毫秒
   )
   ```

3. **运行脚本**：
   - 双击 `record.bat` 开始录制屏幕截图（生成 `.png` 文件）。

4. **停止录制**：
   - 关闭命令行窗口。

#### **方法 2：使用 Xbox Game Bar（录屏+音频）**
1. **开启功能**：
   - 进入 `设置 > 游戏 > Xbox Game Bar`，启用录制功能。

2. **开始录制**：
   - 按 `Win + G` 打开 Game Bar，点击录制按钮（圆形红色图标）。

3. **停止录制**：
   - 再次按 `Win + G` 停止录制，视频保存到 `视频 > 捕获` 文件夹。

---

### **五、注意事项**
1. **游戏录制**：
   - 确保游戏支持控制台指令（如 CSGO 需开启开发者模式）。
   - 录制的 `.dem` 文件可分享或用于回放分析。

2. **命令行录制**：
   - `script` 和 `asciinema` 适合记录命令输入和输出，适合教学或故障排查。
   - `nircmd` 在 Windows 下需管理员权限，且适合简单截图录制。

3. **性能影响**：
   - 高帧率录制会占用较多磁盘空间和 CPU 资源，建议关闭不必要的后台程序。

---

### **总结**
根据需求选择合适的方法：
- **游戏控制台**：使用 `record` 命令。
- **Linux/macOS 命令行**：推荐 `asciinema` 或 `script`。
- **Windows 命令行**：使用 `nircmd` 或 Xbox Game Bar。
- **键盘按键记录**：`macos-key-cast` 或 `nircmd`。

如有具体场景或工具问题，可进一步提问！