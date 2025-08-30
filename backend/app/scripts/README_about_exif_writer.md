
---

# 写入 EXIF 信息的工具与方法

在 Python 中，可以使用以下工具和方法来写入图片的 EXIF 信息。以下是推荐的工具及其使用方法。

---

## 1. **`pyexiv2`**
`pyexiv2` 是一个用于操作图像元数据的库，支持读取和写入 EXIF、IPTC 和 XMP 数据。

### 安装
```bash
pip install pyexiv2
```

### 示例代码（写入 EXIF）
```python
import pyexiv2

def write_exif(image_path, exif_data):
    """写入EXIF信息"""
    try:
        with pyexiv2.Image(image_path) as img:
            for key, value in exif_data.items():
                img[key] = value
            img.write()
        print(f"成功写入EXIF信息到 {image_path}")
    except Exception as e:
        print(f"写入EXIF信息时出错: {str(e)}")

# 示例用法
if __name__ == '__main__':
    image_path = "example.jpg"
    exif_data = {
        'Exif.Photo.DateTimeOriginal': '2023:01:01 12:00:00',
        'Exif.Image.Make': 'Nikon',
        'Exif.Image.Model': 'D5600'
    }
    write_exif(image_path, exif_data)
```

---

## 2. **`Pillow`**
`Pillow` 是 Python 的图像处理库，虽然它的主要功能是图像处理，但也支持部分 EXIF 操作。

### 安装
```bash
pip install Pillow
```

### 示例代码（写入 EXIF）
```python
from PIL import Image

def write_exif_pillow(image_path, exif_data):
    """使用Pillow写入EXIF信息"""
    try:
        with Image.open(image_path) as img:
            # 获取现有的EXIF数据
            exif = img.getexif()
            for key, value in exif_data.items():
                exif[key] = value
            img.save('output.jpg', exif=exif)
        print(f"成功写入EXIF信息到 output.jpg")
    except Exception as e:
        print(f"写入EXIF信息时出错: {str(e)}")

# 示例用法
if __name__ == '__main__':
    image_path = "example.jpg"
    exif_data = {
        36867: '2023:01:01 12:00:00',  # DateTimeOriginal
        271: 'Nikon',                   # Make
        272: 'D5600'                    # Model
    }
    write_exif_pillow(image_path, exif_data)
```

> 注意：`Pillow` 支持的 EXIF 标签有限，建议参考 [EXIF Tags](https://pillow.readthedocs.io/en/stable/ref/exif.html) 文档。

---

## 3. **`exiftool`**
`exiftool` 是一个功能强大的命令行工具，支持几乎所有类型的元数据操作。可以通过 Python 调用 `exiftool` 来实现 EXIF 写入。

### 安装 `exiftool`
- Windows: 下载并安装 [exiftool](https://exiftool.org/)。
- macOS: 使用 Homebrew 安装：
  ```bash
  brew install exiftool
  ```
- Linux: 使用包管理器安装：
  ```bash
  sudo apt-get install libimage-exiftool-perl
  ```

### 方法 1: 使用 `subprocess`
```python
import subprocess

def write_exif_with_exiftool(image_path, exif_data):
    """使用exiftool写入EXIF信息"""
    try:
        command = ['exiftool']
        for key, value in exif_data.items():
            command.extend([f'-{key}={value}'])
        command.append(image_path)
        result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(result.stdout.decode())
    except subprocess.CalledProcessError as e:
        print(f"写入EXIF信息时出错: {e.stderr.decode()}")

# 示例用法
if __name__ == '__main__':
    image_path = "example.jpg"
    exif_data = {
        'DateTimeOriginal': '2023:01:01 12:00:00',
        'Make': 'Nikon',
        'Model': 'D5600'
    }
    write_exif_with_exiftool(image_path, exif_data)
```

### 方法 2: 使用 `pyexiftool`
`pyexiftool` 是一个封装了 `exiftool` 的 Python 库，简化了调用过程。

#### 安装
```bash
pip install pyexiftool
```

#### 示例代码
```python
import pyexiftool

def write_exif_with_pyexiftool(image_path, exif_data):
    """使用pyexiftool写入EXIF信息"""
    try:
        with pyexiftool.ExifTool() as et:
            for key, value in exif_data.items():
                et.execute(f'-{key}={value}', image_path)
            print("成功写入EXIF信息")
    except Exception as e:
        print(f"写入EXIF信息时出错: {str(e)}")

# 示例用法
if __name__ == '__main__':
    image_path = "example.jpg"
    exif_data = {
        'DateTimeOriginal': '2023:01:01 12:00:00',
        'Make': 'Nikon',
        'Model': 'D5600'
    }
    write_exif_with_pyexiftool(image_path, exif_data)
```

---

## 总结
- 如果你希望完全在 Python 中实现，推荐使用 **`pyexiv2`**。
- 如果你需要更强大的功能，并且不介意调用外部工具，推荐使用 **`exiftool`**。
- 如果只需要简单的 EXIF 操作，**`Pillow`** 也可以满足需求，但功能较为有限。

--- 

