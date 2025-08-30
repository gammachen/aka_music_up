import os
from pathlib import Path

# 配置文件路径
MAPPING_FILE = "md_files.txt"
ROOT_DIR = "./"  # 公共目录根路径

def rename_files():
    with open(MAPPING_FILE, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or '|' not in line:
                continue
            
            src_rel, dst_rel = line.split('|', 1)
            src_path = Path(ROOT_DIR) / src_rel
            dst_path = Path(ROOT_DIR) / dst_rel

            if not src_path.exists():
                print(f"⚠️ 源文件不存在: {src_path}")
                continue

            # 创建目标目录（如果不存在）
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            
            try:
                src_path.rename(dst_path)
                print(f"✅ 重命名成功: {src_rel} -> {dst_rel}")
            except Exception as e:
                print(f"❌ 重命名失败 [{src_rel}]: {str(e)}")

if __name__ == "__main__":
    rename_files()