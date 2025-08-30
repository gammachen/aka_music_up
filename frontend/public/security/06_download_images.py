import os
import requests
from urllib.parse import urlparse

# 目标下载目录（可修改）
DOWNLOAD_DIR = "qingflow_images"

# 图片URL列表
IMAGE_URLS = [
    "https://file.qingflow.com/official-page/index/v2021/section2-1-1.png",
    "https://file.qingflow.com/official-page/index/v2021/section2-3-2.png",
    "https://file.qingflow.com/official-page/index/customer-case/production-manage.png",
    "https://file.qingflow.com/official-page/index/customer-case/quality-control.png",
    "https://file.qingflow.com/official-page/index/customer-case/oa-coworking-platform.png",
    "https://file.qingflow.com/official-page/index/v2021/section2-1-4.png",
    "https://file.qingflow.com/official-page/index/customer-case/customer-manage.png",
    "https://file.qingflow.com/official-page/index/customer-case/internet-management.png",
    "https://file.qingflow.com/official-page/index/customer-case/project-manage.png",
    "https://file.qingflow.com/official-page/index/customer-case/invoicing-2.0.png",
    "https://file.qingflow.com/official-page/index/customer-case/after-manage.png",
    "https://file.qingflow.com/official-page/index/customer-case/functionality-updates.png",
    "https://file.qingflow.com/official-page/index/customer-case/asset-management.png",
    "https://file.qingflow.com/official-page/index/customer-case/process-manage.png",
    "https://file.qingflow.com/official-page/index/customer-case/plm-product-lifecycle.png",
    "https://file.qingflow.com/official-page/index/customer-case/okr-manage-2.0.png",
    "https://file.qingflow.com/official-page/index/customer-case/more.png"
]

def download_image(url, save_dir):
    """下载单个图片"""
    try:
        # 解析URL，获取文件名
        parsed_url = urlparse(url)
        filename = os.path.basename(parsed_url.path)
        save_path = os.path.join(save_dir, filename)

        # 设置请求头（Referer 防止403）
        headers = {
            "Referer": "https://qingflow.com",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }

        # 发送请求并下载
        response = requests.get(url, headers=headers, stream=True)
        response.raise_for_status()  # 检查请求是否成功

        # 保存文件
        with open(save_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        print(f"下载成功: {filename}")
    except Exception as e:
        print(f"下载失败: {url} - {e}")

def main():
    # 创建下载目录
    if not os.path.exists(DOWNLOAD_DIR):
        os.makedirs(DOWNLOAD_DIR)

    # 遍历下载所有图片
    for url in IMAGE_URLS:
        download_image(url, DOWNLOAD_DIR)

    print("所有文件下载完成！")

if __name__ == "__main__":
    main()