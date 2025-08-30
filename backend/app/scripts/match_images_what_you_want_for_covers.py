import requests
import logging
import time
from datetime import datetime
import random
from functools import lru_cache
import logging
import os
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import os
import shutil
import random
import datetime
from pathlib import Path
import logging


# 配置日志记录
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# API配置
# Google Custom Search API配置
GOOGLE_CSE_API_KEY = 'AIzaSyC7__VDs8N8MNcEkO6GENa7MXE67b86Rrc'
GOOGLE_CSE_ID = '15126bfec71e54938'

# 百度API配置
BAIDU_API_KEY = 'aKlzbdZKqkRt2qvFYtGrRxRf'
BAIDU_SECRET_KEY = 'YKgD2VD6yFtIkZhZ7Y5bBTpqgnuwMQrs'

# Bing Image Search API配置
BING_API_KEY = 'your_bing_api_key'

# Unsplash API配置
UNSPLASH_ACCESS_KEY = 'aMjYbSUSn8YHQyMf18UpoSD5O27b9Os4_8gjPA6Reyk'
UNSPLASH_SECRET_KEY = 'JnH7Msr2Psm7HrL_OL_AKBzy6HkN8JDvcHdczmzKJ3A'

# Pixabay API配置
# google account and email account
PIXABAY_API_KEYS = ['48396708-29d77c1d96c217e8c0b68e14b','49026138-54948971e85cc6b3e737b18ea','49026270-e9bd986c9dea4d617cedbcf27','49026359-a9e3f244c8dff62103c67d70e']

# Flickr API配置
FLICKR_API_KEY = '7392619bf9a599602a3e92a27b0e32a1'
FLICKR_SECRET_KEY = '3a8807156a38fe39'

# Pexels API配置
PEXELS_API_KEY = 'nUA7LFYIAC3chlkqexnO86B1tB8c8WqVkAlOj17aff9J7X1OSalsI1VJ'

max_retries=1
wait_timeout=0.3
start_index=0

def check_browser_session(driver):
    """检查浏览器会话是否有效"""
    try:
        if driver:
            driver.current_url
            return True
    except Exception:
        return False
    return False

def initialize_browser():
    """初始化浏览器实例"""
    options = webdriver.ChromeOptions()
    options.add_argument('--start-maximized')
    options.add_argument('--headless')
    options.add_argument('--disable-gpu')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    
    user_data_dir = os.path.expanduser('~/CustomChromeProfile_for_baidu_image_search')
    options.add_argument(f'--user-data-dir={user_data_dir}')
    
    # 设置下载目录
    download_dir = (Path(__file__).parent.parent / 'static/uploads/baidu_images').resolve()
    logger.info(f"下载目录: {download_dir}")
    os.makedirs(str(download_dir), exist_ok=True)
    
    prefs = {
        "download.default_directory": str(download_dir),
        "download.prompt_for_download": False,
        "download.directory_upgrade": True,
        "safebrowsing.enabled": False,
        "profile.default_content_settings.popups": 0,
        "profile.content_settings.exceptions.automatic_downloads.*.setting": 1
    }
    options.add_experimental_option("prefs", prefs)
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    
    logger.info("正在初始化Chrome WebDriver...")
    return webdriver.Chrome(options=options), download_dir

def wait_and_find_element(driver, by, value, parent=None, timeout=None):
    """封装的等待和查找元素的函数，带重试机制"""
    retry = 0
    while retry < max_retries:
        try:
            if parent is None:
                parent = driver
            actual_timeout = timeout if timeout else wait_timeout
            element = WebDriverWait(parent, actual_timeout).until(
                EC.presence_of_element_located((by, value))
            )
            return element
        except Exception as e:
            retry += 1
            # if retry >= max_retries:
            #     raise e
            # time.sleep(1)
    return None

def wait_and_find_elements(driver, by, value, parent=None, timeout=None):
    """封装的等待和查找多个元素的函数，带重试机制"""
    logger.info(f"开始查找元素，参数：by={by}, value={value}, parent={'root' if parent is None else 'child'}, timeout={timeout if timeout else wait_timeout}")
    retry = 0
    while retry < max_retries:
        try:
            if parent is None:
                parent = driver
            actual_timeout = timeout if timeout else wait_timeout
            logger.info(f"第{retry + 1}次尝试查找元素...")
            elements = WebDriverWait(parent, actual_timeout).until(
                EC.presence_of_all_elements_located((by, value))
            )
            if elements:
                logger.info(f"成功找到{len(elements)}个元素")
                # for idx, element in enumerate(elements):
                #     logger.info(f"元素{idx + 1}信息：")
                #     logger.info(f"- 标签：{element.tag_name}")
                #     logger.info(f"- 类名：{element.get_attribute('class')}")
                #     logger.info(f"- 可见性：{element.is_displayed()}")
                #     logger.info(f"- 位置：{element.location}")
                #     logger.info(f"- HTML：{element.get_attribute('outerHTML')}")
            else:
                logger.warning("未找到任何元素")
            return elements
        except Exception as e:
            retry += 1
            logger.error(f"第{retry}次查找失败：{str(e)}")
            # logger.error(f"当前页面源码：\n{driver.page_source}")
            if retry >= max_retries:
                logger.error(f"达到最大重试次数{max_retries}，抛出异常")
                # raise e
                
            logger.info(f"等待1秒后重试...")
            # time.sleep(1)
    logger.warning("查找元素失败，返回空列表")
    return []

def get_downloaded_file_path(download_dir):
    """获取最新下载的文件路径
    
    Args:
        download_dir: 下载目录路径
        
    Returns:
        str: 最新下载的文件路径，如果目录为空或没有有效文件则返回None
    """
    try:
        if not os.path.exists(download_dir) or not os.path.isdir(download_dir):
            logger.error(f"下载目录 {download_dir} 不存在或不是目录")
            return None
            
        files = os.listdir(download_dir)
        if not files:
            logger.warning(f"下载目录 {download_dir} 为空")
            return None
            
        # 过滤掉临时下载文件和隐藏文件
        valid_files = [f for f in files if not f.startswith('.') and not f.endswith('.crdownload')]
        if not valid_files:
            logger.warning(f"下载目录 {download_dir} 中没有有效的下载文件")
            return None
            
        paths = [os.path.join(download_dir, basename) for basename in valid_files]
        newest_file = max(paths, key=os.path.getctime)
        
        if os.path.exists(newest_file) and os.path.isfile(newest_file):
            return newest_file
        else:
            logger.error(f"最新文件 {newest_file} 不存在或不是文件")
            return None
            
    except Exception as e:
        logger.error(f"获取下载文件路径时出错: {str(e)}")
        return None

def baidu_image_search_3(driver, download_dir, query, query_prefix=None, query_posfix=None, limit_count=20, width_height=7, imgratio=None, imgformat=None, hd='1'):
    '''
    百度图片搜索，使用selenium模拟浏览器操作，将会直接下载到浏览器指定的目录下，当前是：../static/uploads/baidu_images
    增加了更多可控的参数来支持更细致的搜索
    
    参数:
        query: str, 搜索关键词
        query_prefix: str, 搜索关键词前缀，默认空，非空的情况下，必须是以空格结尾的，比如：”漫画 “
        query_posfix: str, 搜索关键词后缀，默认空，非空的情况下，必须是以空格开始的，比如：” 海报“
        limit_count: int, 最大下载的数量，默认20，浏览器满屏有几个就几个
        width_height: str, 图片尺寸，可选值：6=大图（600px-1080px） 7=超大图（1080px以上），默认空为全部
        imgratio: str, 图片比例，可选值：1=细长竖图 2=竖图 3=方图 4=横图 5=细长横图，默认空为全部
        imgformat: str, 图片格式，可选值：2=BMP 3=JPG 4=PNG 5=JPEG 6=GIF，默认空为全部
        hd: str, 是否高清，可选值：1=高清, 2=图集，默认1为高清
    
    返回值:
        返回本地的图片路径列表
    '''
    logger.info(f"开始百度图片搜索，关键词: {query}")
    
    # 对输入的query_prefix和query_posfix进行严格处理，如果非空，则必须以空格结尾和以空格开始，如果不满足，进行修改
    if query_prefix is not None and not query_prefix.endswith(' '):
        query_prefix = query_prefix.strip() + ' '
    if query_posfix is not None and not query_posfix.startswith(' '):
        query_posfix = ''' ''' + query_posfix.strip()
    
    urls = [f'https://image.baidu.com/search/index?tn=baiduimage&z={width_height}&imgratio={imgratio}&imgformat={imgformat}&hd={hd}&word={query_prefix}{query}{query_posfix}']

    try:
        for url in urls:
            try:
                # 访问URL并等待页面加载
                logger.info(f"正在访问URL: {url}")
                driver.get(url)
                
                if not check_browser_session(driver):
                    logger.error("浏览器会话已失效，尝试重新初始化...")
                    session_retry_count += 1
                    continue
                
                # 实现滚动加载机制(limit_count大于20才滚屏)
                if limit_count > 20:
                    last_height = driver.execute_script("return document.body.scrollHeight")
                    
                    # 设置滚动暂停时间
                    scroll_pause_time = 1
                    # 设置滚动次数
                    scroll_attempts = limit_count % 20 + 1
                    
                    for _ in range(scroll_attempts):
                        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                        time.sleep(scroll_pause_time)
                
                # 模式一的定位尝试（最有可能成功的模式）
                image_class_name = 'waterfall-item_2GzCn'
                
                image_elements = wait_and_find_elements(driver, By.CLASS_NAME, image_class_name)
                total_images = len(image_elements)
                logger.info(f"通过class=’waterfall-item_2GzCn’找到 {total_images} 个图片元素")
                
                # 模式一的尝试
                if total_images > 0:
                    # 从断点位置开始处理图片
                    for i in range(start_index, min(total_images, limit_count)):
                        try:
                            logger.info(f"正在处理第 {i + 1}/{total_images} 个图片")
                            
                            # 重新获取当前图片元素，避免stale element
                            image = wait_and_find_elements(driver, By.CLASS_NAME, image_class_name)[i]
                            
                            # 模拟鼠标移动到图片元素上
                            actions = ActionChains(driver)
                            actions.move_to_element(image).perform()
                            logger.info("已将鼠标移动到图片元素上")
                            
                            # 等待页面响应
                            time.sleep(0.3)
                            logger.info("等待(0.3s)页面响应完成")
                            
                            logger.error("尝试使用class='download-text_rXaz4'的进行定位下载按钮...")
                            
                            download_button = wait_and_find_element(driver, By.CLASS_NAME, 'download-text_rXaz4', parent=image)
                            logger.info("重新定位下载按钮定位成功：")
                            logger.info(f"- 下载按钮类名：{download_button.get_attribute('class')}")
                            logger.info(f"- 下载按钮可见性：{download_button.is_displayed()}")
                            logger.info(f"- 下载按钮HTML：{download_button.get_attribute('outerHTML')}")
                            logger.info("准备点击下载按钮...")
                            
                            driver.execute_script("arguments[0].click();", download_button)
                            logger.info("下载按钮点击完成(会sleep1s)")
                            
                            time.sleep(1)  # 增加下载等待时间
                        except Exception as e:
                            logger.error(f"模式一种处理第 {i + 1} 个图片时出错: {str(e)}")
                else:
                    # 模式二的定位尝试
                    image_class_name = 'waterfall-item_2GzCn'
                    
                    image_elements = wait_and_find_elements(By.CLASS_NAME, image_class_name)
                    total_images = len(image_elements)
                    logger.info(f"通过class=’waterfall-item_2GzCn’找到 {total_images} 个图片元素")
                    
                    if total_images > 0:
                        # 从断点位置开始处理图片
                        for i in range(start_index, min(total_images, limit_count)):
                            try:
                                logger.info(f"正在处理第 {i + 1}/{total_images} 个图片")
                                
                                # 重新获取当前图片元素，避免stale element
                                image = wait_and_find_elements(By.CLASS_NAME, image_class_name)[i]
                                
                                # 模拟鼠标移动到图片元素上
                                actions = ActionChains(driver)
                                actions.move_to_element(image).perform()
                                logger.info("已将鼠标移动到图片元素上")
                                
                                # 等待页面响应
                                time.sleep(0.3)
                                logger.info("等待(0.3s)页面响应完成")
                                
                                logger.error("尝试使用class='download-text_rXaz4'的进行定位下载按钮...")
                                
                                download_button = wait_and_find_element(By.CLASS_NAME, 'download-text_rXaz4', parent=image)
                                logger.info("重新定位下载按钮定位成功：")
                                logger.info(f"- 下载按钮类名：{download_button.get_attribute('class')}")
                                logger.info(f"- 下载按钮可见性：{download_button.is_displayed()}")
                                logger.info(f"- 下载按钮HTML：{download_button.get_attribute('outerHTML')}")
                                logger.info("准备点击下载按钮...")
                                
                                driver.execute_script("arguments[0].click();", download_button)
                                logger.info("下载按钮点击完成(会sleep1s)")
                                # time.sleep(1)  # 增加下载等待时间
                            except Exception as e:
                                logger.error(f"模式一种处理第 {i + 1} 个图片时出错: {str(e)}")
                    
                    else:
                        # 模式三的定位尝试
                        image_class_name = 'imgitem'
                        # 等待图片元素加载
                        logger.info(f"等待图片元素加载...")
                        image_elements = wait_and_find_elements(By.CLASS_NAME, image_class_name)
                        total_images = len(image_elements)
                        
                        logger.info(f"找到 {total_images} 个图片元素")
                        
                        # 模式三的尝试
                        if total_images > 0:
                            # 从断点位置开始处理图片
                            for i in range(start_index, min(total_images, limit_count)):
                                try:
                                    logger.info(f"正在处理第 {i + 1}/{total_images} 个图片")
                                    
                                    # 重新获取当前图片元素，避免stale element
                                    image = wait_and_find_elements(By.CLASS_NAME, image_class_name)[i]
                                    
                                    # 模拟鼠标移动到图片元素上
                                    actions = ActionChains(driver)
                                    actions.move_to_element(image).perform()
                                    logger.info("已将鼠标移动到图片元素上")
                                    
                                    # 等待页面响应
                                    time.sleep(0.3)
                                    logger.info("等待(0.3s)页面响应完成")
                                    
                                    # 查找hover元素并模拟鼠标悬停
                                    logger.info("开始查找hover元素...")
                                    hover_element = wait_and_find_element(By.CLASS_NAME, 'hover', parent=image)
                                    if hover_element:
                                        logger.info("hover元素定位成功：")
                                        logger.info(f"- hover元素类名：{hover_element.get_attribute('class')}")
                                        logger.info(f"- hover元素可见性：{hover_element.is_displayed()}")
                                        logger.info(f"- hover元素HTML：{hover_element.get_attribute('outerHTML')}")
                                        
                                        actions = ActionChains(driver)
                                        logger.info("准备执行鼠标悬停操作...")
                                        actions.move_to_element(hover_element).perform()
                                        time.sleep(0.3)
                                        logger.info("找到hover并操作鼠标悬停完成")
                                        
                                        # 查找并点击下载按钮
                                        logger.info("开始查找下载按钮...")
                                        download_button = wait_and_find_element(By.CLASS_NAME, 'down', parent=hover_element)
                                        if download_button:
                                            logger.info("下载按钮定位成功：")
                                            logger.info(f"- 下载按钮类名：{download_button.get_attribute('class')}")
                                            logger.info(f"- 下载按钮可见性：{download_button.is_displayed()}")
                                            logger.info(f"- 下载按钮HTML：{download_button.get_attribute('outerHTML')}")
                                            logger.info("准备点击下载按钮...")
                                            
                                            driver.execute_script("arguments[0].click();", download_button)
                                            logger.info("下载按钮点击完成(会sleep1s)")
                                            # time.sleep(1)  # 增加下载等待时间
                                        else:
                                            logger.error("未找到下载按钮，模式一尝试失败")
                                    else:
                                        logger.error("未找到hover元素，模式一尝试失败")
                                except Exception as e:
                                    logger.error(f"模式一种处理第 {i + 1} 个图片时出错: {str(e)}")
                
                time.sleep(1)
                logger.info("所有图片处理完成")
            
            except Exception as e:
                logger.error(f"爬取过程中出错: {str(e)}")
                logger.error(f"当前页面源码：{driver.page_source}")
                continue
    finally:
        # 等待文件下载完成
        downloaded_file_path = None
        max_wait_time = 30  # 最大等待时间（秒）
        wait_start_time = time.time()
        wait_interval = 1  # 等待间隔（秒）
        download_success = False
        
        while time.time() - wait_start_time < max_wait_time:
            downloaded_file_path = get_downloaded_file_path(download_dir)
            if downloaded_file_path and not downloaded_file_path.endswith('.crdownload'):
                download_success = True
                logger.info(f"成功下载文件: {downloaded_file_path}")
                break
            time.sleep(wait_interval)
            
        if not download_success:
            logger.warning(f"等待下载文件超时或未找到有效下载文件，继续执行")
    
    return download_dir

def baidu_images_search_and_restore_for_covers():
    """
    通过百度的图片搜索下载图片，并且手动移动到指定的目录:../static/covers
    专门为封面图片使用
    
    处理流程：
    1. 读取crawler_categories目录下的配置文件获取关键词列表
    2. 使用百度图片搜索接口下载图片
    3. 将图片重命名并移动到指定目录

    Returns:
        None
    """
    import os
    import shutil
    import random
    import datetime
    from pathlib import Path
    import logging

    logger = logging.getLogger(__name__)

    # 获取crawler_categories目录的路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    categories_dir = os.path.join(current_dir, 'crawler_categories')
    
    # 确保目标目录存在
    spider_base_dir = os.path.join(current_dir, '..', 'static', 'covers')
    os.makedirs(spider_base_dir, exist_ok=True)
    
    driver, download_dir = initialize_browser()

    # 遍历crawler_categories目录下的所有文件
    for category_file in os.listdir(categories_dir):
        if not category_file.endswith('.txt'):
            continue

        category_name = os.path.splitext(category_file)[0]
        category_file_path = os.path.join(categories_dir, category_file)
        target_dir = os.path.join(spider_base_dir, category_name)
        os.makedirs(target_dir, exist_ok=True)

        logger.info(f'处理类别：{category_name}')

        try:
            # 读取关键词列表
            with open(category_file_path, 'r', encoding='utf-8') as f:
                keywords = [line.strip() for line in f if line.strip()]

            # 处理每个关键词
            for keyword in keywords:
                logger.info(f'搜索关键词：{keyword}')
                try:
                    # 调用百度图片搜索接口
                    download_dir = baidu_image_search_3(driver=driver, download_dir=download_dir, query=keyword, query_prefix="漫画", query_posfix="海报", limit_count=2, width_height=7, imgratio=None, imgformat=3, hd=1)

                    # 获取当前日期
                    current_date = datetime.datetime.now().strftime('%Y%m%d')

                    # 检查下载目录是否有文件
                    download_files = [f for f in os.listdir(download_dir) if os.path.isfile(os.path.join(download_dir, f)) and not f.startswith('.') and not f.endswith('.crdownload')]
                    
                    if not download_files:
                        logger.warning(f'关键词 {keyword} 没有成功下载任何文件，跳过处理')
                        continue

                    # 移动并重命名图片
                    for idx, src_path in list(enumerate(Path(download_dir).glob('*'), start=1)):
                        if not os.path.exists(src_path) or not os.path.isfile(src_path) or src_path.name.startswith('.') or src_path.name.endswith('.crdownload'):
                            logger.warning(f'跳过无效文件：{src_path}')
                            continue

                        # 生成新的文件名
                        random_num = random.randint(1000, 9999)
                        new_filename = f'{keyword}_{current_date}_{random_num}.jpg'
                        final_target_dir = os.path.join(target_dir, keyword)
                        os.makedirs(final_target_dir, exist_ok=True)
                        
                        dst_path = os.path.join(final_target_dir, new_filename)

                        try:
                            shutil.move(src_path, dst_path)
                            logger.info(f'已移动并重命名图片：{dst_path}')
                        except Exception as e:
                            logger.error(f'移动文件失败 {src_path} -> {dst_path}: {str(e)}')

                except Exception as e:
                    logger.error(f'处理关键词 {keyword} 时出错：{str(e)}')

        except Exception as e:
            logger.error(f'处理类别 {category_name} 时出错：{str(e)}')

        finally:
            # 等待文件下载完成
            driver.quit()

    logger.info('图片搜索和存储处理完成')    

import json
def load_keywords():
    """
    从words_mapping_for_image_crawler.json中加载需要抓取的目录与关键词的映射内容
    """
    # 从当前脚本所在目录加载words_mapping_for_image_crawler.json
    current_dir = os.path.dirname(os.path.abspath(__file__))
    words_mapping_file = os.path.join(current_dir, 'words_mapping_for_image_crawler.json')
    # 加载JSON文件
    with open(words_mapping_file, 'r', encoding='utf-8') as f:
        words_mapping = json.load(f)
    return words_mapping

# 使用示例
if __name__ == '__main__':
    test_keyword = 'cat'
    
    baidu_images_search_and_restore_for_covers()