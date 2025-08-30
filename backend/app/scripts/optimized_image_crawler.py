import os
import time
import random
import logging
import datetime
import concurrent.futures
from pathlib import Path
import shutil
from functools import lru_cache
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
from selenium.common.exceptions import TimeoutException, NoSuchElementException, StaleElementReferenceException

# 配置日志记录
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def initialize_browser(download_dir=None):
    """初始化浏览器实例，优化配置参数
    
    Args:
        download_dir: 下载目录路径
        
    Returns:
        tuple: (webdriver实例, 下载目录)
    """
    options = webdriver.ChromeOptions()
    
    # 基本优化参数
    options.add_argument('--headless')  # 无头模式
    options.add_argument('--disable-gpu')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    
    # 性能优化参数
    options.add_argument('--disable-extensions')  # 禁用扩展
    options.add_argument('--disable-infobars')  # 禁用信息栏
    options.add_argument('--disable-notifications')  # 禁用通知
    options.add_argument('--disable-popup-blocking')  # 禁用弹窗阻止
    options.add_argument('--blink-settings=imagesEnabled=false')  # 禁用图片加载，提高速度
    options.add_argument('--disable-default-apps')  # 禁用默认应用
    
    # 内存优化
    options.add_argument('--js-flags=--expose-gc')  # 启用垃圾回收
    options.add_argument('--disable-features=site-per-process')  # 禁用每个站点一个进程
    
    # 设置用户数据目录
    user_data_dir = os.path.expanduser('~/CustomChromeProfile_for_baidu_image_search')
    options.add_argument(f'--user-data-dir={user_data_dir}')
    
    # 设置下载目录
    if not download_dir:
        download_dir = (Path(__file__).parent.parent / 'static/uploads/baidu_images').resolve()
    
    os.makedirs(str(download_dir), exist_ok=True)
    
    prefs = {
        "download.default_directory": str(download_dir),
        "download.prompt_for_download": False,
        "download.directory_upgrade": True,
        "safebrowsing.enabled": False,
        "profile.default_content_settings.popups": 0,
        "profile.content_settings.exceptions.automatic_downloads.*.setting": 1,
        "browser.enable_spellchecking": False,  # 禁用拼写检查
        "browser.sessionstore.resume_from_crash": False,  # 禁用会话恢复
        "credentials_enable_service": False,  # 禁用凭据服务
        "profile.password_manager_enabled": False,  # 禁用密码管理器
    }
    options.add_experimental_option("prefs", prefs)
    options.add_experimental_option("excludeSwitches", ["enable-automation", "enable-logging"])
    
    logger.info(f"下载目录: {download_dir}")
    return webdriver.Chrome(options=options), download_dir


def wait_for_element(driver, by, value, timeout=10, parent=None, condition=EC.presence_of_element_located):
    """等待元素出现的通用函数，使用显式等待
    
    Args:
        driver: WebDriver实例
        by: 定位方式
        value: 定位值
        timeout: 超时时间
        parent: 父元素
        condition: 等待条件
        
    Returns:
        找到的元素或None
    """
    try:
        wait = WebDriverWait(parent or driver, timeout)
        return wait.until(condition((by, value)))
    except (TimeoutException, NoSuchElementException) as e:
        logger.debug(f"等待元素 {value} 超时: {str(e)}")
        return None


def wait_for_elements(driver, by, value, timeout=10, parent=None):
    """等待多个元素出现的通用函数"""
    return wait_for_element(driver, by, value, timeout, parent, EC.presence_of_all_elements_located)


def scroll_page(driver, pause_time=0.5, max_attempts=2):
    """优化的页面滚动函数
    
    Args:
        driver: WebDriver实例
        pause_time: 每次滚动后的暂停时间
        max_attempts: 最大滚动尝试次数
        
    Returns:
        None
    """
    last_height = driver.execute_script("return document.body.scrollHeight")
    attempts = 0
    
    while attempts < max_attempts:
        # 滚动到页面底部
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        
        # 等待页面加载
        time.sleep(pause_time)
        
        # 计算新的滚动高度并比较
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            break
            
        last_height = new_height
        attempts += 1


def download_image(driver, image, retry_count=2, wait_timeout=5):
    """下载单个图片的优化函数
    
    Args:
        driver: WebDriver实例
        image: 图片元素
        retry_count: 重试次数
        wait_timeout: 等待超时时间
        
    Returns:
        bool: 是否成功下载
    """
    for attempt in range(retry_count):
        try:
            # 使用JavaScript滚动到元素位置，更可靠
            driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", image)
            
            # 使用ActionChains移动到元素
            actions = ActionChains(driver)
            actions.move_to_element(image).perform()
            
            # 查找hover元素，使用显式等待
            hover_element = wait_for_element(driver, By.CLASS_NAME, 'hover', wait_timeout, image)
            if not hover_element:
                logger.warning("未找到hover元素，重试...")
                continue
                
            # 移动到hover元素
            actions = ActionChains(driver)
            actions.move_to_element(hover_element).perform()
            
            # 查找并点击下载按钮，使用显式等待
            download_button = wait_for_element(driver, By.CLASS_NAME, 'down', wait_timeout, hover_element)
            if not download_button:
                logger.warning("未找到下载按钮，重试...")
                continue
                
            # 使用JavaScript点击，更可靠
            driver.execute_script("arguments[0].click();", download_button)
            
            # 使用短暂等待确保下载开始
            time.sleep(0.5)
            return True
            
        except StaleElementReferenceException:
            # 元素已过时，需要重新获取
            logger.warning("元素已过时，重试...")
            if attempt == retry_count - 1:
                logger.error("元素持续过时，放弃下载")
                return False
                
        except Exception as e:
            logger.error(f"下载图片时出错: {str(e)}")
            if attempt == retry_count - 1:
                return False
    
    return False


def process_images(driver, start_index=0, max_images=40, wait_timeout=5):
    """处理页面上的图片
    
    Args:
        driver: WebDriver实例
        start_index: 起始索引
        max_images: 最大处理图片数
        wait_timeout: 等待超时时间
        
    Returns:
        int: 成功处理的图片数量
    """
    # 等待图片元素加载
    image_elements = wait_for_elements(driver, By.CLASS_NAME, 'imgitem', wait_timeout)
    if not image_elements:
        logger.warning("未找到图片元素")
        return 0
        
    total_images = min(len(image_elements), max_images)
    logger.info(f"找到 {total_images} 个图片元素，准备处理")
    
    success_count = 0
    
    # 从断点位置开始处理图片
    for i in range(start_index, total_images):
        try:
            # 重新获取当前图片元素，避免stale element
            image_elements = wait_for_elements(driver, By.CLASS_NAME, 'imgitem', wait_timeout)
            if not image_elements or i >= len(image_elements):
                break
                
            image = image_elements[i]
            logger.info(f"正在处理第 {i + 1}/{total_images} 个图片")
            
            if download_image(driver, image):
                success_count += 1
                
        except Exception as e:
            logger.error(f"处理第 {i + 1} 个图片时出错: {str(e)}")
            # 保存当前进度，便于断点续传
            continue
    
    return success_count


def baidu_image_search_optimized(query, max_images=40, wait_timeout=5):
    """优化的百度图片搜索函数
    
    Args:
        query: 搜索关键词
        max_images: 最大下载图片数量
        wait_timeout: 等待超时时间
        
    Returns:
        str: 下载目录路径
    """
    logger.info(f"开始百度图片搜索，关键词: {query}")
    driver = None
    
    try:
        # 初始化浏览器
        driver, download_dir = initialize_browser()
        
        # 构建搜索URL
        url = f'https://image.baidu.com/search/index?tn=baiduimage&word={query}'
        logger.info(f"正在访问URL: {url}")
        
        # 访问URL
        driver.get(url)
        
        # 等待页面加载完成
        body = wait_for_element(driver, By.TAG_NAME, 'body', wait_timeout)
        if not body:
            logger.error("页面加载失败")
            return download_dir
            
        # 滚动页面以加载更多图片
        scroll_page(driver, 0.5, 2)
        
        # 处理图片
        success_count = process_images(driver, 0, max_images, wait_timeout)
        logger.info(f"成功处理 {success_count} 张图片")
        
        return download_dir
        
    except Exception as e:
        logger.error(f"搜索过程中出错: {str(e)}")
        return download_dir if 'download_dir' in locals() else None
        
    finally:
        # 确保关闭浏览器
        if driver:
            driver.quit()
            logger.info("浏览器已关闭")


def process_keyword(category_name, keyword, target_dir):
    """处理单个关键词的图片搜索和保存
    
    Args:
        category_name: 类别名称
        keyword: 搜索关键词
        target_dir: 目标目录
        
    Returns:
        int: 成功处理的图片数量
    """
    try:
        logger.info(f'搜索关键词：{keyword}')
        
        # 调用优化后的百度图片搜索接口
        download_dir = baidu_image_search_optimized(keyword, max_images=20)
        if not download_dir:
            return 0
            
        # 获取当前日期
        current_date = datetime.datetime.now().strftime('%Y%m%d')
        
        # 确保目标目录存在
        final_target_dir = os.path.join(target_dir, keyword)
        os.makedirs(final_target_dir, exist_ok=True)
        
        # 移动并重命名图片
        success_count = 0
        for src_path in Path(download_dir).glob('*'):
            if not os.path.exists(src_path):
                continue
                
            # 生成新的文件名
            random_num = random.randint(1000, 9999)
            new_filename = f'{keyword}_{current_date}_{random_num}.jpg'
            dst_path = os.path.join(final_target_dir, new_filename)
            
            try:
                shutil.move(src_path, dst_path)
                logger.info(f'已移动并重命名图片：{dst_path}')
                success_count += 1
            except Exception as e:
                logger.error(f'移动文件失败 {src_path} -> {dst_path}: {str(e)}')
                
        return success_count
    except Exception as e:
        logger.error(f'处理关键词 {keyword} 时出错：{str(e)}')
        return 0


def baidu_images_search_and_restore_optimized():
    """优化版的百度图片搜索和存储函数
    
    使用并行处理多个关键词，提高效率
    
    处理流程：
    1. 读取crawler_categories目录下的配置文件获取关键词列表
    2. 使用优化后的百度图片搜索接口下载图片
    3. 将图片重命名并移动到指定目录
    
    Returns:
        None
    """
    # 获取crawler_categories目录的路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    categories_dir = os.path.join(current_dir, 'crawler_categories')
    
    # 确保目标目录存在
    spider_base_dir = os.path.join(current_dir, '..', 'static', 'uploads', 'images')
    os.makedirs(spider_base_dir, exist_ok=True)
    
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
                
            # 使用并行处理多个关键词
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                # 创建任务列表
                future_to_keyword = {executor.submit(process_keyword, category_name, keyword, target_dir): keyword for keyword in keywords}
                
                # 处理完成的任务
                for future in concurrent.futures.as_completed(future_to_keyword):
                    keyword = future_to_keyword[future]
                    try:
                        success_count = future.result()
                        logger.info(f'关键词 {keyword} 处理完成，成功下载 {success_count} 张图片')
                    except Exception as e:
                        logger.error(f'处理关键词 {keyword} 时出错：{str(e)}')
                        
        except Exception as e:
            logger.error(f'处理类别 {category_name} 时出错：{str(e)}')
            
    logger.info('图片搜索和存储处理完成')


# 如果直接运行此脚本，则执行优化版的搜索和存储函数
if __name__ == "__main__":
    baidu_images_search_and_restore_optimized()