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

# 令牌桶算法实现请求频率限制
class TokenBucket:
    def __init__(self, tokens, fill_rate):
        self.capacity = tokens
        self.tokens = tokens
        self.fill_rate = fill_rate
        self.last_update = time.time()

    def consume(self):
        now = time.time()
        tokens_to_add = (now - self.last_update) * self.fill_rate
        self.tokens = min(self.capacity, self.tokens + tokens_to_add)
        self.last_update = now

        if self.tokens < 1:
            return False
        self.tokens -= 1
        return True

# 创建令牌桶实例
_google_token_bucket = TokenBucket(tokens=60, fill_rate=1)
_baidu_token_bucket = TokenBucket(tokens=60, fill_rate=1)
_bing_token_bucket = TokenBucket(tokens=60, fill_rate=1)
_unsplash_token_bucket = TokenBucket(tokens=50, fill_rate=1)
_pixabay_token_bucket = TokenBucket(tokens=100, fill_rate=1)
_flickr_token_bucket = TokenBucket(tokens=60, fill_rate=1)
_pexels_token_bucket = TokenBucket(tokens=50, fill_rate=1)

# Google图片搜索
@lru_cache(maxsize=100)
def google_image_search(query):
    return google_image_search_by_count(query, 2)
    
@lru_cache(maxsize=100)
def google_image_search_by_count(query, count=2):
    logger.info(f"开始Google图片搜索，关键词: {query}")
    
    while not _google_token_bucket.consume():
        logger.info("达到Google API请求限制，等待...")
        time.sleep(1)
    
    url = "https://www.googleapis.com/customsearch/v1"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.9,zh-CN;q=0.8,zh;q=0.7'
    }
    
    params = {
        'q': f"{query}",
        'key': GOOGLE_CSE_API_KEY,
        'cx': GOOGLE_CSE_ID,
        'searchType': 'image',
        'imgType': 'photo',
        'safe': 'active',
        'imgSize': 'large',
        'count': count,
        'num': count,
    }
    
    '''
    https://developers.google.com/custom-search/v1/reference/rest/v1/ImgType?hl=zh-cn
    
    imgTypeUndefined	未指定映像类型。
    clipart	仅限剪贴画样式的图片。
    face	仅限人脸的图片。
    lineart	仅限线条艺术图片。
    stock	仅限图库图片。
    photo	仅限照片图片。
    animated	仅限动画图片。
    '''
    
    try:
        response = requests.get(url, params=params, headers=headers)
        response.raise_for_status()
        results = response.json()
        
        print(results)
        
        if 'items' in results:
            return [item['link'] for item in results['items']]
        return []
    except Exception as e:
        logger.error(f"Google图片搜索出错: {str(e)}")
        return []

# 百度图片搜索
def get_baidu_access_token():
    token_url = "https://aip.baidubce.com/oauth/2.0/token"
    params = {
        'grant_type': 'client_credentials',
        'client_id': BAIDU_API_KEY,
        'client_secret': BAIDU_SECRET_KEY
    }
    try:
        response = requests.post(token_url, params=params)
        return response.json().get('access_token')
    except Exception as e:
        logger.error(f"获取百度access_token失败: {str(e)}")
        return None

@lru_cache(maxsize=100)
def baidu_image_search(query):
    '''
    baidu的图片搜索，垃圾，没有提供通用的搜索服务，只提供类似相似图片的搜索服务，不可用，所以联合搜索中没有加入这个搜索引擎
    '''
    logger.info(f"开始百度图片搜索，关键词: {query}")
    
    while not _baidu_token_bucket.consume():
        logger.info("达到百度API请求限制，等待...")
        time.sleep(1)
    
    access_token = get_baidu_access_token()
    if not access_token:
        return []
    
    search_url = "https://apistore.baidu.com/supertopic/api/search"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'application/json',
        'Content-Type': 'application/x-www-form-urlencoded'
    }
    
    params = {
        'keyword': query,
        'apikey': BAIDU_API_KEY,
        'pn': '0',
        'rn': '2'
    }
    
    try:
        response = requests.post(search_url, data=params, headers=headers)
        response.raise_for_status()
        result = response.json()
        
        if 'result' in result and 'image_url' in result['result']:
            return result['result']['image_url']
        return []
    except Exception as e:
        logger.error(f"百度图片搜索出错: {str(e)}")
        return []
    
from pathlib import Path
def baidu_image_search_2(query):
    '''
    百度图片搜索，使用selenium模拟浏览器操作，将会直接下载到浏览器指定的目录下，当前是：../static/uploads/baidu_images

    返回本地的图片路径列表
    '''
    logger.info(f"开始百度图片搜索，关键词: {query}")
    
    max_retries=1
    wait_timeout=0.3
    dynamic_wait=True
    start_index=0
    driver = None
    
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
        # options.add_argument('--headless')
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
    
    urls = ['https://image.baidu.com/search/index?tn=baiduimage&z=7&word=' + query]
    session_retry_count = 0
    max_session_retries = 3
    
    while session_retry_count < max_session_retries:
        try:
            if not check_browser_session(driver):
                if driver:
                    try:
                        driver.quit()
                    except Exception:
                        pass
                driver, download_dir = initialize_browser()
                wait = WebDriverWait(driver, wait_timeout)
                logger.info("Chrome WebDriver初始化完成")
                break
            else:
                logger.info("Chrome WebDriver已初始化")
                break
        except Exception as e:
            logger.error(f"初始化Chrome WebDriver出错: {str(e)}")
            session_retry_count += 1
            continue
    try:
        for url in urls:
            try:
                logger.info(f"开始处理URL: {url}")

                retry_count = 0
                # 访问URL并等待页面加载
                logger.info(f"正在访问URL: {url}")
                driver.get(url)
                
                if not check_browser_session(driver):
                    logger.error("浏览器会话已失效，尝试重新初始化...")
                    session_retry_count += 1
                    continue
                
                def wait_and_find_element(by, value, parent=None, timeout=None):
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
                
                def wait_and_find_elements(by, value, parent=None, timeout=None):
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
                
                # 等待页面加载完成并点击空白处
                body = wait_and_find_element(By.TAG_NAME, 'body')
                actions = ActionChains(driver)
                # actions.move_to_element(body).click().perform()
                
                # 实现滚动加载机制
                # last_height = driver.execute_script("return document.body.scrollHeight")
                # 设置滚动暂停时间
                scroll_pause_time = 1
                # 设置滚动次数
                scroll_attempts = 0
                # 设置最大滚动次数(暂时设置为1，滚动一次，等于说是两个屏幕内的数据，预计是有40张图片)
                max_scroll_attempts = 0
                
                while scroll_attempts < max_scroll_attempts:
                    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                    time.sleep(scroll_pause_time)
                    
                    new_height = driver.execute_script("return document.body.scrollHeight")
                    if new_height == last_height:
                        break
                    last_height = new_height
                    scroll_attempts += 1
                
                image_class_name = 'imgitem'
                # 等待图片元素加载
                logger.info(f"等待图片元素加载...")
                image_elements = wait_and_find_elements(By.CLASS_NAME, image_class_name)
                total_images = len(image_elements)
                
                logger.info(f"找到 {total_images} 个图片元素")
                
                if total_images == 0:
                    image_class_name = 'waterfall-item_2GzCn'
                    logger.warning(f"未找到任何图片元素，尝试使用{image_class_name}当前页面")
                    
                    image_elements = wait_and_find_elements(By.CLASS_NAME, image_class_name)
                    logger.info(f"通过class=’waterfall-item_2GzCn’重新找到 {total_images} 个图片元素")
                    
                    if not image_elements:
                        image_class_name = 'waterfall-item_3xi6q'
                        logger.warning(f"仍然未找到任何图片元素，再找{image_class_name}")
                        
                        
                        image_elements = wait_and_find_elements(By.CLASS_NAME, image_class_name)
                        logger.info(f"通过class=’waterfall-item_3xi6q’重新找到 {total_images} 个图片元素")
                        
                        if not image_elements:
                            logger.warning("仍然未找到任何图片元素，再找")
                            
                            continue
                
                logger.info(f"最终尝试获得的image_elements数量：{len(image_elements)}")
                
                total_images = len(image_elements)
                
                # 从断点位置开始处理图片
                for i in range(start_index, total_images):
                    try:
                        logger.info(f"正在处理第 {i + 1}/{total_images} 个图片")
                        logger.info("开始尝试定位图片元素...")
                        
                        # 重新获取当前图片元素，避免stale element
                        image = wait_and_find_elements(By.CLASS_NAME, image_class_name)[i]
                        logger.info(f"图片元素定位结果：")
                        logger.info(f"- 元素标签：{image.tag_name}")
                        logger.info(f"- 元素类名：{image.get_attribute('class')}")
                        logger.info(f"- 元素可见性：{image.is_displayed()}")
                        logger.info(f"- 元素位置：{image.location}")
                        logger.info(f"- 元素大小：{image.size}")
                        logger.info(f"- 元素HTML：{image.get_attribute('outerHTML')}")
                        
                        # 滚动到图片元素位置
                        logger.info("准备滚动到图片元素位置...")
                        # driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", image)
                        # time.sleep(1)
                        # logger.info("滚动完成，当前页面位置：" + str(driver.execute_script("return window.pageYOffset")))
                        
                        # 模拟鼠标移动到图片元素上
                        actions = ActionChains(driver)
                        actions.move_to_element(image).perform()
                        logger.info("已将鼠标移动到图片元素上")
                        
                        # 等待页面响应
                        time.sleep(1)
                        logger.info("等待(1s)页面响应完成")
                        
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
                            time.sleep(1)
                            logger.info("鼠标悬停操作完成")
                        else:
                            logger.error("未找到hover元素，说明没有Hover元素，可能已经变成其他元素了，继续后续的一些操作，可能会成功的，不成功也不管他")
                            
                            # raise Exception("hover元素定位失败")
                        
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
                            time.sleep(1)  # 增加下载等待时间
                        else:
                            logger.error("未找到class='down'的下载按钮，尝试使用class='download-text_rXaz4'的进行定位下载按钮...")
                            
                            download_button = wait_and_find_element(By.CLASS_NAME, 'download-text_rXaz4', parent=image)
                            logger.info("重新定位下载按钮定位成功：")
                            logger.info(f"- 下载按钮类名：{download_button.get_attribute('class')}")
                            logger.info(f"- 下载按钮可见性：{download_button.is_displayed()}")
                            logger.info(f"- 下载按钮HTML：{download_button.get_attribute('outerHTML')}")
                            logger.info("准备点击下载按钮...")
                            
                            driver.execute_script("arguments[0].click();", download_button)
                            logger.info("下载按钮点击完成(会sleep1s)")
                            
                            # raise Exception("下载按钮定位失败")
                        
                    except Exception as e:
                        logger.error(f"处理第 {i + 1} 个图片时出错: {str(e)}")
                        # logger.error("当前页面源码：")
                        # logger.error(driver.page_source)
                        
                        # 保存当前进度，便于断点续传
                        with open('crawler_checkpoint.txt', 'w') as f:
                            f.write(f"{url}\n{i}")
                        continue
                
                logger.info("所有图片处理完成")
            
            except Exception as e:
                logger.error(f"爬取过程中出错: {str(e)}")
                logger.error(f"当前页面源码：{driver.page_source}")
                continue
    finally:
        if driver:
            driver.quit()
            logger.info("浏览器已关闭")   
    
    return download_dir


def get_downloaded_file_path(download_dir):
    """获取最新下载的文件路径"""
    files = os.listdir(download_dir)
    paths = [os.path.join(download_dir, basename) for basename in files]
    return max(paths, key=os.path.getctime)
def baidu_image_search_3(query, query_prefix=None, query_posfix=None, limit_count=20, width_height=7, imgratio=None, imgformat=None, hd='1'):
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
    
    max_retries=1
    wait_timeout=0.3
    start_index=0
    driver = None
    
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
    
    # 对输入的query_prefix和query_posfix进行严格处理，如果非空，则必须以空格结尾和以空格开始，如果不满足，进行修改
    if query_prefix is not None and not query_prefix.endswith(' '):
        query_prefix = query_prefix.strip() + ' '
    if query_posfix is not None and not query_posfix.startswith(' '):
        query_posfix = ''' ''' + query_posfix.strip()
    
    urls = [f'https://image.baidu.com/search/index?tn=baiduimage&z={width_height}&imgratio={imgratio}&imgformat={imgformat}&hd={hd}&word={query_prefix}{query}{query_posfix}']
    session_retry_count = 0
    max_session_retries = 3
    
    while session_retry_count < max_session_retries:
        try:
            if not check_browser_session(driver):
                if driver:
                    try:
                        driver.quit()
                    except Exception:
                        pass
                driver, download_dir = initialize_browser()
                wait = WebDriverWait(driver, wait_timeout)
                logger.info("Chrome WebDriver初始化完成")
                break
            else:
                logger.info("Chrome WebDriver已初始化")
                break
        except Exception as e:
            logger.error(f"初始化Chrome WebDriver出错: {str(e)}")
            session_retry_count += 1
            continue
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
                
                def wait_and_find_element(by, value, parent=None, timeout=None):
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
                
                def wait_and_find_elements(by, value, parent=None, timeout=None):
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
                
                image_elements = wait_and_find_elements(By.CLASS_NAME, image_class_name)
                total_images = len(image_elements)
                logger.info(f"通过class=’waterfall-item_2GzCn’找到 {total_images} 个图片元素")
                
                # 模式一的尝试
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
        
        while True:
            downloaded_file_path = get_downloaded_file_path(download_dir)
            if downloaded_file_path and not downloaded_file_path.endswith('.crdownload'):
                break
            time.sleep(1)
        
        if driver:
            driver.quit()
            logger.info("浏览器已关闭")   
    
    return download_dir
def baidu_images_search_and_restore():
    """
    通过百度的图片搜索下载图片，并且手动移动到指定的目录:../static/spider
    
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

            # 处理每个关键词
            for keyword in keywords:
                logger.info(f'搜索关键词：{keyword}')
                try:
                    # 调用百度图片搜索接口
                    download_dir = baidu_image_search_2(keyword)

                    # 获取当前日期
                    current_date = datetime.datetime.now().strftime('%Y%m%d')

                    # 移动并重命名图片
                    for idx, src_path in list(enumerate(Path(download_dir).glob('*'), start=1)):
                        if not os.path.exists(src_path):
                            logger.warning(f'源文件不存在：{src_path}')
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

    logger.info('图片搜索和存储处理完成')    


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
                    download_dir = baidu_image_search_3(query=keyword, query_prefix="漫画", query_posfix="海报", limit_count=3, width_height=7, imgratio=None, imgformat=3, hd=1)

                    # 获取当前日期
                    current_date = datetime.datetime.now().strftime('%Y%m%d')

                    # 移动并重命名图片
                    for idx, src_path in list(enumerate(Path(download_dir).glob('*'), start=1)):
                        if not os.path.exists(src_path):
                            logger.warning(f'源文件不存在：{src_path}')
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

    logger.info('图片搜索和存储处理完成')    
    
# Bing图片搜索
@lru_cache(maxsize=100)
def bing_image_search(query):
    '''
    微软的BingSearch，同样没有搞定，有一个Authorization问题，暂时放弃。账号开通中遇到云账号认证的问题，要等级Visa卡，真的是费劲，没搞定。
    '''
    logger.info(f"开始Bing图片搜索，关键词: {query}")
    
    while not _bing_token_bucket.consume():
        logger.info("达到Bing API请求限制，等待...")
        time.sleep(1)
    
    url = "https://api.bing.microsoft.com/v7.0/images/search"
    headers = {
        'Ocp-Apim-Subscription-Key': BING_API_KEY
    }
    
    params = {
        'q': query,
        'count': 2,
        'safeSearch': 'Strict'
    }
    
    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        results = response.json()
        
        if 'value' in results:
            return [item['contentUrl'] for item in results['value']]
        return []
    except Exception as e:
        logger.error(f"Bing图片搜索出错: {str(e)}")
        return []

# Unsplash图片搜索
@lru_cache(maxsize=100)
def unsplash_image_search(query):
    logger.info(f"开始Unsplash图片搜索，关键词: {query}")
    
    while not _unsplash_token_bucket.consume():
        logger.info("达到Unsplash API请求限制，等待...")
        time.sleep(1)
    
    url = "https://api.unsplash.com/search/photos"
    headers = {
        'Authorization': f'Client-ID {UNSPLASH_ACCESS_KEY}'
    }
    
    params = {
        'query': query,
        'per_page': 2
    }
    
    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        results = response.json()
        
        if 'results' in results:
            return [item['urls']['regular'] for item in results['results']]
        return []
    except Exception as e:
        logger.error(f"Unsplash图片搜索出错: {str(e)}")
        return []

# Pixabay图片搜索
@lru_cache(maxsize=100)
def pixabay_image_search(query):
    logger.info(f"开始Pixabay图片搜索，关键词: {query}")
    
    while not _pixabay_token_bucket.consume():
        logger.info("达到Pixabay API请求限制，等待...")
        time.sleep(1)
    
    # 从API_KEYS中随机选择一个API_KEY
    PIXABAY_API_KEY = random.choice(PIXABAY_API_KEYS)
    
    url = "https://pixabay.com/api/"
    params = {
        'key': PIXABAY_API_KEY,
        'q': query,
        'per_page': 4
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        results = response.json()
        
        if 'hits' in results:
            return [item['largeImageURL'] for item in results['hits']]
        return []
    except Exception as e:
        logger.error(f"Pixabay图片搜索出错: {str(e)}")
        return []

# Flickr图片搜索
@lru_cache(maxsize=100)
def flickr_image_search(query):
    logger.info(f"开始Flickr图片搜索，关键词: {query}")
    
    while not _flickr_token_bucket.consume():
        logger.info("达到Flickr API请求限制，等待...")
        time.sleep(1)
    
    url = "https://www.flickr.com/services/rest/"
    params = {
        'method': 'flickr.photos.search',
        'api_key': FLICKR_API_KEY,
        'text': query,
        'format': 'json',
        'nojsoncallback': 1,
        'per_page': 2,
        'media': 'photos',
        'sort': 'relevance'
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        results = response.json()
        
        if 'photos' in results and 'photo' in results['photos']:
            photos = results['photos']['photo']
            return [f"https://farm{p['farm']}.staticflickr.com/{p['server']}/{p['id']}_{p['secret']}_b.jpg" 
                    for p in photos]
        return []
    except Exception as e:
        logger.error(f"Flickr图片搜索出错: {str(e)}")
        return []

# Pexels图片搜索
@lru_cache(maxsize=100)
def pexels_image_search(query):
    logger.info(f"开始Pexels图片搜索，关键词: {query}")
    
    while not _pexels_token_bucket.consume():
        logger.info("达到Pexels API请求限制，等待...")
        time.sleep(1)
    
    url = "https://api.pexels.com/v1/search"
    headers = {
        'Authorization': PEXELS_API_KEY
    }
    
    params = {
        'query': query,
        'per_page': 2
    }
    
    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        results = response.json()
        
        if 'photos' in results:
            return [photo['src']['large'] for photo in results['photos']]
        return []
    except Exception as e:
        logger.error(f"Pexels图片搜索出错: {str(e)}")
        return []

def search_pixabay_2(keyword, page=1, per_page=20):
    """
    Pixabay图片搜索
    :param keyword: 搜索关键词
    :return: 图片URL列表
    """
    
    try:
        logger.info(f"开始Pixabay图片搜索，关键词: {keyword}")

        while not _pixabay_token_bucket.consume():
            logger.info("达到Pixabay API请求限制，等待...")
            time.sleep(1)
            
        # 从API_KEYS中随机选择一个API_KEY
        PIXABAY_API_KEY = random.choice(PIXABAY_API_KEYS)
        
        url = "https://pixabay.com/api/?key={}&q={}&image_type=photo&page={}&per_page={}".format(PIXABAY_API_KEY, keyword, page, per_page)
        response = requests.get(url)
        data = response.json()
        if data['totalHits'] > 0:
            return [hit['largeImageURL'] for hit in data['hits']]
        return []
    except Exception as e:
        logger.error(f"Pixabay图片搜索出错: {str(e)}")
        return []

# 统一的图片搜索接口
def unified_image_search(keyword):
    """
    统一的图片搜索接口，随机选择搜索引擎进行搜索（没有百度，它是垃圾，不支持query搜索）
    :param keyword: 搜索关键词
    :return: 图片URL列表
    """
    logger.info(f"开始统一图片搜索，关键词: {keyword}")
    
    # 定义可用的搜索引擎及其对应的搜索函数
    search_engines = [
        ('Google', google_image_search),
        ('Unsplash', unsplash_image_search),
        ('Pixabay', pixabay_image_search),
        ('Flickr', flickr_image_search),
        ('Pexels', pexels_image_search)
    ]
    
    # 随机打乱搜索引擎顺序
    random.shuffle(search_engines)
    
    # 依次尝试每个搜索引擎
    for engine_name, search_func in search_engines:
        logger.info(f"尝试使用 {engine_name} 搜索引擎")
        
        try:
            results = search_func(keyword)
            
            # 如果获取到结果，直接返回
            if results:
                logger.info(f"{engine_name} 搜索成功，找到 {len(results)} 个结果")
                return results
            else:
                logger.info(f"{engine_name} 搜索未找到结果，尝试下一个搜索引擎")
                
        except Exception as e:
            logger.error(f"{engine_name} 搜索出错: {str(e)}")
            continue
    
    logger.warning("所有搜索引擎均未找到结果")
    return []

import os
import requests

def store_image(url, filepath):
    """存储图片到本地
    Args:
        url: 图片URL
        filepath: 保存路径
    """
    try:
        # 确保目录存在
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # 先检查文件是否存在
        if os.path.exists(filepath):
            logger.info(f"文件已存在，跳过下载: {filepath}")
            return
        
        # 下载图片
        response = requests.get(url)
        sleep(1)
        
        # 检查响应状态
        response.raise_for_status()
        
        # 保存图片
        with open(filepath, 'wb') as f:
            f.write(response.content)
            
        logger.info(f"图片保存成功: {filepath}")
    except Exception as e:
        logger.error(f"保存图片失败 {url} -> {filepath}: {str(e)}")
        raise

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

from time import sleep
def search_pixabay_and_store():
    """
    搜索Pixabay并存储到本地
    """
    logger.info("开始Pixabay图片搜索并存储到本地")

    # 定义搜索关键词
    keywords = ['healthcare']
    keywords = ['healthcare', 'medical', 'medicine', 'doctor', 'nurse', 'hospital', 'clinic', 'pharmacy', 'lab', 'dentist', 'surgeon', 'physician', 'patient', 'health', 'wellness', 'meditation', 'yoga', 'fitness', 'exercise', 'sport', 'gym', 'fitness center', 'gym equipment', 'diet','nutrition', 'dietician', 'diet plan']
    
    # 从words_mapping_for_image_crawler.json中加载需要抓取的目录与关键词的映射内容
    keywords_mapping = load_keywords()
    if keywords_mapping:
        # 获取key与value列表
        for key, keywords in keywords_mapping.items():
            logger.info(f"从words_mapping_for_image_crawler.json中加载关键词: {keywords}")
            
            # 获取当前脚本所在目录
            current_dir = os.path.dirname(os.path.abspath(__file__))
            # 构建基础存储路径
            base_dir = os.path.join(current_dir, '..', 'static', 'beauty', key)
            
            # 遍历关键词
            for keyword in keywords:
                logger.info(f"开始搜索关键词: {keyword}")
                # 定义每页的图片数量
                per_page = 200
                # 400张图的话其实虽然不多，但是开始阶段可以考虑不用过于纠结，因为还有多个词在这里，所以每个词都可以爬取到400张图，所以可以考虑先爬取400张图，然后再考虑爬取更多的图。
                for page in range(1, 2):
                    # 搜索Pixabay
                    results = search_pixabay_2(keyword, page=page, per_page=per_page)
                    sleep(1)
                    
                    # 存储图片到本地
                    for i, url in enumerate(results):
                        filename = f"{keyword}_{page}_{i}.jpg"
                        filepath = os.path.join(base_dir, filename)
                        try:
                            store_image(url, filepath)
                        except Exception as e:
                            logger.error(f"存储图片 {filename} 失败(Ignore first): {e}")
                        logger.info(f"图片 {filename} 存储成功")
                    logger.info(f"关键词 {keyword} 搜索完成")
                logger.info("Pixabay图片搜索并存储到本地完成")
    else:
        logger.warning("未从words_mapping_for_image_crawler.json中加载关键词，使用默认关键词")

def crawl_images_for_cover():
    test_keywords = ['douyin',
'online-pop',
'classic-surround',
'cantonese',
'chinese-dj',
'famous-songs',
'cover-songs',
'western-rhythm',
'sad-songs',
'english-pop',
'grassland',
'minnan',
'car-music',
'studio-dj',
'mandarin-classic',
'8090-memory',
'light-music',
'bass-music',
'douyin',
'online-pop',
'classic-surround',
'cantonese',
'chinese-dj',
'famous-songs',
'cover-songs',
'western-rhythm',
'sad-songs',
'english-pop',
'grassland',
'minnan',
'car-music',
'studio-dj',
'mandarin-classic',
'8090-memory',
'light-music',
'bass-music',
'action-movie',
'action-movie',
'sci-fi',
'sci-fi',
'comedy',
'comedy',
'horror',
'horror',
'romance',
'romance',
'animation',
'animation',
'documentary',
'documentary',
'war',
'war',
'action-movie',
'sci-fi',
'comedy',
'horror',
'romance',
'animation',
'documentary',
'war',
'programming',
'programming',
'language',
'language',
'exam',
'exam',
'career',
'career',
'hobby',
'hobby',
'art-edu',
'art-edu',
'kids-edu',
'kids-edu',
'online-course',
'online-course',
'programming',
'language',
'exam',
'career',
'hobby',
'art-edu',
'kids-edu',
'online-course',
'action-comic',
'action-comic',
'romance-comic',
'romance-comic',
'sci-fi-comic',
'sci-fi-comic',
'funny-comic',
'funny-comic',
'mystery-comic',
'mystery-comic',
'school-comic',
'school-comic',
'adventure-comic',
'adventure-comic',
'fantasy-comic',
'fantasy-comic',
'action-comic',
'romance-comic',
'sci-fi-comic',
'funny-comic',
'mystery-comic',
'school-comic',
'adventure-comic',
'fantasy-comic',
'listening',
'listening',
'speaking',
'speaking',
'reading',
'reading',
'writing',
'writing',
'grammar',
'grammar',
'vocabulary',
'vocabulary',
'business-english',
'business-english',
'exam-english',
'exam-english',
'listening',
'speaking',
'reading',
'writing',
'grammar',
'vocabulary',
'business-english',
'exam-english',
'football-news',
'football-news',
'football-skills',
'football-skills',
'football-tactics',
'football-tactics',
'football-equipment',
'football-league',
'football-training',
'football-stars',
'football-clubs',
'football-news',
'football-skills',
'football-tactics',
'football-equipment',
'football-league',
'football-training',
'football-stars',
'football-clubs',
'nba',
'cba',
'basketball-skills',
'basketball-equipment',
'basketball-tactics',
'basketball-training',
'basketball-stars',
'street-basketball',
'nba',
'cba',
'basketball-skills',
'basketball-equipment',
'basketball-tactics',
'basketball-training',
'basketball-stars',
'street-basketball',
]
    
    test_keywords = [
    'photography',  # 写真
    'business',     # 商务
    'family',       # 家庭
    'team',         # 团队
    'student',      # 学生
    'medical',      # 医疗
    'city',         # 城市
    'photography',  # 写真（重复）
    'farmland',     # 农田
    'architecture', # 建筑
    'animal',       # 动物
    'machinery',    # 机械
    'technology',   # 科技
    'home_decor'    # 家居
]
    
    test_keywords = [
    'family',       # 家庭
    'architecture'    # 家居
]
    
    test_keywords = [
    'family',       # 家庭
]
    for test_keyword in test_keywords:
        # 根据关键字搜索图片，并将其保存为以关键字命名的图片文件
        results = unified_image_search(test_keyword)
        print(f"搜索结果: {results}")
        # 取第一张就行
        # results = results[:1]
        for i, url in enumerate(results):
            # filename = f"{test_keyword}_{i}.jpg"
            filename = f"{test_keyword}.jpg"
            try:
                response = requests.get(url)
                with open(filename, 'wb') as f:
                    f.write(response.content)
                print(f"已保存图片: {filename}")
                # 只要有一张成功就可以了
                break
            except Exception as e:
                print(f"保存图片时出错: {str(e)}")
# 使用示例
if __name__ == '__main__':
    test_keyword = 'cat'
    # results = unified_image_search(test_keyword)
    # print(f"搜索结果: {results}")
    
    # test_keyword = '张爱玲 《倾城之恋》小说'
    # # results = baidu_image_search(test_keyword)
    # # print(f"搜索结果: {results}")
    
    # results = bing_image_search(query=test_keyword)
    # print(f"搜索结果: {results}")
    
    # test_keyword = '王朔 《一点正经都没有》中的生活哲学'
    
    # results = google_image_search(query=test_keyword)
    # print(f"搜索结果: {results}")
    
    
    # 根据关键词获取图片的方法调用，用于生成封面
    # crawl_images_for_cover()
    
    # result = search_pixabay('healthcare')
    # print(result)
    
    # search_pixabay_and_store()
    
    # test_keyword = 'imagesize:140x500 路飞' # google的api接口不支持这样的搜索
    # results = google_image_search(query=test_keyword)
    # print(f"搜索结果: {results}")
    
    # baidu_images_search_and_restore()
    
    baidu_images_search_and_restore_for_covers()