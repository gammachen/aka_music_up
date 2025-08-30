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
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('baidu_crawler.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('BaiduPicture')

def crawl_baidu_images(urls=None, max_retries=3, wait_timeout=10, dynamic_wait=True):
    """抓取百度图片分享内容
    Args:
        urls (list, optional): 百度图片分享链接列表. 默认为None
        max_retries (int, optional): 最大重试次数. 默认为3
        wait_timeout (int, optional): 等待超时时间. 默认为10秒
        dynamic_wait (bool, optional): 是否动态等待. 默认为True
    """
    if not urls:
        urls = ['https://image.baidu.com/search/index?tn=baiduimage&ipn=r&ct=201326592&cl=2&lm=&st=-1&fm=index&fr=&hs=0&xthttps=111110&sf=1&fmq=&pv=&nc=1&z=7&se=&showtab=0&fb=0&face=0&istype=2&ie=utf-8&word=%E8%B7%AF%E9%A3%9E']
    elif isinstance(urls, str):
        urls = [urls]
    
    for url in urls:
        logger.info(f"开始处理URL: {url}")
        driver = None
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                # 初始化Chrome WebDriver
                options = webdriver.ChromeOptions()
                options.add_argument('--disable-gpu')
                options.add_argument('--no-sandbox')
                options.add_argument('--disable-dev-shm-usage')
                options.add_argument('--start-maximized')
                
                # 添加user-data-dir参数
                # user_data_dir = os.path.expanduser('~/CustomChromeProfile_for_baidu_picture_crawler')
                user_data_dir = os.path.expanduser('~/CustomChromeProfile')
                options.add_argument(f'--user-data-dir={user_data_dir}')
                
                # 添加下载设置
                download_dir = os.path.join(os.getcwd(), "static", "baidu_images")
                os.makedirs(download_dir, exist_ok=True)
                
                prefs = {
                    "download.default_directory": os.path.join(os.getcwd(), "downloads"),
                    "download.prompt_for_download": False,
                    "download.directory_upgrade": True,
                    "safebrowsing.enabled": False,
                    "profile.default_content_settings.popups": 0,
                    "profile.content_settings.exceptions.automatic_downloads.*.setting": 1
                }
                options.add_experimental_option("prefs", prefs)
                
                # 添加实验性选项以允许多文件下载
                options.add_experimental_option("excludeSwitches", ["enable-automation"])
                options.add_experimental_option("useAutomationExtension", False)
                
                logger.info("正在初始化Chrome WebDriver...")
                driver = webdriver.Chrome(options=options)
                logger.info("Chrome WebDriver初始化成功")
                
                # 访问URL
                logger.info(f"正在访问URL: {url}")
                driver.get(url)
                logger.info("页面加载完成")
                
                # 等待图片容器加载
                wait = WebDriverWait(driver, wait_timeout)
                logger.info("等待图片容器加载...")
                
                # 尝试多个可能的选择器
                selectors = [
                    (By.CLASS_NAME, 'imgitem'),
                    (By.CLASS_NAME, 'hover'),  # 添加hover类选择器
                    (By.CLASS_NAME, 'waterfall-horizontal-line_2E5Fy'),
                    (By.CSS_SELECTOR, '.graph-waterfall-item'),
                    (By.CSS_SELECTOR, '.waterfall-item'),
                    (By.CSS_SELECTOR, '[class*="waterfall"]')
                ]
                
                image_elements = None
                for selector in selectors:
                    try:
                        logger.info(f"尝试使用选择器: {selector}")
                        # // 实现滚动加载机制
                        last_height = driver.execute_script("return document.body.scrollHeight")
                        scroll_pause_time = 2  # 滚动暂停时间
                        scroll_attempts = 0
                        max_scroll_attempts = 3  # 最大滚动次数
                        
                        while scroll_attempts < max_scroll_attempts:
                            # 滚动到页面底部
                            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                            time.sleep(scroll_pause_time)
                            
                            # 计算新的滚动高度并比较
                            new_height = driver.execute_script("return document.body.scrollHeight")
                            if new_height == last_height:
                                # 如果高度没有变化，说明可能已经到底部
                                break
                            last_height = new_height
                            scroll_attempts += 1
                            logger.info(f"第 {scroll_attempts} 次滚动，页面高度: {new_height}")
                            
                            # 每次滚动后重新获取图片元素
                            try:
                                new_elements = wait.until(
                                    EC.presence_of_all_elements_located(selector)
                                )
                                if len(new_elements) > len(image_elements or []):
                                    image_elements = new_elements
                                    logger.info(f"已加载 {len(image_elements)} 个图片元素")
                            except Exception as e:
                                logger.warning(f"获取新元素时出错: {str(e)}")
                    except Exception as e:
                        logger.warning(f"使用选择器 {selector} 时出错: {str(e)}")
                    if image_elements:
                        break
                    if not image_elements:
                        raise Exception("即使在滚动加载后也未能找到任何图片元素")
                    # 输出页面结构以帮助调试
                    logger.error("页面结构:")
                    logger.error(driver.page_source)
                    raise Exception("未能找到任何图片元素，所有选择器都失败了")
                logger.info(f"找到 {len(image_elements)} 个图片元素")
                
                # 处理每个图片元素
                total_images = len(image_elements)
                for i in range(1, total_images + 1):
                    try:
                        logger.info(f"正在处理第 {i}/{total_images} 个图片")
                        
                        # 重新获取当前图片元素，并输出详细信息
                        selector = ".down"
                        logger.info(f"尝试获取元素，选择器: {selector}")
                        
                        current_image = wait.until(
                            EC.presence_of_element_located((By.CLASS_NAME, "hover"))
                        )
                        
                        # 输出当前图片元素的详细信息
                        logger.info("当前图片元素信息:")
                        logger.info(f"- 标签名: {current_image.tag_name}")
                        logger.info(f"- 类名: {current_image.get_attribute('class')}")
                        logger.info(f"- HTML: {current_image.get_attribute('outerHTML')}")
                        logger.info(f"- 是否显示: {current_image.is_displayed()}")
                        logger.info(f"- 位置: {current_image.location}")
                        logger.info(f"- 大小: {current_image.size}")
                        
                        # 滚动到图片元素位置
                        driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", current_image)
                        time.sleep(2)  # 增加等待时间，确保页面加载完成
                        
                        # 创建 ActionChains 实例来模拟鼠标操作
                        actions = ActionChains(driver)
                        
                        # 移动到图片元素并悬停
                        actions.move_to_element(current_image).perform()
                        logger.info(f"鼠标已悬停在第 {i} 个图片上")
                        
                        if dynamic_wait:
                            time.sleep(2)  # 等待下载按钮出现
                        
                        # 查找下载按钮
                        download_button = wait.until(
                            EC.presence_of_element_located((By.CLASS_NAME, 'down'))
                        )
                        
                        # 点击下载按钮
                        driver.execute_script("arguments[0].click();", download_button)
                        logger.info(f"已点击第 {i} 个图片的下载按钮")
                        
                        # 等待下载完成
                        time.sleep(2)
                        
                    except Exception as e:
                        logger.error(f"处理第 {i} 个图片时出错: {str(e)}")
                        continue
                
                logger.info("所有图片处理完成")
                break  # 成功完成，跳出重试循环
                
            except Exception as e:
                retry_count += 1
                logger.error(f"爬取过程中出错 (尝试 {retry_count}/{max_retries}): {str(e)}")
                if retry_count >= max_retries:
                    logger.error(f"达到最大重试次数，爬取失败: {str(e)}")
                    break
                time.sleep(5)  # 重试前等待
                
            finally:
                if driver:
                    try:
                        logger.info("正在关闭浏览器...")
                        driver.quit()
                        logger.info("浏览器已关闭")
                    except Exception as e:
                        logger.error(f"关闭浏览器时出错: {str(e)}")

def crawl_baidu_images_v3(query=None, wait_timeout=1, max_retries=1, start_index=0):
    """抓取百度图片分享内容 - 增强版
    Args:
        query (str, optional): 搜索关键词. 默认为None
        wait_timeout (int, optional): 等待超时时间. 默认为10秒
        max_retries (int, optional): 单个元素最大重试次数. 默认为3
        start_index (int, optional): 断点续传起始索引. 默认为0
    """
    if not query:
        query = '美女'

    logger.info(f"开始处理关键词: {query}")
    
    urls = ['https://image.baidu.com/search/index?tn=baiduimage&word=%E8%B7%AF%E9%A3%9E']
    
    
    
def crawl_baidu_images_v2(urls=None, wait_timeout=1, max_retries=1, start_index=0):
    """抓取百度图片分享内容 - 增强版
    Args:
        urls (list, optional): 百度图片分享链接列表. 默认为None
        wait_timeout (int, optional): 等待超时时间. 默认为10秒
        max_retries (int, optional): 单个元素最大重试次数. 默认为3
        start_index (int, optional): 断点续传起始索引. 默认为0
    """
    if not urls:
        urls = ['https://image.baidu.com/search/index?tn=baiduimage&word=%E8%B7%AF%E9%A3%9E']
    elif isinstance(urls, str):
        urls = [urls]
    
    for url in urls:
        logger.info(f"开始处理URL: {url}")
        driver = None
        retry_count = 0
        
        try:
            # 初始化Chrome WebDriver，使用普通模式
            options = webdriver.ChromeOptions()
            options.add_argument('--start-maximized')
            options.add_argument('--disable-gpu')
            options.add_argument('--no-sandbox')
            options.add_argument('--disable-dev-shm-usage')
            
            # 设置下载目录
            download_dir = os.path.join(os.getcwd(), "static", "baidu_images")
            os.makedirs(download_dir, exist_ok=True)
            
            prefs = {
                "download.default_directory": download_dir,
                "download.prompt_for_download": False,
                "download.directory_upgrade": True,
                "safebrowsing.enabled": False,
                "profile.default_content_settings.popups": 0,
                "profile.content_settings.exceptions.automatic_downloads.*.setting": 1
                }
            options.add_experimental_option("prefs", prefs)
            options.add_experimental_option("excludeSwitches", ["enable-automation"])
            
            logger.info("正在初始化Chrome WebDriver...")
            driver = webdriver.Chrome(options=options)
            wait = WebDriverWait(driver, wait_timeout)
            
            # 访问URL并等待页面加载
            logger.info(f"正在访问URL: {url}")
            driver.get(url)
            
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
                        if retry >= max_retries:
                            raise e
                        time.sleep(1)
                return None
            
            def wait_and_find_elements(by, value, parent=None, timeout=None):
                """封装的等待和查找多个元素的函数，带重试机制"""
                retry = 0
                while retry < max_retries:
                    try:
                        if parent is None:
                            parent = driver
                        actual_timeout = timeout if timeout else wait_timeout
                        elements = WebDriverWait(parent, actual_timeout).until(
                            EC.presence_of_all_elements_located((by, value))
                        )
                        return elements
                    except Exception as e:
                        retry += 1
                        if retry >= max_retries:
                            raise e
                        time.sleep(1)
                return []
            
            # 等待页面加载完成并点击空白处
            body = wait_and_find_element(By.TAG_NAME, 'body')
            actions = ActionChains(driver)
            actions.move_to_element(body).click().perform()
            
            # 实现滚动加载机制
            last_height = driver.execute_script("return document.body.scrollHeight")
            scroll_pause_time = 2
            scroll_attempts = 0
            max_scroll_attempts = 3
            
            while scroll_attempts < max_scroll_attempts:
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(scroll_pause_time)
                
                new_height = driver.execute_script("return document.body.scrollHeight")
                if new_height == last_height:
                    break
                last_height = new_height
                scroll_attempts += 1
            
            # 等待图片元素加载
            logger.info("等待图片元素加载...")
            image_elements = wait_and_find_elements(By.CLASS_NAME, 'imgitem')
            total_images = len(image_elements)
            logger.info(f"找到 {total_images} 个图片元素")
            
            # 从断点位置开始处理图片
            for i in range(start_index, total_images):
                try:
                    logger.info(f"正在处理第 {i + 1}/{total_images} 个图片")
                    logger.info("开始尝试定位图片元素...")
                    
                    # 重新获取当前图片元素，避免stale element
                    image = wait_and_find_elements(By.CLASS_NAME, 'imgitem')[i]
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
                    time.sleep(2)
                    logger.info("等待页面响应完成")
                    
                    # 查找hover元素并模拟鼠标悬停
                    logger.info("开始查找hover元素...")
                    hover_element = wait_and_find_element(By.CLASS_NAME, 'hover', parent=image)
                    if hover_element:
                        logger.info("hover元素定位成功：")
                        logger.info(f"- hover元素类名：{hover_element.get_attribute('class')}")
                        logger.info(f"- hover元素可见性：{hover_element.is_displayed()}")
                        logger.info(f"- hover元素HTML：{hover_element.get_attribute('outerHTML')}")
                    else:
                        logger.error("未找到hover元素")
                        raise Exception("hover元素定位失败")
                    
                    actions = ActionChains(driver)
                    logger.info("准备执行鼠标悬停操作...")
                    actions.move_to_element(hover_element).perform()
                    time.sleep(1)
                    logger.info("鼠标悬停操作完成")
                    
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
                        logger.info("下载按钮点击完成")
                        time.sleep(1)  # 增加下载等待时间
                    else:
                        logger.error("未找到下载按钮")
                        raise Exception("下载按钮定位失败")
                    
                except Exception as e:
                    logger.error(f"处理第 {i + 1} 个图片时出错: {str(e)}")
                    logger.error("当前页面源码：")
                    logger.error(driver.page_source)
                    # 保存当前进度，便于断点续传
                    with open('crawler_checkpoint.txt', 'w') as f:
                        f.write(f"{url}\n{i}")
                    continue
            
            logger.info("所有图片处理完成")
            
        except Exception as e:
            logger.error(f"爬取过程中出错: {str(e)}")
            logger.error(f"当前页面源码：{driver.page_source}")
            
        finally:
            if driver:
                driver.quit()
                logger.info("浏览器已关闭")

def crawl_baidu_images_and_store():
    """抓取百度图片分享内容并存储到本地"""
    '''
    crawler_categories目录下的所有文件，文件名将作为后续保存的目录的名称，文件内容是对应的搜索关键字列表
    构造百度图片搜索URL，并使用Selenium爬取并保存图片，Selenium会自动处理翻页，并保存图片到本地的指定目录
    对每个关键字搜索完成之后，开始将已经存储的文件转移到工程的app/static/spider/${keyword}目录下，图片名称进行重命名：关键词_年月日_随机数.jpg  
    '''
    

if __name__ == '__main__':
    # 示例：批量处理多个分享链接
    urls = [
        # 'https://pan.quark.cn/s/22130dc18514#/list/share/12908f55af354afa9b8c5d262ddc17f7-%E7%9F%AD%E8%A7%86%E9%A2%91%E5%89%AA%E8%BE%91%E8%AF%AD%E9%9F%B3%E5%8C%85%E7%B4%A0%E6%9D%90%E5%90%88%E9%9B%86/85c0ac483d4943acb28b84bdfd5a8053-%E5%A4%A7%E5%8F%B8%E9%A9%AC'],
        # 添加更多分享链接
        'https://pan.quark.cn/s/22130dc18514#/list/share/12908f55af354afa9b8c5d262ddc17f7-%E7%9F%AD%E8%A7%86%E9%A2%91%E5%89%AA%E8%BE%91%E8%AF%AD%E9%9F%B3%E5%8C%85%E7%B4%A0%E6%9D%90%E5%90%88%E9%9B%86/6b7c99aa2bd348c1bf6ab7835cad8c66-%E6%9D%8E%E4%BA%91%E9%BE%99',
     
    ]
    # crawl_quark_share(urls)
    
    # urls = [
    #     # 'https://pan.quark.cn/s/22130dc18514#/list/share/12908f55af354afa9b8c5d262ddc17f7-%E7%9F%AD%E8%A7%86%E9%A2%91%E5%89%AA%E8%BE%91%E8%AF%AD%E9%9F%B3%E5%8C%85%E7%B4%A0%E6%9D%90%E5%90%88%E9%9B%86/85c0ac483d4943acb28b84bdfd5a8053-%E5%A4%A7%E5%8F%B8%E9%A9%AC'],
    #     # 添加更多分享链接
    #     'https://image.baidu.com/search/index?tn=baiduimage&ipn=r&ct=201326592&cl=2&lm=&st=-1&fm=index&fr=&hs=0&xthttps=111110&sf=1&fmq=&pv=&nc=1&z=7&se=&showtab=0&fb=0&face=0&istype=2&ie=utf-8&word=%E8%B7%AF%E9%A3%9E', 
    # ]
    
    # crawl_baidu_images_v2(urls)
    
    
    