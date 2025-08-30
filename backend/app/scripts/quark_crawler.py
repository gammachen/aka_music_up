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
        logging.FileHandler('quark_crawler.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('QuarkCrawler')

def crawl_quark_share(urls=None, max_retries=3, wait_timeout=10, dynamic_wait=True):
    """抓取夸克网盘分享内容
    Args:
        urls (list, optional): 夸克网盘分享链接列表. 默认为None
        max_retries (int, optional): 最大重试次数. 默认为3
        wait_timeout (int, optional): 元素等待超时时间(秒). 默认为10
        dynamic_wait (bool, optional): 是否启用动态等待. 默认为True
    """
    if not urls:
        urls = ['https://pan.quark.cn/s/22130dc18514#/list/share/12908f55af354afa9b8c5d262ddc17f7-%E7%9F%AD%E8%A7%86%E9%A2%91%E5%89%AA%E8%BE%91%E8%AF%AD%E9%9F%B3%E5%8C%85%E7%B4%A0%E6%9D%90%E5%90%88%E9%9B%86/85c0ac483d4943acb28b84bdfd5a8053-%E5%A4%A7%E5%8F%B8%E9%A9%AC']
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
                # options.add_argument('--headless')  # 无头模式，视情况启用
                options.add_argument('--disable-gpu')
                options.add_argument('--no-sandbox')
                options.add_argument('--disable-dev-shm-usage')
                options.add_argument('--start-maximized')
                
                # 添加user-data-dir参数，使用默认的Chrome用户配置目录
                # user_data_dir = os.path.expanduser('~/Library/Application Support/Google/Chrome')
                user_data_dir = os.path.expanduser('~/CustomChromeProfile')
                options.add_argument(f'--user-data-dir={user_data_dir}')
                
                # 添加下载设置
                prefs = {
                    "download.default_directory": os.path.join(os.getcwd(), "downloads"),
                    "download.prompt_for_download": False,
                    "download.directory_upgrade": True,
                    "safebrowsing.enabled": True,
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
                
                # 等待表格加载，使用显式等待
                wait = WebDriverWait(driver, wait_timeout)
                logger.info("等待表格元素加载...")
                
                # 使用更精确的元素定位策略
                tbody = wait.until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, '.ant-table-tbody'))
                )
                logger.info("成功定位表格主体")
                
                # 获取所有行
                rows = tbody.find_elements(By.CSS_SELECTOR, 'tr[data-row-key]')
                logger.info(f"找到 {len(rows)} 个文件行")
                
                # 处理每一行
                for i, row in enumerate(rows, 1):
                    max_row_retries = 3
                    row_retry_count = 0
                    
                    while row_retry_count < max_row_retries:
                        try:
                            logger.info(f"正在处理第 {i}/{len(rows)} 个文件 (尝试 {row_retry_count + 1}/{max_row_retries})")
                            
                            # 确保元素可见和可点击
                            file_cell = wait.until(
                                EC.element_to_be_clickable((By.CSS_SELECTOR, f"tr[data-row-key='{row.get_attribute('data-row-key')}'] td:nth-child(4)"))
                            )
                            
                            # 打印更详细的file_cell信息
                            logger.info(f"file_cell详细信息:")
                            logger.info(f"- 标签名: {file_cell.tag_name}")
                            logger.info(f"- 文本内容: {file_cell.text}")
                            logger.info(f"- class属性: {file_cell.get_attribute('class')}")
                            logger.info(f"- data-row-key: {file_cell.get_attribute('data-row-key')}")
                            
                            # 首先找到并点击复选框
                            checkbox = row.find_element(By.CSS_SELECTOR, '.ant-checkbox-input')
                            driver.execute_script("arguments[0].click();", checkbox)
                            
                            # 创建 ActionChains 实例来模拟鼠标操作
                            actions = ActionChains(driver)
                            
                            # 移动到文件单元格并点击
                            actions.move_to_element(file_cell).click().perform()
                            
                            # 确保点击事件被正确触发
                            driver.execute_script("arguments[0].click();", file_cell)
                            
                            logger.info(f"已点击第 {i} 行的file-click-wrap元素以显示操作按钮")
                            
                            # 动态等待下载按钮出现
                            if dynamic_wait:
                                wait_time = min(1 + (row_retry_count * 0.5), 3)  # 动态增加等待时间
                            else:
                                wait_time = 1
                            time.sleep(wait_time)
                            
                            # 使用显式等待查找下载按钮
                            download_button = wait.until(
                                EC.presence_of_element_located((By.CSS_SELECTOR, '.hoitem-down'))
                            )
                            
                            # 等待下载开始并完成
                            download_dir = os.path.join(os.getcwd(), "downloads")
                            initial_files = set(os.listdir(download_dir)) if os.path.exists(download_dir) else set()
                            
                            # 使用更精确的选择器和更长的等待时间来定位下载列表
                            try:
                                # 尝试多个选择器策略来定位下载按钮
                                selectors = [
                                    (By.CSS_SELECTOR, '.filename-text'),
                                    (By.CSS_SELECTOR, '.ant-table-cell .filename-text'),
                                    (By.CSS_SELECTOR, '[data-row-key] .filename-text')
                                ]
                                
                                filename_cell = None
                                for selector in selectors:
                                    try:
                                        filename_cell = row.find_element(*selector)
                                        logger.info(f"成功使用选择器 {selector[1]} 定位到文件名元素")
                                        break
                                    except Exception as e:
                                        logger.debug(f"使用选择器 {selector[1]} 定位失败: {str(e)}")
                                        continue
                                
                                if not filename_cell:
                                    raise Exception("无法找到文件名元素")
                                
                                # 尝试多个路径定位下载列表
                                download_selectors = [
                                    (By.CSS_SELECTOR, '.filename > .share-download-list'),
                                    (By.CSS_SELECTOR, 'td.td-file .filename > .share-download-list'),
                                    (By.CSS_SELECTOR, '.file-click-wrap + .hover-oper + .share-download-list'),
                                    (By.CSS_SELECTOR, '[data-row-key] td.td-file .share-download-list')
                                ]
                                
                                share_download_list = None
                                for selector in download_selectors:
                                    try:
                                        share_download_list = row.find_element(*selector)
                                        logger.info(f"成功使用选择器 {selector[1]} 定位到下载列表元素")
                                        break
                                    except Exception as e:
                                        logger.debug(f"使用选择器 {selector[1]} 定位失败: {str(e)}")
                                        continue
                                
                                if not share_download_list:
                                    raise Exception("无法找到下载列表元素")
                                
                                # 确保元素可见和可交互
                                driver.execute_script("arguments[0].scrollIntoView(true);", share_download_list)
                                time.sleep(0.5)  # 等待滚动完成
                                
                                driver.execute_script("arguments[0].click();", share_download_list)
                                logger.info(f"已点击第 {i} 个文件的下载列表")
                                
                                # 等待新文件开始下载，使用更严格的超时控制
                                max_wait_time = 30  # 最大等待时间（秒）
                                check_interval = 1  # 检查间隔（秒）
                                start_time = time.time()
                                new_file = None
                                
                                while time.time() - start_time < max_wait_time:
                                    current_files = set(os.listdir(download_dir)) if os.path.exists(download_dir) else set()
                                    new_files = current_files - initial_files
                                    
                                    if new_files:
                                        new_file = list(new_files)[0]
                                        logger.info(f"检测到新文件开始下载: {new_file}")
                                        break
                                    
                                    time.sleep(check_interval)
                                
                                if not new_file:
                                    logger.warning(f"等待第 {i} 个文件开始下载超时")
                                    raise Exception("文件下载未开始")
                                
                                # 等待文件下载完成，使用更可靠的完成状态判断
                                file_path = os.path.join(download_dir, new_file)
                                last_size = -1
                                unchanged_count = 0
                                max_unchanged_count = 5  # 增加连续相同大小的检查次数
                                size_check_interval = 1  # 文件大小检查间隔（秒）
                                download_timeout = 15  # 下载总超时时间（秒）
                                download_start_time = time.time()
                                
                                while time.time() - download_start_time < download_timeout:
                                    if os.path.exists(file_path):
                                        current_size = os.path.getsize(file_path)
                                        
                                        if current_size == last_size:
                                            unchanged_count += 1
                                            if unchanged_count >= max_unchanged_count:
                                                logger.info(f"文件 {new_file} 下载完成（大小稳定）")
                                                break
                                        else:
                                            unchanged_count = 0
                                            last_size = current_size
                                            logger.info(f"文件 {new_file} 当前大小: {current_size} 字节")
                                    
                                    time.sleep(size_check_interval)
                                
                                if time.time() - download_start_time >= download_timeout:
                                    logger.error(f"文件 {new_file} 下载超时")
                                    raise Exception("文件下载超时")
                                
                                logger.info(f"文件 {new_file} 下载成功完成")
                                break  # 成功处理，跳出重试循环
                            except Exception as e:
                                row_retry_count += 1
                                logger.warning(f"处理第 {i} 行时出错 (尝试 {row_retry_count}/{max_row_retries}): {str(e)}")
                                if row_retry_count >= max_row_retries:
                                    logger.error(f"处理第 {i} 行失败，达到最大重试次数: {str(e)}")
                                    break
                                time.sleep(5)  # 重试前等待
                        except Exception as e:
                                row_retry_count += 1
                                logger.warning(f"xxx处理第 {i} 行时出错 (尝试 {row_retry_count}/{max_row_retries}): {str(e)}")
                                if row_retry_count >= max_row_retries:
                                    logger.error(f"处理第 {i} 行失败，达到最大重试次数: {str(e)}")
                                    break
                                time.sleep(5)  # 重试前等待
                logger.info("所有文件处理完成")
                break  # 成功完成，跳出主重试循环
                
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


if __name__ == '__main__':
    # 示例：批量处理多个分享链接
    urls = [
        # 'https://pan.quark.cn/s/22130dc18514#/list/share/12908f55af354afa9b8c5d262ddc17f7-%E7%9F%AD%E8%A7%86%E9%A2%91%E5%89%AA%E8%BE%91%E8%AF%AD%E9%9F%B3%E5%8C%85%E7%B4%A0%E6%9D%90%E5%90%88%E9%9B%86/85c0ac483d4943acb28b84bdfd5a8053-%E5%A4%A7%E5%8F%B8%E9%A9%AC',
        # 添加更多分享链接
        'https://pan.quark.cn/s/22130dc18514#/list/share/12908f55af354afa9b8c5d262ddc17f7-%E7%9F%AD%E8%A7%86%E9%A2%91%E5%89%AA%E8%BE%91%E8%AF%AD%E9%9F%B3%E5%8C%85%E7%B4%A0%E6%9D%90%E5%90%88%E9%9B%86/6b7c99aa2bd348c1bf6ab7835cad8c66-%E6%9D%8E%E4%BA%91%E9%BE%99',
     
    ]
    crawl_quark_share(urls)