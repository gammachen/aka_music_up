# HTTP请求相关的配置

# 通用User-Agent配置
# DEFAULT_USER_AGENT = 'AkaMusic/1.0 (https://akamusic.com; contact@akamusic.com) Python/3.x'
DEFAULT_USER_AGENT = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'

# 通用请求头
DEFAULT_HEADERS = {
    'User-Agent': DEFAULT_USER_AGENT,
    'Accept': '*/*',
    'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8'
}