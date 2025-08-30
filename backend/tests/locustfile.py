import time
import random
from locust import HttpUser, task, between, tag

class AlphaGoUser(HttpUser):
    # 设置思考时间在1-5秒之间，模拟真实用户行为
    wait_time = between(1, 5)
    
    # 初始化用户
    def on_start(self):
        # 用户登录信息
        self.login_payload = {
            "email": "a@s.com",
            "password": "123123csd"
        }
        # 可以在这里执行登录操作，获取token
        # self.client.post("/api/auth/login/email", json=self.login_payload)
    
    @tag('homepage')
    @task(10)  # 权重为10，表示这个任务执行的概率较高
    def visit_homepage(self):
        # 访问首页
        try:
            with self.client.get("/", name="首页", catch_response=True, verify=False, allow_redirects=True) as response:
                print(f"首页访问成功: {response.status_code} {response._content} {response.cookies}")
                if response.status_code != 200:
                    response.failure(f"首页访问失败: {response.status_code}")
        except Exception as e:
            print(f"首页访问异常: {str(e)}")
            # 创建一个模拟的响应对象来记录失败
            self.environment.events.request_failure.fire(
                request_type="GET",
                name="首页",
                response_time=0,
                exception=e,
            )

    @tag('genre')
    @task(5)  # 权重为5，表示这个任务执行的概率中等
    def visit_genre(self):
        # 访问全分类页面
        try:
            with self.client.get("/genre", name="全分类", catch_response=True, verify=False, allow_redirects=True) as response:
                if response.status_code != 200:
                    response.failure(f"全分类访问失败: {response.status_code}")
        except Exception as e:
            print(f"全分类访问异常: {str(e)}")
            # 创建一个模拟的响应对象来记录失败
            self.environment.events.request_failure.fire(
                request_type="GET",
                name="全分类",
                response_time=0,
                exception=e,
            )
                
    @tag('beaulista')
    @task(3)  # 权重为3，表示这个任务执行的概率较低
    def visit_beaulista(self):
        # 访问写真页面
        try:
            with self.client.get("/beaulist/37e3808047553cedb34daa9b1d7ab2a3", name="写真", catch_response=True, verify=False, allow_redirects=True) as response:
                if response.status_code != 200:
                    response.failure(f"写真访问失败: {response.status_code}")
        except Exception as e:
            print(f"写真访问异常: {str(e)}")
            # 创建一个模拟的响应对象来记录失败
            self.environment.events.request_failure.fire(
                request_type="GET",
                name="写真",
                response_time=0,
                exception=e,
            )

'''                    
    @tag('category')
    @task(5)
    def browse_categories(self):
        # 浏览分类
        with self.client.get("/api/category/tree", name="分类树", catch_response=True) as response:
            if response.status_code != 200:
                response.failure(f"分类树访问失败: {response.status_code}")
    
    @tag('comic')
    @task(3)
    def browse_comics(self):
        # 浏览漫画列表
        with self.client.get("/api/comic/all_comic_genre", name="漫画列表", catch_response=True) as response:
            if response.status_code != 200:
                response.failure(f"漫画列表访问失败: {response.status_code}")
                return
            
            # 如果成功获取漫画列表，随机选择一个漫画查看详情
            try:
                data = response.json()
                if data.get('code') == 200 and data.get('data') and len(data['data']) > 0:
                    # 随机选择一个漫画ID
                    comic_id = random.choice(data['data'])['id']
                    # 查看漫画详情
                    self.view_comic_detail(comic_id)
            except Exception as e:
                response.failure(f"处理漫画列表数据失败: {str(e)}")
    
    def view_comic_detail(self, comic_id):
        # 查看漫画详情
        with self.client.get(f"/api/comic/contentDetail?id={comic_id}", name="漫画详情", catch_response=True) as response:
            if response.status_code != 200:
                response.failure(f"漫画详情访问失败: {response.status_code}")
                return
            
            # 如果成功获取漫画详情，查看章节列表
            try:
                data = response.json()
                if data.get('code') == 200:
                    # 查看章节列表
                    self.view_chapter_list(comic_id)
            except Exception as e:
                response.failure(f"处理漫画详情数据失败: {str(e)}")
    
    def view_chapter_list(self, comic_id):
        # 查看章节列表
        with self.client.get(f"/api/comic/chapterList?id={comic_id}", name="章节列表", catch_response=True) as response:
            if response.status_code != 200:
                response.failure(f"章节列表访问失败: {response.status_code}")
                return
            
            # 如果成功获取章节列表，随机选择一个章节查看
            try:
                data = response.json()
                if data.get('code') == 200 and data.get('data') and len(data['data']) > 0:
                    # 随机选择一个章节
                    chapter_id = random.choice(data['data'])['id']
                    # 查看章节详情
                    self.view_chapter_detail(chapter_id)
            except Exception as e:
                response.failure(f"处理章节列表数据失败: {str(e)}")
    
    def view_chapter_detail(self, chapter_id):
        # 查看章节详情
        with self.client.get(f"/api/comic/chapterDetail?id={chapter_id}", name="章节详情", catch_response=True) as response:
            if response.status_code != 200:
                response.failure(f"章节详情访问失败: {response.status_code}")
    
    @tag('music')
    @task(3)
    def browse_music(self):
        # 浏览音乐列表
        with self.client.get("/api/music/mulist", name="音乐列表", catch_response=True) as response:
            if response.status_code != 200:
                response.failure(f"音乐列表访问失败: {response.status_code}")
    
    @tag('auth')
    @task(1)
    def login_flow(self):
        # 模拟登录流程
        with self.client.post("/api/auth/login/email", json=self.login_payload, name="邮箱登录", catch_response=True) as response:
            if response.status_code != 200:
                response.failure(f"登录失败: {response.status_code}")
                return
            
            try:
                data = response.json()
                if data.get('code') != 200:
                    response.failure(f"登录失败: {data.get('message')}")
            except Exception as e:
                response.failure(f"处理登录响应失败: {str(e)}")
    
    @tag('topic')
    @task(2)
    def browse_topics(self):
        # 浏览话题列表
        with self.client.get("/api/topic/", name="话题列表", catch_response=True) as response:
            if response.status_code != 200:
                response.failure(f"话题列表访问失败: {response.status_code}")
                return
            
            # 如果成功获取话题列表，随机选择一个话题查看详情
            try:
                data = response.json()
                if data.get('code') == 200 and data.get('data') and len(data['data']) > 0:
                    # 随机选择一个话题
                    topic_id = random.choice(data['data'])['id']
                    # 查看话题详情
                    self.view_topic_detail(topic_id)
            except Exception as e:
                response.failure(f"处理话题列表数据失败: {str(e)}")
    
    def view_topic_detail(self, topic_id):
        # 查看话题详情
        with self.client.get(f"/api/topic/{topic_id}", name="话题详情", catch_response=True) as response:
            if response.status_code != 200:
                response.failure(f"话题详情访问失败: {response.status_code}")
                return
            
            # 如果成功获取话题详情，查看评论
            try:
                data = response.json()
                if data.get('code') == 200:
                    # 查看评论
                    self.view_comments(topic_id)
            except Exception as e:
                response.failure(f"处理话题详情数据失败: {str(e)}")
    
    def view_comments(self, topic_id):
        # 查看评论
        with self.client.get(f"/api/comment/?topic_id={topic_id}", name="评论列表", catch_response=True) as response:
            if response.status_code != 200:
                response.failure(f"评论列表访问失败: {response.status_code}")
'''