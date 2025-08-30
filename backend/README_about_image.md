## 模型建设

### DB库表

## 初始化

### 库表建设

### 数据脚本

图片的处理更加复杂（妈蛋）

其中构建”佳人“构建初始化的sql脚本：
backend/app/scripts/generate_image_category_sql.py

```java
    # 读取image_mapping.json文件
    with open('backend/app/resource/image_mapping.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 生成SQL语句
    sql_statements = process_directory(data)
    
    # 将SQL语句写入文件
    with open('backend/app/scripts/category_init.sql', 'w', encoding='utf-8') as f:
        f.write('-- 初始化分类数据\n\n')
        f.write('-- 确保外键约束开启\nPRAGMA foreign_keys = ON;\n\n')
        for sql in sql_statements:
            f.write(sql + '\n')
```

### 涉及到参考的很多站点

### 图片预览

### 图片分享

### 图片下载

### 图片对话

### 图片去除背景

### 图片编辑

### 佳人这个特殊分类的图片内容的初始化

脚本image_scanner.py将扫描目录下的所有子目录结构，将子目录构建成目录树，提供给前端直接解析使用，待完善这个复杂的逻辑的说明

### google搜索图片引擎的升级

### 通过关键字搜索进行图片卡片式的展示

构造了ImageGallery组件
--- 涉及到scripts中的image_dims.json的构造（不同比例（9:16,9:26等），不同风格（grid等）、不同边框（border、round）、不同动画fold_in等）
构造了CharacterGallery组件（comic的landing中引用了）
构造了DiagonalCharacterGallery组件（comic的landing中引用了）

### 增加了一个抓取图片的脚本

通过让DeepSeek提供写真、医疗的关键字内容，通过Unsplash提供的API（每页200条）下载图片

### 增加了百度图片搜索服务

百度没有提供api服务，只提供了

### 抓取封面的图片数据

### 调用pixabay的API获取高质量的图片

其中涉及到哪些分类（都是图片下的二级分类，比如机械、科技、医疗等，这些分类是让DeepSeek辅助生成的，之后再让DeepSeek根据这些分类生成对应的关键词，就是领域内的关键词列表）

```json
words_mapping_for_image_crawler.json
"医疗": [
      "healthcare",
      "medical care",
      "hospital staff",
      "doctor consultation",
      "medical treatment",
      "healthcare professional",
      "medical equipment",
      "patient care",
      "clinical practice",
      "medical facility"
    ]
```

详情见：search_pixabay_and_store函数的实现

### 非常好的pixabay的图片搜索服务

### 抓取更多分类的图片数据

强迫症似的要抓取历史人物、动漫人物等（主要是想要作人物等相关的图片资料库）

构建了多个分类的映射内容：backend/app/scripts/crawler_categories

内容是通过DeepSeek提供的分类关键词

```json
(translate-env) (base) shhaofu@shhaofudeMacBook-Pro scripts % tree crawler_categories 
crawler_categories
├── JPAV.txt
├── 狗.txt
├── 猫.txt
├── 山川.txt
├── 河流.txt
├── 三国人物.txt
├── 励志电影.txt
├── 大陆作家.txt
├── 大陆歌星.txt
├── 日本明星.txt
├── 日本歌星.txt
├── 武侠电影.txt
├── 水浒人物.txt
├── 科幻电影.txt
├── 港澳台作家.txt
├── 港澳台歌星.txt
├── 红楼梦人物.txt
├── 西游记人物.txt
├── 近代哲学家.txt
├── 诺贝尔获奖者.txt
├── 近代体育明星.txt
└── 中国古代哲学家.txt
```

```json
(translate-env) (base) shhaofu@shhaofudeMacBook-Pro scripts % more crawler_categories/JPAV.txt 
三上悠亚
河北彩伽
石川澪
桃乃木香奈
凪光（凪ひかる）
伊藤舞雪
金松季步（金松季歩）
村上悠华
宫下玲奈
田野忧
瀬戸环奈
石原希
Amane Mahina
Rikka Ono（小野立花）
Riri Nanatsumori（松本铃鹿）
Hibiki Natsume（夏目响）
深田恭子
武藤蓝
饭岛爱
吉泽明步
```

通过百度search进行图片内容的下载，跑的有点慢就是了，毕竟是使用了浏览器模拟人为操作！

### 涉及到static/beauty目录与佳人这个有颜色的图库的特殊处理

### 初始化过程中涉及到对文件目录的名字进行MD5加密编码的步骤

这里不使用uuid的形式是为了能够在后续作Category的名字与目录作映射的逻辑的，使用uuid基本上就比较割裂了

```shell
_scan_directory_recursive
def _scan_directory_recursive(self, current_dir):
        """递归扫描目录，返回树形结构的数据"""
        # 使用 MD5 对目录路径进行哈希
        md5_hash = hashlib.md5(str(current_dir.name).encode()).hexdigest()
        
        result = {
            'dir_name': current_dir.name if current_dir != self.root_dir else '',
            'children': [],
            # 'refer_id': str(hash(str(current_dir.absolute())))
            'refer_id': md5_hash
        }
```

