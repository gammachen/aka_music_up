## 模型建设

### DB库表

## 初始化

### 库表建设

### 数据脚本

resource目录下的comic_mapping_*.json

这些文件是通过咨询AI得到的，先让其生成漫画的分类，然后再让其生成这些分类下的推荐的漫画集合

现在的漫画分类包含了8种，分别通过comic_mapping_x.json文件的key表示，要打开文件来查看或者读取所有文件内的key才能够得到所有的分类集合

但是在scripts目录下的init_comic_data.py中我们作了一个初始化对应分类脚本，将comic_mapping_*.json中的分类写入了数据库的category表，使用数据库中对应的category也有对应的分类的，在分类页面中能够直接有系统读取透出的：/genre


```java
24	热血漫画	4	2	1	background: linear-gradient(45deg, #FF9A9E, #FAD0C4)	/static/category/action-comic.jpg				
25	恋爱漫画	4	2	2	background: linear-gradient(45deg, #A18CD1, #FBC2EB)	/static/category/romance-comic.jpg				
26	科幻漫画	4	2	3	background: linear-gradient(45deg, #6B8DD6, #8E37D7)	/static/category/sci-fi-comic.jpg				
27	搞笑漫画	4	2	4	background: linear-gradient(45deg, #434343, #000000)	/static/category/funny-comic.jpg				
28	悬疑漫画	4	2	5	background: linear-gradient(45deg, #D4FC79, #96E6A1)	/static/category/mystery-comic.jpg				
29	校园漫画	4	2	6	background: linear-gradient(45deg, #84FAB0, #8FD3F4)	/static/category/school-comic.jpg				
30	冒险漫画	4	2	7	background: linear-gradient(45deg, #FF9A9E, #FAD0C4)	/static/category/adventure-comic.jpg				
31	魔幻漫画	4	2	8	background: linear-gradient(45deg, #A18CD1, #FBC2EB)	/static/category/fantasy-comic.jpg				
```

```python
-- 插入漫画的二级分类
INSERT INTO category (name, parent_id, level, sort_order, background_style, desc_image)
SELECT 
    '热血漫画', id, 2, 1, 'background: linear-gradient(45deg, #FF9A9E, #FAD0C4)', 'https://example.com/images/action-comic.jpg'
FROM category WHERE name = '漫画';

INSERT INTO category (name, parent_id, level, sort_order, background_style, desc_image)
SELECT 
    '恋爱漫画', id, 2, 2, 'background: linear-gradient(45deg, #A18CD1, #FBC2EB)', 'https://example.com/images/romance-comic.jpg'
FROM category WHERE name = '漫画';

INSERT INTO category (name, parent_id, level, sort_order, background_style, desc_image)
SELECT 
    '科幻漫画', id, 2, 3, 'background: linear-gradient(45deg, #6B8DD6, #8E37D7)', 'https://example.com/images/sci-fi-comic.jpg'
FROM category WHERE name = '漫画';

INSERT INTO category (name, parent_id, level, sort_order, background_style, desc_image)
SELECT 
    '搞笑漫画', id, 2, 4, 'background: linear-gradient(45deg, #434343, #000000)', 'https://example.com/images/funny-comic.jpg'
FROM category WHERE name = '漫画';

INSERT INTO category (name, parent_id, level, sort_order, background_style, desc_image)
SELECT 
    '悬疑漫画', id, 2, 5, 'background: linear-gradient(45deg, #D4FC79, #96E6A1)', 'https://example.com/images/mystery-comic.jpg'
FROM category WHERE name = '漫画';

INSERT INTO category (name, parent_id, level, sort_order, background_style, desc_image)
SELECT 
    '校园漫画', id, 2, 6, 'background: linear-gradient(45deg, #84FAB0, #8FD3F4)', 'https://example.com/images/school-comic.jpg'
FROM category WHERE name = '漫画';

INSERT INTO category (name, parent_id, level, sort_order, background_style, desc_image)
SELECT 
    '冒险漫画', id, 2, 7, 'background: linear-gradient(45deg, #FF9A9E, #FAD0C4)', 'https://example.com/images/adventure-comic.jpg'
FROM category WHERE name = '漫画';

INSERT INTO category (name, parent_id, level, sort_order, background_style, desc_image)
SELECT 
    '魔幻漫画', id, 2, 8, 'background: linear-gradient(45deg, #A18CD1, #FBC2EB)', 'https://example.com/images/fantasy-comic.jpg'
FROM category WHERE name = '漫画';
```

### 初始化与导入漫画数据

漫画数据的模型我们定义成是Content与Chapter

将某部漫画直接作为Content（将来会扩展到小说、课程），只要是能够划分这种主题与章节的形态的都可以这样放

漫画的Content表示的是漫画主题，名称等信息

脚本scripts/init_comic_directory.py在static/comic目录下建立了直接的漫画级别的目录结构，使用的原数据就是source
```json
../static/comic
├── 20世纪少年
├── Aama
├── Akira
├── All You Need Is Kill
├── 子不语
├── 守望者
├── 寄生兽
├── 山海师
├── 彼岸岛
├── 恶之华
├── 恶魔人
├── 海贼王
│   ├── 海贼王-第001话
```

脚本scripts/init_comic_data.py将扫描漫画目录下的章节目录与图片文件（TODO 后续视频等应该也是这种思路来处理的！！！）

#### 初始化漫画列表

通过DeepSeek生成一系列的漫画名称（这里涉及到一个关于漫画领域的多个站点的一个数据源）（虽然参考像腾讯漫画站点等内容会比较多一点）

TODO 数据源

### 