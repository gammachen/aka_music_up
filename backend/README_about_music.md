## 模型建设

### DB库表

## 初始化

### 库表建设

### 数据脚本

音乐的处理更加复杂

涉及到将原音乐文件转换成m3u8格式，再将目录映射成json格式的数据，数据库那边的category还需要配置你已经处理过了的目录项目，将其refer_id设置为对应的处理目录，并且建立的目录的名称需要对应上，这样才能够在导航页面得到正确的映射路径

涉及到的脚本有scripts/song_data_processor.py, mp3_converter.py

生成的音乐映射文件在resouce目录下的x_song_mapping.json

比如category中定义的音乐为一级目录（DB脚本初始化的）
1	音乐		1	1	background: linear-gradient(45deg, #FF6B6B, #FFE66D)	/static/category/music.jpg				

音乐下面的子分类是；
```sql
SELECT * from category where parent_id = 1;
```

```python
146	抖音网红歌曲	1	2	1	background: linear-gradient(45deg, #FF9A9E, #FAD0C4)	/static/category/douyin.jpg			1	mulist
148	网络流行歌曲	1	2	2	background: linear-gradient(45deg, #A18CD1, #FBC2EB)	/static/category/online-pop.jpg			2	mulist
150	经典流行环绕歌曲	1	2	3	background: linear-gradient(45deg, #6B8DD6, #8E37D7)	/static/category/classic-surround.jpg			3	mulist
152	经典粤语歌曲	1	2	4	background: linear-gradient(45deg, #434343, #000000)	/static/category/cantonese.jpg			4	mulist
154	抖友中文DJ歌曲	1	2	5	background: linear-gradient(45deg, #D4FC79, #96E6A1)	/static/category/chinese-dj.jpg			5	mulist
156	一人一首成名曲	1	2	6	background: linear-gradient(45deg, #84FAB0, #8FD3F4)	/static/category/famous-songs.jpg			6	mulist
158	经典翻唱发烧人声	1	2	7	background: linear-gradient(45deg, #FF9A9E, #FAD0C4)	/static/category/cover-songs.jpg			7	mulist
160	欧美节奏控	1	2	8	background: linear-gradient(45deg, #A18CD1, #FBC2EB)	/static/category/western-rhythm.jpg			8	mulist
162	伤感网络情歌	1	2	9	background: linear-gradient(45deg, #FA709A, #FEE140)	/static/category/sad-songs.jpg			9	mulist
164	英文流行歌曲	1	2	10	background: linear-gradient(45deg, #43E97B, #38F9D7)	/static/category/english-pop.jpg			10	mulist
166	草原天籁	1	2	11	background: linear-gradient(45deg, #FF9A9E, #FAD0C4)	/static/category/grassland.jpg			11	mulist
168	闽南语歌曲	1	2	12	background: linear-gradient(45deg, #A18CD1, #FBC2EB)	/static/category/minnan.jpg			12	mulist
170	车载4D环绕8D魔音	1	2	13	background: linear-gradient(45deg, #6B8DD6, #8E37D7)	/static/category/car-music.jpg			13	mulist
172	音乐工作室制作抖音DJ专辑	1	2	14	background: linear-gradient(45deg, #434343, #000000)	/static/category/studio-dj.jpg			14	mulist
174	国语经典成名曲	1	2	15	background: linear-gradient(45deg, #D4FC79, #96E6A1)	/static/category/mandarin-classic.jpg			15	mulist
176	8090后回忆录歌曲	1	2	16	background: linear-gradient(45deg, #84FAB0, #8FD3F4)	/static/category/8090-memory.jpg			16	mulist
178	轻音乐发烧人声歌曲	1	2	17	background: linear-gradient(45deg, #FF9A9E, #FAD0C4)	/static/category/light-music.jpg			17	mulist
180	重低音硬曲	1	2	18	background: linear-gradient(45deg, #A18CD1, #FBC2EB)	/static/category/bass-music.jpg			18	mulist
```

在sources目录下的init_category.sql中是这样构建的：
```sql
-- 插入音乐的二级分类
INSERT INTO category (name, parent_id, level, sort_order, background_style, desc_image) 
SELECT 
    '抖音网红歌曲', id, 2, 1, 'background: linear-gradient(45deg, #FF9A9E, #FAD0C4)', 'https://example.com/images/douyin.jpg'
FROM category WHERE name = '音乐';

INSERT INTO category (name, parent_id, level, sort_order, background_style, desc_image)
SELECT 
    '网络流行歌曲', id, 2, 2, 'background: linear-gradient(45deg, #A18CD1, #FBC2EB)', 'https://example.com/images/online-pop.jpg'
FROM category WHERE name = '音乐';

INSERT INTO category (name, parent_id, level, sort_order, background_style, desc_image)
SELECT 
    '经典流行环绕歌曲', id, 2, 3, 'background: linear-gradient(45deg, #6B8DD6, #8E37D7)', 'https://example.com/images/classic-surround.jpg'
FROM category WHERE name = '音乐';

INSERT INTO category (name, parent_id, level, sort_order, background_style, desc_image)
SELECT 
    '经典粤语歌曲', id, 2, 4, 'background: linear-gradient(45deg, #434343, #000000)', 'https://example.com/images/cantonese.jpg'
FROM category WHERE name = '音乐';

INSERT INTO category (name, parent_id, level, sort_order, background_style, desc_image)
SELECT 
    '抖友中文DJ歌曲', id, 2, 5, 'background: linear-gradient(45deg, #D4FC79, #96E6A1)', 'https://example.com/images/chinese-dj.jpg'
FROM category WHERE name = '音乐';

INSERT INTO category (name, parent_id, level, sort_order, background_style, desc_image)
SELECT 
    '一人一首成名曲', id, 2, 6, 'background: linear-gradient(45deg, #84FAB0, #8FD3F4)', 'https://example.com/images/famous-songs.jpg'
FROM category WHERE name = '音乐';

INSERT INTO category (name, parent_id, level, sort_order, background_style, desc_image)
SELECT 
    '经典翻唱发烧人声', id, 2, 7, 'background: linear-gradient(45deg, #FF9A9E, #FAD0C4)', 'https://example.com/images/cover-songs.jpg'
FROM category WHERE name = '音乐';

INSERT INTO category (name, parent_id, level, sort_order, background_style, desc_image)
SELECT 
    '欧美节奏控', id, 2, 8, 'background: linear-gradient(45deg, #A18CD1, #FBC2EB)', 'https://example.com/images/western-rhythm.jpg'
FROM category WHERE name = '音乐';

INSERT INTO category (name, parent_id, level, sort_order, background_style, desc_image)
SELECT 
    '伤感网络情歌', id, 2, 9, 'background: linear-gradient(45deg, #FA709A, #FEE140)', 'https://example.com/images/sad-songs.jpg'
FROM category WHERE name = '音乐';

INSERT INTO category (name, parent_id, level, sort_order, background_style, desc_image)
SELECT 
    '英文流行歌曲', id, 2, 10, 'background: linear-gradient(45deg, #43E97B, #38F9D7)', 'https://example.com/images/english-pop.jpg'
FROM category WHERE name = '音乐';

INSERT INTO category (name, parent_id, level, sort_order, background_style, desc_image)
SELECT 
    '草原天籁', id, 2, 11, 'background: linear-gradient(45deg, #FF9A9E, #FAD0C4)', 'https://example.com/images/grassland.jpg'
FROM category WHERE name = '音乐';

INSERT INTO category (name, parent_id, level, sort_order, background_style, desc_image)
SELECT 
    '闽南语歌曲', id, 2, 12, 'background: linear-gradient(45deg, #A18CD1, #FBC2EB)', 'https://example.com/images/minnan.jpg'
FROM category WHERE name = '音乐';

INSERT INTO category (name, parent_id, level, sort_order, background_style, desc_image)
SELECT 
    '车载4D环绕8D魔音', id, 2, 13, 'background: linear-gradient(45deg, #6B8DD6, #8E37D7)', 'https://example.com/images/car-music.jpg'
FROM category WHERE name = '音乐';

INSERT INTO category (name, parent_id, level, sort_order, background_style, desc_image)
SELECT 
    '音乐工作室制作抖音DJ专辑', id, 2, 14, 'background: linear-gradient(45deg, #434343, #000000)', 'https://example.com/images/studio-dj.jpg'
FROM category WHERE name = '音乐';

INSERT INTO category (name, parent_id, level, sort_order, background_style, desc_image)
SELECT 
    '国语经典成名曲', id, 2, 15, 'background: linear-gradient(45deg, #D4FC79, #96E6A1)', 'https://example.com/images/mandarin-classic.jpg'
FROM category WHERE name = '音乐';

INSERT INTO category (name, parent_id, level, sort_order, background_style, desc_image)
SELECT 
    '8090后回忆录歌曲', id, 2, 16, 'background: linear-gradient(45deg, #84FAB0, #8FD3F4)', 'https://example.com/images/8090-memory.jpg'
FROM category WHERE name = '音乐';

INSERT INTO category (name, parent_id, level, sort_order, background_style, desc_image)
SELECT 
    '轻音乐发烧人声歌曲', id, 2, 17, 'background: linear-gradient(45deg, #FF9A9E, #FAD0C4)', 'https://example.com/images/light-music.jpg'
FROM category WHERE name = '音乐';

INSERT INTO category (name, parent_id, level, sort_order, background_style, desc_image)
SELECT 
    '重低音硬曲', id, 2, 18, 'background: linear-gradient(45deg, #A18CD1, #FBC2EB)', 'https://example.com/images/bass-music.jpg'
FROM category WHERE name = '音乐';
```
这些脚本是根据我对AI（Trae和DeepSeek）的描述让其构造生成的，其中包括背景色、背景图片（对应的Icon的英文），并且我是给了它关于二级目录的名称列表，这个列表是根据我们本地已经拥有的音乐资源构造的，在下载的大几十G的音乐文件中将文件名等提取出来将其构造成二级分类的！

音乐的原始文件是在夸克网盘中搜索到的内容（百度云也有，只是夸克升级了svip，下载非常迅速）


### 分类与歌曲的映射

通过脚本scripts中的song_data_processor.py文件来对某个目录下的音乐文件进行转码并且生成对应的映射文件，比如对4目录下的文件转码，生成对应的m3u8格式的文件到4目录下的子目录下（还要有子目录是更好的分类与生成的m3u8格式的文件的唯一性，避免文件名冲突）

最后生成的歌曲与m3u8格式文件的映射文件，可以参考下面的示例：
```json
{
  "118.Beyond-谁伴我闯荡": "/118/1740070000_3569.m3u8",
  "25.陈奕迅-单车": "/25/1740070005_8789.m3u8",
  "122.Beyond -  光辉岁月": "/122/1740070008_7195.m3u8",
  "0359.古巨基 - 眼睛不能没眼泪": "/0359/1740070012_3652.m3u8",
  "0371.李国祥&伦永亮 - 总有你鼓励": "/0371/1740070017_1268.m3u8",
  "0390.王菲 - 迷魂记": "/0390/1740070020_8971.m3u8"
}
```

这里还必须说的是，脚本生辰过程还会带上日期，表明生成的时间，避免文件的覆盖

生成的是:年月日_processed_songs.json

在recommend.py路由中会去找最近生成的songs.json文件，并且缓存一份online_musics_xxxxxx_processed_songs.json文件，等于说是本地缓存映射关系，这个主要是因为推荐的音乐列表（在Landing页面使用）应该是比较固定的，但是我们的音乐数据的映射是歌曲名称:歌曲路径，是一个Map，每次取值可能都会是变化了的，所以重新生成了一个数组形式的数据！！！



