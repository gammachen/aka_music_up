```bash

使用以下命令将MP4视频转换为M3U8格式：

ffmpeg -i demo.mp4 -profile:v baseline -level 3.0 -start_number 0 -hls_time 10 -hls_list_size 0 -f hls demo.m3u8
其中，demo.mp4是原始MP4文件，demo.m3u8是生成的M3U8文件。参数解释如下：

-profile:v baseline：设置视频质量为基本画质。

-level 3.0：设置视频级别。

-start_number 0：从0开始编号。

-hls_time 10：每10秒切一个片段。

-hls_list_size 0：保存所有片段信息。


```

ffmpeg -i liang_jing_ru.mp4 -profile:v baseline -level 3.0 -start_number 0 -hls_time 10 -hls_list_size 0 -f hls online/liang_jing_ru.m3u8

ffmpeg -i jieba_chunli.mp4 -profile:v baseline -level 3.0 -start_number 0 -hls_time 10 -hls_list_size 0 -f hls online/jieba_chunli.m3u8

ffmpeg -i demo.mp4 -profile:v baseline -level 3.0 -start_number 0 -hls_time 10 -hls_list_size 0 -f hls /Users/shhaofu/Code/cursor-projects/aka_music/backend/app/static/uploads/online/banma_1.m3u8

ffmpeg -i 167\ Alphabet\ Song\ -\ Alphabet\ ‘Q’\ So.mp4  -profile:v baseline -level 3.0 -start_number 0 -hls_time 10 -hls_list_size 0 -f hls /Users/shhaofu/Code/cursor-projects/aka_music/backend/app/static/uploads/online/ln_2.m3u8

ffmpeg -i 167\ Alphabet\ Song\ -\ Alphabet\ ‘Q’\ So.mp4  -profile:v baseline -level 3.0 -start_number 0 -hls_time 10 -hls_list_size 0 -f hls /Users/shhaofu/Code/cursor-projects/aka_music/backend/app/static/uploads/online/ln_2.m3u8






