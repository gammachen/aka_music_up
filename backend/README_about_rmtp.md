```shell
docker run -d --name nginx-hls -p 1935:1935 -p 8080:8080 -v /root/aka_music/backend/app/resource/01_nginx_config_for_rtmp.conf:/etc/nginx/nginx.conf alqutami/rtmp-hls
```

```shell
(aka_music) root@iZbp1cqdx0u3g3wriko37cZ:~# whereis ffmpeg
ffmpeg: /usr/bin/ffmpeg /usr/share/ffmpeg /usr/share/man/man1/ffmpeg.1.gz
```

```shell
docker run -d --name nginx-hls -p 1935:1935 -p 8080:8080 -v $HOME/aka_music/backend/app/resource/01_nginx_config_for_rtmp.conf:/etc/nginx/nginx.conf alqutami/rtmp-hls

ffmpeg -re -stream_loop -1 -i /root/aka_music/backend/app/static/def/jieba_chunli.mp4 -c copy -f flv rtmp://127.0.0.1:1935/live/test

docker run -d --name nginx-hls -p 1935:1935 -p 8080:8080 -v $HOME/Code/cursor-projects/aka_music/backend/app/resource/01_nginx_config_for_rtmp.conf:/etc/nginx/nginx.conf alqutami/rtmp-hls

ffmpeg -re -stream_loop -1 -i $HOME/Code/cursor-projects/aka_music/backend/app/static/def/jieba_chunli.mp4 -c copy -f flv rtmp://127.0.0.1:1935/live/test

docker run -d --name nginx-hls-2 -p 1936:1935 -p 8081:8080 -v $HOME/Code/cursor-projects/aka_music/backend/app/resource/01_nginx_config_for_rtmp.conf:/etc/nginx/nginx.conf alqutami/rtmp-hls

ffmpeg -re -stream_loop -1 -i $HOME/Code/cursor-projects/aka_music/backend/app/static/def/jieba_chunli.mp4 -c copy -f flv rtmp://127.0.0.1:1935/live/test

ffmpeg -re -stream_loop -1 -i $HOME/Code/cursor-projects/aka_music/backend/app/static/def/jieba_haogui.mp4 -c copy -f flv rtmp://127.0.0.1:1936/live/testhaogui



rtmp://http://alphago.ltd:1935/live/test
```

```shell
观测流媒体服务器：
http://47.98.62.98:8080/stat

```

```shell
rtmp://http://alphago.ltd:1935/live/stream
```

使用镜像自带的players播放

To play RTMP content (requires Flash): http://47.98.62.98:8080/players/rtmp.html
To play HLS content: http://47.98.62.98:8080/players/hls.html
To play HLS content using hls.js library: http://47.98.62.98:8080/players/hls_hlsjs.html
To play DASH content: http://47.98.62.98:8080/players/dash.html
To play RTMP and HLS contents on the same page: http://47.98.62.98:8080/players/rtmp_hls.html

```shell
构建live/stream
ffmpeg -re -stream_loop -1 -i /root/aka_music/backend/app/static/def/video_2.mp4 -c copy -f flv rtmp://127.0.0.1:1935/live/stream
```

```bash
生产环境中上传视频文件发生413异常，文件太大了，但是上传的大小限制在后端是100M之内，所以可能是nginx的配置限制了的，而不是backend的api接口限制，要排查的

112.10.202.62 - - [13/Mar/2025:11:20:39 +0800] "POST /api/backend/rtmp/upload_video HTTP/1.1" 413 594 "https://alphago.ltd/backend/rtmp/adm" "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36"

2025/03/13 11:20:38 [error] 514299#514299: *245 client intended to send too large body: 15985909 bytes, client: 112.10.202.62, server: localhost, request: "POST /api/backend/rtmp/upload_video HTTP/1.1", host: "alphago.ltd", referrer: "https://alphago.ltd/backend/rtmp/adm"

Request URL:
https://alphago.ltd/api/backend/rtmp/upload_video
Request Method:
POST
Status Code:
413 Request Entity Too Large
Remote Address:
127.0.0.1:7897
Referrer Policy:
strict-origin-when-cross-origin
```




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

```bash
将视频转换为m3u8格式

ffmpeg -i liang_jing_ru.mp4 -profile:v baseline -level 3.0 -start_number 0 -hls_time 10 -hls_list_size 0 -f hls online/liang_jing_ru.m3u8

ffmpeg -i jieba_chunli.mp4 -profile:v baseline -level 3.0 -start_number 0 -hls_time 10 -hls_list_size 0 -f hls online/jieba_chunli.m3u8

ffmpeg -i demo.mp4 -profile:v baseline -level 3.0 -start_number 0 -hls_time 10 -hls_list_size 0 -f hls /Users/shhaofu/Code/cursor-projects/aka_music/backend/app/static/uploads/online/banma_1.m3u8

ffmpeg -i 167\ Alphabet\ Song\ -\ Alphabet\ ‘Q’\ So.mp4  -profile:v baseline -level 3.0 -start_number 0 -hls_time 10 -hls_list_size 0 -f hls /Users/shhaofu/Code/cursor-projects/aka_music/backend/app/static/uploads/online/ln_2.m3u8

ffmpeg -i 167\ Alphabet\ Song\ -\ Alphabet\ ‘Q’\ So.mp4  -profile:v baseline -level 3.0 -start_number 0 -hls_time 10 -hls_list_size 0 -f hls /Users/shhaofu/Code/cursor-projects/aka_music/backend/app/static/uploads/online/ln_2.m3u8

ffmpeg -i /Users/shhaofu/Downloads/cai_dian_4.mp4  -profile:v baseline -level 3.0 -start_number 0 -hls_time 10 -hls_list_size 0 -f hls /Users/shhaofu/Code/cursor-projects/aka_music/backend/app/static/uploads/online/cai_dian_4.m3u8

ffmpeg -i /Users/shhaofu/Downloads/cai_dian_2.mp4  -profile:v baseline -level 3.0 -start_number 0 -hls_time 10 -hls_list_size 0 -f hls /Users/shhaofu/Code/cursor-projects/aka_music/backend/app/static/uploads/online/cai_dian_2.m3u8
```




