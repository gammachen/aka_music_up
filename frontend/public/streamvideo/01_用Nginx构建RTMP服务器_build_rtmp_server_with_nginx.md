## 构建rtmp服务器

```shell
docker run -d --name nginx-hls -p 1935:1935 -p 8080:8080 -v $HOME/aka_music/backend/app/resource/01_nginx_config_for_rtmp.conf:/etc/nginx/nginx.conf alqutami/rtmp-hls

ffmpeg -re -stream_loop -1 -i /root/aka_music/backend/app/static/def/jieba_chunli.mp4 -c copy -f flv rtmp://127.0.0.1:1935/live/test

# 最原始的一个rtmp-nginx.conf的配置内容
docker run -d --name nginx-hls \
  -p 1935:1935 \
  -p 8080:8080 \
  -v $HOME/Code/cursor-projects/aka_music/backend/app/resource/01_nginx_config_for_rtmp.conf:/etc/nginx/nginx.conf \
  alqutami/rtmp-hls



ffmpeg -re -stream_loop -1 -i $HOME/Code/cursor-projects/aka_music/backend/app/static/def/jieba_chunli.mp4 -c copy -f flv rtmp://127.0.0.1:1935/live/test

docker run -d --name nginx-hls-2 -p 1936:1935 -p 8081:8080 -v $HOME/Code/cursor-projects/aka_music/backend/app/resource/01_nginx_config_for_rtmp.conf:/etc/nginx/nginx.conf alqutami/rtmp-hls

单独推送流到127.0.0.1:1935
ffmpeg -re -stream_loop -1 -i $HOME/Code/cursor-projects/aka_music/backend/app/static/def/jieba_chunli.mp4 -c copy -f flv rtmp://127.0.0.1:1935/live/testhaogui

单独推送流到127.0.0.1:1936
ffmpeg -re -stream_loop -1 -i $HOME/Code/cursor-projects/aka_music/backend/app/static/def/jieba_haogui.mp4 -c copy -f flv rtmp://127.0.0.1:1936/live/testhaogui

同时推送两个流过去
ffmpeg -re -stream_loop -1 -i $HOME/Code/cursor-projects/aka_music/backend/app/static/def/jieba_haogui.mp4 -c copy -f flv rtmp://127.0.0.1:1936/live/testhaogui -c copy -f flv rtmp://127.0.0.1:1935/live/testhaogui

rtmp://http://alphago.ltd:1935/live/test

docker run -d --name nginx-hls-3 -p 1937:1935 -p 8082:8080 -v $HOME/Code/cursor-projects/aka_music/backend/app/resource/01_nginx_config_for_rtmp_v2.conf:/etc/nginx/nginx.conf alqutami/rtmp-hls

ffmpeg -re -stream_loop -1 -i $HOME/Code/cursor-projects/aka_music/backend/app/static/def/jieba_haogui.mp4 -c copy -f flv rtmp://127.0.0.1:1937/live/testhaogui
```

```shell
构建普通的nginx服务进行反向代理

docker-compose -f docker-compose-proxy-rtmp-v2.yml up -d

# 文件名：docker-compose.yml
# version: '3'
services:
  nginx-rtmp:
    image: nginx:latest  # 
    # image: alqutami/rtmp-hls # 使用支持RTMP的镜像
    container_name: nginx-rtmp-proxy
    ports:
      - "1934:1934"    # 映射RTMP端口到宿主机
      # - "8090:8090"        # 映射HTTP健康检查端口（可选）
    volumes:
      # - ./conf.d/nginx-rtmp-proxy.conf:/etc/nginx/nginx.conf
      # - ./conf.d:/etc/nginx/conf.d
      # - ./conf.d/nginx.conf:/etc/nginx/conf.d/rtmp-nginx.conf
      - ./conf.d/nginx.conf:/etc/nginx/nginx.conf
      - ./logs:/var/log/nginx       # 挂载日志目录
    networks:
      - default
    restart: unless-stopped

一定一定注意nginx.conf的配置，否则nginx无法启动，这里面映射的是nginx.conf的文件，不是映射文件夹

并且本来将该文件映射到/etc/nginx/conf.d的某个文件的配置之后，nginx能够将其作为特殊配置进行加载的，但是实际情况并不是这样，nginx能够识别该文件，但是里面配置的内容与其加载的顺序有关系，导致其对其中的指令无法识别，和其加载顺序有关！（导致指令加载的层次错误） ---- 这种错误简直是灾难
```