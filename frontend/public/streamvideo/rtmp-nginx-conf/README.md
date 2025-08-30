ffmpeg -i rtmp://localhost:1938/live/testhaogui \
                        -c:v libx264 -b:v 2500k -s 1280x720 \
                        -c:a aac -b:a 128k \
                        -f dash -seg_duration 3 -window_size 5 -use_template 1 -use_timeline 0 \
                        -init_seg_name testhaogui-init.m4s -media_seg_name testhaogui-%d.m4s \
                        -dash_segment_type mp4 -adaptation_sets "id=0,streams=v id=1,streams=a" \
                        testhaogui.mpd


ffmpeg -i rtmp://localhost:1935/live/testhaogui \
                        -async 1 -vsync -1 -c:v libx264 -b:v 2500k -s 1280x720 \
                        -c:a aac -b:a 128k \
                        -f dash -seg_duration 3 -window_size 5 -use_template 1 -use_timeline 0 \
                        -init_seg_name testhaogui-init.m4s -media_seg_name testhaogui-%d.m4s \
                        -dash_segment_type mp4 -adaptation_sets "id=0,streams=v id=1,streams=a" \
                        testhaogui.mpd

ffmpeg -fflags +genpts -analyzeduration 10M -probesize 10M \
  -i rtmp://localhost:1935/live/testhaogui \
  -async 1 -vsync -1 -c:v libx264 -preset ultrafast -tune zerolatency -b:v 2500k -s 1280x720 \
  -c:a aac -b:a 128k -ar 44100 \
  -f dash -seg_duration 3 -window_size 5 -use_template 1 \
  -init_seg_name testhaogui-init.m4s -media_seg_name testhaogui-%d.m4s \
  -max_muxing_queue_size 1024 -avioflags direct \
  -adaptation_sets "id=0,streams=v id=1,streams=a" \
  /mnt/dash/testhaogui.mpd


ffmpeg -i rtmp://localhost:1938/live/testhaogui 
-c:v libx264 -preset veryfast -profile:v baseline -level 3.0 
-c:a aac -b:a 128k 
-f dash -window_size 5 -extra_window_size 3 
-remove_at_exit 1 
manifest.mpd;

ffmpeg -i rtmp://192.168.31.109:1938/live/testhaogui 
-c:v libx264 -preset veryfast -profile:v baseline -level 3.0 
-c:a aac -b:a 128k 
-f dash -window_size 5 -extra_window_size 3 
-remove_at_exit 1 
manifest.mpd

ffmpeg -i rtmp://192.168.31.109:1938/live/testhaogui 
  -map 0:v:0 -c:v:0 libx264 -b:v:0 2500k -s:v:0 1280x720 
  -map 0:v:0 -c:v:1 libx264 -b:v:1 1000k -s:v:1 640x360 
  -map 0:a:0 -c:a aac -b:a 128k 
  -f dash -adaptation_sets "id=0,streams=v id=1,streams=a" 
testhaogui.mpd;


ffmpeg -i input.mp4 \
  -map 0:v:0 -c:v libx264 -b:v 2500k -vf "scale=1280:720" -r 30 -g 60 -keyint_min 60 \
  -map 0:a:0 -c:a aac -b:a 128k -ar 44100 \
  -f dash \
  -seg_duration 4 \
  -window_size 5 \
  -use_template 1 \
  -use_timeline 0 \
  -init_seg_name "\$RepresentationID\$-init.mp4" \
  -media_seg_name "\$RepresentationID\$-\$Number%05d\$.mp4" \
  -adaptation_sets "id=0,streams=v id=1,streams=a" \
  -max_muxing_queue_size 1024 \
  -avioflags direct \
  output.mpd


nginx的几个目录：
/usr/local/nginx

/mnt/dash
/mnt/hls

ffmpeg -re -stream_loop -1 -i $HOME/Code/cursor-projects/aka_music/backend/app/static/def/jieba_haogui.mp4 -c copy -f flv rtmp://127.0.0.1:1938/live/testhaogui

ffmpeg -i $HOME/Code/cursor-projects/aka_music/backend/app/static/def/jieba_haogui.mp4     -c:v libx264 -b:v 3M -g 60 -keyint_min 60     -c:a aac -b:a 128k     -f dash     -seg_duration 4     -use_template 1     -use_timeline 1     -init_seg_name init-\$RepresentationID\$.mp4     -media_seg_name chunk-\$RepresentationID\$-\$Number%05d\$.mp4     dashdata/testhaogui.mpd

ffmpeg -i $HOME/Code/cursor-projects/aka_music/backend/app/static/def/jieba_haogui.mp4 \
    -map 0:v:0 -map 0:a:0 \
    # 视频流1（高清）
    -c:v:0 libx264 \
    -b:v:0 3000k -maxrate:v:0 3000k -bufsize:v:0 6000k \
    -vf "scale=1280:720" \
    -profile:v:0 high -preset slower \
    -crf 23 \
    -g 120 -keyint_min 120 \
    -c:v:1 libx264 \
    -b:v:1 1500k -maxrate:v:1 1500k -bufsize:v:1 3000k \
    -vf "scale=854:480" \
    -profile:v:1 main -preset medium \
    -crf 25 \
    -g 120 -keyint_min 120 \
    -c:a aac -b:a 128k \
    -f dash \
    -seg_duration 4 \
    -adaptation_sets "id=0,streams=v id=1,streams=a" \
    -use_template 1 \
    -use_timeline 1 \
    dashdata/testhaogui_high.mpd


ffprobe -i $HOME/Code/cursor-projects/aka_music/backend/app/static/def/jieba_haogui.mp4 -show_streams -select_streams v:0 -show_entries stream=width,height:format=bit_rate -v quiet -of json