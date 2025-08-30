## 直播、推流（Index）

<a href="/rtc/about_architecture.html">架构</a>

<a href="/rtc/about_video.html">关于音频</a>

<a href="/rtc/5-2_音视频编码格式简介.html">5-2_音视频编码格式简介</a>

<a href="/rtc/5-3_音视频封装格式.html">5-3_音视频封装格式</a>

<a href="/rtc/5-4_流媒体协议.html">5-4_流媒体协议</a>

<a href="/rtc/5-5_直播业务介绍.html">5-5_直播业务介绍</a>

<a href="/rtc/5-6_一个直播的整个流程及背后的故事.html">5-6_一个直播的整个流程及背后的故事</a>

<a href="/rtc/5-7_本章知识点总结.html">知识点总结</a>


pip install opencv-python numpy kivy -i http://pypi.douban.com/simple --trusted-host pypi.douban.com --trusted-host files.pythonhosted.org

pip install opencv-python numpy kivy -i https://pypi.tuna.tsinghua.edu.cn/simple

pip install opencv-python numpy kivy -i http://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn

ffmpeg -re -stream_loop -1 -i  -c copy -f flv rtmp://127.0.0.1:1935/live/test

ffmpeg -re -stream_loop -1 -i $HOME/Code/cursor-projects/aka_music/backend/app/static/def/jieba_chunli.mp4 -c copy -f flv rtmp://127.0.0.1:1935/live/test

ffmpeg -re -stream_loop -1 -i VID_20250521_212312.mp4 -c copy -f flv rtmp://127.0.0.1:1935/live/test

ffmpeg -re -stream_loop -1 -i VID_20250521_212312.mp4 -c copy -f flv rtmp://192.168.31.150:1935/live/test

ffmpeg \
    -f jpeg_pipe -i <(termux-camera-photo -c 0 - | while :; do cat; done) \
    -framerate 30 \
    -vf "scale=640:480,format=yuv420p" \
    -c:v libx264 \
    -preset ultrafast \
    -tune zerolatency \
    -g 60 \
    -b:v 1000k \
    -f flv \
    rtmp://192.168.31.150:1935/live/test

ffmpeg -f avfoundation -i "0:0" \
-vcodec libx264 -preset ultrafast -tune zerolatency -b:v 2048k -pix_fmt yuv420p -g 25 -r 30 \
-acodec aac -b:a 128k \
-f flv rtmp://192.168.31.150:1935/live/test

(base) shhaofu@shhaofudeMacBook-Pro Downloads % ffmpeg -f avfoundation -list_devices true -i ""
ffmpeg version 7.1.1 Copyright (c) 2000-2025 the FFmpeg developers
  built with Apple clang version 16.0.0 (clang-1600.0.26.6)
  configuration: --prefix=/opt/homebrew/Cellar/ffmpeg/7.1.1_1 --enable-shared --enable-pthreads --enable-version3 --cc=clang --host-cflags= --host-ldflags='-Wl,-ld_classic' --enable-ffplay --enable-gnutls --enable-gpl --enable-libaom --enable-libaribb24 --enable-libbluray --enable-libdav1d --enable-libharfbuzz --enable-libjxl --enable-libmp3lame --enable-libopus --enable-librav1e --enable-librist --enable-librubberband --enable-libsnappy --enable-libsrt --enable-libssh --enable-libsvtav1 --enable-libtesseract --enable-libtheora --enable-libvidstab --enable-libvmaf --enable-libvorbis --enable-libvpx --enable-libwebp --enable-libx264 --enable-libx265 --enable-libxml2 --enable-libxvid --enable-lzma --enable-libfontconfig --enable-libfreetype --enable-frei0r --enable-libass --enable-libopencore-amrnb --enable-libopencore-amrwb --enable-libopenjpeg --enable-libspeex --enable-libsoxr --enable-libzmq --enable-libzimg --disable-libjack --disable-indev=jack --enable-videotoolbox --enable-audiotoolbox --enable-neon
  libavutil      59. 39.100 / 59. 39.100
  libavcodec     61. 19.101 / 61. 19.101
  libavformat    61.  7.100 / 61.  7.100
  libavdevice    61.  3.100 / 61.  3.100
  libavfilter    10.  4.100 / 10.  4.100
  libswscale      8.  3.100 /  8.  3.100
  libswresample   5.  3.100 /  5.  3.100
  libpostproc    58.  3.100 / 58.  3.100
2025-05-30 20:52:15.451 ffmpeg[99928:6707010] WARNING: Add NSCameraUseContinuityCameraDeviceType to your Info.plist to use AVCaptureDeviceTypeContinuityCamera.
[AVFoundation indev @ 0x131904820] AVFoundation video devices:
[AVFoundation indev @ 0x131904820] [0] FaceTime高清摄像头
[AVFoundation indev @ 0x131904820] [1] Capture screen 0
[AVFoundation indev @ 0x131904820] [2] Capture screen 1
[AVFoundation indev @ 0x131904820] AVFoundation audio devices:
[AVFoundation indev @ 0x131904820] [0] MacBook Pro麦克风

ffmpeg \
-f avfoundation \
-i video="FaceTime高清摄像头" \
-vcodec libx264 \
-preset ultrafast \
-tune zerolatency \
-b:v 2048k \
-pix_fmt yuv420p \
-g 25 \
-r 30 \
-f flv \
rtmp://192.168.31.150:1935/live/test

ffmpeg \
-f avfoundation \
-i "video=FaceTime高清摄像头:audio=MacBook Pro麦克风" \
-vcodec libx264 \
-preset ultrafast \
-tune zerolatency \
-b:v 2048k \
-pix_fmt yuv420p \
-r 30 \
-g 30 \
-acodec aac \
-b:a 128k \
-f flv \
rtmp://192.168.31.150:1935/live/test

ffmpeg \
-f avfoundation \
-i "" \
-vcodec libx264 \
-preset ultrafast \
-tune zerolatency \
-b:v 2048k \
-pix_fmt yuv420p \
-r 30 \
-g 30 \
-acodec aac \
-b:a 128k \
-f flv \
rtmp://192.168.31.150:1935/live/test

ffmpeg \
-f avfoundation \
-i "0:0" \
-vcodec libx264 \
-preset ultrafast \
-tune zerolatency \
-b:v 2048k \
-pix_fmt yuv420p \
-r 30 \
-g 30 \
-acodec aac \
-b:a 128k \
-f flv \
rtmp://192.168.31.150:1935/live/test

ffmpeg \
-f rawvideo \
-i "0:0" \
-vcodec libx264 \
-preset ultrafast \
-tune zerolatency \
-b:v 2048k \
-pix_fmt yuv420p \
-r 30 \
-g 30 \
-acodec aac \
-b:a 128k \
-f flv \
rtmp://192.168.31.150:1935/live/test

'ffmpeg',
        '-y',
        '-f', 'rawvideo',
        '-vcodec', 'rawvideo',
        '-pix_fmt', 'bgr24',
        '-s', f'{WIDTH}x{HEIGHT}',
        '-r', str(FPS),
        '-i', '-',
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-preset', 'ultrafast',
        '-tune', 'zerolatency',
        '-g', str(FPS * 2),  # 关键帧间隔(2秒)
        '-b:v', '500k',      # 降低比特率
        '-f', 'flv',
        RTMP_URL