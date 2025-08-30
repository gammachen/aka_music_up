


```bash
nginx中配置不同的代理路径的不同表现：

# WebSocket支持
    location /api/meeting/socket.io {
        proxy_pass https://127.0.0.1:5000/;
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto https;
    }

    # WebSocket支持
    location /api/meeting/socket.io/ {
        proxy_pass https://127.0.0.1:5000/;
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto https;
    }

112.10.202.62,127.0.0.1 - - [13/Mar/2025 15:33:55] "GET //?EIO=4&transport=websocket HTTP/1.1" 404 331 0.000609

多了一个/

(aka_music) root@iZbp1cqdx0u3g3wriko37cZ:~/aka_music/backend# vi /etc/nginx/sites-available/aka_music_https
(aka_music) root@iZbp1cqdx0u3g3wriko37cZ:~/aka_music/backend# systemctl restart nginx
(aka_music) root@iZbp1cqdx0u3g3wriko37cZ:~/aka_music/backend# tail -f nohup.out

(508036) accepted ('127.0.0.1', 33766)
112.10.202.62,127.0.0.1 - - [13/Mar/2025 15:35:18] "GET /?EIO=4&transport=websocket HTTP/1.1" 404 331 0.000564
(508036) accepted ('127.0.0.1', 54908)

理论上这个才是正常的

但是结果却是：两个都是404的结果，没有正确的路由到我们的后端使用socketio启动的wsgi启动的flask应用

结合vite中的proxy来看：
    '/api/meeting/socket.io': {
        target: process.env.VITE_API_URL || 'https://127.0.0.1:5000',
        changeOrigin: true,
        ws: true,
        secure: false
      }


```