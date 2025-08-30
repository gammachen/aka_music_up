

```shell
from frontend request:
Error: connect ECONNREFUSED ::1:5000
    at TCPConnectWrap.afterConnect [as oncomplete] (node:net:1555:16) (x4)

 This error message indicates that the frontend is trying to connect to the backend server on port 5000, but the backend server is not listening on that port.

from backend response:
 a production deployment. Use a production WSGI server instead.
 * Running on all addresses (0.0.0.0)
 * Running on http://127.0.0.1:5000
 * Running on http://192.168.31.108:5000
2025-03-10 08:56:51 - INFO - Press CTRL+C to quit
2025-03-10 08:56:51 - INFO -  * Restarting with stat
2025-03-10 08:56:52 - WARNING -  * Debugger is active!
2025-03-10 08:56:52 - INFO -  * Debugger PIN: 436-366-312
2025-03-10 08:57:17 - INFO - 127.0.0.1 - - [10/Mar/2025 08:57:17] "GET /api/recommend/online_musics HTTP/1.1" 200 -
2025-03-10 09:00:35 - INFO - 127.0.0.1 - - [10/Mar/2025 09:00:35] "GET /static/def/a8.png HTTP/1.1" 200 -
2025-03-10 09:00:35 - INFO - 127.0.0.1 - - [10/Mar/2025 09:00:35] "GET /favicon.ico HTTP/1.1" 404 -
2025-03-10 09:03:34 - INFO - 127.0.0.1 - - [10/Mar/2025 09:03:34] "GET /api/recommend/online_musics HTTP/1.1" 200 -

```

```html
Request URL:
http://127.0.0.1:5173/static/def/a1.png
Request Method:
GET
Status Code:
500 Internal Server Error
Remote Address:
127.0.0.1:5173
Referrer Policy:
strict-origin-when-cross-origin
```

比较奇怪：backend server 运行在 5000 端口，前端运行在 5173 端口，为什么前端请求 backend server 5000 端口会报错？

开发环境中我们是通过vite.config.ts中的proxy来转发/static的请求到后端的5000端口的

```js

```

