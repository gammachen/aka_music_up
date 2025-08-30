# AKA Music 项目部署文档

## 1. 服务器环境准备

### 1.1 基础系统包安装
```bash
# 更新系统包
sudo apt update
sudo apt upgrade -y

# 安装基础工具
sudo apt install -y build-essential git curl wget rsync screen nginx
```

### 1.2 安装 Node.js 环境
```bash
# 使用 nvm 安装 Node.js
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash

```bash
root@iZbp1cqdx0u3g3wriko37cZ:~# curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash
  % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                 Dload  Upload   Total   Spent    Left  Speed
100 14984  100 14984    0     0  18768      0 --:--:-- --:--:-- --:--:-- 18753
=> Downloading nvm from git to '/root/.nvm'
=> Cloning into '/root/.nvm'...
remote: Enumerating objects: 381, done.
remote: Counting objects: 100% (381/381), done.
remote: Compressing objects: 100% (324/324), done.
remote: Total 381 (delta 43), reused 176 (delta 29), pack-reused 0 (from 0)
Receiving objects: 100% (381/381), 383.82 KiB | 1.08 MiB/s, done.
Resolving deltas: 100% (43/43), done.
* (HEAD detached at FETCH_HEAD)
  master
=> Compressing and cleaning up git repository

=> Appending nvm source string to /root/.bashrc
=> Appending bash_completion source string to /root/.bashrc
=> Close and reopen your terminal to start using nvm or run the following to use it now:

export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"  # This loads nvm
[ -s "$NVM_DIR/bash_completion" ] && \. "$NVM_DIR/bash_completion"  # This loads nvm bash_completion

# 重新加载 shell 配置
export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"

# 安装 Node.js LTS 版本
nvm install --lts
nvm use --lts

# 验证安装
node --version
npm --version
```

```bash
root@iZbp1cqdx0u3g3wriko37cZ:~# npm --version
10.9.2
root@iZbp1cqdx0u3g3wriko37cZ:~# node --version
v22.14.0
```

### 1.3 安装 Python 环境
```bash
# 安装 Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b

# 初始化 conda
~/miniconda3/bin/conda init bash

# 重新加载 shell 配置
source ~/.bashrc

# 创建项目虚拟环境
conda create -n aka_music python=3.11 -y
conda activate aka_music
```

```bash
root@iZbp1cqdx0u3g3wriko37cZ:~# ~/miniconda3/bin/conda init bash
no change     /root/miniconda3/condabin/conda
no change     /root/miniconda3/bin/conda
no change     /root/miniconda3/bin/conda-env
no change     /root/miniconda3/bin/activate
no change     /root/miniconda3/bin/deactivate
no change     /root/miniconda3/etc/profile.d/conda.sh
no change     /root/miniconda3/etc/fish/conf.d/conda.fish
no change     /root/miniconda3/shell/condabin/Conda.psm1
no change     /root/miniconda3/shell/condabin/conda-hook.ps1
no change     /root/miniconda3/lib/python3.12/site-packages/xontrib/conda.xsh
no change     /root/miniconda3/etc/profile.d/conda.csh
modified      /root/.bashrc

==> For changes to take effect, close and re-open your current shell. <==
```

## 2. 代码同步

### 2.1 创建 rsync 排除文件
在本地项目根目录创建 `.rsyncignore` 文件：
```
.git/
.history/
node_modules/
__pycache__/
*.pyc
.env
venv/
.vscode/
.idea/
```

### 2.2 同步代码到服务器
```bash
# 在本地执行，将代码同步到服务器
# 请替换 {YOUR_SERVER_IP} 为实际的服务器IP
rsync -avz --exclude-from='.rsyncignore' ./ root@{YOUR_SERVER_IP}:/root/aka_music/

rsync -avz --exclude-from='.rsyncignore' ./ root@47.98.62.98:/root/aka_music/
```

```bash
ECS的监控：
https://cloudmonitor.console.aliyun.com/hostMonitoring/details/i-bp1cqdx0u3g3wriko37c?product=acs_ecs_dashboard&key=Basemonitoring
```

## 3. 前端部署

### 3.1 开发环境配置
```bash
# 在服务器上执行
cd /root/aka_music/frontend

# 安装依赖
npm install

# 启动开发服务器
npm run dev
```

开发环境下，Vite的代理配置（vite.config.ts）已经设置了对`/api`和`/static`的代理转发到后端服务，无需额外的Nginx配置。

### 3.2 生产环境部署
```bash
# 在服务器上执行
cd /root/aka_music/frontend

# 安装依赖（如果未安装）
npm install
```

```bash
(base) root@iZbp1cqdx0u3g3wriko37cZ:~/aka_music/frontend# npm install
npm warn deprecated rimraf@3.0.2: Rimraf versions prior to v4 are no longer supported
npm warn deprecated lodash.isequal@4.5.0: This package is deprecated. Use require('node:util').isDeepStrictEqual instead.
npm warn deprecated inflight@1.0.6: This module is not supported, and leaks memory. Do not use it. Check out lru-cache if you want a good and tested way to coalesce async requests by a key value, which is much more comprehensive and powerful.
npm warn deprecated glob@7.2.3: Glob versions prior to v9 are no longer supported
npm warn deprecated @humanwhocodes/object-schema@2.0.3: Use @eslint/object-schema instead
npm warn deprecated @humanwhocodes/config-array@0.13.0: Use @eslint/config-array instead
npm warn deprecated eslint@8.57.1: This version is no longer supported. Please see https://eslint.org/version-support for other options.

added 441 packages, and audited 442 packages in 17s

72 packages are looking for funding
  run `npm fund` for details

2 moderate severity vulnerabilities

To address all issues, run:
  npm audit fix

Run `npm audit` for details.
npm notice
npm notice New major version of npm available! 10.9.2 -> 11.1.0
npm notice Changelog: https://github.com/npm/cli/releases/tag/v11.1.0
npm notice To update run: npm install -g npm@11.1.0
npm notice
```

# 构建生产环境代码
npm run build
```bash
(base) root@iZbp1cqdx0u3g3wriko37cZ:~/aka_music/frontend# npm run build

> frontend@0.0.0 build
> vite build

vite v6.1.0 building for production...
✓ 1169 modules transformed.
x Build failed in 5.70s
error during build:
[vite]: Rollup failed to resolve import "element-plus/es/locale" from "/root/aka_music/frontend/src/views/comic/ComicDetail.vue?vue&type=script&setup=true&lang.ts".
This is most likely unintended because it can break your application at runtime.
If you do want to externalize this module explicitly add it to
`build.rollupOptions.external`
    at viteLog (file:///root/aka_music/frontend/node_modules/vite/dist/node/chunks/dep-CfG9u7Cn.js:51487:15)
    at onRollupLog (file:///root/aka_music/frontend/node_modules/vite/dist/node/chunks/dep-CfG9u7Cn.js:51537:5)
    at onLog (file:///root/aka_music/frontend/node_modules/vite/dist/node/chunks/dep-CfG9u7Cn.js:51185:7)
    at file:///root/aka_music/frontend/node_modules/rollup/dist/es/shared/node-entry.js:20487:32
    at Object.logger [as onLog] (file:///root/aka_music/frontend/node_modules/rollup/dist/es/shared/node-entry.js:22230:9)
    at ModuleLoader.handleInvalidResolvedId (file:///root/aka_music/frontend/node_modules/rollup/dist/es/shared/node-entry.js:21113:26)
    at file:///root/aka_music/frontend/node_modules/rollup/dist/es/shared/node-entry.js:21071:26
```

```bash
(base) root@iZbp1cqdx0u3g3wriko37cZ:~/aka_music/frontend# npm run build

> frontend@0.0.0 build
> vite build

vite v6.1.0 building for production...
✓ 1213 modules transformed.
x Build failed in 5.68s
error during build:
Could not resolve "../assets/adb-1.webp" from "src/views/music/Mulist.vue?vue&type=script&setup=true&lang.ts"
file: /root/aka_music/frontend/src/views/music/Mulist.vue?vue&type=script&setup=true&lang.ts
    at getRollupError (file:///root/aka_music/frontend/node_modules/rollup/dist/es/shared/parseAst.js:397:41)
    at error (file:///root/aka_music/frontend/node_modules/rollup/dist/es/shared/parseAst.js:393:42)
    at ModuleLoader.handleInvalidResolvedId (file:///root/aka_music/frontend/node_modules/rollup/dist/es/shared/node-entry.js:21111:24)
    at file:///root/aka_music/frontend/node_modules/rollup/dist/es/shared/node-entry.js:21071:26
```

```bash
(base) root@iZbp1cqdx0u3g3wriko37cZ:~/aka_music/frontend# npm run build

> frontend@0.0.0 build
> vite build

vite v6.1.0 building for production...
✓ 1213 modules transformed.
x Build failed in 5.65s
error during build:
Could not resolve "../assets/adb-1.webp" from "src/views/music/Mugrid.vue?vue&type=script&setup=true&lang.ts"
file: /root/aka_music/frontend/src/views/music/Mugrid.vue?vue&type=script&setup=true&lang.ts
    at getRollupError (file:///root/aka_music/frontend/node_modules/rollup/dist/es/shared/parseAst.js:397:41)
    at error (file:///root/aka_music/frontend/node_modules/rollup/dist/es/shared/parseAst.js:393:42)
    at ModuleLoader.handleInvalidResolvedId (file:///root/aka_music/frontend/node_modules/rollup/dist/es/shared/node-entry.js:21111:24)
    at file:///root/aka_music/frontend/node_modules/rollup/dist/es/shared/node-entry.js:21071:26
```

```bash
(base) root@iZbp1cqdx0u3g3wriko37cZ:~/aka_music/frontend# npm run build

> frontend@0.0.0 build
> vite build

vite v6.1.0 building for production...
transforming (3739) node_modules/element-plus/es/components/table/src/filter-panel.mjs
/static/def/images/hero-bg.jpg referenced in /static/def/images/hero-bg.jpg didn't resolve at build time, it will remain unchanged to be resolved at runtime
✓ 4105 modules transformed.
dist/index.html                                              0.50 kB │ gzip:   0.32 kB
dist/assets/alipay-D6T3nOZF.png                              7.98 kB
dist/assets/ad1-DOVhyQZL.png                                10.44 kB
dist/assets/logo-DftGc78M.png                               12.35 kB
dist/assets/ad2-DnL0QlPU.png                                17.27 kB
dist/assets/qrcode-DWGss9-I.png                             33.18 kB
dist/assets/wechat-S2qlHaDh.png                             39.51 kB
dist/assets/sky-CzVYdHjI.webp                               41.18 kB
dist/assets/adb-CHJdcgOw.png                                51.48 kB
dist/assets/phone-D_4qGpZG.webp                             55.97 kB
dist/assets/adb-1-D18y_yML.webp                             81.92 kB
dist/assets/adb-3-Db_Q06fb.webp                             95.49 kB
dist/assets/adb-2-REc0F20D.webp                             96.91 kB
dist/assets/tooltip-animation-compressed-D3Z8y5aB.mp4      507.39 kB
dist/assets/background-removal-tooltip-B1FzNs4D.mp4        992.02 kB
dist/assets/login-banner-2-XST3Qqjy.jpg                  1,368.97 kB
dist/assets/video_2--fs5dBWP.mp4                        10,135.39 kB
dist/assets/MyTopics-6Ft28BQ2.css                            0.05 kB │ gzip:   0.07 kB
dist/assets/MyPoints-C_vNO3X7.css                            0.05 kB │ gzip:   0.07 kB
dist/assets/MyMessages-BiNhNpnS.css                          0.05 kB │ gzip:   0.07 kB
dist/assets/MyRecharge-BRX7wraq.css                          0.05 kB │ gzip:   0.07 kB
dist/assets/MyFavorites-CjwIC0mH.css                         0.05 kB │ gzip:   0.07 kB
dist/assets/MyStatistics-B0C-HRjR.css                        0.41 kB │ gzip:   0.23 kB
dist/assets/AdCarousel-L7blsXeY.css                          0.81 kB │ gzip:   0.38 kB
dist/assets/PaymentSuccess-QCDskm2X.css                      0.91 kB │ gzip:   0.43 kB
dist/assets/Profile-CFe6w38d.css                             0.97 kB │ gzip:   0.40 kB
dist/assets/Register-BkgQ0tKQ.css                            1.14 kB │ gzip:   0.39 kB
dist/assets/Topics-Bx9txZup.css                              1.35 kB │ gzip:   0.53 kB
dist/assets/Recharge-BcCqTmL7.css                            1.42 kB │ gzip:   0.50 kB
dist/assets/Login-DqbJdfy-.css                               1.44 kB │ gzip:   0.50 kB
dist/assets/Home-D-HbmkZ8.css                                1.97 kB │ gzip:   0.66 kB
dist/assets/DiagonalSplitCharacterGallery-oL1MzDv_.css       3.30 kB │ gzip:   0.86 kB
dist/assets/AlbumDetail-DZasvtB4.css                         3.42 kB │ gzip:   0.93 kB
dist/assets/ComicDetail-DbsTnHC4.css                         3.55 kB │ gzip:   1.01 kB
dist/assets/HeroSearch-BcmRmc8J.css                          4.15 kB │ gzip:   1.21 kB
dist/assets/TopicDetail-wXxONzml.css                         4.80 kB │ gzip:   1.27 kB
dist/assets/AppPromotion-20xSWPqJ.css                        6.08 kB │ gzip:   1.50 kB
dist/assets/BeautyLanding-BdhSuLvj.css                       7.08 kB │ gzip:   1.65 kB
dist/assets/Landing-CAdN8QFG.css                             8.16 kB │ gzip:   1.91 kB
dist/assets/Mugrid-DRWCm_NF.css                              8.16 kB │ gzip:   1.91 kB
dist/assets/ComicLanding-CTvgQSG2.css                        8.16 kB │ gzip:   1.91 kB
dist/assets/ComicGenre-Dp0Wbc9t.css                          8.45 kB │ gzip:   1.99 kB
dist/assets/Genre-BTrSqlDp.css                               8.62 kB │ gzip:   2.05 kB
dist/assets/Mulist-C0hUAmLL.css                              9.23 kB │ gzip:   2.05 kB
dist/assets/CreateTopic-D56MSq-6.css                        15.94 kB │ gzip:   3.16 kB
dist/assets/index-CtB8LF8q.css                             337.41 kB │ gzip:  47.12 kB
dist/assets/music-BoLWMq_I.js                                0.18 kB │ gzip:   0.16 kB
dist/assets/video_2-Du6Npi_L.js                              0.24 kB │ gzip:   0.18 kB
dist/assets/comic-BAKg20qs.js                                0.56 kB │ gzip:   0.26 kB
dist/assets/AdCarousel-Ops5ACNR.js                           0.67 kB │ gzip:   0.44 kB
dist/assets/MyMessages-CgaNib9v.js                           1.09 kB │ gzip:   0.73 kB
dist/assets/MyTopics-y6kqHTyy.js                             1.10 kB │ gzip:   0.74 kB
dist/assets/request-CHlkatbA.js                              1.38 kB │ gzip:   0.76 kB
dist/assets/MyFavorites-xLC8qVpX.js                          1.38 kB │ gzip:   0.88 kB
dist/assets/MyPoints-CEm2fIgA.js                             1.38 kB │ gzip:   0.90 kB
dist/assets/HeartOutlined-Dug1fNHA.js                        1.57 kB │ gzip:   0.93 kB
dist/assets/LikeOutlined-B0kaaqDp.js                         1.62 kB │ gzip:   0.89 kB
dist/assets/MessageOutlined-C4hHiLME.js                      1.69 kB │ gzip:   0.97 kB
dist/assets/PaymentSuccess-COjfAZt6.js                       1.74 kB │ gzip:   1.00 kB
dist/assets/MyRecharge-CL8A2oAQ.js                           1.94 kB │ gzip:   1.23 kB
dist/assets/MyStatistics--EHeKQWk.js                         2.03 kB │ gzip:   0.93 kB
dist/assets/WalletOutlined-Z1ZrwVNc.js                       2.24 kB │ gzip:   0.97 kB
dist/assets/auth-DQ6CsakJ.js                                 2.58 kB │ gzip:   1.08 kB
dist/assets/Home-DHXIw-VV.js                                 2.63 kB │ gzip:   1.52 kB
dist/assets/Genre-DEcKOn8F.js                                3.31 kB │ gzip:   1.40 kB
dist/assets/Topics-Elo5-WIl.js                               3.54 kB │ gzip:   1.84 kB
dist/assets/category-CCwik_RV.js                             3.79 kB │ gzip:   1.60 kB
dist/assets/BeautyLanding-DppQxVTK.js                        4.11 kB │ gzip:   1.68 kB
dist/assets/Recharge-eVlFefb7.js                             4.75 kB │ gzip:   2.62 kB
dist/assets/qrcode-BkrCSOM9.js                               4.80 kB │ gzip:   2.02 kB
dist/assets/Profile-Dld66BGn.js                              7.60 kB │ gzip:   2.74 kB
dist/assets/Register-DMg-ocCP.js                             7.67 kB │ gzip:   2.75 kB
dist/assets/Landing-18Vsb_za.js                              8.68 kB │ gzip:   3.17 kB
dist/assets/ComicGenre-eZB5TU-2.js                           8.70 kB │ gzip:   3.03 kB
dist/assets/Mugrid-BCCTQ2CG.js                               8.86 kB │ gzip:   3.31 kB
dist/assets/Mulist-DHvbouMr.js                               8.93 kB │ gzip:   3.32 kB
dist/assets/AlbumDetail-p2-a2sR4.js                          9.02 kB │ gzip:   4.19 kB
dist/assets/ComicDetail-06bQMFJP.js                          9.76 kB │ gzip:   4.48 kB
dist/assets/Login-0P2Lq90L.js                               10.08 kB │ gzip:   3.83 kB
dist/assets/ComicLanding-Cy4G7Hrz.js                        12.56 kB │ gzip:   4.54 kB
dist/assets/TopicDetail-tbZwskD4.js                         14.79 kB │ gzip:   5.82 kB
dist/assets/HeroSearch-CNeUPVxz.js                         199.97 kB │ gzip:  49.06 kB
dist/assets/CreateTopic-Qln7MI2_.js                        816.30 kB │ gzip: 285.34 kB
dist/assets/index-DLgJt36u.js                            2,782.72 kB │ gzip: 872.80 kB

(!) Some chunks are larger than 500 kB after minification. Consider:
- Using dynamic import() to code-split the application
- Use build.rollupOptions.output.manualChunks to improve chunking: https://rollupjs.org/configuration-options/#output-manualchunks
- Adjust chunk size limit for this warning via build.chunkSizeWarningLimit.
✓ built in 24.01s
```



### 3.3 配置 Nginx（生产环境）
```bash
# 创建 Nginx 配置文件
sudo vim /etc/nginx/sites-available/aka_music
```

添加以下配置：
```nginx
server {
    listen 80;
    server_name alphago.ltd;  # 替换为实际域名

    # 前端静态文件
    location / {
        root /root/aka_music/frontend/dist;
        try_files $uri $uri/ /index.html;
        # 添加缓存配置
        expires 1d;
        add_header Cache-Control "public, no-cache";
    }

    # 静态资源缓存配置
    #location /assets/ {
    #    root /root/aka_music/frontend/dist/assets;
    #    expires 7d;
    #    add_header Cache-Control "public, no-cache";
    #}

    # 后端 API 代理（生产环境可选，如果前端构建时已配置正确的API地址则不需要）
    # 注意：在开发环境中，这个代理是由Vite的devServer配置处理的
    location /api/ {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }

    # 静态资源static代理
    location /static/ {
        proxy_pass http://127.0.0.1:5000/static/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }

    # graphql代理
    location /graphql/ {
        proxy_pass http://127.0.0.1:11800/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```
```bash
注意nginx默认的default配置项可能对工程或者项目的影响，最好是手动删除，或者脚本删除
```

```bash
# 创建符号链接并重启 Nginx
sudo ln -s /etc/nginx/sites-available/aka_music /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

```bash
(base) root@iZbp1cqdx0u3g3wriko37cZ:/etc/nginx/sites-available# sasystemctl status nginx.service
sasystemctl: command not found
(base) root@iZbp1cqdx0u3g3wriko37cZ:/etc/nginx/sites-available# systemctl status nginx.service
× nginx.service - A high performance web server and a reverse proxy server
     Loaded: loaded (/usr/lib/systemd/system/nginx.service; enabled; preset: enabled)
     Active: failed (Result: exit-code) since Mon 2025-03-03 15:19:08 CST; 26s ago
   Duration: 25min 41.642s
       Docs: man:nginx(8)
    Process: 34378 ExecStartPre=/usr/sbin/nginx -t -q -g daemon on; master_process on; (code=exited, status=1/FAILURE)
        CPU: 4ms

Mar 03 15:19:08 iZbp1cqdx0u3g3wriko37cZ systemd[1]: Starting nginx.service - A high performance web server and a reverse proxy server...
Mar 03 15:19:08 iZbp1cqdx0u3g3wriko37cZ nginx[34378]: 2025/03/03 15:19:08 [emerg] 34378#34378: open() "/etc/nginx/sites-enabled/default" failed (2: No such file or director>
Mar 03 15:19:08 iZbp1cqdx0u3g3wriko37cZ nginx[34378]: nginx: configuration file /etc/nginx/nginx.conf test failed
Mar 03 15:19:08 iZbp1cqdx0u3g3wriko37cZ systemd[1]: nginx.service: Control process exited, code=exited, status=1/FAILURE
Mar 03 15:19:08 iZbp1cqdx0u3g3wriko37cZ systemd[1]: nginx.service: Failed with result 'exit-code'.
Mar 03 15:19:08 iZbp1cqdx0u3g3wriko37cZ systemd[1]: Failed to start nginx.service - A high performance web server and a reverse proxy server.
```

## 4. 后端部署

### 4.1 安装依赖
```bash
# 在服务器上执行
cd /root/aka_music/backend

# 安装项目依赖
pip install -r requirements.txt
```

### 4.2 使用 Screen 管理后端进程
```bash
# 创建新的 screen 会话
screen -S aka_music_backend

# 在 screen 会话中启动后端服务
conda activate aka_music
cd /root/aka_music/backend
flask run --host=127.0.0.1 --port=5000

```bash
Requirement already satisfied: greenlet!=0.4.17 in /root/miniconda3/envs/aka_music/lib/python3.11/site-packages (from sqlalchemy>=2.0.16->Flask-SQLAlchemy==3.1.1->-r requirements.txt (line 2)) (3.1.1)
Installing collected packages: Werkzeug, blinker, Flask, Flask-SQLAlchemy
  Attempting uninstall: Werkzeug
    Found existing installation: Werkzeug 2.0.3
    Uninstalling Werkzeug-2.0.3:
      Successfully uninstalled Werkzeug-2.0.3
  Attempting uninstall: Flask
    Found existing installation: Flask 2.0.1
    Uninstalling Flask-2.0.1:
      Successfully uninstalled Flask-2.0.1
  Attempting uninstall: Flask-SQLAlchemy
    Found existing installation: Flask-SQLAlchemy 2.5.1
    Uninstalling Flask-SQLAlchemy-2.5.1:
      Successfully uninstalled Flask-SQLAlchemy-2.5.1
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
flask-jwt-extended 4.4.4 requires Flask<3.0,>=2.0, but you have flask 3.1.0 which is incompatible.
```

# 按 Ctrl+A+D 分离 screen 会话
```

### 4.3 管理 Screen 会话
```bash
# 列出所有 screen 会话
screen -ls

# 重新连接到会话
screen -r aka_music_backend

# 终止会话
screen -X -S aka_music_backend quit
```

## 5. 自动化部署脚本

### 5.1 创建部署脚本
在本地项目根目录创建 `deploy.sh`：
```bash
#!/bin/bash

# 设置服务器信息
SERVER_IP="your_server_ip"
SERVER_USER="root"
PROJECT_DIR="/root/aka_music"

# 同步代码
rsync -avz --exclude-from='.rsyncignore' ./ $SERVER_USER@$SERVER_IP:$PROJECT_DIR/

# 通过 SSH 在服务器上执行命令
ssh $SERVER_USER@$SERVER_IP << 'ENDSSH'
    # 前端构建
    cd /root/aka_music/frontend
    npm install
    npm run build

    # 重启后端服务
    screen -X -S aka_music_backend quit || true
    screen -dmS aka_music_backend bash -c 'cd /root/aka_music/backend && conda activate aka_music && flask run --host=127.0.0.1 --port=5000'

    # 重启 Nginx
    sudo systemctl restart nginx
ENDSSH
```

### 5.2 使用部署脚本
```bash
# 添加执行权限
chmod +x deploy.sh

# 执行部署
./deploy.sh
```

## 6. 注意事项

1. 安全配置：
   - 建议配置防火墙，只开放必要端口（80, 443, SSH）
   - 使用 SSL 证书启用 HTTPS
   - 定期更新系统和依赖包

2. 备份策略：
   - 定期备份数据库和用户上传的文件
   - 可以使用 crontab 设置自动备份任务

3. 监控：
   - 配置服务器监控（CPU、内存、磁盘使用情况）
   - 设置日志轮转，避免日志文件过大

4. 优化建议：
   - 后续可以考虑使用 Docker 容器化部署
   - 可以使用 PM2 或 Supervisor 替代 Screen 管理进程
   - 配置 CDN 加速静态资源访问


```bash
2025/03/03 15:35:09 [crit] 34437#34437: *4 stat() "/root/aka_music/frontend/dist/index.html" failed (13: Permission denied), client: 125.75.66.97, server: localhost, request: "GET / HTTP/1.1", host: "www.alphago.ltd"
2025/03/03 15:35:09 [crit] 34437#34437: *4 stat() "/root/aka_music/frontend/dist/index.html" failed (13: Permission denied), client: 125.75.66.97, server: localhost, request: "GET / HTTP/1.1", host: "www.alphago.ltd"
2025/03/03 15:35:09 [error] 34437#34437: *4 rewrite or internal redirection cycle while internally redirecting to "/index.html", client: 125.75.66.97, server: localhost, request: "GET / HTTP/1.1", host: "www.alphago.ltd"
```
域名解析那边的请求


screen ls

screen -r aka_music_backend

```bash
注意conda环境要匹配，否则嗝屁

Error: No such option: --debug (Possible options: --debugger, --no-debugger)
 * Environment: production
   WARNING: This is a development server. Do not use it in a production deployment.
   Use a production WSGI server instead.
 * Debug mode: off
Traceback (most recent call last):
  File "/root/miniconda3/bin/flask", line 8, in <module>
    sys.exit(main())
             ^^^^^^
  File "/root/miniconda3/lib/python3.12/site-packages/flask/cli.py", line 990, in main
    cli.main(args=sys.argv[1:])
  File "/root/miniconda3/lib/python3.12/site-packages/flask/cli.py", line 596, in main
    return super().main(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/root/miniconda3/lib/python3.12/site-packages/click/core.py", line 1082, in main
    rv = self.invoke(ctx)
         ^^^^^^^^^^^^^^^^
  File "/root/miniconda3/lib/python3.12/site-packages/click/core.py", line 1697, in invoke
    return _process_result(sub_ctx.command.invoke(sub_ctx))
                           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/root/miniconda3/lib/python3.12/site-packages/click/core.py", line 1443, in invoke
    return ctx.invoke(self.callback, **ctx.params)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/root/miniconda3/lib/python3.12/site-packages/click/core.py", line 788, in invoke
    return __callback(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/root/miniconda3/lib/python3.12/site-packages/click/decorators.py", line 92, in new_func
    return ctx.invoke(f, obj, *args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/root/miniconda3/lib/python3.12/site-packages/click/core.py", line 788, in invoke
    return __callback(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/root/miniconda3/lib/python3.12/site-packages/flask/cli.py", line 845, in run_command
    app = DispatchingApp(info.load_app, use_eager_loading=eager_loading)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/root/miniconda3/lib/python3.12/site-packages/flask/cli.py", line 321, in __init__
    self._load_unlocked()
  File "/root/miniconda3/lib/python3.12/site-packages/flask/cli.py", line 346, in _load_unlocked
    self._app = rv = self.loader()
                     ^^^^^^^^^^^^^
  File "/root/miniconda3/lib/python3.12/site-packages/flask/cli.py", line 406, in load_app
    app = locate_app(self, import_name, None, raise_if_not_found=False)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/root/miniconda3/lib/python3.12/site-packages/flask/cli.py", line 256, in locate_app
    __import__(module_name)
  File "/root/aka_music/backend/app/__init__.py", line 6, in <module>
    from .models.user import db, User
  File "/root/aka_music/backend/app/models/user.py", line 4, in <module>
    db = SQLAlchemy()
         ^^^^^^^^^^^^
  File "/root/miniconda3/lib/python3.12/site-packages/flask_sqlalchemy/__init__.py", line 758, in __init__
    _include_sqlalchemy(self, query_class)
  File "/root/miniconda3/lib/python3.12/site-packages/flask_sqlalchemy/__init__.py", line 112, in _include_sqlalchemy
    for key in module.__all__:
               ^^^^^^^^^^^^^^
  File "/root/miniconda3/lib/python3.12/site-packages/sqlalchemy/__init__.py", line 294, in __getattr__
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
AttributeError: module 'sqlalchemy' has no attribute '__all__'. Did you mean: '__file__'?
```

```bash
build的时候提示assets资源的特殊处理

(base) root@iZbp1cqdx0u3g3wriko37cZ:~/aka_music/frontend# npm run build

> frontend@0.0.0 build
> vite build

vite v6.1.0 building for production...
transforming (3656) node_modules/@ant-design/icons-svg/es/asn/UsbOutlined.js
/static/def/images/hero-bg.jpg referenced in /static/def/images/hero-bg.jpg didn't resolve at build time, it will remain unchanged to be resolved at runtime
```


```bash
(aka_music) root@iZbp1cqdx0u3g3wriko37cZ:/etc/nginx/sites-available# systemctl status nginx.service
× nginx.service - A high performance web server and a reverse proxy server
     Loaded: loaded (/usr/lib/systemd/system/nginx.service; enabled; preset: enabled)
     Active: failed (Result: exit-code) since Thu 2025-03-13 09:29:36 CST; 32s ago
   Duration: 8min 9.548s
       Docs: man:nginx(8)
    Process: 511295 ExecStartPre=/usr/sbin/nginx -t -q -g daemon on; master_process on; (code=exited, status=1/FAILURE)
        CPU: 3ms

Mar 13 09:29:36 iZbp1cqdx0u3g3wriko37cZ systemd[1]: Starting nginx.service - A high performance web server and a reverse proxy server...
Mar 13 09:29:36 iZbp1cqdx0u3g3wriko37cZ nginx[511295]: 2025/03/13 09:29:36 [emerg] 511295#511295: open() "/etc/nginx/sites-enabled/aka_music" failed (2: No such file or directory) in /etc/nginx/nginx.conf:60
Mar 13 09:29:36 iZbp1cqdx0u3g3wriko37cZ nginx[511295]: nginx: configuration file /etc/nginx/nginx.conf test failed
Mar 13 09:29:36 iZbp1cqdx0u3g3wriko37cZ systemd[1]: nginx.service: Control process exited, code=exited, status=1/FAILURE
Mar 13 09:29:36 iZbp1cqdx0u3g3wriko37cZ systemd[1]: nginx.service: Failed with result 'exit-code'.
Mar 13 09:29:36 iZbp1cqdx0u3g3wriko37cZ systemd[1]: Failed to start nginx.service - A high performance web server and a reverse proxy server.
```

```bash
被扫描了，可能有风险（TODO）

84.252.135.0 - - [13/Mar/2025:09:48:38 +0800] "HEAD /.env.conf HTTP/1.1" 301 0 "-" "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:90.0) Gecko/20100101 Firefox/90.0"
84.252.135.0 - - [13/Mar/2025:09:48:40 +0800] "HEAD /.env.yml HTTP/1.1" 301 0 "-" "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.2 Safari/605.1.15"
84.252.135.0 - - [13/Mar/2025:09:48:40 +0800] "HEAD /.env.yaml HTTP/1.1" 301 0 "-" "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.131 Safari/537.36 Edg/92.0.902.67"
84.252.135.0 - - [13/Mar/2025:09:48:41 +0800] "HEAD /.env.json HTTP/1.1" 301 0 "-" "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
84.252.135.0 - - [13/Mar/2025:09:48:41 +0800] "HEAD /.env.ini HTTP/1.1" 301 0 "-" "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.131 Safari/537.36"
84.252.135.0 - - [13/Mar/2025:09:48:42 +0800] "HEAD /.env.toml HTTP/1.1" 301 0 "-" "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.2 Safari/605.1.15"
84.252.135.0 - - [13/Mar/2025:09:48:43 +0800] "HEAD /.env.php HTTP/1.1" 301 0 "-" "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:90.0) Gecko/20100101 Firefox/90.0"
84.252.135.0 - - [13/Mar/2025:09:48:43 +0800] "HEAD /.env.js HTTP/1.1" 301 0 "-" "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36"
84.252.135.0 - - [13/Mar/2025:09:48:49 +0800] "HEAD /.env.config.php HTTP/1.1" 301 0 "-" "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36"
84.252.135.0 - - [13/Mar/2025:09:48:49 +0800] "HEAD /.env.config.js HTTP/1.1" 301 0 "-" "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.131 Safari/537.36"
84.252.135.0 - - [13/Mar/2025:09:48:50 +0800] "HEAD /.env.config.json HTTP/1.1" 301 0 "-" "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36"
84.252.135.0 - - [13/Mar/2025:09:48:51 +0800] "HEAD /.env.production1 HTTP/1.1" 301 0 "-" "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:90.0) Gecko/20100101 Firefox/90.0"
84.252.135.0 - - [13/Mar/2025:09:48:51 +0800] "HEAD /.env.production2 HTTP/1.1" 301 0 "-" "Mozilla/5.0 (iPhone; CPU iPhone OS 14_7_1 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.2 Mobile/15E148 Safari/604.1"
84.252.135.0 - - [13/Mar/2025:09:48:52 +0800] "HEAD /.env.production3 HTTP/1.1" 301 0 "-" "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.131 Safari/537.36"
84.252.135.0 - - [13/Mar/2025:09:48:52 +0800] "HEAD /.env.prod1 HTTP/1.1" 301 0 "-" "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36"
84.252.135.0 - - [13/Mar/2025:09:48:53 +0800] "HEAD /.env.prod2 HTTP/1.1" 301 0 "-" "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36"
84.252.135.0 - - [13/Mar/2025:09:48:57 +0800] "HEAD /.env.prod3 HTTP/1.1" 301 0 "-" "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:90.0) Gecko/20100101 Firefox/90.0"
84.252.135.0 - - [13/Mar/2025:09:48:57 +0800] "HEAD /.env.staging1 HTTP/1.1" 301 0 "-" "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36"
84.252.135.0 - - [13/Mar/2025:09:48:58 +0800] "HEAD /.env.staging2 HTTP/1.1" 301 0 "-" "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.131 Safari/537.36 Edg/92.0.902.67"
84.252.135.0 - - [13/Mar/2025:09:48:58 +0800] "HEAD /.env.staging3 HTTP/1.1" 301 0 "-" "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:90.0) Gecko/20100101 Firefox/90.0"
84.252.135.0 - - [13/Mar/2025:09:48:59 +0800] "HEAD /.env.dev1 HTTP/1.1" 301 0 "-" "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:90.0) Gecko/20100101 Firefox/90.0"
84.252.135.0 - - [13/Mar/2025:09:48:59 +0800] "HEAD /.env.dev2 HTTP/1.1" 301 0 "-" "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:90.0) Gecko/20100101 Firefox/90.0"
84.252.135.0 - - [13/Mar/2025:09:49:00 +0800] "HEAD /.env.dev3 HTTP/1.1" 301 0 "-" "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:90.0) Gecko/20100101 Firefox/90.0"
84.252.135.0 - - [13/Mar/2025:09:49:00 +0800] "HEAD /.env.development1 HTTP/1.1" 301 0 "-" "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.131 Safari/537.36"
84.252.135.0 - - [13/Mar/2025:09:49:01 +0800] "HEAD /.env.development2 HTTP/1.1" 301 0 "-" "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:90.0) Gecko/20100101 Firefox/90.0"
84.252.135.0 - - [13/Mar/2025:09:49:01 +0800] "HEAD /.env.development3 HTTP/1.1" 301 0 "-" "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36"
84.252.135.0 - - [13/Mar/2025:09:49:02 +0800] "HEAD /.env.test1 HTTP/1.1" 301 0 "-" "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.131 Safari/537.36"
84.252.135.0 - - [13/Mar/2025:09:49:02 +0800] "HEAD /.env.test2 HTTP/1.1" 301 0 "-" "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.131 Safari/537.36 Edg/92.0.902.67"
84.252.135.0 - - [13/Mar/2025:09:49:03 +0800] "HEAD /.env.test3 HTTP/1.1" 301 0 "-" "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:90.0) Gecko/20100101 Firefox/90.0"
84.252.135.0 - - [13/Mar/2025:09:49:03 +0800] "HEAD /phpinfo.php HTTP/1.1" 301 0 "-" "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.131 Safari/537.36"
84.252.135.0 - - [13/Mar/2025:09:49:04 +0800] "HEAD /info.php HTTP/1.1" 301 0 "-" "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.131 Safari/537.36 Edg/92.0.902.67"
84.252.135.0 - - [13/Mar/2025:09:49:05 +0800] "HEAD /i.php HTTP/1.1" 301 0 "-" "Mozilla/5.0 (iPhone; CPU iPhone OS 14_7_1 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.2 Mobile/15E148 Safari/604.1"
84.252.135.0 - - [13/Mar/2025:09:49:05 +0800] "HEAD /test_phpinfo.php HTTP/1.1" 301 0 "-" "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
84.252.135.0 - - [13/Mar/2025:09:49:06 +0800] "HEAD /phpinfo_test.php HTTP/1.1" 301 0 "-" "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.131 Safari/537.36"
84.252.135.0 - - [13/Mar/2025:09:49:06 +0800] "HEAD /info_test.php HTTP/1.1" 301 0 "-" "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
84.252.135.0 - - [13/Mar/2025:09:49:07 +0800] "HEAD /test_info.php HTTP/1.1" 301 0 "-" "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.131 Safari/537.36"
84.252.135.0 - - [13/Mar/2025:09:49:07 +0800] "HEAD /php_info.php HTTP/1.1" 301 0 "-" "Mozilla/5.0 (iPhone; CPU iPhone OS 14_7_1 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.2 Mobile/15E148 Safari/604.1"
84.252.135.0 - - [13/Mar/2025:09:49:11 +0800] "HEAD /php-info.php HTTP/1.1" 301 0 "-" "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:90.0) Gecko/20100101 Firefox/90.0"
84.252.135.0 - - [13/Mar/2025:09:49:16 +0800] "HEAD /phpversion.php HTTP/1.1" 301 0 "-" "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:90.0) Gecko/20100101 Firefox/90.0"
84.252.135.0 - - [13/Mar/2025:09:49:17 +0800] "HEAD /phpv.php HTTP/1.1" 301 0 "-" "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.131 Safari/537.36 Edg/92.0.902.67"
84.252.135.0 - - [13/Mar/2025:09:49:17 +0800] "HEAD /phptest.php HTTP/1.1" 301 0 "-" "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.131 Safari/537.36"
84.252.135.0 - - [13/Mar/2025:09:49:18 +0800] "HEAD /test.php HTTP/1.1" 301 0 "-" "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.131 Safari/537.36"
84.252.135.0 - - [13/Mar/2025:09:49:18 +0800] "HEAD /php.php HTTP/1.1" 301 0 "-" "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.131 Safari/537.36 Edg/92.0.902.67"
84.252.135.0 - - [13/Mar/2025:09:49:19 +0800] "HEAD /php-test.php HTTP/1.1" 301 0 "-" "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.131 Safari/537.36 Edg/92.0.902.67"
84.252.135.0 - - [13/Mar/2025:09:49:19 +0800] "HEAD /php_test.php HTTP/1.1" 301 0 "-" "Mozilla/5.0 (iPhone; CPU iPhone OS 14_7_1 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.2 Mobile/15E148 Safari/604.1"
84.252.135.0 - - [13/Mar/2025:09:49:23 +0800] "HEAD /infophp.php HTTP/1.1" 301 0 "-" "Mozilla/5.0 (iPhone; CPU iPhone OS 14_7_1 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.2 Mobile/15E148 Safari/604.1"
```

```bash
Remote机器上通过5000直接能够得到数据，说明后端服务貌似也是ok的，问题可能出现在nginx没有正确的将/api的请求转发到后端

(aka_music) root@iZbp1cqdx0u3g3wriko37cZ:~/aka_music/frontend# curl -k 'https://localhost:5000/api/recommend/online_musics'
{
  "code": 200,
  "data": [
    {
      "artist": "\u5b59\u8000\u5a01",
      "coverUrl": "/static/def/a8.png",
      "id": 69,
      "plays": "12.4K",
      "title": "69.\u5b59\u8000\u5a01-\u7231\u7684\u6545\u4e8b\u4e0a\u96c6[FLAC/MP3-320K]",
      "url": "static/videos/4/69/1740070293_6384.m3u8"
    },
    {
      "artist": "\u738b\u83f2",
      "coverUrl": "/static/def/a4.png",
      "id": 366,
      "plays": "15.1K",
      "title": "0366.\u738b\u83f2 - \u90ae\u5dee[FLAC/MP3-320K]",
      "url": "static/videos/4/0366/1740070465_3155.m3u8"
    },
```    