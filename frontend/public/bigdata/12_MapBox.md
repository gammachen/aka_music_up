`MAPBOX_API_KEY` 是用于在 Apache Superset 或其他支持 Mapbox 的数据可视化工具中启用 Mapbox 地图功能的 API 密钥。它通常是一个以 `pk.` 开头的公共访问令牌（public access token），而不是以 `sk.` 开头的秘密访问令牌（secret access token）。以下是关于 `MAPBOX_API_KEY` 的详细说明和使用方法：

### 1. **MAPBOX_API_KEY 的作用**
`MAPBOX_API_KEY` 是用于在 Apache Superset 或其他支持 Mapbox 的数据可视化工具中启用 Mapbox 地图功能的 API 密钥。它允许用户在地图上显示数据、进行交互式探索等操作。
![How to Configure Mapbox API? | TRAVELER THEME](https://oss.metaso.cn/metaso/thumbnail/fd79a517a10311811510e60915f05530.jpg)

### 2. **如何获取 MAPBOX_API_KEY**
要获取 `MAPBOX_API_KEY`，用户需要在 [Mapbox 官网](https://account.mapbox.com/access-tokens/ ) 注册一个账户，并创建一个访问令牌。通常，用户可以选择使用公共访问令牌（`pk.*`）或秘密访问令牌（`sk.*`）。根据一些错误信息，建议使用公共访问令牌，而不是秘密访问令牌 。

### 3. **如何设置 MAPBOX_API_KEY**
在 Apache Superset 中，`MAPBOX_API_KEY` 通常通过以下方式设置：

#### 方法一：通过环境变量设置
在启动 Superset 之前，可以通过设置环境变量来指定 `MAPBOX_API_KEY`。例如，在 Linux 系统中，可以使用以下命令设置环境变量：

```bash
export MAPBOX_API_KEY="pk.eyJ1IjoiZXphYW45MDIiLCJhIjoiY2xhdHI4NzI3MDQwazNwcDg1bDdyN3ZzMCJ9.8BOAE-IFmp6PeellMppXsA"
```


#### 方法二：在配置文件中设置
在 Superset 的配置文件 `superset_config.py ` 中，可以手动设置 `MAPBOX_API_KEY`：

```python
import os
MAPBOX_API_KEY = os.getenv('MAPBOX_API_KEY', 'pk.eyJ1xxxxxxtTGhkFm3KaTf_WQ')
```


#### 方法三：通过 Docker 配置
如果使用 Docker 部署 Superset，可以在 `docker-compose.yml` 文件中设置环境变量：

```yaml
environment:
  - MAPBOX_API_KEY=pk.eyJ1xxxxxxtTGhkFm3KaTf_WQ
```


### 4. **常见错误及解决方法**
#### 错误：使用秘密访问令牌（`sk.*`）导致错误
如果在使用 `MAPBOX_API_KEY` 时，错误地使用了秘密访问令牌（`sk.*`），可能会导致错误。例如，在使用 `deck.gl ` 时，错误地使用了 `sk.eyJ1IjoieGFuaWN1IiwiYSI6ImNsMnp2aW9qbjFqaTYzaX********`，而正确的做法是使用公共访问令牌（`pk.*`）。

#### 错误：未正确设置 MAPBOX_API_KEY
如果未正确设置 `MAPBOX_API_KEY`，可能会导致 Mapbox 地图无法加载。例如，在 Apache Superset 的配置中，如果 `MAPBOX_API_KEY` 未设置，可能会导致地图无法显示 。

### 5. **其他相关配置**
在 Apache Superset 的配置中，`MAPBOX_API_KEY` 通常与其他配置项一起使用，例如：

- `SECRET_KEY`：用于加密会话密钥，建议使用一个长随机字符串。
- `CACHE_CONFIG`：用于配置缓存，以提高性能。
- `SQLALCHEMY_DATABASE_URI`：用于指定数据库连接信息。

### 6. **总结**
`MAPBOX_API_KEY` 是 Apache Superset 或其他支持 Mapbox 的数据可视化工具中启用 Mapbox 地图功能的关键配置项。它通常是一个公共访问令牌（`pk.*`），而不是秘密访问令牌（`sk.*`）。用户可以通过环境变量、配置文件或 Docker 配置等方式设置 `MAPBOX_API_KEY`，并确保其正确性以避免错误。



