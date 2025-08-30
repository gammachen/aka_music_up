# Immich 服务器架构分析

## 技术栈

Immich 服务器使用以下技术栈：

1. **后端框架**：NestJS（基于 Node.js 的框架）
2. **数据库**：PostgreSQL（从代码中可以推断）
3. **WebSocket**：Socket.IO（用于实时通信）
4. **依赖注入**：NestJS 内置的 DI 系统
5. **API 文档**：Swagger（通过 `@ApiTags`、`@ApiProperty` 等装饰器可以看出）
6. **日期处理**：Luxon（`DateTime` 类）
7. **任务队列**：自定义队列系统（通过 `JobName`、`QueueName` 等枚举可以看出）
8. **文件存储**：自定义存储系统（`StorageRepository`、`StorageCore` 等）

## 架构设计

Immich 服务器采用了模块化的架构设计，主要遵循 NestJS 的架构模式：

1. **控制器层（Controllers）**：处理 HTTP 请求，如 `ServerController`、`AppController` 等
2. **服务层（Services）**：包含业务逻辑，如 `ServerService`、`SessionService` 等
3. **仓库层（Repositories）**：处理数据访问，如 `UserRepository`、`AssetRepository` 等
4. **DTO（数据传输对象）**：定义请求和响应的数据结构，如 `ServerAboutResponseDto`
5. **中间件（Middleware）**：处理认证等横切关注点，如 `WebSocketAdapter`
6. **守卫（Guards）**：控制路由访问，如 `@Authenticated()` 装饰器
7. **事件系统**：使用 `@OnEvent` 装饰器处理系统事件

## 模块设计

从代码中可以识别出以下主要模块：

1. **认证模块**：处理用户登录、会话管理（`SessionService`）
2. **资产模块**：处理照片、视频等媒体资产（`AssetRepository`）
3. **用户模块**：用户管理（`UserRepository`）
4. **服务器模块**：服务器配置和信息（`ServerService`）
5. **存储模块**：文件存储管理（`StorageRepository`）
6. **备份模块**：数据备份（`BackupService`）
7. **版本模块**：版本管理（`VersionService`）
8. **同步模块**：数据同步（`SyncService`）
9. **机器学习模块**：智能特性支持（`MachineLearningRepository`）
10. **地图模块**：地理位置功能（`MapRepository`）

## 服务接口

主要的服务接口包括：

### ServerController
```typescript
@Get('about')
getAboutInfo(): Promise<ServerAboutResponseDto>

@Get('storage')
getStorage(): Promise<ServerStorageResponseDto>

@Get('ping')
pingServer(): ServerPingResponse

@Get('version')
getServerVersion(): ServerVersionResponseDto

@Get('features')
getServerFeatures(): Promise<ServerFeaturesDto>

@Get('theme')
getTheme(): Promise<ServerThemeDto>

@Get('config')
getServerConfig(): Promise<ServerConfigDto>

@Get('statistics')
getServerStatistics(): Promise<ServerStatsResponseDto>

@Get('media-types')
getSupportedMediaTypes(): ServerMediaTypesResponseDto

@Put('license')
setServerLicense(@Body() license: LicenseKeyDto): Promise<LicenseResponseDto>

@Delete('license')
deleteServerLicense(): Promise<void>

@Get('license')
getServerLicense(): Promise<LicenseResponseDto>
```

### AppController
```typescript
@Get('.well-known/immich')
getImmichWellKnown()

@Get('custom.css')
getCustomCss()
```

### SessionService
```typescript
async getAll(auth: AuthDto): Promise<SessionResponseDto[]>
async delete(auth: AuthDto, id: string): Promise<void>
async deleteAll(auth: AuthDto): Promise<void>
```

## 数据模型

从代码中可以推断出以下主要数据模型：

1. **User**：用户信息
2. **Asset**：媒体资产（照片、视频等）
3. **Album**：相册
4. **Session**：用户会话
5. **SystemConfig**：系统配置
6. **VersionHistory**：版本历史
7. **License**：许可证信息

## 主要流程

### 启动流程
1. 在 `api.ts` 中，通过 `bootstrap()` 函数启动 NestJS 应用
2. 设置中间件、CORS、WebSocket 适配器等
3. 设置全局前缀 `/api`
4. 如果存在 Web 资源，则提供静态文件服务
5. 启动 HTTP 服务器

### 认证流程
1. 通过 `@Authenticated()` 装饰器保护路由
2. 使用 `SessionService` 管理用户会话
3. 定期清理过期的会话令牌

### 备份流程
1. 通过 `BackupService` 管理数据库备份
2. 使用 cron 表达式定时执行备份
3. 根据配置保留指定数量的备份

### 同步流程
1. 通过 `SyncService` 处理数据同步
2. 支持用户、资产、合作伙伴等数据的同步
3. 使用检查点机制确保增量同步

### WebSocket 通信
1. 使用 `EventRepository` 管理 WebSocket 事件
2. 通过 Redis 适配器支持多实例部署
3. 使用装饰器定义事件处理程序

## 特性标志系统

服务器实现了特性标志系统，通过 `ServerFeaturesDto` 控制功能的启用/禁用：

```typescript
{
  smartSearch: boolean;
  facialRecognition: boolean;
  duplicateDetection: boolean;
  map: boolean;
  reverseGeocoding: boolean;
  importFaces: boolean;
  sidecar: boolean;
  search: boolean;
  trash: boolean;
  oauth: boolean;
  oauthAutoLaunch: boolean;
  passwordLogin: boolean;
  configFile: boolean;
  email: boolean;
}
```

这种设计允许灵活地控制功能的可用性，并可能支持不同级别的服务订阅。

总结来说，Immich 服务器是一个功能丰富的媒体管理系统，采用现代化的架构设计和技术栈，支持多种高级功能如机器学习、地理位置、备份和同步等。