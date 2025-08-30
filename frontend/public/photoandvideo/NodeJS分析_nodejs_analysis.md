# Immich服务器中Node.js技术栈详细分析

## 1. 核心框架：NestJS

Immich服务器使用NestJS作为后端框架，这是一个基于Node.js构建的企业级框架，具有以下特点：

### 1.1 架构组件

#### 控制器层（Controllers）
- **功能**：处理HTTP请求，定义API端点
- **实现方式**：使用装饰器（如`@Controller()`、`@Get()`、`@Post()`等）定义路由
- **示例**：
  ```typescript
  @Controller('server')
  export class ServerController {
    @Get('about')
    getAboutInfo(): Promise<ServerAboutResponseDto> { ... }
    
    @Get('storage')
    getStorage(): Promise<ServerStorageResponseDto> { ... }
  }
  ```

#### 服务层（Services）
- **功能**：包含业务逻辑，由控制器调用
- **实现方式**：使用`@Injectable()`装饰器标记为可注入的服务
- **示例**：
  ```typescript
  @Injectable()
  export class ServerService {
    constructor(
      private readonly configService: ConfigService,
      private readonly storageRepository: StorageRepository
    ) {}
    
    async getAboutInfo(): Promise<ServerAboutResponseDto> { ... }
  }
  ```

#### 仓库层（Repositories）
- **功能**：处理数据访问逻辑，与数据库交互
- **实现方式**：通常结合TypeORM使用，封装数据库操作
- **示例**：
  ```typescript
  @Injectable()
  export class AssetRepository {
    constructor(
      @InjectRepository(AssetEntity)
      private assetRepository: Repository<AssetEntity>
    ) {}
    
    async findById(id: string): Promise<AssetEntity | null> { ... }
  }
  ```

#### 模块（Modules）
- **功能**：组织相关组件，形成功能单元
- **实现方式**：使用`@Module()`装饰器定义模块结构
- **示例**：
  ```typescript
  @Module({
    imports: [TypeOrmModule.forFeature([AssetEntity])],
    controllers: [AssetController],
    providers: [AssetService, AssetRepository],
    exports: [AssetService]
  })
  export class AssetModule {}
  ```

### 1.2 依赖注入系统

- **实现方式**：NestJS使用基于装饰器的依赖注入系统
- **注入方式**：
  - 构造函数注入：最常用的方式
  - 属性注入：使用`@Inject()`装饰器
  - 方法注入：较少使用
- **作用域**：
  - 默认单例（Singleton）：整个应用共享一个实例
  - 请求作用域（Request）：每个请求创建新实例
  - 瞬态（Transient）：每次注入创建新实例
- **示例**：
  ```typescript
  @Injectable()
  export class ServerService {
    constructor(
      @Inject(CONFIG_TOKEN) private config: ConfigType,
      private readonly storageService: StorageService
    ) {}
  }
  ```

### 1.3 中间件和拦截器

#### 中间件（Middleware）
- **功能**：处理请求前的预处理逻辑
- **实现方式**：实现`NestMiddleware`接口
- **示例**：
  ```typescript
  @Injectable()
  export class AuthMiddleware implements NestMiddleware {
    constructor(private readonly sessionService: SessionService) {}
    
    async use(req: Request, res: Response, next: NextFunction) {
      const token = req.headers.authorization?.split(' ')[1];
      if (token) {
        const session = await this.sessionService.validate(token);
        if (session) req.user = session.user;
      }
      next();
    }
  }
  ```

#### 拦截器（Interceptors）
- **功能**：拦截请求和响应，实现横切关注点
- **实现方式**：实现`NestInterceptor`接口
- **示例**：
  ```typescript
  @Injectable()
  export class TransformInterceptor implements NestInterceptor {
    intercept(context: ExecutionContext, next: CallHandler): Observable<any> {
      return next.handle().pipe(
        map(data => ({ success: true, data }))
      );
    }
  }
  ```

#### 守卫（Guards）
- **功能**：控制路由访问权限
- **实现方式**：实现`CanActivate`接口
- **示例**：
  ```typescript
  @Injectable()
  export class AuthGuard implements CanActivate {
    canActivate(context: ExecutionContext): boolean | Promise<boolean> {
      const request = context.switchToHttp().getRequest();
      return !!request.user;
    }
  }
  ```
  
  在Immich中，通过自定义装饰器简化了守卫的使用：
  ```typescript
  export function Authenticated() {
    return applyDecorators(
      UseGuards(AuthGuard),
      ApiBearerAuth()
    );
  }
  ```

## 2. 实时通信：Socket.IO

### 2.1 WebSocket适配器

- **功能**：提供实时双向通信
- **实现方式**：通过自定义`WebSocketAdapter`集成Socket.IO
- **示例**：
  ```typescript
  @WebSocketGateway()
  export class EventsGateway implements OnGatewayConnection {
    @WebSocketServer()
    server: Server;
    
    handleConnection(client: Socket) {
      const token = client.handshake.auth.token;
      // 验证连接
    }
    
    @SubscribeMessage('event')
    handleEvent(client: Socket, payload: any) {
      // 处理事件
      this.server.emit('response', { data: 'response data' });
    }
  }
  ```

### 2.2 事件系统

- **功能**：处理系统内部事件
- **实现方式**：使用`@OnEvent`装饰器订阅事件
- **示例**：
  ```typescript
  @Injectable()
  export class AssetEventListener {
    constructor(private readonly socketService: SocketService) {}
    
    @OnEvent('asset.created')
    handleAssetCreatedEvent(payload: AssetCreatedEvent) {
      this.socketService.emitToUser(payload.userId, 'asset:created', {
        id: payload.assetId
      });
    }
  }
  ```

## 3. 前后端分离架构

### 3.1 API接口设计

- **RESTful API**：遵循REST原则设计API
- **API文档**：使用Swagger（通过`@ApiTags`、`@ApiProperty`等装饰器）自动生成API文档
- **版本控制**：通过URL路径或HTTP头部实现API版本控制
- **示例**：
  ```typescript
  @ApiTags('assets')
  @Controller('assets')
  export class AssetsController {
    @Get(':id')
    @ApiOperation({ summary: 'Get asset by id' })
    @ApiResponse({ status: 200, type: AssetResponseDto })
    @ApiResponse({ status: 404, description: 'Asset not found' })
    async getAssetById(@Param('id') id: string): Promise<AssetResponseDto> { ... }
  }
  ```

### 3.2 数据传输对象（DTOs）

- **功能**：定义请求和响应的数据结构
- **实现方式**：使用类和装饰器定义DTO
- **验证**：使用`class-validator`库进行请求验证
- **示例**：
  ```typescript
  export class CreateAssetDto {
    @ApiProperty()
    @IsString()
    deviceAssetId: string;
    
    @ApiProperty({ required: false })
    @IsOptional()
    @IsBoolean()
    isFavorite?: boolean;
    
    @ApiProperty()
    @IsDateString()
    fileCreatedAt: string;
  }
  ```

### 3.3 前后端通信

- **HTTP通信**：使用RESTful API进行常规通信
- **WebSocket通信**：使用Socket.IO进行实时通信
- **文件上传**：使用multipart/form-data格式上传文件
- **认证方式**：使用JWT（JSON Web Token）进行身份验证

## 4. 任务队列系统

### 4.1 队列实现

- **功能**：处理异步任务和后台作业
- **实现方式**：自定义队列系统，通过枚举定义任务类型
- **示例**：
  ```typescript
  export enum JobName {
    ASSET_DELETION = 'asset-deletion',
    ASSET_UPLOAD = 'asset-upload',
    METADATA_EXTRACTION = 'metadata-extraction',
    THUMBNAIL_GENERATION = 'thumbnail-generation'
  }
  
  export enum QueueName {
    ASSET_DELETION = 'asset-deletion',
    ASSET_UPLOAD = 'asset-upload',
    BACKGROUND_TASK = 'background-task',
    THUMBNAIL_GENERATION = 'thumbnail-generation'
  }
  ```

### 4.2 任务处理器

- **功能**：执行队列中的任务
- **实现方式**：使用处理器类处理特定类型的任务
- **示例**：
  ```typescript
  @Injectable()
  export class ThumbnailGenerationProcessor {
    constructor(private readonly assetService: AssetService) {}
    
    async process(job: Job<ThumbnailGenerationJobData>) {
      const { assetId } = job.data;
      await this.assetService.generateThumbnail(assetId);
      return { success: true };
    }
  }
  ```

## 5. 文件存储系统

### 5.1 存储抽象

- **功能**：提供统一的文件存储接口
- **实现方式**：通过`StorageCore`和`StorageRepository`抽象存储操作
- **示例**：
  ```typescript
  @Injectable()
  export class StorageRepository {
    constructor(private readonly storageCore: StorageCore) {}
    
    async writeFile(path: string, data: Buffer): Promise<void> {
      await this.storageCore.writeFile(path, data);
    }
    
    async readFile(path: string): Promise<Buffer> {
      return this.storageCore.readFile(path);
    }
    
    async deleteFile(path: string): Promise<void> {
      await this.storageCore.deleteFile(path);
    }
  }
  ```

### 5.2 存储策略

- **本地存储**：存储在本地文件系统
- **S3兼容存储**：支持AWS S3或兼容S3的存储服务
- **配置方式**：通过环境变量或配置文件配置存储策略

## 6. 特性标志系统

### 6.1 特性控制

- **功能**：控制功能的启用/禁用
- **实现方式**：通过`ServerFeaturesDto`定义可用特性
- **示例**：
  ```typescript
  export class ServerFeaturesDto {
    @ApiProperty()
    smartSearch: boolean;
    
    @ApiProperty()
    facialRecognition: boolean;
    
    @ApiProperty()
    duplicateDetection: boolean;
    
    @ApiProperty()
    map: boolean;
    
    // 其他特性...
  }
  ```

### 6.2 特性访问

- **功能**：在代码中检查特性是否启用
- **实现方式**：通过服务方法检查特性状态
- **示例**：
  ```typescript
  @Injectable()
  export class FeatureService {
    constructor(private readonly configService: ConfigService) {}
    
    isFeatureEnabled(feature: keyof ServerFeaturesDto): boolean {
      const features = this.configService.get<ServerFeaturesDto>('features');
      return features?.[feature] ?? false;
    }
  }
  ```

## 7. 启动流程

### 7.1 应用引导

- **入口文件**：`api.ts`中的`bootstrap()`函数
- **实现步骤**：
  1. 创建NestJS应用实例
  2. 设置全局前缀（`/api`）
  3. 配置中间件、CORS、WebSocket适配器
  4. 提供静态文件服务（如果存在Web资源）
  5. 启动HTTP服务器
- **示例**：
  ```typescript
  async function bootstrap() {
    const app = await NestFactory.create(AppModule);
    
    app.setGlobalPrefix('api');
    app.enableCors();
    app.useWebSocketAdapter(new SocketIoAdapter(app));
    
    // 配置Swagger
    const config = new DocumentBuilder()
      .setTitle('Immich API')
      .setDescription('Immich API documentation')