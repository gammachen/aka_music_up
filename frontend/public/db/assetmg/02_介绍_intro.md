# 数据库实体关系文档

根据提供的DDL文件，以下是系统中的实体和它们之间的关系详细文档。

## 核心实体

### 1. 用户 (users)
- **主键**: `id` (UUID)
- **主要属性**: 
  - `email`: 用户邮箱
  - `password`: 密码
  - `name`: 用户名称
  - `isAdmin`: 是否为管理员
  - `quotaSizeInBytes`: 存储配额大小
  - `quotaUsageInBytes`: 已使用的存储空间
  - `status`: 用户状态

### 2. 资产 (assets)
- **主键**: `id` (UUID)
- **主要属性**:
  - `deviceAssetId`: 设备资产ID
  - `ownerId`: 所有者ID (外键关联users)
  - `type`: 资产类型
  - `originalPath`: 原始路径
  - `fileCreatedAt`: 文件创建时间
  - `isFavorite`: 是否收藏
  - `isArchived`: 是否归档
  - `isVisible`: 是否可见
  - `checksum`: 校验和
  - `libraryId`: 所属库ID (外键关联libraries)
  - `stackId`: 堆栈ID (外键关联asset_stack)
  - `status`: 资产状态

### 3. 相册 (albums)
- **主键**: `id` (UUID)
- **主要属性**:
  - `ownerId`: 所有者ID (外键关联users)
  - `albumName`: 相册名称
  - `albumThumbnailAssetId`: 相册缩略图资产ID
  - `description`: 描述
  - `isActivityEnabled`: 是否启用活动
  - `order`: 排序方式

### 4. 库 (libraries)
- **主键**: `id` (UUID)
- **主要属性**:
  - `name`: 库名称
  - `ownerId`: 所有者ID (外键关联users)
  - `importPaths`: 导入路径数组
  - `exclusionPatterns`: 排除模式数组
  - `refreshedAt`: 刷新时间

### 5. 人物 (person)
- **主键**: `id` (UUID)
- **主要属性**:
  - `ownerId`: 所有者ID (外键关联users)
  - `name`: 人物名称
  - `thumbnailPath`: 缩略图路径
  - `isHidden`: 是否隐藏
  - `birthDate`: 出生日期
  - `faceAssetId`: 面部资产ID (外键关联asset_faces)
  - `isFavorite`: 是否收藏

### 6. 标签 (tags)
- **主键**: `id` (UUID)
- **主要属性**:
  - `userId`: 用户ID (外键关联users)
  - `value`: 标签值
  - `color`: 颜色
  - `parentId`: 父标签ID (外键关联tags)

## 关系实体

### 1. 相册-资产关系 (albums_assets_assets)
- **复合主键**: `albumsId`, `assetsId`
- 建立相册与资产的多对多关系

### 2. 相册-共享用户关系 (albums_shared_users_users)
- **复合主键**: `albumsId`, `usersId`
- **属性**: `role` (角色，如editor)
- 建立相册与共享用户的多对多关系

### 3. 标签-资产关系 (tag_asset)
- **复合主键**: `assetsId`, `tagsId`
- 建立标签与资产的多对多关系

### 4. 伙伴关系 (partners)
- **复合主键**: `sharedById`, `sharedWithId`
- **属性**: `inTimeline` (是否在时间线中显示)
- 建立用户之间的共享关系

### 5. 记忆-资产关系 (memories_assets_assets)
- **复合主键**: `memoriesId`, `assetsId`
- 建立记忆与资产的多对多关系

### 6. 共享链接-资产关系 (shared_link__asset)
- **复合主键**: `assetsId`, `sharedLinksId`
- 建立共享链接与资产的多对多关系

## 功能实体

### 1. 资产文件 (asset_files)
- **主键**: `id` (UUID)
- **主要属性**:
  - `assetId`: 资产ID (外键关联assets)
  - `type`: 文件类型
  - `path`: 文件路径
- 存储资产的不同类型文件

### 2. 资产堆栈 (asset_stack)
- **主键**: `id` (UUID)
- **主要属性**:
  - `primaryAssetId`: 主资产ID (外键关联assets)
  - `ownerId`: 所有者ID (外键关联users)
- 管理资产堆栈关系

### 3. 资产面部 (asset_faces)
- **主键**: `id` (UUID)
- **主要属性**:
  - `assetId`: 资产ID (外键关联assets)
  - `personId`: 人物ID (外键关联person)
  - 面部位置坐标: `boundingBoxX1`, `boundingBoxY1`, `boundingBoxX2`, `boundingBoxY2`
  - `sourceType`: 来源类型
- 存储资产中检测到的面部信息

### 4. EXIF数据 (exif)
- **主键**: `assetId` (外键关联assets)
- **主要属性**:
  - 相机信息: `make`, `model`, `lensModel`
  - 图像信息: `exifImageWidth`, `exifImageHeight`, `fileSizeInByte`
  - 拍摄信息: `dateTimeOriginal`, `fNumber`, `focalLength`, `iso`, `exposureTime`
  - 地理信息: `latitude`, `longitude`, `city`, `state`, `country`
- 存储资产的元数据信息

### 5. 记忆 (memories)
- **主键**: `id` (UUID)
- **主要属性**:
  - `ownerId`: 所有者ID (外键关联users)
  - `type`: 记忆类型
  - `data`: 记忆数据
  - `isSaved`: 是否保存
  - `memoryAt`: 记忆时间
  - `seenAt`: 查看时间
- 管理用户的记忆功能

### 6. 共享链接 (shared_links)
- **主键**: `id` (UUID)
- **主要属性**:
  - `userId`: 用户ID (外键关联users)
  - `key`: 密钥
  - `type`: 链接类型
  - `expiresAt`: 过期时间
  - `allowUpload`: 是否允许上传
  - `albumId`: 相册ID
  - `allowDownload`: 是否允许下载
  - `showExif`: 是否显示EXIF信息
  - `password`: 密码
- 管理资产和相册的共享链接

## 审计和历史记录

### 1. 资产审计 (assets_audit)
- **主键**: `id` (UUID)
- **主要属性**:
  - `assetId`: 资产ID
  - `ownerId`: 所有者ID
  - `deletedAt`: 删除时间
- 记录已删除资产的审计信息

### 2. 用户审计 (users_audit)
- **主键**: `id` (UUID)
- **主要属性**:
  - `userId`: 用户ID
  - `deletedAt`: 删除时间
- 记录已删除用户的审计信息

### 3. 伙伴审计 (partners_audit)
- **主键**: `id` (UUID)
- **主要属性**:
  - `sharedById`: 共享者ID
  - `sharedWithId`: 被共享者ID
  - `deletedAt`: 删除时间
- 记录已删除伙伴关系的审计信息

### 4. 移动历史 (move_history)
- **主键**: `id` (UUID)
- **主要属性**:
  - `entityId`: 实体ID
  - `pathType`: 路径类型
  - `oldPath`: 旧路径
  - `newPath`: 新路径
- 记录实体路径变更历史

### 5. 版本历史 (version_history)
- **主键**: `id` (UUID)
- **主要属性**:
  - `version`: 版本号
  - `createdAt`: 创建时间
- 记录系统版本历史

## 系统和配置

### 1. 系统配置 (system_config)
- **主键**: `key`
- **主要属性**: `value`
- 存储系统配置键值对

### 2. 系统元数据 (system_metadata)
- **主键**: `key`
- **主要属性**: `value` (JSON)
- 存储系统元数据

### 3. 用户元数据 (user_metadata)
- **复合主键**: `userId`, `key`
- **主要属性**: `value` (JSON)
- 存储用户相关的元数据

### 4. API密钥 (api_keys)
- **主键**: `id` (UUID)
- **主要属性**:
  - `name`: 名称
  - `key`: 密钥
  - `userId`: 用户ID (外键关联users)
  - `permissions`: 权限数组
- 管理API访问权限

## 特殊功能

### 1. 智能搜索 (smart_search)
- **主键**: `assetId`
- **主要属性**: `embedding` (向量)
- 支持基于向量的智能搜索功能

### 2. 面部搜索 (face_search)
- **主键**: `faceId`
- **主要属性**: `embedding` (向量)
- 支持基于向量的面部搜索功能

### 3. 地理数据 (geodata_places)
- **主键**: `id`
- **主要属性**:
  - `name`: 地点名称
  - `longitude`, `latitude`: 经纬度
  - `countryCode`: 国家代码
  - `admin1Name`, `admin2Name`: 行政区划名称
- 存储地理位置数据

### 4. 自然地理 (naturalearth_countries)
- **主键**: `id`
- **主要属性**:
  - `admin`: 行政区名称
  - `admin_a3`: 三字母行政区代码
  - `coordinates`: 多边形坐标
- 存储国家地理边界数据

## 主要关系图

```
users
 ├── assets (一对多)
 ├── albums (一对多)
 ├── libraries (一对多)
 ├── person (一对多)
 ├── tags (一对多)
 ├── partners (多对多)
 ├── memories (一对多)
 ├── shared_links (一对多)
 └── api_keys (一对多)

assets
 ├── asset_files (一对多)
 ├── exif (一对一)
 ├── asset_faces (一对多)
 ├── albums (多对多，通过albums_assets_assets)
 ├── tags (多对多，通过tag_asset)
 ├── memories (多对多，通过memories_assets_assets)
 ├── shared_links (多对多，通过shared_link__asset)
 └── asset_stack (多对一)

albums
 ├── assets (多对多，通过albums_assets_assets)
 ├── users (多对多，通过albums_shared_users_users)
 └── shared_links (一对多)

person
 └── asset_faces (一对多)

tags
 ├── assets (多对多，通过tag_asset)
 └── tags (自引用，父子关系)
```

## 总结

数据库设计围绕用户、资产、相册、人物和标签等核心实体构建，通过多种关系表实现复杂的数据关联。系统还包含丰富的审计、历史记录和元数据表，以支持完整的数据管理和追踪功能。特殊功能如向量搜索和地理数据支持增强了系统的智能化和空间分析能力。

整体设计遵循关系数据库的最佳实践，使用UUID作为主键，合理设置外键约束，并通过索引优化查询性能。系统还采用了触发器来自动更新时间戳和维护审计记录，确保数据的完整性和可追溯性。