我来帮你设计 MySQL 数据库模型，包括表结构、索引和关联关系：

```sql
-- 内容主表
CREATE TABLE `content` (
  `id` char(36) NOT NULL COMMENT '内容ID',
  `title` varchar(255) NOT NULL COMMENT '标题',
  `type` enum('NOVEL','COMIC') NOT NULL COMMENT '内容类型：小说/漫画',
  `author_id` char(36) NOT NULL COMMENT '作者ID',
  `cover_url` varchar(255) DEFAULT NULL COMMENT '封面图片URL',
  `description` text COMMENT '内容描述',
  `status` enum('DRAFT','PUBLISHED') NOT NULL DEFAULT 'DRAFT' COMMENT '状态：草稿/已发布',
  `price_strategy` enum('PER_CHAPTER','BUNDLE') NOT NULL COMMENT '定价策略：按章/打包',
  `created_at` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `updated_at` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (`id`),
  KEY `idx_author_id` (`author_id`),
  KEY `idx_type_status` (`type`, `status`),
  KEY `idx_created_at` (`created_at`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='内容主表';

-- 章节表
CREATE TABLE `chapter` (
  `id` char(36) NOT NULL COMMENT '章节ID',
  `content_id` char(36) NOT NULL COMMENT '关联的内容ID',
  `chapter_no` int NOT NULL COMMENT '章节序号',
  `title` varchar(255) NOT NULL COMMENT '章节标题',
  `pages` json NOT NULL COMMENT '章节内容：小说为文本段落数组，漫画为图片URL数组',
  `price` decimal(10,2) NOT NULL DEFAULT '0.00' COMMENT '章节价格',
  `is_free` tinyint(1) NOT NULL DEFAULT '0' COMMENT '是否免费',
  `unlock_type` enum('PURCHASE','VIP') NOT NULL COMMENT '解锁类型：购买/VIP',
  `created_at` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `updated_at` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (`id`),
  UNIQUE KEY `uk_content_chapter` (`content_id`,`chapter_no`),
  KEY `idx_content_id` (`content_id`),
  KEY `idx_created_at` (`created_at`),
  CONSTRAINT `fk_chapter_content` FOREIGN KEY (`content_id`) REFERENCES `content` (`id`) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='章节表';

-- 用户购买记录表
CREATE TABLE `user_purchase` (
  `id` char(36) NOT NULL COMMENT '购买记录ID',
  `user_id` char(36) NOT NULL COMMENT '用户ID',
  `content_id` char(36) NOT NULL COMMENT '内容ID',
  `chapter_id` char(36) DEFAULT NULL COMMENT '章节ID（按章购买时使用）',
  `amount` decimal(10,2) NOT NULL COMMENT '支付金额',
  `purchase_type` enum('CHAPTER','BUNDLE') NOT NULL COMMENT '购买类型：单章/整本',
  `created_at` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '购买时间',
  PRIMARY KEY (`id`),
  KEY `idx_user_content` (`user_id`,`content_id`),
  KEY `idx_chapter_id` (`chapter_id`),
  KEY `idx_created_at` (`created_at`),
  CONSTRAINT `fk_purchase_content` FOREIGN KEY (`content_id`) REFERENCES `content` (`id`),
  CONSTRAINT `fk_purchase_chapter` FOREIGN KEY (`chapter_id`) REFERENCES `chapter` (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='用户购买记录表';
```

设计说明：

1. `content` 表：
   - 使用 UUID 作为主键
   - 增加了创建和更新时间
   - 添加了作者ID、类型和状态的索引
   - 使用 ENUM 限制内容类型和状态

2. `chapter` 表：
   - 使用 UUID 作为主键
   - 使用 JSON 类型存储章节内容
   - 添加了内容ID和章节号的唯一约束
   - 设置了外键关联到 content 表
   - 增加了价格和解锁类型字段

3. `user_purchase` 表（新增）：
   - 记录用户的购买历史
   - 支持按章购买和整本购买
   - 添加了相关索引提升查询效率
   - 设置了外键关联

主要特点：
1. 使用外键保证数据完整性
2. 合理的索引设计提升查询效率
3. 灵活支持小说和漫画两种内容类型
4. 支持多种付费模式
5. 完整的时间戳记录
6. 使用 JSON 类型灵活存储内容

