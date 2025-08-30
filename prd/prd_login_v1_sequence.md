sequenceDiagram
    participant User
    participant Frontend
    participant Backend
    participant Database

    User->>Frontend: 访问登录页面
    Frontend->>User: 显示登录页面

    User->>Frontend: 输入用户名和密码
    Frontend->>Backend: 发送登录请求（用户名, 密码）
    Backend->>Database: 查询用户信息（用户名）
    Database-->>Backend: 返回用户信息
    Backend->>Backend: 校验密码
    alt 密码正确
        Backend->>Database: 更新登录时间
        Database-->>Backend: 确认更新
        Backend->>Frontend: 返回登录成功
        Frontend->>User: 显示首页
    else 密码错误
        Backend->>Frontend: 返回密码错误
        Frontend->>User: 显示错误提示
    end

    alt 登录失败次数 >= 5
        Backend->>Backend: 检查登录失败次数
        Backend->>Frontend: 返回账号锁定
        Frontend->>User: 显示账号锁定提示
    else 登录失败次数 < 5
        Backend->>Frontend: 返回登录失败
        Frontend->>User: 显示登录失败提示
    end