---

### **用户会话传递的完整案例：从登录到ThreadLocal的使用**

---

#### **一、背景与需求**
在Web应用中，用户登录后需要在多个服务层（如Controller、Service、DAO）中访问用户身份信息（如用户ID、权限等）。直接通过参数层层传递会增加代码复杂度，而**ThreadLocal**可以为每个请求线程提供独立的用户会话存储，实现线程隔离。

---

#### **二、核心流程**
1. **用户登录**：生成并返回Token（如JWT）。
2. **拦截器解析Token**：在请求入口拦截器中解析Token，提取用户信息，并存入**ThreadLocal**。
3. **业务层直接获取**：在Controller、Service等层通过**ThreadLocal**直接获取用户信息。
4. **清理ThreadLocal**：在请求结束后，确保清理ThreadLocal，避免内存泄漏。

---

#### **三、代码实现与步骤详解**

---

##### **1. 用户登录接口**
```java
@RestController
public class AuthController {
    @PostMapping("/login")
    public ResponseEntity<String> login(@RequestBody LoginRequest request) {
        // 1. 验证用户名和密码
        User user = userService.validateUser(request.getUsername(), request.getPassword());
        if (user == null) {
            throw new RuntimeException("登录失败");
        }
        
        // 2. 生成JWT Token（包含用户信息）
        String token = JwtUtils.generateToken(user.getId(), user.getUsername());
        
        // 3. 返回Token给客户端
        return ResponseEntity.ok("Token: " + token);
    }
}
```

---

##### **2. 拦截器（设置ThreadLocal）**
在请求入口拦截器中，解析Token并设置用户信息到**ThreadLocal**：
```java
@Component
public class AuthInterceptor implements HandlerInterceptor {
    // ThreadLocal容器：存储当前线程的用户信息
    private static final ThreadLocal<UserContext> USER_CONTEXT = new ThreadLocal<>();

    @Override
    public boolean preHandle(HttpServletRequest request, HttpServletResponse response, Object handler) {
        try {
            // 1. 从请求头获取Token
            String token = request.getHeader("Authorization");
            if (token == null) {
                throw new RuntimeException("Token缺失");
            }

            // 2. 解析Token，获取用户信息
            Claims claims = JwtUtils.parseToken(token);
            Long userId = Long.parseLong(claims.getSubject());
            String username = (String) claims.get("username");

            // 3. 创建用户上下文对象
            UserContext userContext = new UserContext(userId, username);

            // 4. 将用户信息存入ThreadLocal
            USER_CONTEXT.set(userContext);

            return true;
        } catch (Exception e) {
            response.sendError(HttpServletResponse.SC_UNAUTHORIZED, "认证失败");
            return false;
        }
    }

    @Override
    public void afterCompletion(HttpServletRequest request, HttpServletResponse response, Object handler, Exception ex) {
        // 5. 请求结束后清理ThreadLocal
        USER_CONTEXT.remove();
    }
}
```

---

##### **3. 用户上下文类（UserContext）**
```java
public class UserContext {
    private final Long userId;
    private final String username;

    public UserContext(Long userId, String username) {
        this.userId = userId;
        this.username = username;
    }

    // 提供Getter方法
    public Long getUserId() {
        return userId;
    }

    public String getUsername() {
        return username;
    }
}
```

---

##### **4. 业务层直接获取用户信息**
在Service或DAO层中，无需参数传递，直接通过**ThreadLocal**获取用户信息：
```java
@Service
public class OrderService {
    public void createOrder(Product product) {
        // 1. 获取当前线程的用户信息
        UserContext userContext = AuthInterceptor.USER_CONTEXT.get();
        Long userId = userContext.getUserId();

        // 2. 业务逻辑：关联用户ID创建订单
        Order order = new Order(userId, product);
        orderRepository.save(order);
    }
}
```

---

##### **5. 配置拦截器**
在Spring Boot中配置拦截器，确保所有请求经过拦截器处理：
```java
@Configuration
public class WebConfig implements WebMvcConfigurer {
    @Autowired
    private AuthInterceptor authInterceptor;

    @Override
    public void addInterceptors(InterceptorRegistry registry) {
        registry.addInterceptor(authInterceptor)
                .addPathPatterns("/**") // 应用到所有路径
                .excludePathPatterns("/login"); // 排除登录接口
    }
}
```

---

#### **四、关键点解析**
1. **ThreadLocal的作用**：
   - **线程隔离**：每个请求线程在拦截器中设置独立的`UserContext`，确保不同线程的用户信息互不干扰。
   - **简化代码**：业务层无需接收用户信息参数，直接通过`AuthInterceptor.USER_CONTEXT.get()`获取。

2. **内存泄漏防范**：
   - **在拦截器的`afterCompletion`中清理**：确保每个请求结束后调用`USER_CONTEXT.remove()`，避免线程池复用线程时残留数据。
   - **静态ThreadLocal**：将`USER_CONTEXT`声明为静态变量，确保单例，避免重复创建。

3. **线程安全**：
   - `ThreadLocal`的`get`和`set`方法是线程本地操作，无需额外同步。

---

#### **五、错误使用案例与修复**
**错误场景**：未在拦截器中清理ThreadLocal。
```java
// 错误代码：未实现afterCompletion方法
public class FaultyInterceptor implements HandlerInterceptor {
    @Override
    public boolean preHandle(...) {
        // 设置ThreadLocal
        USER_CONTEXT.set(userContext);
        return true;
    }

    // 未实现afterCompletion，导致线程池中的线程保留旧数据
}
```

**修复方案**：
```java
@Override
public void afterCompletion(...) {
    USER_CONTEXT.remove();
}
```

---

#### **六、扩展：跨线程传递的挑战**
在某些场景（如异步任务、线程池），原始请求线程可能被复用或切换，此时`ThreadLocal`的值会丢失。  
**解决方案**：
- **使用TTL（Thread-Transient Local）**：通过`TTL`库（如`com.github.kristofa：threadlocal-transaction`）自动管理跨线程传递。
- **手动传递参数**：在异步任务中显式传递用户信息。

---

#### **七、总结**
通过ThreadLocal实现用户会话传递的核心步骤为：
1. **登录生成Token** → 2. **拦截器解析Token并设置ThreadLocal** → 3. **业务层直接获取** → 4. **请求结束后清理**。  
这种方式显著简化了代码结构，但需严格遵守清理规范，避免内存泄漏。在复杂场景中，可结合TTL等工具解决跨线程问题。

如果需要进一步探讨具体实现细节或扩展场景，请随时提问！