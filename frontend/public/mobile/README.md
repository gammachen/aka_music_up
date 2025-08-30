在 Termux 中启动 SSH 服务，可以让你通过手机连接到其他设备，或者让其他设备通过 SSH 连接到你的 Termux。Termux 支持 `openssh` 包，提供了完整的 `sshd` 和 `ssh` 命令。

以下是 **在 Termux 中启动 SSH 服务** 的完整步骤：

---

## 📱 步骤 1：安装 OpenSSH

首先确保你已经安装了 `openssh` 包：

```bash
pkg install openssh
```

---

## 🔐 步骤 2：设置 SSH 登录密码（可选）

Termux 默认的用户是 `u0_aXXX`（类似 Linux 用户），你可以为这个用户设置密码用于 SSH 登录：

```bash
passwd
```

输入并确认新密码。这将作为你通过 SSH 登录 Termux 时使用的密码。

> ⚠️ 注意：如果你不设置密码，将无法通过密码方式登录 SSH。

---

## 🚀 步骤 3：启动 SSHD 服务

Termux 不会自动创建 SSH 主机密钥，所以首次运行前需要生成它们：

```bash
sshd
```

如果提示找不到密钥文件（如 `/data/data/com.termux/files/usr/etc/ssh/ssh_host_*` 缺失），执行以下命令自动生成：

```bash
reset-ssh-host-keys
```

然后再次启动 SSH 服务：

```bash
sshd
```

---

## 📡 步骤 4：查看本机 IP 地址

要从其他设备连接到 Termux，你需要知道 Termux 所在设备的局域网 IP 地址：

```bash
ifconfig
```

或使用简化的：

```bash
ip a
```

找到 `wlan0` 接口下的 IP 地址，通常是 `192.168.x.x` 或 `10.0.0.x` 格式。

---

## 📲 步骤 5：从其他设备连接 Termux

在另一台设备（如电脑）上，使用 SSH 客户端连接 Termux：

```bash
ssh your-username@your-termux-ip -p 8022
```

其中：
- `your-username` 是你在 Termux 中的用户名（默认是 `u0_aXXX`，可以通过 `whoami` 查看）
- `your-termux-ip` 是 Termux 所在设备的局域网 IP 地址
- `-p 8022` 是 Termux 的 SSH 端口号，默认监听在 8022 端口

---

## 🔧 可选：修改 SSH 配置（如端口、允许 root 登录等）

编辑配置文件：

```bash
nano $PREFIX/etc/ssh/sshd_config
```

常见修改项：

- 修改端口：`Port 8022`
- 允许 root 登录（Termux 中没有 root，但可以启用允许 UID 0 登录）：
  ```bash
  PermitRootLogin yes
  ```

保存后重启 `sshd`：

```bash
pkill sshd
sshd
```

---

## 拷贝文件到本地

```bash
scp -P 8022 u0_a142@192.168.31.71:/data/data/com.termux/files/home/storage/downloads/cam_1.jpg .
```

---

## ✅ 总结

| 操作 | 命令 |
|------|------|
| 安装 openssh | `pkg install openssh` |
| 设置密码 | `passwd` |
| 生成主机密钥 | `reset-ssh-host-keys` |
| 启动 SSH 服务 | `sshd` |
| 查看 IP 地址 | `ip a` |
| 从其他设备连接 | `ssh username@termux_ip -p 8022` |

---

## 💡 提示

- Termux 的 SSH 服务不会开机自启，每次都需要手动运行 `sshd`。
- 如果你想后台运行 Termux 并保持 SSH 服务开启，建议使用 `tmux` 或者保持 Termux 在前台运行。
- Termux 的 SSH 只能在局域网中使用，不能直接暴露公网访问，除非你做了端口转发或使用反向隧道工具（如 frp、ngrok）。

---

Termux 成为一个永久运行的 SSH 服务器、设置公钥登录、或者通过公网访问 Termux

