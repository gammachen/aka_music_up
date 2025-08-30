# 计算机基础技术文档

本文档集合涵盖了计算机科学与技术的核心基础知识，包括操作系统、网络协议、Java虚拟机、并发编程等重要领域。这些基础知识是构建高性能、可靠系统的重要基石。

## 核心内容领域

1. **网络协议基础**
   - TCP三次握手机制详解
   - SYN Flood攻击防御
   - HTTP协议深入解析
   - TCP/UDP在iOS中的应用

2. **操作系统原理**
   - 处理器与线程管理
   - 操作系统内存机制
   - 系统性能监控与优化
   - 内存交换与资源管理

3. **Java技术栈**
   - Java基础数据结构
   - 设计模式应用
   - JVM性能调优
   - Arthas工具使用

4. **并发编程**
   - 线程基础知识
   - 线程池与文件下载
   - 可重入锁机制
   - ThreadLocal应用
   - JMM与volatile关键字

## 文档目录

### TCP/IP网络协议
- `01_tcp_three_handshake.md` - TCP三次握手详解
- `01_2_tcp_three_handshake.md` - TCP三次握手进阶
- `01_3_tcp_defend_syn_flood.md` - SYN Flood防御策略
- `01_7_ios_tcp_udp.md` - iOS中的TCP/UDP应用
- `01_8_http.md` - HTTP协议详解

### 操作系统与性能
- `02_processor_thread.md` - 处理器与线程管理
- `03_os_memory.md` - 操作系统内存管理
- `07_2_online_problem.md` - 线上问题排查
- `07_3_online_problem.md` - 线上问题分析
- `07_5_online_problem_find.md` - 问题定位方法

### Java与JVM
- `04_java_structure.md` - Java数据结构
- `05_patterns.md` - 设计模式
- `06_jvm.md` - JVM基础
- `06_2_jvm_performance.md` - JVM性能优化
- `06_3_jvm_arthas.md` - Arthas工具使用

### 并发编程
- `08_thread_basic.md` - 线程基础
- `08_threadexcutor_filedownload.md` - 线程池应用
- `09_reentrant_lock.md` - 可重入锁
- `10_1_threadlocal.md` - ThreadLocal基础
- `11_jmm_volatile.md` - Java内存模型

### 性能监控
- `07_swap_memory_monitoring.md` - 内存交换监控
- `07_7_script_record.md` - 监控脚本记录
- `07_memory_eater.py` - 内存压测工具

## 实践指南

1. **问题诊断**
   - 系统性能分析方法
   - JVM问题排查流程
   - 内存泄漏定位技术
   - 线程死锁检测方案

2. **性能优化**
   - JVM参数调优
   - 线程池配置优化
   - 内存管理最佳实践
   - 网络通信性能提升

3. **安全防护**
   - 网络攻击防御
   - 资源隔离策略
   - 性能监控告警
   - 故障恢复机制

本文档集合不仅提供了理论知识，还包含了大量实践经验和工具使用指南，帮助开发者构建高质量的软件系统。通过系统学习这些基础知识，能够更好地理解和解决实际开发中遇到的各种技术挑战。