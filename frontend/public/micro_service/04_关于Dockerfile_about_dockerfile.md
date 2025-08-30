# Docker 与 Docker Compose 使用指南

## 1. Docker 简介

Docker 是一个开源的应用容器引擎，允许开发者将应用及其依赖打包到一个轻量级、可移植的容器中，然后发布到任何流行的 Linux 机器上，也可以实现虚拟化。容器是完全使用沙箱机制，相互之间不会有任何接口。

## 2. Dockerfile 基础

Dockerfile 是一个文本文件，包含了一系列指令，用于构建 Docker 镜像。以下是一个简单的 Dockerfile 示例：