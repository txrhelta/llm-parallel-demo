# 大模型并行策略实验

## 项目简介
本项目实现大模型训练中4种并行策略原理文档、Mermaid可视化流程图，以及PyTorch DDP数据并行演示代码。并行策略：数据并行、流水线并行、张量并行、3D并行;同时讲解ZeRO内存优化技术。

## 环境依赖
- Python >=3.10
- PyTorch >=2.0
- Git

安装命令：
```bash
pip install torch
## 使用的Prompt
帮我完成大模型并行策略课程实验，需要：
1. 整理4种并行策略（数据并行、流水线并行、张量并行、3D并行）原理文档；
2. 绘制Mermaid可视化流程图；
3. 编写PyTorch DDP数据并行演示代码；
4. 讲解ZeRO内存优化技术；
5. 项目结构规范，附带README文档，适合课程作业提交。

## 运行方式
```bash
python src/ddp_demo.py
