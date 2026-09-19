\# 并行策略对比：数据并行 / 流水线并行 / 张量并行 / 3D并行

\## 1. Mermaid可视化四种并行策略

```mermaid

flowchart TD

&#x20;   %% 数据并行 DP

&#x20;   subgraph 数据并行 Data Parallel

&#x20;       DP1\[GPU1<br/>完整模型 + 样本分片1]

&#x20;       DP2\[GPU2<br/>完整模型 + 样本分片2]

&#x20;       DP3\[GPU3<br/>完整模型 + 样本分片3]

&#x20;       DP4\[GPU4<br/>完整模型 + 样本分片4]

&#x20;       DP1 <-->|梯度AllReduce通信| DP2

&#x20;       DP2 <--> DP3

&#x20;       DP3 <--> DP4

&#x20;   end



&#x20;   %% 流水线并行 PP

&#x20;   subgraph 流水线并行 Pipeline Parallel

&#x20;       PP1\[GPU1<br/>模型层1\~2]

&#x20;       PP2\[GPU2<br/>模型层3\~4]

&#x20;       PP3\[GPU3<br/>模型层5\~6]

&#x20;       PP4\[GPU4<br/>模型层7\~8]

&#x20;       PP1 -->|传递激活值| PP2 --> PP3 --> PP4

&#x20;   end



&#x20;   %% 张量并行 TP

&#x20;   subgraph 张量并行 Tensor Parallel

&#x20;       TP1\[GPU1<br/>权重矩阵左半部分]

&#x20;       TP2\[GPU2<br/>权重矩阵右半部分]

&#x20;       TP1 <-->|AllGather/ReduceScatter| TP2

&#x20;   end



&#x20;   %% 3D并行 = DP+PP+TP组合

&#x20;   subgraph 3D并行 3D Parallel

&#x20;       direction TB

&#x20;       A\[数据并行分组]

&#x20;       B\[流水线层切分]

&#x20;       C\[张量权重切分]

&#x20;       A --> B --> C

&#x20;   end



