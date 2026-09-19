"""
并行策略对比实验：DDP数据并行最小Demo（Windows CPU版本）
Python >=3.10
运行命令：python parallelism_demo.py
"""
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler


class SimpleDataset(Dataset):
    def __init__(self, sample_num, feature_dim):
        self.data = torch.randn(sample_num, feature_dim)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class SimpleNet(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, in_dim)
        )

    def forward(self, x):
        return self.net(x)


def main():
    # Windows CPU 分布式初始化
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group(backend="gloo", rank=0, world_size=1)

    device = torch.device("cpu")
    model = SimpleNet(1024, 2048).to(device)
    model = DDP(model)

    dataset = SimpleDataset(sample_num=1000, feature_dim=1024)
    train_sampler = DistributedSampler(dataset, num_replicas=1, rank=0)
    train_loader = DataLoader(dataset, batch_size=16, sampler=train_sampler)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # 训练循环
    for epoch in range(3):
        train_sampler.set_epoch(epoch)
        total_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            pred = model(batch)
            loss = torch.mean((pred - batch) ** 2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch}, Average Loss: {avg_loss:.4f}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
