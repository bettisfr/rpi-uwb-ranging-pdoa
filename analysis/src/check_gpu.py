#!/usr/bin/env python3
"""Verify PyTorch CUDA availability and a small synthetic training step."""

import json
from pathlib import Path

import torch


def main():
    if not torch.cuda.is_available():
        raise RuntimeError('PyTorch cannot access CUDA')
    torch.manual_seed(42)
    device = torch.device('cuda')
    x = torch.randn(256, 6, device=device)
    y = 2*x[:, :1] - x[:, 1:2]
    model = torch.nn.Sequential(torch.nn.Linear(6, 32), torch.nn.ReLU(),
                                torch.nn.Linear(32, 1)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=.01)
    initial = float(torch.nn.functional.mse_loss(model(x), y).detach())
    for _ in range(100):
        optimizer.zero_grad()
        loss = torch.nn.functional.mse_loss(model(x), y)
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize()
    final = float(torch.nn.functional.mse_loss(model(x), y).detach())
    if not final < initial:
        raise RuntimeError('Synthetic CUDA training did not reduce loss')
    report = dict(torch_version=torch.__version__, cuda_runtime=torch.version.cuda,
                  device=torch.cuda.get_device_name(0),
                  device_memory_bytes=torch.cuda.get_device_properties(0).total_memory,
                  initial_loss=initial, final_loss=final,
                  status='passed', scope='Synthetic CUDA smoke test, not a paper experiment')
    output = Path(__file__).resolve().parents[1] / 'outputs/gpu_check.json'
    output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
