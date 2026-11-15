import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from ultralytics.nn.modules import DEConv


def main():
    torch.manual_seed(0)
    x = torch.randn(1, 64, 80, 80)
    module = DEConv(64).eval()

    with torch.no_grad():
        train_output = module(x)
        module.switch_to_deploy()
        deploy_output = module(x)

    max_error = (train_output - deploy_output).abs().max().item()
    print(f"output shape: {tuple(deploy_output.shape)}")
    print(f"maximum train/deploy error: {max_error:.6e}")
    assert deploy_output.shape == x.shape
    assert torch.allclose(train_output, deploy_output, atol=1e-5, rtol=1e-4)


if __name__ == "__main__":
    main()
