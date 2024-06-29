import os
import argparse
import torch
import torchvision
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader


def update_ema_params(
    target: torch.nn.Module, source: torch.nn.Module, decay_rate=0.995
):
    targParams = target.parameters()
    srcParams = source.parameters()
    for targParam, srcParam in zip(targParams, srcParams):
        targParam.data.mul_(decay_rate).add_(
            srcParam.data, alpha=1 - decay_rate
        )


def plot_images(images: torch.Tensor):
    plt.figure(figsize=(32, 32))
    plt.imshow(
        torch.cat(
            [
                torch.cat([i for i in images.cpu()], dim=-1),
            ],
            dim=-2,
        )
        .permute(1, 2, 0)
        .cpu()
    )
    plt.show()


def save_images(images: torch.Tensor, path: str, **kwargs):
    grid = torchvision.utils.make_grid(images, **kwargs)
    ndarr = grid.permute(1, 2, 0).cpu().numpy()
    im = Image.fromarray(ndarr)
    im.save(path)


def get_data(args: argparse.Namespace):
    NUM_WORKER = 4

    transforms = T.Compose(
        [
            T.Resize(int(args.size * 1.25)),
            T.RandomResizedCrop(args.size, scale=(0.8, 1.0)),
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    dataset = torchvision.datasets.ImageFolder(
        args.root_dir, transform=transforms
    )

    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=NUM_WORKER,
    )


def setup_logging(run: str = "run"):
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs(os.path.join("models", run), exist_ok=True)
    os.makedirs(os.path.join("results", run), exist_ok=True)
