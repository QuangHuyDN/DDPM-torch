import torch
import torch.nn as nn
from tqdm import tqdm
import argparse
import os
import copy

from utils import get_data, save_images, setup_logging, update_ema_params
from unet import UNet

import logging
from torch.utils.tensorboard import SummaryWriter

logging.basicConfig(
    format="%(asctime)s - %(levelname)s: %(message)s",
    level=logging.INFO,
    datefmt="%I:%M:%S",
)


class Diffusion:
    def __init__(
        self,
        noise_step: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        img_size: int = 64,
        device: str = "cuda",
    ) -> None:
        self.noise_step = noise_step
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.device = device

        self.beta = torch.linspace(beta_start, beta_end, noise_step)
        self.alpha = 1 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def add_noise_to_image(self, x: torch.Tensor, t: int):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[
            :, None, None, None
        ]
        epsilon = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * epsilon, epsilon

    def sample_timesteps(self, n: int):
        return torch.randint(low=1, high=self.noise_step, size=(n,))

    @torch.no_grad()
    def sample(self, model: nn.Module, n: int):
        model.eval()

        x = torch.randn((n, 3, self.img_size, self.img_size)).to(self.device)
        for i in tqdm(reversed(range(1, self.noise_step)), position=0):
            t = (torch.ones(n) * i).long().to(self.device)
            pred_noise = model(x, t)
            alpha = self.alpha[t][:, None, None, None]
            alpha_hat = self.alpha_hat[t][:, None, None, None]
            beta = self.beta[t][:, None, None, None]
            if i > 1:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)
            x = (
                1
                / torch.sqrt(alpha)
                * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * pred_noise)
                + torch.sqrt(beta) * noise
            )

        model.train()
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)
        return x


def train(args: argparse.Namespace):
    setup_logging(args.run)
    device = f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu"
    dataloader = get_data(args)

    model = UNet(
        in_chans=args.in_chans,
        out_chans=args.in_chans,
        img_size=args.size,
        device=device,
    ).to(device)

    ema_model = copy.deepcopy(model).eval().requires_grad_(False)
    ema_steps = 0
    ema_warmups = 2000

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    diffusion = Diffusion(img_size=args.size, device=device)

    logger = SummaryWriter(os.path.join("logs", args.run))
    l = len(dataloader)
    min_loss = torch.inf

    model.train()
    for epoch in range(args.epochs):
        logging.info(f"Starting epoch {epoch + 1}:")
        epoch_loss = 0
        pbar = tqdm(dataloader)
        for i, (images, _) in enumerate(pbar):
            images = images.to(device)
            t = diffusion.sample_timesteps(images.shape[0]).to(device)
            x_t, noise = diffusion.add_noise_to_image(images, t)

            pred_noise = model(x_t, t)
            loss = criterion(pred_noise, noise)
            epoch_loss += loss.item() / images.shape[0]

            opt.zero_grad()
            loss.backward()
            opt.step()

            # EMA update
            if ema_steps >= ema_warmups:
                update_ema_params(ema_model, model, decay_rate=0.995)
            else:
                # update ema model with source model weight
                ema_model.load_state_dict(model.state_dict())
                ema_steps += 1

            pbar.set_postfix(MSE=loss.item())
            logger.add_scalar("MSE", loss.item(), global_step=epoch * l + i)

        epoch_loss = epoch_loss / len(dataloader)
        if epoch_loss < min_loss:
            torch.save(
                model.state_dict(), os.path.join("models", args.run, "best.pth")
            )
            torch.save(
                ema_model.state_dict(),
                os.path.join("models", args.run, "best_ema.pth"),
            )

        if (epoch + 1) % 10 == 0:
            sampled_image = diffusion.sample(model, n=images.shape[0])
            sampled_ema_image = diffusion.sample(ema_model, n=images.shape[0])
            save_images(
                sampled_image,
                os.path.join("results", args.run, f"{epoch}.jpg"),
            )
            save_images(
                sampled_ema_image,
                os.path.join("results", args.run, f"{epoch}_ema.jpg"),
            )

    torch.save(
        model.state_dict(),
        os.path.join("models", args.run, "last.pth"),
    )
    torch.save(
        ema_model.state_dict(),
        os.path.join("models", args.run, "last_ema.pth"),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--run", action="store", type=str, required=True)
    parser.add_argument("--root_dir", action="store", type=str, required=True)
    parser.add_argument(
        "--size", action="store", type=int, required=False, default=64
    )
    parser.add_argument(
        "--in_chans", action="store", type=int, required=False, default=3
    )
    parser.add_argument(
        "--batch_size",
        action="store",
        type=int,
        required=False,
        default=16,
    )
    parser.add_argument(
        "--lr",
        action="store",
        type=float,
        required=False,
        default=3e-4,
    )
    parser.add_argument(
        "--epochs", action="store", type=int, required=False, default=300
    )
    parser.add_argument(
        "--gpu_id", action="store", type=int, required=False, default=0
    )

    args = parser.parse_args()
    train(args)
