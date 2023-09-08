#Credits to Dominic Rampas for the tutorial (aka dome272 on GitHub)

import sys
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import numpy as np
import os
import copy
import torch
import torch.nn as nn
from tqdm import tqdm
from torch import optim
import utils as utils
from modules import conditional_UNET, EMA
import logging
from torch.utils.tensorboard import SummaryWriter

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")


class StableDiffusion:

    def __init__(self, beta_init, beta_end, steps, procedure_type="linear", img_size=256):
        self.img_size = img_size
        self.beta_init = beta_init
        self.beta_end = beta_end
        self.steps = steps
        if procedure_type == "linear":
            self.betas = self.beta_linear()
        else:
            print("Requires a valid time procedure")
            raise TypeError
        
        self.alphas = 1 - self.betas
        self.alphas_cum_prod = torch.cumprod(self.alphas, dim=0)
        self.img_to_tensor = self.transform_to_tensor()
        self.tensor_to_img = self.reverse_transform()
        
        
    def transform_to_tensor(self):
        transformations = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: (x * 2) - 1)
        ])
        return transformations


    def reverse_transform(self):
        reverse_transform = transforms.Compose([
            transforms.Lambda(lambda x: (x + 1) / 2),
            transforms.Lambda(lambda x: x.permute(1, 2, 0)), #channel, height, width --> height, width, channel
            transforms.Lambda(lambda x: x * 255),
            transforms.Lambda(lambda x: x.numpy().astype(np.uint8)),
            transforms.ToPILImage()
        ])
        return reverse_transform
        
        
    def add_noise(self, image, timestep):
        sqrt_alpha_t = torch.sqrt(self.alphas_cum_prod[timestep])[None, None, None]
        sqrt_one_minus_alpha_t = torch.sqrt(1 - self.alphas_cum_prod[timestep])[None,None,None]
        noise = torch.randn_like(image)
        return sqrt_alpha_t * image + noise * sqrt_one_minus_alpha_t, noise
    
    
    def beta_linear(self):
        return torch.linspace(self.beta_init, self.beta_end, self.steps)
    

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.steps, size=(n,))
    
    
    def sampling(self, model, sample_number, labels, cfg_scale):
        model.eval()
        with torch.no_grad():
            img = torch.randn(sample_number, 3, self.img_size, self.img_size)
            for sample in tqdm(reversed(range(1, self.steps)), position = 0):
                t = (torch.ones(sample_number) * sample).long()
                predicted_noise = model(img, t, labels)
                if cfg_scale > 0:
                    unconditional_pred_noise = model(img, t, None)
                    predicted_noise = torch.lerp(unconditional_pred_noise, predicted_noise, cfg_scale)
                alpha = self.alphas[t][:, None, None, None]
                alpha_cum_prod = self.alphas_cum_prod[t][:, None, None, None]
                beta = self.betas[t][:, None, None, None]
                if sample > 1:
                    noise = torch.randn_like(img)
                else:
                    noise = torch.zeros_like(img)
                img = 1 / torch.sqrt(alpha) * (img - ((1 - alpha) / (torch.sqrt(1 - alpha_cum_prod))) * predicted_noise) + torch.sqrt(beta) * noise
        model.train()
        rev = self.reverse_transform()
        img = rev(img)
        return img

def train(args):
    utils.setup_logging(args.run_name)
    dataloader = utils.get_data(args)
    model = conditional_UNET(num_classes=args.num_classes)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    mse = nn.MSELoss()
    diffusion = StableDiffusion(img_size=args.image_size)
    logger = SummaryWriter(os.path.join("runs", args.run_name))
    l = len(dataloader)
    ema = EMA(0.995)
    ema_model = copy.deepcopy(model).eval().requires_grad_(False)

    for epoch in range(args.epochs):
        logging.info(f"Starting epoch {epoch}:")
        pbar = tqdm(dataloader)
        for i, (images, labels) in enumerate(pbar):
            images = images
            labels = labels
            t = diffusion.sample_timesteps(images.shape[0])
            x_t, noise = diffusion.noise_images(images, t)
            if np.random.random() < 0.1:
                labels = None
            predicted_noise = model(x_t, t, labels)
            loss = mse(noise, predicted_noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ema.step_ema(ema_model, model)

            pbar.set_postfix(MSE=loss.item())
            logger.add_scalar("MSE", loss.item(), global_step=epoch * l + i)

        if epoch % 10 == 0:
            labels = torch.arange(10).long()
            sampled_images = diffusion.sample(model, n=len(labels), labels=labels)
            ema_sampled_images = diffusion.sample(ema_model, n=len(labels), labels=labels)
            utils.plot_images(sampled_images)
            utils.save_images(sampled_images, os.path.join("results", args.run_name, f"{epoch}.jpg"))
            utils.save_images(ema_sampled_images, os.path.join("results", args.run_name, f"{epoch}_ema.jpg"))
            torch.save(model.state_dict(), os.path.join("models", args.run_name, f"ckpt.pt"))
            torch.save(ema_model.state_dict(), os.path.join("models", args.run_name, f"ema_ckpt.pt"))
            torch.save(optimizer.state_dict(), os.path.join("models", args.run_name, f"optim.pt"))


def launch():
    import argparse
    parser = argparse.ArgumentParser()
    arguments = parser.parse_args()
    arguments.run_name = "DDPM_conditional"
    arguments.epochs = 300
    arguments.batch_size = 14
    arguments.image_size = 64
    arguments.num_classes = 10
    arguments.dataset_path = "C:/Users/hpghs/Desktop/Research/diffusion/archive/cifar10/cifar10-64/train"
    arguments.lr = 3e-4
    train(arguments)


if __name__ == '__main__':
    launch()
