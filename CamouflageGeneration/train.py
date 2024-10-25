import torch
import config
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dataset import HorseZebraDataset, EnvironmentDataset
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from utils import load_checkpoint, save_checkpoint , get_next_experiment_number
from tqdm import tqdm
from discriminator import Discriminator
from generator import Generator
import os
from skimage.metrics import structural_similarity as ssim
import numpy as np
from torch.utils.tensorboard import SummaryWriter  # TensorBoard import



def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.xavier_normal_(m.weight)
    elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.InstanceNorm2d):
        nn.init.normal_(m.weight, 0.0, 0.02)


def color_retention_loss(real, generated, reconstructed, L1):
    loss_first = L1(real, generated)
    loss_second = L1(real, reconstructed)
    return loss_first + loss_second


def calculate_psnr(img1, img2, max_pixel_value=1.0):
    mse = F.mse_loss(img1, img2)
    psnr = 10 * torch.log10((max_pixel_value ** 2) / mse)
    return psnr


def calculate_ssim(img1, img2):
    if len(img1.shape) == 4:
        img1 = img1.squeeze(0)
        img2 = img2.squeeze(0)
    if len(img1.shape) == 3:
        img1 = img1.detach().cpu().numpy().transpose(1, 2, 0)
        img2 = img2.detach().cpu().numpy().transpose(1, 2, 0)
    ssim_value = ssim(img1, img2, multichannel=True)
    return ssim_value


def gradient_penalty(discriminator, real, fake, device="cpu", lambda_gp=10):
    batch_size, C, H, W = real.shape
    epsilon = torch.rand(batch_size, 1, 1, 1, device=device, requires_grad=True)
    interpolated_images = epsilon * real + (1 - epsilon) * fake
    interpolated_images.requires_grad_(True)
    interpolated_scores = discriminator(interpolated_images)
    gradients = torch.autograd.grad(
        outputs=interpolated_scores,
        inputs=interpolated_images,
        grad_outputs=torch.ones_like(interpolated_scores),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_norm = gradients.norm(2, dim=1)
    gradient_penalty = lambda_gp * ((gradient_norm - 1) ** 2).mean()
    return gradient_penalty


def train_func(disc_H, disc_Z, gen_H, gen_Z, opt_disc, opt_gen, g_scaler, d_scaler, L1, mse, loader, writer, epoch):
    loop = tqdm(loader, leave=True)
    for idx, (zebra, horse) in enumerate(loop):
        zebra = zebra.to(config.DEVICE)
        horse = horse.to(config.DEVICE)

        # ---- Train Discriminators H & Z ----
        with torch.cuda.amp.autocast():
            fake_horse = gen_H(zebra).detach()
            fake_zebra = gen_Z(horse).detach()

            D_H_real = disc_H(horse)
            D_H_fake = disc_H(fake_horse)
            D_H_loss = -(D_H_real.mean() - D_H_fake.mean())
            gp_H = gradient_penalty(disc_H, horse, fake_horse, device=config.DEVICE, lambda_gp=10)
            D_H_loss = D_H_loss + gp_H

            D_Z_real = disc_Z(zebra)
            D_Z_fake = disc_Z(fake_zebra)
            D_Z_loss = -(D_Z_real.mean() - D_Z_fake.mean())
            gp_Z = gradient_penalty(disc_Z, zebra, fake_zebra, device=config.DEVICE, lambda_gp=10)
            D_Z_loss = D_Z_loss + gp_Z

            D_loss = (D_H_loss + D_Z_loss) / 2

        opt_disc.zero_grad()
        d_scaler.scale(D_loss).backward(retain_graph=True)
        d_scaler.step(opt_disc)
        d_scaler.update()

        # ---- Train Generators H & Z ----
        with torch.cuda.amp.autocast():
            fake_horse = gen_H(zebra)
            fake_zebra = gen_Z(horse)

            D_H_fake = disc_H(fake_horse)
            D_Z_fake = disc_Z(fake_zebra)
            loss_G_H = -D_H_fake.mean()
            loss_G_Z = -D_Z_fake.mean()

            cycle_zebra = gen_Z(fake_horse)
            cycle_horse = gen_H(fake_zebra)
            cycle_zebra_loss = L1(zebra, cycle_zebra)
            cycle_horse_loss = L1(horse, cycle_horse)

            color_loss_zebra = color_retention_loss(zebra, fake_horse, cycle_zebra, L1)
            color_loss_horse = color_retention_loss(horse, fake_zebra, cycle_horse, L1)

            if config.LAMBDA_IDENTITY > 0:
                identity_zebra = gen_Z(zebra)
                identity_horse = gen_H(horse)
                identity_zebra_loss = L1(zebra, identity_zebra)
                identity_horse_loss = L1(horse, identity_horse)
            else:
                identity_zebra_loss = 0
                identity_horse_loss = 0

            G_loss = (
                    loss_G_Z
                    + loss_G_H
                    + cycle_zebra_loss * config.LAMBDA_CYCLE
                    + cycle_horse_loss * config.LAMBDA_CYCLE
                    + color_loss_zebra * config.LAMBDA_COLOR
                    + color_loss_horse * config.LAMBDA_COLOR
                    + identity_zebra_loss * config.LAMBDA_IDENTITY
                    + identity_horse_loss * config.LAMBDA_IDENTITY
            )

        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        # ---- Log Losses and Images to TensorBoard ----
        if idx % 200 == 0:
            psnr_horse = calculate_psnr(horse, fake_horse)
            psnr_zebra = calculate_psnr(zebra, fake_zebra)

            writer.add_scalar("PSNR/Horse", psnr_horse.item(), epoch * len(loader) + idx)
            writer.add_scalar("PSNR/Zebra", psnr_zebra.item(), epoch * len(loader) + idx)
            writer.add_scalar("Loss/Discriminator", D_loss.item(), epoch * len(loader) + idx)
            writer.add_scalar("Loss/Generator", G_loss.item(), epoch * len(loader) + idx)

            writer.add_image("Real Horse", horse[0] * 0.5 + 0.5, epoch * len(loader) + idx)
            writer.add_image("Fake Horse", fake_horse[0] * 0.5 + 0.5, epoch * len(loader) + idx)
            writer.add_image("Real Zebra", zebra[0] * 0.5 + 0.5, epoch * len(loader) + idx)
            writer.add_image("Fake Zebra", fake_zebra[0] * 0.5 + 0.5, epoch * len(loader) + idx)


def main():
    exp_number = get_next_experiment_number()
    writer = SummaryWriter(log_dir=f"runs/experiment{exp_number}")
    disc_H = Discriminator(in_channels=3).to(config.DEVICE)
    disc_Z = Discriminator(in_channels=3).to(config.DEVICE)
    gen_H = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)
    gen_Z = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)

    opt_disc = optim.Adam(
        list(disc_H.parameters()) + list(disc_Z.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999)
    )
    opt_gen = optim.Adam(
        list(gen_H.parameters()) + list(gen_Z.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999)
    )

    L1 = nn.L1Loss()
    mse = nn.MSELoss()

    if config.LOAD_MODEL:
        load_checkpoint(config.CHECKPOINT_GEN_H, gen_H, opt_gen, config.LEARNING_RATE)
        load_checkpoint(config.CHECKPOINT_GEN_Z, gen_Z, opt_gen, config.LEARNING_RATE)
        load_checkpoint(config.CHECKPOINT_CRITIC_H, disc_H, opt_disc, config.LEARNING_RATE)
        load_checkpoint(config.CHECKPOINT_CRITIC_Z, disc_Z, opt_disc, config.LEARNING_RATE)

    train_dataset = EnvironmentDataset(places=config.TRAIN_DIR + "/trainA", textures=config.TRAIN_DIR + "/trainB",
                                       transform=config.transforms)
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, pin_memory=True, shuffle=True,
                              num_workers=config.NUM_WORKERS)

    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()

    writer.add_text("Config", str(config.__dict__))  # Log config parameters

    if not os.path.exists(config.RESULTS):
        os.makedirs(config.RESULTS)

    for epoch in range(config.NUM_EPOCHS):
        train_func(disc_H, disc_Z, gen_H, gen_Z, opt_disc, opt_gen, g_scaler, d_scaler, L1, mse, train_loader, writer,
                   epoch)

        if config.SAVE_MODEL:
            save_checkpoint(gen_H, opt_gen, filename=config.CHECKPOINT_GEN_H)
            save_checkpoint(gen_Z, opt_gen, filename=config.CHECKPOINT_GEN_Z)
            save_checkpoint(disc_H, opt_disc, filename=config.CHECKPOINT_CRITIC_H)
            save_checkpoint(disc_Z, opt_disc, filename=config.CHECKPOINT_CRITIC_Z)

    writer.close()  # Close the writer when training is done


if __name__ == "__main__":
    main()
