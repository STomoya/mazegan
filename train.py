
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler

from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

from data import MazeDataset
from model import U2Net, Discriminator, init_weights_normal, init_weights_xavier

softplus = nn.Softplus()
l1_loss = nn.L1Loss()

def d_loss(real_prob, fake_prob):
    real_loss = softplus(- real_prob).mean()
    fake_loss = softplus(  fake_prob).mean()
    return real_loss + fake_loss

def g_loss(fake_prob):
    return softplus(- fake_prob).mean()

def train_func(
    max_iter, dataset,
    D, G, optimizer_D, optimizer_G,
    l1_lambda, use_amp, device,
    result_folder='./result',
    save_interval=1000
):

    scaler = GradScaler()
    batches_done = 0
    bar = tqdm(total=max_iter)

    while True:
        for src, dst in dataset:
            optimizer_D.zero_grad()
            optimizer_G.zero_grad()

            src = src.to(device)
            dst = dst.to(device)

            ''''Discriminator'''
            with autocast(use_amp):
                # D(to)
                real_prob = D(dst)
                # D(G(from))
                fake = G(src)
                fake_prob = D(fake.detach())
                # loss
                adv_loss = d_loss(real_prob, fake_prob)
                L1_loss = l1_loss(fake, dst)
                D_loss = adv_loss + l1_lambda * L1_loss

            if use_amp:
                scaler.scale(D_loss).backward()
                scaler.step(optimizer_D)
            else:
                D_loss.backward()
                optimizer_D.step()

            '''Generator'''
            with autocast(use_amp):
                # D(G(from))
                fake = G(src)
                fake_prob = D(fake)
                # loss
                G_loss = g_loss(fake_prob)
            
            if use_amp:
                scaler.scale(G_loss).backward()
                scaler.step(optimizer_G)
            else:
                G_loss.backward()
                optimizer_G.step()

            # save
            if batches_done % save_interval == 0:
                image_grid = make_grid(src, fake, dst)
                save_image(image_grid, os.path.join(result_folder, f'{batches_done}.png'), nrow=9, normalize=True, range=(-1, 1))
                torch.save(G.state_dict(), os.path.join(result_folder, f'{batches_done}.pt'))

            # updates
            batches_done += 1
            scaler.update()
            bar.set_postfix_str('G Loss : {:.5f} D Loss : {:.5f}'.format(G_loss.item(), D_loss.item()))
            bar.update(1)

            if batches_done == max_iter:
                break

        if batches_done == max_iter:
            break
    
    torch.save(G.state_dict(), os.path.join(result_folder, 'final.pt'))

def make_grid(src, gen, dst, num_images=6):
    srcs = src.chunk(src.size(0), dim=0)
    dsts = dst.chunk(dst.size(0), dim=0)
    gens = gen.chunk(gen.size(0), dim=0)

    images = []
    for index, (src, gen, dst) in enumerate(zip(srcs, gens, dsts)):
        images.extend([
            src, gen, dst
        ])
        if index == num_images-1:
            break
    return torch.cat(images, dim=0)



def main(args):

    if not args.name in args.result_folder:
        args.result_folder = os.path.join(args.name, args.result_folder)

    if not os.path.exists(args.result_folder):
        os.makedirs(args.result_folder)
        os.chmod(args.result_folder, 0o777)

    g_kwargs = dict(
        depth=args.depth, in_channels=args.src_channels, out_channels=args.dst_channels,
        channels=args.channels,
        use_sn=args.use_sn_g, norm_name=args.norm_g, act_name=args.act_g,
        up_mode=args.up_mode, down_mode=args.down_mode, img_output=args.rgb_output
    )
    d_kwargs = dict(
        in_channels=args.dst_channels, channels=args.channels, n_layers=args.num_d_layers,
        use_sn=args.use_sn_d, norm_name=args.norm_d, act_name=args.act_d
    )
    G = U2Net(**g_kwargs)
    D = Discriminator(**d_kwargs)

    if args.init == 'normal': init_weight = init_weights_normal
    elif args.init == 'xavier': init_weight = init_weights_xavier

    G.apply(init_weight)
    D.apply(init_weight)

    optimizer_G = optim.Adam(G.parameters(), lr=args.learning_rate_g, betas=(args.beta1, args.beta2))
    optimizer_D = optim.Adam(D.parameters(), lr=args.learning_rate_d, betas=(args.beta1, args.beta2))

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    G.to(device)
    D.to(device)

    dataset = MazeDataset(image_size=args.image_size)
    dataset = DataLoader(dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=os.cpu_count(),
        pin_memory=torch.cuda.is_available())

    if args.max_iter < 0:
        args.max_iter = len(dataset) * 100

    train_kwargs = dict(
        max_iter=args.max_iter, dataset=dataset,
        D=D, G=G, optimizer_D=optimizer_D, optimizer_G=optimizer_G,
        l1_lambda=args.l1_lambda, use_amp=args.use_amp, device=device,
        result_folder=args.result_folder, save_interval=args.save_interval
    )
    train_func(**train_kwargs)