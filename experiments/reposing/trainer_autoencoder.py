import os
import json
import torch
import torch.nn as nn
import random
import numpy as np
import matplotlib.pyplot as plt

from torchvision.utils import save_image
from tqdm import tqdm
from torch.utils.data import DataLoader
from networks.PoissonNet import PoissonNet
from .dataset import MOYOBakedDataset
from geometry.operators import construct_mesh_operators
from utils.helpers import cycle, seed_everything, count_parameters, MSE_loss, to_np
from viz.helpers import render_mesh, render_overlayed_meshes, add_text, image_grid
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Train a PoissonNet autoencoder on MOYO/SMPL data")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    return parser.parse_args()
seed_everything(31415)

device = torch.device('cuda:0')
args = parse_args()
config = json.load(open(args.config, 'r'))
print("loaded config from", args.config)
print("config:", config)
config['exp_name'] = config.get('exp_name', 'repose_autoencoder_true')
batch_size = config['batch_size']
grad_accum = config['grad_accum']
lr = config['lr']
clip_grad_norm = config['clip_grad_norm']
train_steps = config['train_steps']
mass_mse = config['mass_mse']
viz_steps = config['viz_steps']

lambda_v = config.get('lambda_v', 1.0)
lambda_g = config.get('lambda_g', 1.0)

# === Autoencoder Settings ===
autoencoding = True # Set to True to enable autoencoder mode
head_dim = config.get('head_dim', 32) # Dimension for Q and K heads
use_mass_attention = config.get('use_mass_attention', True)
use_softmax = config.get('use_softmax',True)
exp_name = config['exp_name']
os.makedirs(os.path.join('results', exp_name), exist_ok=True)
outfile = lambda x: os.path.join('results', exp_name, x)

# === Data ===
data_dir='./smplx_data'
train_dataset = MOYOBakedDataset(data_dir=data_dir, train=True, config=config)
test_dataset = MOYOBakedDataset(data_dir=data_dir, train=False, config=config)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
train_loader = cycle(train_loader)
test_loader = cycle(test_loader)

# === Models ===
pose_ndim = train_dataset.num_pose_params

if autoencoding:
    # If autoencoding, we learn a latent from target mesh instead of using pose params
    latent_dim = head_dim * head_dim
    feature_dim = latent_dim
    
    # Encoder: Operates on target mesh -> generates c -> latent
    encoder = PoissonNet(C_in=3,
                         C_out=2 * head_dim, # Q and K
                         C_width=config['width'],
                         n_blocks=config['nblocks'],
                         head='linear',
                         extra_features=0,
                         outputs_at='vertices',
                         last_activation=nn.Identity(),
                         config=config)
    encoder = encoder.to(device)
    print('Encoder parameters:', count_parameters(encoder))
else:
    feature_dim = pose_ndim
    encoder = None

# Decoder (Original Model): Operates on source mesh + extra features -> target mesh
model = PoissonNet( C_in=3,
                    C_out=3,
                    C_width=config['width'],
                    n_blocks=config['nblocks'],
                    head='njf',
                    extra_features=feature_dim,
                    config=config,)
    
print('Decoder parameters:', count_parameters(model))
model = model.to(device)

params = list(model.parameters())
if encoder is not None:
    params += list(encoder.parameters())

optimizer = torch.optim.Adam(params, lr=lr)

def generate_latent(c, mass=None, use_mass_attention=True, use_softmax=True):
    """
    Generates latent vector from encoder output c.
    c: (B, V, 2*head_dim)
    mass: (B, V) or None
    Returns: (B, head_dim*head_dim)
    """
    B, V, C = c.shape
    d = C // 2
    Q = c[..., :d]
    K = c[..., d:]
    
    Q_T = Q.transpose(1, 2) # (B, d, V)
    
    if mass is not None and use_mass_attention:
        if mass.dim() == 1:
            mass = mass.unsqueeze(0)
        mass_diag = torch.diag_embed(mass) # (B, V, V)
        scores = torch.bmm(torch.bmm(Q_T, mass_diag), K) # (B, d, d)
    else:
        scores = torch.bmm(Q_T, K) # (B, d, d)
        
    scores = scores / (d ** 0.5)
    if use_softmax:
        attention_map = torch.softmax(scores, dim=-1) # (B, d, d)
    else:
        attention_map = scores
    latent = attention_map.flatten(1) # (B, d*d)
    
    return latent

@torch.no_grad()
def form_batch(data_loader, augment=False, compute_tar_ops=False):
    verts_src, verts_tar, faces, pose_params = next(data_loader)

    verts_src = verts_src.to(device)
    verts_tar = verts_tar.to(device)
    faces = faces.to(device)
    pose_params = pose_params.to(device)

    if augment: 
        # augment src/target global scale before computing operators
        scale_xyz = torch.rand(verts_src.shape[0], 1, 1, device=verts_src.device) * 0.6 + 0.7
        verts_src = verts_src * scale_xyz
        verts_tar = verts_tar * scale_xyz

        # augment mesh position:
        shift_xyz = torch.randn(verts_src.shape[0], 1, 3, device=verts_src.device) * 0.15
        verts_src = verts_src + shift_xyz
        verts_tar = verts_tar + shift_xyz

    mass_src, solver_src, G_src, M_src = construct_mesh_operators(verts_src, faces, high_precision=True)
    
    mass_tar, solver_tar, G_tar, M_tar = None, None, None, None
    if compute_tar_ops:
        mass_tar, solver_tar, G_tar, M_tar = construct_mesh_operators(verts_tar, faces, high_precision=True)
        
    return verts_src, verts_tar, faces, mass_src, solver_src, G_src, M_src, pose_params, (mass_tar, solver_tar, G_tar, M_tar)

def compute_loss(pred_v, pred_grad, tar_v, G, v_mass, f_mass, mass_weighted=True):
    tar_grad = torch.bmm(G, tar_v)
    loss_v = MSE_loss(pred_v, tar_v, v_mass, mass_weighted=True)
    loss_g = MSE_loss(pred_grad, tar_grad, f_mass, mass_weighted=True)
    return loss_v, loss_g

def train_batch(batch_i):
    model.train()
    if encoder is not None:
        encoder.train()

    batch_loss_v = 0
    batch_loss_g = 0
    batch_loss = 0
    accums = 0

    while accums < grad_accum:
        verts_src, verts_tar, faces, mass_src, solver_src, G_src, M_src, pose_params, tar_ops = form_batch(train_loader, augment=True, compute_tar_ops=autoencoding)
        
        extra_features = pose_params
        
        if autoencoding:
            mass_tar, solver_tar, G_tar, M_tar = tar_ops
            
            # Encoder pass on target
            c = encoder(
                x_in=verts_tar,
                M=M_tar,
                G=G_tar,
                solver=solver_tar,
                faces=faces,
                vertex_mass=mass_tar,# Encoder takes no extra features
            )
            
            # Generate latent
            extra_features = generate_latent(c, mass_tar, use_mass_attention, use_softmax)

        preds, preds_grad = model(
            x_in=verts_src,
            M=M_src, 
            G=G_src, 
            solver=solver_src, 
            faces=faces, 
            vertex_mass=mass_src,
            extra_features=extra_features
        )

        # Align pred centroid with target centroid to stabilize training:
        preds_mean = preds.mean(dim=1, keepdim=True) # (B, 1, 3)
        tar_mean = verts_tar.mean(dim=1, keepdim=True) # (B, 1, 3)
        verts_tar = verts_tar - tar_mean
        preds = preds - preds_mean

        # L = λ_v * ‖v_tar - v_pred‖^2 + λ_g * ‖∇_src v_tar - ∇_src v_pred‖^2
        loss_v, loss_g = compute_loss(preds, preds_grad, verts_tar, G_src, mass_src, M_src, mass_weighted=mass_mse)
        loss_v = loss_v * lambda_v
        loss_g = loss_g * lambda_g
        loss = (loss_v + loss_g) / grad_accum
        loss.backward() 

        batch_loss += loss.item()
        batch_loss_v += loss_v.item() / grad_accum
        batch_loss_g += loss_g.item() / grad_accum
        accums += 1

    if clip_grad_norm is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
        if encoder is not None:
             torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip_grad_norm)

    optimizer.step()
    optimizer.zero_grad()

    if batch_i % viz_steps == 0:
        vsrc_np = to_np(verts_src[0])
        f_np = to_np(faces[0])
        vtar_np = to_np(verts_tar[0])
        preds_np = to_np(preds[0])
        
        render_src = add_text(render_mesh(vsrc_np, f_np), caption='source')
        render_tar = add_text(render_mesh(vtar_np, f_np), caption='target')
        render_pred = add_text(render_mesh(preds_np, f_np), caption='output')
        render_overlayed = add_text(render_overlayed_meshes([vtar_np, preds_np], [f_np, f_np]), caption='overlayed')
        render = torch.cat([render_src, render_tar, render_pred, render_overlayed], dim=-1)
        save_image(render, outfile('viz_train.png'))

    return batch_loss, batch_loss_v, batch_loss_g

@torch.no_grad()
def test():
    model.eval()
    if encoder is not None:
        encoder.eval()
    
    total_loss = 0
    total_loss_v = 0
    total_loss_g = 0
    total_samples = 0

    renders = []
    render_idxs = random.sample(range(len(test_dataset)), 9) if len(test_dataset) > 9 else range(len(test_dataset))
    for i in range(len(test_dataset)):
        verts_src, verts_tar, faces, mass_src, solver_src, G_src, M_src, pose_params, tar_ops = form_batch(test_loader, augment=False, compute_tar_ops=autoencoding)
        
        extra_features = pose_params
        
        if autoencoding:
            mass_tar, solver_tar, G_tar, M_tar = tar_ops
            c = encoder(
                x_in=verts_tar,
                M=M_tar,
                G=G_tar,
                solver=solver_tar,
                faces=faces,
                vertex_mass=mass_tar
            )
            extra_features = generate_latent(c, mass_tar)

        preds, preds_grad = model(
            x_in=verts_src,
            M=M_src, 
            G=G_src, 
            solver=solver_src, 
            faces=faces, 
            vertex_mass=mass_src,
            extra_features=extra_features
        )

        preds_mean = preds.mean(dim=1, keepdim=True)
        tar_mean = verts_tar.mean(dim=1, keepdim=True)
        verts_tar = verts_tar - tar_mean
        preds = preds - preds_mean

        loss_v, loss_g = compute_loss(preds, preds_grad, verts_tar, G_src, mass_src, M_src, mass_weighted=mass_mse)
        loss_v = loss_v * lambda_v
        loss_g = loss_g * lambda_g
        loss = loss_v + loss_g

        total_loss += loss.item()
        total_loss_v += loss_v.item()
        total_loss_g += loss_g.item()
        total_samples += 1

        if i in render_idxs:
            vsrc_np = to_np(verts_src[0])
            f_np = to_np(faces[0])
            vtar_np = to_np(verts_tar[0])
            preds_np = to_np(preds[0])

            render_src = add_text(render_mesh(vsrc_np, f_np), caption='source')
            render_tar = add_text(render_mesh(vtar_np, f_np), caption='target')
            render_pred = add_text(render_mesh(preds_np, f_np), caption='output')
            render_overlayed = add_text(render_overlayed_meshes([vtar_np, preds_np], [f_np, f_np]), caption='overlayed')
            render = torch.cat([render_src, render_tar, render_pred, render_overlayed], dim=-1)
            renders += [render]

    renders = image_grid(renders)
    save_image(renders, outfile('viz_test.png'))

    total_loss /= total_samples
    total_loss_v /= total_samples
    total_loss_g /= total_samples

    return total_loss


train_losses = []
train_losses_v = []
train_losses_g = []
test_losses = []
test_steps = []
pbar = tqdm(range(train_steps), dynamic_ncols=True)
for step_i in pbar:
    train_loss, train_loss_v, train_loss_g = train_batch(step_i)
    
    train_losses += [train_loss]
    train_losses_v += [train_loss_v]
    train_losses_g += [train_loss_g]

    if step_i % viz_steps == 0 and step_i > 0:
        test_loss = test()
        test_losses += [test_loss]
        test_steps += [step_i]
        
        save_dict = {
            'model': model.state_dict(),
            'encoder': encoder.state_dict() if encoder else None,
            'config': config,
            'step': step_i
        }
        torch.save(save_dict, outfile(f'poissonnet_repose_{step_i}_{test_loss:.4f}.pt'))

    if step_i % 500 == 0:
        fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(5, 20))
        ax[0].plot(train_losses, label='Train')
        ax[0].plot(train_losses_v, label='vertex')
        ax[0].plot(train_losses_g, label='gradient')
        ax[0].set_ylim(0, 0.8)
        ax[0].legend()
        ax[0].set_title('Train loss')
        ax[1].plot(test_steps, test_losses, label='Test')
        ax[1].set_title('Test loss')
        plt.tight_layout()
        plt.savefig(outfile('loss.png'))
        plt.close()

    pbar.set_description(f"Train loss: {train_loss:.5f}")

final_save = {
    'model': model.state_dict(),
    'encoder': encoder.state_dict() if encoder else None,
    'config': config
}
torch.save(final_save, outfile(f'poissonnet_repose_final.pt'))
print("Training complete")
