"""
Script to visualize SMPLX dataset entries.
Loads an entry from the baked .pt file, saves source and target meshes,
and visualizes them using viser.

Usage:
    python visualize_dataset.py --data_path train_baked_hands_100_5.0.pt --index 0
"""

import argparse
import torch
import numpy as np
import viser
import trimesh
from pathlib import Path


def save_mesh(vertices, faces, filepath):
    """Save mesh as PLY file."""
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    mesh.export(filepath)
    print(f"Saved mesh to {filepath}")


def visualize_meshes(source_verts, target_verts, faces):
    """Visualize source and target meshes using viser."""
    server = viser.ViserServer(port=8080)
    
    # Create source mesh with blue color
    source_mesh = trimesh.Trimesh(vertices=source_verts, faces=faces, process=False)
    source_mesh.visual.vertex_colors = np.array([0, 100, 255, 255])  # Blue RGBA
    
    server.scene.add_mesh_trimesh(
        name="/source_mesh",
        mesh=source_mesh,
    )
    
    # Create target mesh with orange color - offset for better visibility
    offset = np.array([1.5, 0, 0])
    target_mesh = trimesh.Trimesh(vertices=target_verts + offset, faces=faces, process=False)
    target_mesh.visual.vertex_colors = np.array([255, 140, 0, 255])  # Orange RGBA
    
    server.scene.add_mesh_trimesh(
        name="/target_mesh",
        mesh=target_mesh,
    )
    
    # Add text labels
    server.scene.add_label(
        name="/source_label",
        text="Source Mesh",
        position=(0, -1.5, 0),
    )
    
    server.scene.add_label(
        name="/target_label",
        text="Target Mesh",
        position=(1.5, -1.5, 0),
    )
    
    print("\n" + "="*60)
    print("Viser server running at http://localhost:8080")
    print("Press Ctrl+C to exit")
    print("="*60 + "\n")
    
    # Keep server running
    try:
        while True:
            pass
    except KeyboardInterrupt:
        print("\nShutting down...")


def main():
    parser = argparse.ArgumentParser(description="Visualize SMPLX dataset entries")
    parser.add_argument(
        "--data_path",
        type=str,
        default="train_baked_hands_100_5.0.pt",
        help="Path to the .pt file (relative to smplx_data folder or absolute path)"
    )
    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="Index of the entry to visualize"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./visualized_meshes",
        help="Directory to save meshes"
    )
    parser.add_argument(
        "--no_visualize",
        action="store_true",
        help="Skip visualization, only save meshes"
    )
    
    args = parser.parse_args()
    
    # Load data
    data_path = Path(args.data_path)
    if not data_path.is_absolute():
        # Assume it's relative to smplx_data folder
        script_dir = Path(__file__).parent
        data_path = script_dir / data_path
    
    print(f"Loading data from: {data_path}")
    data = torch.load(data_path)
    
    # Print dataset info
    print(f"\nDataset structure:")
    for key, value in data.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape} (dtype: {value.dtype})")
        else:
            print(f"  {key}: {value}")
    
    # Extract entry
    index = args.index
    num_samples = data['src_verts'].shape[0]
    
    if index >= num_samples:
        raise ValueError(f"Index {index} out of range. Dataset has {num_samples} samples.")
    
    src_verts = data['src_verts'][index].cpu().numpy()
    tar_verts = data['tar_verts'][index].cpu().numpy()
    faces = data['faces'].cpu().numpy()
    
    print(f"\nLoading entry {index}/{num_samples-1}")
    print(f"  Source vertices: {src_verts.shape}")
    print(f"  Target vertices: {tar_verts.shape}")
    print(f"  Faces: {faces.shape}")
    
    # Print pose and shape info if available
    if 'src_betas' in data:
        src_betas = data['src_betas'][index].cpu().numpy()
        tar_betas = data['tar_betas'][index].cpu().numpy()
        print(f"  Source body shape (first 3 betas): {src_betas[:3]}")
        print(f"  Target body shape (first 3 betas): {tar_betas[:3]}")
    
    if 'genders' in data:
        gender_map = {0: 'neutral', 1: 'male', 2: 'female'}
        gender = data['genders'][index].item()
        print(f"  Gender: {gender_map.get(gender, 'unknown')}")
    
    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    
    # Save meshes
    src_path = save_dir / f"source_mesh_{index}.ply"
    tar_path = save_dir / f"target_mesh_{index}.ply"
    
    save_mesh(src_verts, faces, src_path)
    save_mesh(tar_verts, faces, tar_path)
    
    # Visualize
    if not args.no_visualize:
        visualize_meshes(src_verts, tar_verts, faces)
    else:
        print("\nSkipping visualization (--no_visualize flag set)")


if __name__ == "__main__":
    main()

