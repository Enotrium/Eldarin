#!/usr/bin/env python3
"""Generate white-background PNG previews of the HTML figures for README embedding."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

OUT_DIR = Path(__file__).parent.parent / "images_white"
OUT_DIR.mkdir(exist_ok=True)

COLORS_LIGHT = {
    'bg': '#ffffff',
    'white': '#222222',
    'gray': '#6f6f6f', 
    'accent_blue': '#025e8d',
    'accent_green': '#27ae60',
    'accent_orange': '#d35400',
    'accent_red': '#c0392b',
    'accent_purple': '#8e44ad',
    'accent_cyan': '#16a085',
    'accent_pink': '#e91e63',
    'modality_rgb': '#E74C3C',
    'modality_event': '#d2991d',
    'modality_audio': '#2980b9',
    'modality_imu': '#27ae60',
    'swarm_agent_0': '#e74c3c',
    'swarm_agent_1': '#2ecc71',
    'swarm_agent_2': '#3498db',
    'swarm_agent_3': '#f39c12',
    'twin_physical': '#e67e22',
    'twin_virtual': '#3498db',
    'fpga_green': '#27ae60',
}

def white_thumbnail(fig_num, title, width=800, height=500):
    """Generate a simple white-background preview image."""
    fig, ax = plt.subplots(figsize=(width/100, height/100), dpi=100)
    fig.patch.set_facecolor('#ffffff')
    ax.set_facecolor('#ffffff')
    
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.axis('off')
    
    # Card background
    from matplotlib.patches import FancyBboxPatch
    card = FancyBboxPatch((40, 40), width-80, height-80, 
                           boxstyle="round,pad=15", 
                           facecolor='#f8f9fa', edgecolor='#d5d5d5', linewidth=1.5)
    ax.add_patch(card)
    
    # Title
    ax.text(width/2, height-100, f"Figure {fig_num}: {title}", 
            fontsize=18, fontweight='bold', color='#222222',
            horizontalalignment='center', fontfamily='sans-serif')
    
    # Description  
    descriptions = {
        1: "System Architecture — Multi-modal input through\nhierarchical encoding to 4D detection & tracking",
        2: "Multi-Modal Encoders — ResNet+FPN, Event CNN,\nAudio Mel-Spec, IMU LSTM",
        3: "VSA/HDC Operations — Binding(⊗), Bundling(⊕),\nPermutation(ρ), Similarity",
        4: "Digital Twin — Virtual-physical synchronization\nwith slot-based HD memory",
        5: "Swarm Consensus — 4-UAV collaborative perception\nwith communication-aware weighting",
        6: "4D Detection & Tracking — BBoxes, 3D position,\nvelocity arrows, trajectory trails",
        7: "Communication Adaptation — Link quality effects\non modality weighting and accuracy",
        8: "FPGA Event Pipeline — Polarity split → spatial\naccumulation → quantization → voxel output",
        9: "Ablation Studies — Component contributions,\ntraining convergence, occlusion, FPS benchmarks",
        10: "UAV Deployment — FPGA + GPU + multi-sensor\nintegration with ground station",
    }
    
    ax.text(width/2, height-300, descriptions.get(fig_num, ""),
            fontsize=13, color='#6f6f6f', horizontalalignment='center',
            fontfamily='sans-serif', linespacing=1.8)
    
    # Interactive badge
    badge = FancyBboxPatch((width/2-100, 60), 200, 30, 
                            boxstyle="round,pad=5",
                            facecolor='#025e8d', edgecolor='none', alpha=0.1)
    ax.add_patch(badge)
    ax.text(width/2, 75, "🖱 Click for interactive HTML version →", 
            fontsize=10, color='#025e8d', fontweight='bold',
            horizontalalignment='center')
    
    path = OUT_DIR / f"fig_{fig_num:02d}_white.png"
    fig.savefig(path, dpi=100, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return path

if __name__ == "__main__":
    titles = {
        1: "System Architecture",
        2: "Encoder Architecture", 
        3: "VSA/HDC Operations",
        4: "Digital Twin Framework",
        5: "Swarm Consensus",
        6: "4D Detection & Tracking",
        7: "Communication Adaptation",
        8: "FPGA Event Pipeline",
        9: "Ablation Studies",
        10: "UAV Deployment",
    }
    for num, title in titles.items():
        path = white_thumbnail(num, title)
        print(f"  ✓ {path}")
    print(f"\nDone — {len(titles)} white preview images in {OUT_DIR}/")