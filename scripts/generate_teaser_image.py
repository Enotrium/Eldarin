#!/usr/bin/env python3
"""
Generate Eldarin teaser image (Figure 1 style  project page).
Shows UAV 4D object detection & tracking with detection boxes, trajectories,
and multi-modal annotations.
"""
import matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.lines as mlines
from pathlib import Path
import os

# Output directory
OUTPUT_DIR = Path(__file__).parent.parent / "images"
OUTPUT_DIR.mkdir(exist_ok=True)


def draw_drone_scene(ax):
    """Draw a simulated UAV aerial view scene."""
    # Background gradient (aerial view)
    bg = np.zeros((480, 640, 3))
    # Sky/ground gradient
    for y in range(480):
        t = y / 480
        bg[y, :] = [
            0.3 + 0.2 * t,   # R
            0.4 + 0.2 * t,   # G
            0.5 + 0.15 * t,  # B
        ]
    ax.imshow(bg, extent=[0, 640, 0, 480], aspect='auto')

    # Road (perspective trapezoid)
    road_pts = np.array([[200, 0], [440, 0], [500, 480], [140, 480]])
    road = patches.Polygon(road_pts, facecolor='#555555', alpha=0.6, edgecolor='#666', linewidth=1)
    ax.add_patch(road)

    # Road markings (dashed center line)
    for y in range(20, 460, 40):
        x1 = 200 + (440 - 200) * y / 480
        x2 = 140 + (500 - 140) * y / 480
        cx = (x1 + x2) / 2
        ax.plot([cx - 5, cx + 5], [y, y], color='#FFD700', linewidth=2, alpha=0.7)

    # Buildings
    buildings = [
        (50, 100, 80, 60), (520, 80, 90, 70), (30, 250, 60, 50),
        (550, 200, 70, 80), (80, 350, 100, 60), (500, 350, 100, 50),
    ]
    for bx, by, bw, bh in buildings:
        b = patches.Rectangle((bx, by), bw, bh, facecolor='#8B7355', alpha=0.5,
                              edgecolor='#6B5335', linewidth=0.5, linestyle='--')
        ax.add_patch(b)

    # Trees
    for tx, ty in [(120, 150), (400, 100), (80, 280), (560, 300), (250, 380)]:
        tree = patches.Circle((tx, ty), 15, facecolor='#2D5A27', alpha=0.4,
                              edgecolor='#1A3A15', linewidth=0.5)
        ax.add_patch(tree)

    return bg


def draw_detection_boxes(ax):
    """Draw detection boxes with tracking IDs and trajectories."""
    # Define some objects with positions over time
    objects = {
        'car_1': {
            'trajectory': [(300, 400), (310, 370), (320, 340), (330, 310), (340, 280)],
            'color': '#FF4444',
            'label': 'Car ID:1',
            'bbox_sizes': [(45, 25), (45, 25), (42, 22), (40, 20), (38, 18)],
        },
        'car_2': {
            'trajectory': [(150, 420), (180, 380), (210, 340), (240, 300), (270, 260)],
            'color': '#4488FF',
            'label': 'Car ID:2',
            'bbox_sizes': [(48, 26), (46, 24), (44, 22), (42, 20), (40, 18)],
        },
        'truck': {
            'trajectory': [(450, 380), (470, 340), (490, 300), (510, 260), (530, 220)],
            'color': '#FF8800',
            'label': 'Truck ID:3',
            'bbox_sizes': [(60, 32), (58, 30), (55, 28), (52, 26), (50, 24)],
        },
        'bus': {
            'trajectory': [(100, 370), (120, 330), (140, 290), (160, 250), (180, 210)],
            'color': '#44CC44',
            'label': 'Bus ID:4',
            'bbox_sizes': [(70, 35), (68, 33), (65, 30), (62, 28), (60, 25)],
        },
        'pedestrian': {
            'trajectory': [(380, 430), (385, 410), (390, 390), (395, 370), (400, 350)],
            'color': '#FF44FF',
            'label': 'Person ID:5',
            'bbox_sizes': [(12, 25), (12, 25), (12, 24), (12, 23), (12, 22)],
        },
    }

    # Draw trajectories
    for obj_id, obj in objects.items():
        traj = np.array(obj['trajectory'])
        color = obj['color']

        # Trajectory trail (fading)
        for i in range(1, len(traj)):
            alpha = 0.15 + 0.85 * (i / len(traj))
            width = 1 + 3 * (i / len(traj))
            ax.plot(traj[i-1:i+1, 0], traj[i-1:i+1, 1],
                    color=color, alpha=alpha, linewidth=width)

        # Current bounding box (most recent position)
        last_pos = traj[-1]
        bw, bh = obj['bbox_sizes'][-1]
        bbox = patches.Rectangle(
            (last_pos[0] - bw/2, last_pos[1] - bh/2),
            bw, bh,
            linewidth=3, edgecolor=color, facecolor=color, alpha=0.15,
        )
        ax.add_patch(bbox)

        # Label
        ax.text(last_pos[0], last_pos[1] - bh/2 - 8, obj['label'],
                color=color, fontsize=11, fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor=color, pad=2),
                horizontalalignment='center', verticalalignment='bottom')

    return objects


def draw_velocity_arrows(ax, objects):
    """Draw velocity arrows for each tracked object."""
    for obj_id, obj in objects.items():
        traj = np.array(obj['trajectory'])
        if len(traj) >= 2:
            dx = traj[-1, 0] - traj[-2, 0]
            dy = traj[-1, 1] - traj[-2, 1]
            speed = np.sqrt(dx**2 + dy**2)
            if speed > 5:
                dx_norm = dx / speed * 25
                dy_norm = dy / speed * 25
                ax.arrow(traj[-1, 0], traj[-1, 1],
                         dx_norm, dy_norm,
                         head_width=6, head_length=8,
                         fc=obj['color'], ec=obj['color'], alpha=0.9,
                         linewidth=2)


def draw_3d_info_panel(ax, objects):
    """Draw 3D position and velocity info panel."""
    info_x, info_y = 10, 10
    panel_width, panel_height = 220, 180

    panel = FancyBboxPatch(
        (info_x, info_y), panel_width, panel_height,
        boxstyle="round,pad=5",
        facecolor='white', edgecolor='#333', linewidth=2, alpha=0.85
    )
    ax.add_patch(panel)

    ax.text(info_x + 10, info_y + 165, "📊 4D Tracking Info",
            fontsize=11, fontweight='bold', color='#222')

    # Simulated 3D info
    info_lines = [
        "Car ID:1  3D:(12.3, 8.7, 0.5)  v=4.2m/s",
        "Car ID:2  3D:(15.8, 5.2, 0.4)  v=6.1m/s",
        "Truck ID:3  3D:(20.1, -3.4, 0.8)  v=5.8m/s",
        "Bus ID:4  3D:(-5.6, 10.3, 0.6)  v=7.2m/s",
        "Person ID:5  3D:(2.1, 12.5, 1.7)  v=1.1m/s",
    ]
    for i, line in enumerate(info_lines):
        ax.text(info_x + 10, info_y + 145 - i * 25, line,
                fontsize=8, color='#444', family='monospace')


def draw_modality_indicators(ax):
    """Show active modalities as a badge."""
    x, y = 400, 10
    modalities = [
        ('RGB', '#E74C3C'), ('Event', '#F39C12'),
        ('Audio', '#3498DB'), ('IMU', '#2ECC71'),
    ]
    for i, (name, color) in enumerate(modalities):
        badge = patches.Rectangle(
            (x + i * 55, y), 50, 24,
            facecolor=color, edgecolor='white', linewidth=1, alpha=0.9
        )
        ax.add_patch(badge)
        ax.text(x + i * 55 + 25, y + 12, name,
                color='white', fontsize=9, fontweight='bold',
                horizontalalignment='center', verticalalignment='center')


def draw_event_stream_inset(ax):
    """Draw a small event stream inset (sparse event camera view)."""
    # Inset axes
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    inset_ax = inset_axes(ax, width="15%", height="15%", loc='lower right',
                          borderpad=2)

    # Simulated event frame
    event_img = np.zeros((100, 100))
    # Random events (sparse)
    np.random.seed(42)
    for _ in range(300):
        ex = np.random.randint(0, 100)
        ey = np.random.randint(0, 100)
        event_img[ey, ex] = 1 if np.random.random() > 0.5 else -1

    inset_ax.imshow(event_img, cmap='RdYlBu_r', vmin=-1, vmax=1, aspect='auto')
    inset_ax.set_title('Event Stream', fontsize=7, pad=2)
    inset_ax.axis('off')


def generate_teaser():
    """Generate the main teaser image (Figure 1 style)."""
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    fig.patch.set_facecolor('#1a1a2e')

    # Main scene
    draw_drone_scene(ax)
    objects = draw_detection_boxes(ax)
    draw_velocity_arrows(ax, objects)
    draw_3d_info_panel(ax, objects)
    draw_modality_indicators(ax)
    draw_event_stream_inset(ax)

    # Title
    ax.text(320, 475, "Eldarin: Hierarchical Multimodal 4D Object Detection & Tracking for UAVs",
            fontsize=16, fontweight='bold', color='white',
            horizontalalignment='center', verticalalignment='bottom',
            bbox=dict(facecolor='#1a1a2e', alpha=0.7, edgecolor='none', pad=8))

    # Subtitle
    ax.text(320, 462, " · Visual + Event + Audio + IMU · VSA/HDC · SNN",
            fontsize=10, color='#aaa', horizontalalignment='center',
            verticalalignment='bottom',
            bbox=dict(facecolor='#1a1a2e', alpha=0.5, edgecolor='none', pad=4))

    ax.set_xlim(0, 640)
    ax.set_ylim(0, 480)
    ax.axis('off')

    plt.tight_layout()
    output_path = OUTPUT_DIR / "eldarin-teaser.png"
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"Teaser image saved: {output_path}")


if __name__ == "__main__":
    generate_teaser()
