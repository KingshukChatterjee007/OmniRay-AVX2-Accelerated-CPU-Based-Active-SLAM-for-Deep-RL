import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Set dark background theme
plt.style.use('dark_background')

fig, ax = plt.subplots(figsize=(16, 7.5), dpi=300)
fig.patch.set_facecolor('#0B0F19')
ax.set_facecolor('#0B0F19')

# Turn off axes
ax.set_xlim(0, 100)
ax.set_ylim(0, 50)
ax.axis('off')

# Helper function to draw rounded boxes
def draw_box(ax, x, y, w, h, title, subtitle="", bg_color="#1E293B", border_color="#38BDF8", title_color="#F8FAFC", sub_color="#94A3B8", fontsize=10, radius=1.5):
    box = patches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.3,rounding_size={radius}",
        ec=border_color, fc=bg_color, lw=1.8, zorder=2
    )
    ax.add_patch(box)
    
    if subtitle:
        ax.text(x + w/2, y + h*0.62, title, color=title_color, fontsize=fontsize, fontweight='bold', ha='center', va='center', zorder=3)
        ax.text(x + w/2, y + h*0.35, subtitle, color=sub_color, fontsize=fontsize-2, ha='center', va='center', zorder=3)
    else:
        ax.text(x + w/2, y + h/2, title, color=title_color, fontsize=fontsize, fontweight='bold', ha='center', va='center', zorder=3)

# Helper function to draw section bounding boxes (subgraphs)
def draw_section(ax, x, y, w, h, label, border_color="#334155", bg_color="#0F172A"):
    box = patches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.5,rounding_size=2.0",
        ec=border_color, fc=bg_color, lw=1.2, linestyle="--", zorder=1
    )
    ax.add_patch(box)
    ax.text(x + 2, y + h - 2.5, label.upper(), color="#64748B", fontsize=9, fontweight='bold', ha='left', va='top', zorder=3)

# 1. Sections
draw_section(ax, 2, 4, 21, 42, "State Observations")
draw_section(ax, 26, 4, 23, 42, "Neural Feature Extraction")
draw_section(ax, 52, 4, 18, 42, "Policy Network")
draw_section(ax, 73, 4, 25, 42, "Active SLAM Environment")

# 2. Observation Boxes (Left Column)
draw_box(ax, 4, 36, 17, 7, "2D Coverage Map", "Occupancy Grid (H x W)", bg_color="#131C2E", border_color="#38BDF8")
draw_box(ax, 4, 26, 17, 7, "2D SLAM Map", "Particle Filter Map", bg_color="#131C2E", border_color="#38BDF8")
draw_box(ax, 4, 16, 17, 7, "1D LiDAR Sweep", "128 Ray Range Scan", bg_color="#131C2E", border_color="#38BDF8")
draw_box(ax, 4, 6,  17, 7, "1D Robot Poses", "Pose & Motion Deltas", bg_color="#131C2E", border_color="#38BDF8")

# 3. Encoders (Second Column)
draw_box(ax, 28, 29, 19, 11, "2D CNN Branch", "Conv2D -> BatchNorm -> ReLU\nMulti-channel Feature Map", bg_color="#1E1B4B", border_color="#818CF8", fontsize=9.5)
draw_box(ax, 28, 9,  19, 11, "1D MLP Branch", "Dense Layer Cascade\nVector Kinematics Embeddings", bg_color="#1E1B4B", border_color="#818CF8", fontsize=9.5)

# 4. Fusion & Policy (Third Column)
draw_box(ax, 54, 16, 14, 20, "Concatenated\nFusion Layer", "256-D Feature Vector", bg_color="#064E3B", border_color="#34D399", fontsize=10)
draw_box(ax, 54, 6,  14, 7, "PPO Policy Head", "Actor-Critic Network", bg_color="#4C1D95", border_color="#F472B6", fontsize=9.5)

# 5. Environment (Right Column)
draw_box(ax, 75, 27, 21, 13, "Simulated Dynamics", "Differential Drive Kinematics\nTire Slippage & Drift", bg_color="#2D1B00", border_color="#F59E0B", fontsize=9.5)
draw_box(ax, 75, 9,  21, 13, "Sensor Disturbances", "Gaussian LiDAR Noise\nRandom Laser Dropouts", bg_color="#2D1B00", border_color="#F59E0B", fontsize=9.5)

# Arrows / Connections
arrow_props = dict(arrowstyle="-|>", color="#94A3B8", lw=1.8, mutation_scale=15)
highlight_arrow = dict(arrowstyle="-|>", color="#38BDF8", lw=2.0, mutation_scale=15)
action_arrow = dict(arrowstyle="-|>", color="#F472B6", lw=2.2, mutation_scale=16)
feedback_arrow = dict(arrowstyle="-|>", color="#64748B", lw=1.5, linestyle="--", mutation_scale=12)

# Obs -> Encoders
ax.annotate("", xy=(28, 34.5), xytext=(21, 39.5), arrowprops=arrow_props)
ax.annotate("", xy=(28, 34.5), xytext=(21, 29.5), arrowprops=arrow_props)

ax.annotate("", xy=(28, 14.5), xytext=(21, 19.5), arrowprops=arrow_props)
ax.annotate("", xy=(28, 14.5), xytext=(21, 9.5),  arrowprops=arrow_props)

# Encoders -> Fusion
ax.annotate("", xy=(54, 28.0), xytext=(47, 34.5), arrowprops=highlight_arrow)
ax.annotate("", xy=(54, 24.0), xytext=(47, 14.5), arrowprops=highlight_arrow)

# Fusion -> Policy
ax.annotate("", xy=(61, 13.0), xytext=(61, 16.0), arrowprops=highlight_arrow)

# Policy -> Env
ax.annotate("", xy=(75, 33.5), xytext=(68, 9.5), arrowprops=action_arrow)
ax.annotate("", xy=(75, 15.5), xytext=(68, 9.5), arrowprops=action_arrow)

# Label on Action Arrow
ax.text(71.5, 23.5, "Action (v, ω)", color="#F472B6", fontsize=9.5, fontweight='bold', ha='center', va='center', rotation=42, zorder=5)

# Feedback Loop (Top & Bottom paths)
# Top feedback
ax.annotate("", xy=(12.5, 43.0), xytext=(85.5, 40.0),
            arrowprops=dict(arrowstyle="-|>", color="#38BDF8", lw=1.5, linestyle="--",
                            connectionstyle="bar,angle=180,fraction=-0.18", mutation_scale=14))
ax.text(50, 48.2, "CLOSED-LOOP STATE FEEDBACK (Observation Step t+1)", color="#38BDF8", fontsize=9, fontweight='bold', ha='center', va='center')

# Title Bar at the Very Top
plt.suptitle("OmniRay Architecture: Deep RL Active SLAM & Noise-Robust Spatial Discovery", color="#F8FAFC", fontsize=14, fontweight='bold', y=0.98)

# Output directory check
os.makedirs("assets", exist_ok=True)
output_path = os.path.join("assets", "architecture_horizontal.png")
plt.tight_layout()
plt.subplots_adjust(top=0.92, bottom=0.05, left=0.02, right=0.98)
plt.savefig(output_path, dpi=300, facecolor=fig.get_facecolor(), edgecolor='none')
plt.close()

print(f"Successfully generated research-grade architecture diagram at: {output_path}")
