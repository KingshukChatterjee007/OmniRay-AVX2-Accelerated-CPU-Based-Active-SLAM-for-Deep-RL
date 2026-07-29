import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Set clean publication style (light background)
plt.style.use('default')

fig, ax = plt.subplots(figsize=(16, 7.5), dpi=300)
fig.patch.set_facecolor('#FFFFFF')
ax.set_facecolor('#FFFFFF')

# Turn off axes
ax.set_xlim(0, 100)
ax.set_ylim(0, 52)
ax.axis('off')

# Helper function to draw rounded boxes
def draw_box(ax, x, y, w, h, title, subtitle="", bg_color="#F8FAFC", border_color="#0EA5E9", title_color="#0F172A", sub_color="#475569", fontsize=10, radius=1.5):
    box = patches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.3,rounding_size={radius}",
        ec=border_color, fc=bg_color, lw=1.8, zorder=3
    )
    ax.add_patch(box)
    
    if subtitle:
        ax.text(x + w/2, y + h*0.64, title, color=title_color, fontsize=fontsize, fontweight='bold', ha='center', va='center', zorder=4)
        ax.text(x + w/2, y + h*0.32, subtitle, color=sub_color, fontsize=fontsize-2.5, ha='center', va='center', zorder=4)
    else:
        ax.text(x + w/2, y + h/2, title, color=title_color, fontsize=fontsize, fontweight='bold', ha='center', va='center', zorder=4)

# Helper function to draw section bounding boxes (subgraphs)
def draw_section(ax, x, y, w, h, label, border_color="#CBD5E1", bg_color="#F1F5F9", label_color="#334155"):
    box = patches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.4,rounding_size=2.0",
        ec=border_color, fc=bg_color, lw=1.5, linestyle="--", zorder=1
    )
    ax.add_patch(box)
    # Place section header clearly above or inside top banner without text overlap
    banner = patches.FancyBboxPatch(
        (x + 1, y + h - 4), w - 2, 3.2,
        boxstyle="round,pad=0.2,rounding_size=1.0",
        ec="none", fc="#E2E8F0", zorder=2
    )
    ax.add_patch(banner)
    ax.text(x + w/2, y + h - 2.4, label.upper(), color=label_color, fontsize=9.5, fontweight='bold', ha='center', va='center', zorder=4)

# 1. Outer Container Sections (Positioned cleanly without overlapping boxes)
draw_section(ax, 2, 3, 21, 43, "State Observations", border_color="#94A3B8", bg_color="#F8FAFC", label_color="#0F172A")
draw_section(ax, 26, 3, 23, 43, "Neural Feature Encoders", border_color="#A5B4FC", bg_color="#EEF2FF", label_color="#3730A3")
draw_section(ax, 52, 3, 18, 43, "Policy Network", border_color="#FBCFE8", bg_color="#FDF2F8", label_color="#831843")
draw_section(ax, 73, 3, 25, 43, "Active SLAM Environment", border_color="#FDE68A", bg_color="#FEFCE8", label_color="#78350F")

# 2. Observation Boxes (Column 1)
draw_box(ax, 4, 34, 17, 6.5, "2D Coverage Map", "Occupancy Grid (H x W)", bg_color="#FFFFFF", border_color="#0284C7")
draw_box(ax, 4, 24, 17, 6.5, "2D SLAM Map", "Particle Filter Map", bg_color="#FFFFFF", border_color="#0284C7")
draw_box(ax, 4, 14, 17, 6.5, "1D LiDAR Sweep", "128 Ray Range Scan", bg_color="#FFFFFF", border_color="#0284C7")
draw_box(ax, 4, 4,  17, 6.5, "1D Robot Poses", "Pose & Motion Deltas", bg_color="#FFFFFF", border_color="#0284C7")

# 3. Encoders (Column 2)
draw_box(ax, 28, 25, 19, 13, "2D CNN Branch", "Conv2D -> BatchNorm -> ReLU\nMulti-channel Feature Maps", bg_color="#FFFFFF", border_color="#4F46E5", fontsize=10)
draw_box(ax, 28, 6,  19, 13, "1D MLP Branch", "Dense Layer Cascade\nKinematics Vector Embeddings", bg_color="#FFFFFF", border_color="#4F46E5", fontsize=10)

# 4. Fusion & Policy (Column 3)
draw_box(ax, 54, 23, 14, 15, "Concatenated\nFusion Layer", "256-D Feature Vector", bg_color="#ECFDF5", border_color="#059669", fontsize=10)
draw_box(ax, 54, 5,  14, 12, "PPO Policy Head", "Actor-Critic Network", bg_color="#FDF4FF", border_color="#C026D3", fontsize=10)

# 5. Environment (Column 4)
draw_box(ax, 75, 23, 21, 15, "Simulated Dynamics", "Differential Drive Kinematics\nTire Slippage & Yaw Drift", bg_color="#FFFFFF", border_color="#D97706", fontsize=10)
draw_box(ax, 75, 5,  21, 15, "Sensor Disturbances", "Gaussian LiDAR Noise\nRandom Laser Dropouts", bg_color="#FFFFFF", border_color="#D97706", fontsize=10)

# 6. Feed-forward Connections & Arrows
arrow_style = dict(arrowstyle="-|>", color="#475569", lw=1.8, mutation_scale=14)
highlight_arrow = dict(arrowstyle="-|>", color="#4F46E5", lw=2.0, mutation_scale=14)
action_arrow = dict(arrowstyle="-|>", color="#C026D3", lw=2.2, mutation_scale=15)

# Obs -> Encoders
ax.annotate("", xy=(28, 33), xytext=(21, 37.25), arrowprops=arrow_style)
ax.annotate("", xy=(28, 30), xytext=(21, 27.25), arrowprops=arrow_style)

ax.annotate("", xy=(28, 14.5), xytext=(21, 17.25), arrowprops=arrow_style)
ax.annotate("", xy=(28, 10.5), xytext=(21, 7.25),  arrowprops=arrow_style)

# Encoders -> Fusion
ax.annotate("", xy=(54, 32), xytext=(47, 31.5), arrowprops=highlight_arrow)
ax.annotate("", xy=(54, 28), xytext=(47, 12.5), arrowprops=highlight_arrow)

# Fusion -> Policy Head
ax.annotate("", xy=(61, 17), xytext=(61, 23), arrowprops=highlight_arrow)

# Policy -> Env Actions
ax.annotate("", xy=(75, 30.5), xytext=(68, 11), arrowprops=action_arrow)
ax.annotate("", xy=(75, 12.5), xytext=(68, 11), arrowprops=action_arrow)

# Action Label (Tilted cleanly along arrow path)
ax.text(71.5, 23, "Action (v, ω)", color="#C026D3", fontsize=10, fontweight='bold', ha='center', va='center', rotation=42, zorder=6,
        bbox=dict(boxstyle="round,pad=0.2", fc="#FFFFFF", ec="#C026D3", lw=1.2))

# 7. Clean Orthogonal Feedback Path (Avoiding ALL text & boxes)
# Route feedback along top exterior: (85.5, 38) -> UP to (85.5, 49) -> LEFT to (12.5, 49) -> DOWN to (12.5, 40.5)
fb_x = [85.5, 85.5, 12.5, 12.5]
fb_y = [38.0, 49.0, 49.0, 40.5]
ax.plot(fb_x, fb_y, color="#0284C7", linestyle="--", lw=2.0, zorder=5)

# Arrowhead pointing DOWN into top observation box
ax.annotate("", xy=(12.5, 40.5), xytext=(12.5, 41.5),
            arrowprops=dict(arrowstyle="-|>", color="#0284C7", lw=2.0, mutation_scale=15), zorder=6)

# Feedback Label (Positioned cleanly on top feedback line)
ax.text(49, 49, " CLOSED-LOOP STATE FEEDBACK (Step t+1) ", color="#0284C7", fontsize=9.5, fontweight='bold',
        ha='center', va='center', zorder=6, bbox=dict(boxstyle="round,pad=0.3", fc="#FFFFFF", ec="#0284C7", lw=1.5))

# Title Bar at the Very Top
plt.suptitle("OmniRay Architecture: Deep RL Active SLAM & Noise-Robust Spatial Discovery", color="#0F172A", fontsize=14, fontweight='bold', y=0.98)

# Output save
os.makedirs("assets", exist_ok=True)
output_path = os.path.join("assets", "architecture_horizontal.png")
plt.tight_layout()
plt.subplots_adjust(top=0.93, bottom=0.03, left=0.01, right=0.99)
plt.savefig(output_path, dpi=300, facecolor=fig.get_facecolor(), edgecolor='none')
plt.close()

print(f"Successfully generated clean white publication architecture diagram at: {output_path}")
