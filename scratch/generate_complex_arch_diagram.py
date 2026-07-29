import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Set clean publication style (light background)
plt.style.use('default')

fig, ax = plt.subplots(figsize=(18, 9.5), dpi=300)
fig.patch.set_facecolor('#FFFFFF')
ax.set_facecolor('#FFFFFF')

# Turn off axes
ax.set_xlim(0, 100)
ax.set_ylim(0, 56)
ax.axis('off')

# Helper function to draw rounded boxes with mathematical details
def draw_box(ax, x, y, w, h, title, formula="", desc="", bg_color="#FFFFFF", border_color="#0EA5E9", title_color="#0F172A", formula_color="#1E40AF", desc_color="#475569", fontsize=9.5, radius=1.5):
    box = patches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.3,rounding_size={radius}",
        ec=border_color, fc=bg_color, lw=1.8, zorder=3
    )
    ax.add_patch(box)
    
    ax.text(x + w/2, y + h - 1.8, title, color=title_color, fontsize=fontsize, fontweight='bold', ha='center', va='top', zorder=4)
    
    if formula:
        ax.text(x + w/2, y + h/2 + 0.2, formula, color=formula_color, fontsize=fontsize-0.5, fontweight='bold', ha='center', va='center', zorder=4)
        
    if desc:
        ax.text(x + w/2, y + 1.5, desc, color=desc_color, fontsize=fontsize-2.2, ha='center', va='bottom', zorder=4)

# Helper function to draw section container boxes
def draw_section(ax, x, y, w, h, label, border_color="#CBD5E1", bg_color="#F8FAFC", label_color="#334155"):
    box = patches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.4,rounding_size=2.0",
        ec=border_color, fc=bg_color, lw=1.5, linestyle="--", zorder=1
    )
    ax.add_patch(box)
    
    banner = patches.FancyBboxPatch(
        (x + 1, y + h - 4.2), w - 2, 3.4,
        boxstyle="round,pad=0.2,rounding_size=1.0",
        ec="none", fc="#E2E8F0", zorder=2
    )
    ax.add_patch(banner)
    ax.text(x + w/2, y + h - 2.5, label.upper(), color=label_color, fontsize=10, fontweight='bold', ha='center', va='center', zorder=4)

# 1. Outer Container Sections
draw_section(ax, 2, 3, 23, 46, "1. Mathematical State Observations", border_color="#94A3B8", bg_color="#F8FAFC", label_color="#0F172A")
draw_section(ax, 27, 3, 23, 46, "2. Neural Feature Encoders", border_color="#A5B4FC", bg_color="#EEF2FF", label_color="#3730A3")
draw_section(ax, 52, 3, 20, 46, "3. Policy Network & PPO Objective", border_color="#FBCFE8", bg_color="#FDF2F8", label_color="#831843")
draw_section(ax, 74, 3, 24, 46, "4. Kinodynamics & VectorSLAM", border_color="#FDE68A", bg_color="#FEFCE8", label_color="#78350F")

# 2. Observation Boxes (Column 1)
draw_box(ax, 4, 34, 19, 9, "2D Coverage Grid Map", r"$M_{cov}^{(t)} \in \{0, 1\}^{H \times W}$", "Explored Frontier Tensor", border_color="#0284C7")
draw_box(ax, 4, 24, 19, 9, "2D Particle SLAM Log-Odds", r"$M_{slam}^{(t)} = \log \frac{P(m_{ij}=1)}{1-P(m_{ij}=1)}$", "Bayesian Grid Belief", border_color="#0284C7")
draw_box(ax, 4, 14, 19, 9, "1D LiDAR Sweep Array", r"$z_t = \{r_k\}_{k=1}^{128}, r_k \in [0, d_{max}]$", "AVX2 SIMD Raycast Ranges", border_color="#0284C7")
draw_box(ax, 4, 4,  19, 9, "1D Robot Kinematic State", r"$p_t = [x, y, \theta, v_{t-1}, \omega_{t-1}]^T$", "Continuous Pose & Vel History", border_color="#0284C7")

# 3. Encoders (Column 2)
draw_box(ax, 29, 25, 19, 18, "2D Spatial CNN Branch", r"$\mathbf{h}_{2D} = \Phi_{CNN}([M_{cov} \ || \ M_{slam}])$", "Conv2D(16,32) -> BN -> ReLU\nOutput Tensor: 128-D Embedding", bg_color="#FFFFFF", border_color="#4F46E5", fontsize=10)
draw_box(ax, 29, 5,  19, 18, "1D Kinematic MLP Branch", r"$\mathbf{h}_{1D} = \Phi_{MLP}([z_t \ || \ p_t])$", "Dense(128) -> ReLU -> Dense(128)\nOutput Tensor: 128-D Embedding", bg_color="#FFFFFF", border_color="#4F46E5", fontsize=10)

# 4. Fusion & Policy (Column 3)
draw_box(ax, 54, 28, 16, 15, "Concatenated Fusion", r"$\mathbf{z}_t = [\mathbf{h}_{2D} \ || \ \mathbf{h}_{1D}] \in \mathbb{R}^{256}$", "Joint Spatial-Kinematic Vector", bg_color="#ECFDF5", border_color="#059669", fontsize=10)
draw_box(ax, 54, 5,  16, 21, "PPO Actor-Critic Head", r"$\pi_\theta(a_t|s_t) = \mathcal{N}(\mu_\theta(\mathbf{z}_t), \Sigma)$"+"\n\n"+r"$V_\phi(s_t) = W_V \mathbf{z}_t + b_V$"+"\n\n"+r"$\mathcal{L}_{PPO} = \hat{\mathbb{E}}_t [\min(r_t \hat{A}_t, \text{clip}(r_t) \hat{A}_t)]$", "Continuous Gaussian Action & Value Loss", bg_color="#FDF4FF", border_color="#C026D3", fontsize=9.5)

# 5. Kinodynamics & VectorSLAM (Column 4)
draw_box(ax, 76, 27, 20, 16, "Kinodynamic Slippage", r"$[x_{t+1}, y_{t+1}, \theta_{t+1}]^T = [x_t, y_t, \theta_t]^T +$"+"\n"+r"$[(v_t+\eta_v)\cos\theta_t \Delta t, (v_t+\eta_v)\sin\theta_t \Delta t, (\omega_t+\eta_\omega)\Delta t]^T$", "Tire Slip & Yaw Drift Noise", bg_color="#FFFFFF", border_color="#D97706", fontsize=8.5)
draw_box(ax, 76, 5,  20, 18, "VectorSLAM Particle Filter", r"$w_i^{(t)} \propto w_i^{(t-1)} \prod_{k=1}^{128} \exp\left(-\frac{(r_k - \hat{r}_k^{(i)})^2}{2\sigma_{lidar}^2}\right)$"+"\n\n"+r"$\hat{x}_{slam}^{(t)} = \sum_{i=1}^N w_i^{(t)} x_i^{(t)}$", "NumPy Vectorized Scan-Matching", bg_color="#FFFFFF", border_color="#D97706", fontsize=9)

# 6. Feed-forward Arrows
arrow_style = dict(arrowstyle="-|>", color="#475569", lw=1.8, mutation_scale=14)
highlight_arrow = dict(arrowstyle="-|>", color="#4F46E5", lw=2.0, mutation_scale=14)
action_arrow = dict(arrowstyle="-|>", color="#C026D3", lw=2.2, mutation_scale=15)

# Obs -> Encoders
ax.annotate("", xy=(29, 36), xytext=(23, 38.5), arrowprops=arrow_style)
ax.annotate("", xy=(29, 31), xytext=(23, 28.5), arrowprops=arrow_style)

ax.annotate("", xy=(29, 16), xytext=(23, 18.5), arrowprops=arrow_style)
ax.annotate("", xy=(29, 11), xytext=(23, 8.5),  arrowprops=arrow_style)

# Encoders -> Fusion
ax.annotate("", xy=(54, 37), xytext=(48, 34), arrowprops=highlight_arrow)
ax.annotate("", xy=(54, 32), xytext=(48, 14), arrowprops=highlight_arrow)

# Fusion -> Policy Head
ax.annotate("", xy=(62, 26), xytext=(62, 28), arrowprops=highlight_arrow)

# Policy -> Env Actions
ax.annotate("", xy=(76, 35), xytext=(70, 15), arrowprops=action_arrow)
ax.annotate("", xy=(76, 14), xytext=(70, 15), arrowprops=action_arrow)

# Action Label (Tilted cleanly along arrow path)
ax.text(73, 25, r"Action $a_t = [v_t, \omega_t]^T$", color="#C026D3", fontsize=9.5, fontweight='bold', ha='center', va='center', rotation=42, zorder=6,
        bbox=dict(boxstyle="round,pad=0.25", fc="#FFFFFF", ec="#C026D3", lw=1.2))

# 7. Clean Orthogonal Feedback Path
fb_x = [86.0, 86.0, 13.5, 13.5]
fb_y = [43.0, 52.0, 52.0, 43.5]
ax.plot(fb_x, fb_y, color="#0284C7", linestyle="--", lw=2.0, zorder=5)

# Arrowhead pointing DOWN into top observation box
ax.annotate("", xy=(13.5, 43.5), xytext=(13.5, 44.5),
            arrowprops=dict(arrowstyle="-|>", color="#0284C7", lw=2.0, mutation_scale=15), zorder=6)

# Feedback Label (Positioned cleanly on top feedback line)
ax.text(49.5, 52.0, r" CLOSED-LOOP STATE TRANSITION: $s_{t+1} \sim P(s_{t+1} | s_t, a_t)$ ", color="#0284C7", fontsize=9.5, fontweight='bold',
        ha='center', va='center', zorder=6, bbox=dict(boxstyle="round,pad=0.3", fc="#FFFFFF", ec="#0284C7", lw=1.5))

# Title Bar at the Very Top
plt.suptitle("OmniRay Mathematical Architecture: Deep RL Active SLAM & Closed-Loop Dynamics", color="#0F172A", fontsize=14, fontweight='bold', y=0.985)

# Output save locally (NOT pushing to git yet as instructed)
os.makedirs("assets", exist_ok=True)
output_path = os.path.join("assets", "architecture_detailed_formulas.png")
plt.subplots_adjust(top=0.93, bottom=0.02, left=0.01, right=0.99)
plt.savefig(output_path, dpi=300, facecolor=fig.get_facecolor(), edgecolor='none')
plt.close()

print(f"Successfully generated detailed mathematical architecture diagram at: {output_path}")
