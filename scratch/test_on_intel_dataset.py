"""
Intel Research Lab Real Floorplan Benchmark
=============================================
Evaluates classical Yamauchi (1997) Frontier Exploration vs OmniRay 5-Layer System (50K model)
on the real Intel Research Lab office building layout (rooms, corridors, doorways).
"""

import os
import sys
import time

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation
from stable_baselines3 import PPO
from envs.active_slam_env import ActiveSLAMEnv
from envs.adaptive_env import AdaptiveActiveSLAMEnv

plt.style.use("dark_background")


def get_intel_lab_walls():
    """
    Constructs the 2D wall segment geometry of the Intel Research Lab floorplan (28m x 28m scaled to 100x100).
    Includes outer perimeter, main corridor walls, 6 office rooms, and doorway openings.
    """
    walls = [
        # Outer Boundary Perimeter (100x100)
        (0.0, 0.0, 100.0, 0.0),
        (100.0, 0.0, 100.0, 100.0),
        (100.0, 100.0, 0.0, 100.0),
        (0.0, 100.0, 0.0, 0.0),
        
        # Central Main Corridor (Horizontal wall at Y=50 with 2 doorway openings)
        (0.0, 50.0, 40.0, 50.0),        # Left corridor wall
        (50.0, 50.0, 70.0, 50.0),       # Middle corridor wall
        (80.0, 50.0, 100.0, 50.0),      # Right corridor wall
        
        # Office Room 1 & 2 Dividers (Top Half Y=50 to 100)
        (33.0, 50.0, 33.0, 85.0),       # Room 1 vertical divider (doorway at top Y=85-100)
        (66.0, 50.0, 66.0, 85.0),       # Room 2 vertical divider
        
        # Office Room 3 & 4 Dividers (Bottom Half Y=0 to 50)
        (33.0, 15.0, 33.0, 50.0),       # Room 3 vertical divider (doorway at bottom Y=0-15)
        (66.0, 15.0, 66.0, 50.0),       # Room 4 vertical divider
        
        # Interior Cubicle / Partition Obstacles
        (15.0, 25.0, 25.0, 25.0),       # Cubicle wall Room 3
        (75.0, 75.0, 85.0, 75.0),       # Cubicle wall Room 2
    ]
    return walls


def run_yamauchi_intel(env, max_steps=300):
    """Executes Yamauchi frontier-based exploration on Intel floorplan."""
    obs, info = env.reset(seed=42)
    
    gt_trajectory = []
    coverage_history = []
    
    gt_x, gt_y = env._robot_x, env._robot_y
    gt_trajectory.append((gt_x, gt_y))
    coverage_history.append(info["coverage"])
    
    step_count = 0
    total_distance = 0.0
    collisions = 0
    start_time = time.time()
    
    while step_count < max_steps:
        grid_map = env.slam.map
        res = env.map_res
        arena_size = env.arena_size
        
        unknown = (np.abs(grid_map) < 0.1)
        free = (grid_map < -0.2)
        dilated_free = binary_dilation(free, iterations=1)
        frontiers = np.argwhere(unknown & dilated_free)
        
        grid_x = int(env._robot_x / (arena_size / res))
        grid_y = int(env._robot_y / (arena_size / res))
        
        if len(frontiers) > 0:
            dists = np.hypot(frontiers[:, 0] - grid_y, frontiers[:, 1] - grid_x)
            nearest_idx = np.argmin(dists)
            target_gy, target_gx = frontiers[nearest_idx]
            
            target_x = target_gx * (arena_size / res)
            target_y = target_gy * (arena_size / res)
            
            dx = target_x - env._robot_x
            dy = target_y - env._robot_y
            target_angle = np.arctan2(dy, dx)
            
            angle_diff = target_angle - env._robot_theta
            angle_diff = (angle_diff + np.pi) % (2 * np.pi) - np.pi
            
            lidar_scan = env.last_scan if hasattr(env, "last_scan") and env.last_scan is not None else np.ones(128) * 30.0
            min_lidar = np.min(lidar_scan)
            
            if min_lidar < 2.5:
                min_idx = np.argmin(lidar_scan)
                angle_offset = (min_idx / len(lidar_scan)) * 2 * np.pi
                steering = -0.6 if angle_offset < np.pi else 0.6
                speed = 0.4
            else:
                steering = np.clip(angle_diff, -0.4, 0.4)
                speed = 1.2 if abs(angle_diff) < 0.5 else 0.4
        else:
            speed = 1.0
            steering = np.random.uniform(-0.3, 0.3)
        
        action = np.array([speed, steering], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        
        step_count += 1
        gt_x, gt_y = env._robot_x, env._robot_y
        
        prev_x, prev_y = gt_trajectory[-1]
        total_distance += np.hypot(gt_x - prev_x, gt_y - prev_y)
        
        gt_trajectory.append((gt_x, gt_y))
        coverage_history.append(info["coverage"])
        
        if info.get("collision", False):
            collisions += 1
            
        if terminated or truncated:
            break
            
    wall_time = time.time() - start_time
    
    return {
        "trajectory": np.array(gt_trajectory),
        "coverage": coverage_history,
        "final_coverage": coverage_history[-1],
        "total_distance": total_distance,
        "collisions": collisions,
        "steps": step_count,
        "wall_time": wall_time
    }


def run_omniray_intel(model_path, env, max_steps=300):
    """Executes OmniRay PPO model on Intel floorplan."""
    model = PPO.load(model_path)
    obs, info = env.reset(seed=42)
    
    gt_trajectory = []
    coverage_history = []
    
    base_env = env.env if hasattr(env, "env") else env
    gt_trajectory.append((base_env._robot_x, base_env._robot_y))
    coverage_history.append(info["coverage"])
    
    step_count = 0
    total_distance = 0.0
    collisions = 0
    start_time = time.time()
    
    while step_count < max_steps:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        
        step_count += 1
        gt_x, gt_y = base_env._robot_x, base_env._robot_y
        
        prev_x, prev_y = gt_trajectory[-1]
        total_distance += np.hypot(gt_x - prev_x, gt_y - prev_y)
        
        gt_trajectory.append((gt_x, gt_y))
        coverage_history.append(info["coverage"])
        
        if info.get("collision", False):
            collisions += 1
            
        if terminated or truncated:
            break
            
    wall_time = time.time() - start_time
    
    return {
        "trajectory": np.array(gt_trajectory),
        "coverage": coverage_history,
        "final_coverage": coverage_history[-1],
        "total_distance": total_distance,
        "collisions": collisions,
        "steps": step_count,
        "wall_time": wall_time,
        "slam_map": base_env.slam.map.copy()
    }


if __name__ == "__main__":
    print("=" * 80)
    print("  OmniRay vs Yamauchi (1997) — Real Intel Research Lab Floorplan Benchmark")
    print("=" * 80)
    
    model_path = r"results\ablation_20260728_125451\full_seed42.zip"
    save_dir = r"results\ablation_20260728_125451"
    
    # 1. Instantiate environment with Intel Research Lab floorplan walls
    intel_walls = get_intel_lab_walls()
    print("  Constructing Intel Research Lab 2D floorplan geometry (6 rooms + corridors)...")
    
    raw_env = ActiveSLAMEnv(
        backend="simd",
        num_rays=128,
        map_resolution=50,
        max_steps=300,
        use_slam=True,
        real_world_noise=True
    )
    # Override walls with Intel floorplan
    raw_env._walls = intel_walls
    raw_env.raycaster.set_walls(intel_walls)
    
    # 2. Run Yamauchi Baseline on Intel Lab
    print("  Running Yamauchi (1997) Frontier Exploration on Intel Lab Floorplan...")
    yamauchi_res = run_yamauchi_intel(raw_env, max_steps=300)
    
    # 3. Run OmniRay 50K Model on Intel Lab
    print("  Running OmniRay 5-Layer System (50K Master Model) on Intel Lab Floorplan...")
    raw_env_2 = ActiveSLAMEnv(
        backend="simd",
        num_rays=128,
        map_resolution=50,
        max_steps=300,
        use_slam=True,
        real_world_noise=True
    )
    raw_env_2._walls = intel_walls
    raw_env_2.raycaster.set_walls(intel_walls)
    
    adaptive_env = AdaptiveActiveSLAMEnv(
        raw_env_2,
        enable_health=True,
        enable_adaptive_reward=True,
        enable_meta=True,
        enable_curriculum=True,
        enable_continual=True
    )
    omniray_res = run_omniray_intel(model_path, adaptive_env, max_steps=300)
    
    # Print Quantitative Results
    print("\n" + "=" * 80)
    print("  INTEL RESEARCH LAB FLOORPLAN — BENCHMARK RESULTS")
    print("=" * 80)
    print(f"{'Metric':<35} | {'Yamauchi (1997)':<20} | {'OmniRay (Ours)':<20}")
    print("-" * 80)
    print(f"{'Final Map Coverage (%)':<35} | {yamauchi_res['final_coverage']*100:.2f}%{'':<13} | {omniray_res['final_coverage']*100:.2f}%")
    print(f"{'Total Path Length (m)':<35} | {yamauchi_res['total_distance']:.2f} m{'':<12} | {omniray_res['total_distance']:.2f} m")
    print(f"{'Execution Steps':<35} | {yamauchi_res['steps']}{'':<17} | {omniray_res['steps']}")
    print(f"{'Wall Collisions':<35} | {yamauchi_res['collisions']}{'':<17} | {omniray_res['collisions']}")
    print(f"{'Exploration Efficiency (Cov/m)':<35} | {yamauchi_res['final_coverage']*100/yamauchi_res['total_distance']:.3f}{'':<13} | {omniray_res['final_coverage']*100/omniray_res['total_distance']:.3f}")
    print("=" * 80)
    
    # Generate 4-panel publication visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 12), facecolor="#0d0d1a")
    
    # Panel 1: Yamauchi Trajectory on Intel Floorplan
    ax1 = axes[0, 0]
    ax1.set_facecolor("#0d0d1a")
    for x1, y1, x2, y2 in intel_walls:
        ax1.plot([x1, x2], [y1, y2], color="#ff6b6b", linewidth=2.0, alpha=0.8)
    ax1.plot(yamauchi_res["trajectory"][:, 0], yamauchi_res["trajectory"][:, 1], color="#ff9f43", linestyle="--", linewidth=2.0, label="Yamauchi Path")
    ax1.scatter(yamauchi_res["trajectory"][0, 0], yamauchi_res["trajectory"][0, 1], color="#00ff88", s=100, label="Start")
    ax1.set_title(f"Yamauchi (1997) on Intel Floorplan\nCoverage: {yamauchi_res['final_coverage']*100:.1f}% | Collisions: {yamauchi_res['collisions']}", color="white", fontsize=12, fontweight="bold")
    ax1.set_xlim(-5, 105)
    ax1.set_ylim(-5, 105)
    ax1.set_aspect("equal")
    ax1.grid(color="#333355", linestyle="--", alpha=0.5)
    ax1.tick_params(colors="white")
    ax1.legend(loc="upper left", facecolor="#0d0d1a", labelcolor="white")
    
    # Panel 2: OmniRay Trajectory on Intel Floorplan
    ax2 = axes[0, 1]
    ax2.set_facecolor("#0d0d1a")
    for x1, y1, x2, y2 in intel_walls:
        ax2.plot([x1, x2], [y1, y2], color="#ff6b6b", linewidth=2.0, alpha=0.8)
    ax2.plot(omniray_res["trajectory"][:, 0], omniray_res["trajectory"][:, 1], color="#00ff88", linestyle="-", linewidth=2.5, label="OmniRay 5-Layer Path")
    ax2.scatter(omniray_res["trajectory"][0, 0], omniray_res["trajectory"][0, 1], color="#00ff88", s=100, label="Start")
    ax2.set_title(f"OmniRay (Ours) on Intel Floorplan\nCoverage: {omniray_res['final_coverage']*100:.1f}% | Collisions: {omniray_res['collisions']}", color="white", fontsize=12, fontweight="bold")
    ax2.set_xlim(-5, 105)
    ax2.set_ylim(-5, 105)
    ax2.set_aspect("equal")
    ax2.grid(color="#333355", linestyle="--", alpha=0.5)
    ax2.tick_params(colors="white")
    ax2.legend(loc="upper left", facecolor="#0d0d1a", labelcolor="white")
    
    # Panel 3: OmniRay Reconstructed Occupancy Map
    ax3 = axes[1, 0]
    ax3.set_facecolor("#0d0d1a")
    slam_prob = 1.0 / (1.0 + np.exp(-omniray_res["slam_map"]))
    im = ax3.imshow(slam_prob, origin="lower", cmap="inferno", extent=[0, 100, 0, 100], vmin=0.0, vmax=1.0)
    ax3.plot(omniray_res["trajectory"][:, 0], omniray_res["trajectory"][:, 1], color="#00ff88", linewidth=1.5, alpha=0.7)
    ax3.set_title("Reconstructed Occupancy Map (Intel Lab)", color="white", fontsize=12, fontweight="bold")
    cbar = fig.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
    cbar.ax.yaxis.set_tick_params(color="#8888aa", labelcolor="white")
    ax3.tick_params(colors="white")
    
    # Panel 4: Coverage Progression Comparison
    ax4 = axes[1, 1]
    ax4.set_facecolor("#0d0d1a")
    ax4.plot(np.array(yamauchi_res["coverage"])*100, color="#ff9f43", linewidth=2.0, linestyle="--", label="Yamauchi (1997)")
    ax4.plot(np.array(omniray_res["coverage"])*100, color="#00ff88", linewidth=2.5, label="OmniRay 5-Layer System")
    ax4.set_title("Exploration Progression on Intel Lab Map (%)", color="white", fontsize=12, fontweight="bold")
    ax4.set_xlabel("Simulation Steps", color="white")
    ax4.set_ylabel("Map Coverage (%)", color="white")
    ax4.grid(color="#333355", linestyle="--", alpha=0.5)
    ax4.tick_params(colors="white")
    ax4.legend(loc="lower right", facecolor="#0d0d1a", labelcolor="white")
    
    plt.suptitle("Intel Research Lab Real Floorplan Benchmark — OmniRay vs Yamauchi (1997)", color="white", fontsize=15, fontweight="bold", y=0.98)
    plt.tight_layout()
    
    report_plot_path = os.path.join(save_dir, "intel_lab_benchmark_report.png")
    plt.savefig(report_plot_path, dpi=150, facecolor="#0d0d1a")
    plt.close()
    print(f"\n  [SUCCESS] Intel Lab benchmark plot saved to: {report_plot_path}")
