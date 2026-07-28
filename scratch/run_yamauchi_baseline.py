"""
Yamauchi (1997) Frontier-Based Exploration Baseline vs OmniRay 5-Layer System
==============================================================================

This script implements the classical Yamauchi (1997) frontier exploration algorithm:
1. Finds frontier cells (free grid cells adjacent to unexplored unknown cells)
2. Clusters frontier cells and selects the nearest centroid via Euclidean/BFS distance
3. Navigates the robot toward the frontier target using reactive obstacle avoidance
4. Measures map coverage %, trajectory length (m), and execution time on ActiveSLAMEnv.

Runs a head-to-head evaluation against the trained OmniRay PPO policy.
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

# Force dark theme for plots
plt.style.use("dark_background")


def run_yamauchi_baseline(env, max_steps=300):
    """Executes classical Yamauchi frontier-based exploration."""
    obs, info = env.reset(seed=42)
    
    gt_trajectory = []
    coverage_history = []
    
    gt_x, gt_y, gt_theta = env._robot_x, env._robot_y, env._robot_theta
    gt_trajectory.append((gt_x, gt_y))
    coverage_history.append(info["coverage"])
    
    step_count = 0
    total_distance = 0.0
    collisions = 0
    
    start_time = time.time()
    
    while step_count < max_steps:
        # Get grid state from env
        grid_map = env.slam.map  # log-odds grid
        res = env.map_res
        arena_size = env.arena_size
        
        # 1. Identify frontiers (unknown cells adjacent to free cells)
        unknown = (np.abs(grid_map) < 0.1)  # log-odds near 0 = unknown
        free = (grid_map < -0.2)             # negative log-odds = free
        dilated_free = binary_dilation(free, iterations=1)
        frontiers = np.argwhere(unknown & dilated_free)
        
        # 2. Select nearest frontier target
        grid_x = int(env._robot_x / (arena_size / res))
        grid_y = int(env._robot_y / (arena_size / res))
        
        if len(frontiers) > 0:
            dists = np.hypot(frontiers[:, 0] - grid_y, frontiers[:, 1] - grid_x)
            nearest_idx = np.argmin(dists)
            target_gy, target_gx = frontiers[nearest_idx]
            
            target_x = target_gx * (arena_size / res)
            target_y = target_gy * (arena_size / res)
            
            # Steering toward target
            dx = target_x - env._robot_x
            dy = target_y - env._robot_y
            target_angle = np.arctan2(dy, dx)
            
            angle_diff = target_angle - env._robot_theta
            angle_diff = (angle_diff + np.pi) % (2 * np.pi) - np.pi
            
            # Reactive collision avoidance
            lidar_scan = env.last_scan if hasattr(env, "last_scan") and env.last_scan is not None else np.ones(128) * 30.0
            min_lidar = np.min(lidar_scan)
            
            if min_lidar < 2.0:
                min_idx = np.argmin(lidar_scan)
                angle_offset = (min_idx / len(lidar_scan)) * 2 * np.pi
                steering = -0.5 if angle_offset < np.pi else 0.5
                speed = 0.5
            else:
                steering = np.clip(angle_diff, -0.3, 0.3)
                speed = 1.5 if abs(angle_diff) < 0.5 else 0.5
        else:
            speed = 1.0
            steering = np.random.uniform(-0.3, 0.3)
        
        action = np.array([speed, steering], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        
        step_count += 1
        gt_x, gt_y = env._robot_x, env._robot_y
        
        # Track distance
        prev_x, prev_y = gt_trajectory[-1]
        dist_moved = np.hypot(gt_x - prev_x, gt_y - prev_y)
        total_distance += dist_moved
        
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


def run_omniray_agent(model_path, env, max_steps=300):
    """Executes OmniRay PPO agent policy."""
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
        dist_moved = np.hypot(gt_x - prev_x, gt_y - prev_y)
        total_distance += dist_moved
        
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


if __name__ == "__main__":
    print("=" * 75)
    print("  OmniRay vs Yamauchi (1997) Frontier Exploration Benchmark Comparison")
    print("=" * 75)
    
    model_path = r"results\ablation_20260728_125451\full_seed42.zip"
    save_dir = r"results\ablation_20260728_125451"
    
    # Create test env
    print("  Initializing environment under real-world noise...")
    raw_env = ActiveSLAMEnv(
        backend="simd",
        num_rays=128,
        map_resolution=50,
        max_steps=300,
        use_slam=True,
        real_world_noise=True
    )
    
    # 1. Run Yamauchi Baseline
    print("  Running Yamauchi (1997) Frontier Exploration Baseline...")
    yamauchi_res = run_yamauchi_baseline(raw_env, max_steps=300)
    
    # 2. Run OmniRay Master Policy
    print("  Running OmniRay 5-Layer Adaptive Autonomy Agent...")
    adaptive_env = AdaptiveActiveSLAMEnv(raw_env, enable_health=True, enable_adaptive_reward=True, enable_meta=True, enable_curriculum=True, enable_continual=True)
    omniray_res = run_omniray_agent(model_path, adaptive_env, max_steps=300)
    
    print("\n" + "=" * 75)
    print("  HEAD-TO-HEAD COMPARISON SUMMARY")
    print("=" * 75)
    print(f"{'Metric':<30} | {'Yamauchi (1997)':<20} | {'OmniRay (Ours)':<20}")
    print("-" * 75)
    print(f"{'Final Map Coverage (%)':<30} | {yamauchi_res['final_coverage']*100:.2f}%{'':<13} | {omniray_res['final_coverage']*100:.2f}%")
    print(f"{'Total Path Traveled (m)':<30} | {yamauchi_res['total_distance']:.2f} m{'':<12} | {omniray_res['total_distance']:.2f} m")
    print(f"{'Execution Steps':<30} | {yamauchi_res['steps']}{'':<17} | {omniray_res['steps']}")
    print(f"{'Collisions Recorded':<30} | {yamauchi_res['collisions']}{'':<17} | {omniray_res['collisions']}")
    print(f"{'Exploration Efficiency (Cov/m)':<30} | {yamauchi_res['final_coverage']*100/yamauchi_res['total_distance']:.3f}{'':<13} | {omniray_res['final_coverage']*100/omniray_res['total_distance']:.3f}")
    print("=" * 75)
    
    # Plot comparative chart
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor="#0d0d1a")
    
    # Subplot 1: Trajectory Comparison
    ax1 = axes[0]
    ax1.set_facecolor("#0d0d1a")
    ax1.plot(yamauchi_res["trajectory"][:, 0], yamauchi_res["trajectory"][:, 1], color="#ff6b6b", linestyle="--", linewidth=2.0, label="Yamauchi (1997) Frontier Path")
    ax1.plot(omniray_res["trajectory"][:, 0], omniray_res["trajectory"][:, 1], color="#00ff88", linestyle="-", linewidth=2.5, label="OmniRay 5-Layer Path")
    ax1.set_title("Robot Trajectory Comparison", color="white", fontsize=13, fontweight="bold")
    ax1.set_xlim(-5, 105)
    ax1.set_ylim(-5, 105)
    ax1.set_aspect("equal")
    ax1.grid(color="#333355", linestyle="--", alpha=0.5)
    ax1.tick_params(colors="white")
    ax1.legend(loc="upper left", facecolor="#0d0d1a", labelcolor="white")
    
    # Subplot 2: Exploration Rate Progression
    ax2 = axes[1]
    ax2.set_facecolor("#0d0d1a")
    ax2.plot(np.array(yamauchi_res["coverage"])*100, color="#ff6b6b", linewidth=2.0, linestyle="--", label="Yamauchi (1997) Frontier")
    ax2.plot(np.array(omniray_res["coverage"])*100, color="#00ff88", linewidth=2.5, label="OmniRay 5-Layer System")
    ax2.set_title("Exploration Rate Progression (%)", color="white", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Simulation Steps", color="white")
    ax2.set_ylabel("Map Coverage (%)", color="white")
    ax2.grid(color="#333355", linestyle="--", alpha=0.5)
    ax2.tick_params(colors="white")
    ax2.legend(loc="lower right", facecolor="#0d0d1a", labelcolor="white")
    
    plt.suptitle("OmniRay vs Yamauchi (1997) Frontier Exploration Benchmark", color="white", fontsize=15, fontweight="bold", y=0.98)
    plt.tight_layout()
    
    comp_plot_path = os.path.join(save_dir, "yamauchi_vs_omniray_comparison.png")
    plt.savefig(comp_plot_path, dpi=150, facecolor="#0d0d1a")
    plt.close()
    print(f"\nSaved benchmark comparison plot: {comp_plot_path}")
