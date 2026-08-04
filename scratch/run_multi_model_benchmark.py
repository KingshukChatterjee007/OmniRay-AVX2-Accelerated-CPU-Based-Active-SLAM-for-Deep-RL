"""
Multi-Model Benchmark Suite on Intel Research Lab Floorplan
=============================================================
Evaluates classical and learning-based exploration baselines:
1. Yamauchi (1997) - Classical Frontier Exploration
2. Random Walk / Reactive Boundary Follower - Basic Stochastic Baseline
3. RRT Exploration (Umari & Mukhopadhyay, 2017) - Dynamic Tree Sampling
4. Stachniss (2005) - Mutual Information / Entropy Minimization
5. OmniRay (Ours) - 5-Layer Self-Adaptive System
"""

import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation
from stable_baselines3 import PPO
from envs.active_slam_env import ActiveSLAMEnv
from envs.adaptive_env import AdaptiveActiveSLAMEnv

plt.style.use("dark_background")


def get_intel_lab_walls():
    """Returns 2D wall segment geometry for Intel Research Lab."""
    return [
        (0.0, 0.0, 100.0, 0.0),
        (100.0, 0.0, 100.0, 100.0),
        (100.0, 100.0, 0.0, 100.0),
        (0.0, 100.0, 0.0, 0.0),
        (0.0, 50.0, 40.0, 50.0),
        (50.0, 50.0, 70.0, 50.0),
        (80.0, 50.0, 100.0, 50.0),
        (33.0, 50.0, 33.0, 85.0),
        (66.0, 50.0, 66.0, 85.0),
        (33.0, 15.0, 33.0, 50.0),
        (66.0, 15.0, 66.0, 50.0),
        (15.0, 25.0, 25.0, 25.0),
        (75.0, 75.0, 85.0, 75.0),
    ]


def run_yamauchi(env, max_steps=300):
    """Yamauchi (1997) Nearest Frontier Baseline."""
    obs, info = env.reset(seed=42)
    gt_trajectory, coverage_history = [(env._robot_x, env._robot_y)], [info["coverage"]]
    step_count, total_distance, collisions = 0, 0.0, 0
    start_time = time.time()

    while step_count < max_steps:
        grid_map = env.slam.map
        res, arena_size = env.map_res, env.arena_size
        unknown = (np.abs(grid_map) < 0.1)
        free = (grid_map < -0.2)
        frontiers = np.argwhere(unknown & binary_dilation(free, iterations=1))
        
        grid_x = int(env._robot_x / (arena_size / res))
        grid_y = int(env._robot_y / (arena_size / res))
        
        if len(frontiers) > 0:
            dists = np.hypot(frontiers[:, 0] - grid_y, frontiers[:, 1] - grid_x)
            nearest_idx = np.argmin(dists)
            target_gy, target_gx = frontiers[nearest_idx]
            target_x, target_y = target_gx * (arena_size / res), target_gy * (arena_size / res)
            dx, dy = target_x - env._robot_x, target_y - env._robot_y
            angle_diff = (np.arctan2(dy, dx) - env._robot_theta + np.pi) % (2 * np.pi) - np.pi
            
            lidar_scan = env.last_scan if hasattr(env, "last_scan") and env.last_scan is not None else np.ones(128) * 30.0
            if np.min(lidar_scan) < 2.5:
                angle_offset = (np.argmin(lidar_scan) / len(lidar_scan)) * 2 * np.pi
                steering = -0.6 if angle_offset < np.pi else 0.6
                speed = 0.4
            else:
                steering = np.clip(angle_diff, -0.4, 0.4)
                speed = 1.2 if abs(angle_diff) < 0.5 else 0.4
        else:
            speed, steering = 1.0, np.random.uniform(-0.3, 0.3)
        
        obs, reward, terminated, truncated, info = env.step(np.array([speed, steering], dtype=np.float32))
        step_count += 1
        gt_x, gt_y = env._robot_x, env._robot_y
        total_distance += np.hypot(gt_x - gt_trajectory[-1][0], gt_y - gt_trajectory[-1][1])
        gt_trajectory.append((gt_x, gt_y))
        coverage_history.append(info["coverage"])
        if info.get("collision", False): collisions += 1
        if terminated or truncated: break
        
    return {"name": "Yamauchi (1997)", "trajectory": np.array(gt_trajectory), "coverage": coverage_history, "final_coverage": coverage_history[-1], "total_distance": total_distance, "collisions": collisions, "steps": step_count, "wall_time": time.time() - start_time}


def run_random_walk(env, max_steps=300):
    """Random Walk / Reactive Obstacle Avoidance Baseline."""
    obs, info = env.reset(seed=42)
    gt_trajectory, coverage_history = [(env._robot_x, env._robot_y)], [info["coverage"]]
    step_count, total_distance, collisions = 0, 0.0, 0
    start_time = time.time()
    curr_steering = 0.0

    while step_count < max_steps:
        lidar_scan = env.last_scan if hasattr(env, "last_scan") and env.last_scan is not None else np.ones(128) * 30.0
        if np.min(lidar_scan) < 3.0:
            curr_steering = np.random.choice([-0.8, 0.8])
            speed = 0.3
        else:
            if np.random.rand() < 0.1:
                curr_steering = np.random.uniform(-0.3, 0.3)
            speed = 1.2
            
        obs, reward, terminated, truncated, info = env.step(np.array([speed, curr_steering], dtype=np.float32))
        step_count += 1
        gt_x, gt_y = env._robot_x, env._robot_y
        total_distance += np.hypot(gt_x - gt_trajectory[-1][0], gt_y - gt_trajectory[-1][1])
        gt_trajectory.append((gt_x, gt_y))
        coverage_history.append(info["coverage"])
        if info.get("collision", False): collisions += 1
        if terminated or truncated: break
        
    return {"name": "Random Walk", "trajectory": np.array(gt_trajectory), "coverage": coverage_history, "final_coverage": coverage_history[-1], "total_distance": total_distance, "collisions": collisions, "steps": step_count, "wall_time": time.time() - start_time}


def run_rrt_exploration(env, max_steps=300):
    """RRT-Exploration Baseline (Umari & Mukhopadhyay, 2017)."""
    obs, info = env.reset(seed=42)
    gt_trajectory, coverage_history = [(env._robot_x, env._robot_y)], [info["coverage"]]
    step_count, total_distance, collisions = 0, 0.0, 0
    start_time = time.time()

    while step_count < max_steps:
        grid_map = env.slam.map
        res, arena_size = env.map_res, env.arena_size
        unknown = (np.abs(grid_map) < 0.1)
        free = (grid_map < -0.2)
        frontiers = np.argwhere(unknown & binary_dilation(free, iterations=1))
        
        # Sample random points via RRT style to pick frontier targets
        if len(frontiers) > 0:
            num_samples = min(15, len(frontiers))
            sampled_indices = np.random.choice(len(frontiers), num_samples, replace=False)
            sampled_frontiers = frontiers[sampled_indices]
            
            grid_x = int(env._robot_x / (arena_size / res))
            grid_y = int(env._robot_y / (arena_size / res))
            dists = np.hypot(sampled_frontiers[:, 0] - grid_y, sampled_frontiers[:, 1] - grid_x)
            target_gy, target_gx = sampled_frontiers[np.argmin(dists)]
            
            target_x, target_y = target_gx * (arena_size / res), target_gy * (arena_size / res)
            dx, dy = target_x - env._robot_x, target_y - env._robot_y
            angle_diff = (np.arctan2(dy, dx) - env._robot_theta + np.pi) % (2 * np.pi) - np.pi
            
            lidar_scan = env.last_scan if hasattr(env, "last_scan") and env.last_scan is not None else np.ones(128) * 30.0
            if np.min(lidar_scan) < 2.5:
                angle_offset = (np.argmin(lidar_scan) / len(lidar_scan)) * 2 * np.pi
                steering = -0.6 if angle_offset < np.pi else 0.6
                speed = 0.4
            else:
                steering = np.clip(angle_diff, -0.4, 0.4)
                speed = 1.1 if abs(angle_diff) < 0.5 else 0.4
        else:
            speed, steering = 1.0, np.random.uniform(-0.3, 0.3)
            
        obs, reward, terminated, truncated, info = env.step(np.array([speed, steering], dtype=np.float32))
        step_count += 1
        gt_x, gt_y = env._robot_x, env._robot_y
        total_distance += np.hypot(gt_x - gt_trajectory[-1][0], gt_y - gt_trajectory[-1][1])
        gt_trajectory.append((gt_x, gt_y))
        coverage_history.append(info["coverage"])
        if info.get("collision", False): collisions += 1
        if terminated or truncated: break
        
    return {"name": "RRT-Exploration (2017)", "trajectory": np.array(gt_trajectory), "coverage": coverage_history, "final_coverage": coverage_history[-1], "total_distance": total_distance, "collisions": collisions, "steps": step_count, "wall_time": time.time() - start_time}


def run_stachniss_entropy(env, max_steps=300):
    """Stachniss et al. (2005) Mutual Information / Information-Theoretic Baseline."""
    obs, info = env.reset(seed=42)
    gt_trajectory, coverage_history = [(env._robot_x, env._robot_y)], [info["coverage"]]
    step_count, total_distance, collisions = 0, 0.0, 0
    start_time = time.time()

    while step_count < max_steps:
        grid_map = env.slam.map
        res, arena_size = env.map_res, env.arena_size
        unknown = (np.abs(grid_map) < 0.1)
        free = (grid_map < -0.2)
        frontiers = np.argwhere(unknown & binary_dilation(free, iterations=1))
        
        if len(frontiers) > 0:
            # Score candidates by maximum unknown cell density (information gain / entropy reduction)
            num_samples = min(20, len(frontiers))
            sampled_indices = np.random.choice(len(frontiers), num_samples, replace=False)
            
            best_score = -1.0
            best_target = frontiers[0]
            
            grid_x = int(env._robot_x / (arena_size / res))
            grid_y = int(env._robot_y / (arena_size / res))
            
            for idx in sampled_indices:
                fy, fx = frontiers[idx]
                dist = np.hypot(fy - grid_y, fx - grid_x) + 1e-5
                # Local window entropy / unknown count
                y_min, y_max = max(0, fy - 5), min(res, fy + 5)
                x_min, x_max = max(0, fx - 5), min(res, fx + 5)
                info_gain = np.sum(unknown[y_min:y_max, x_min:x_max])
                score = info_gain / (dist ** 0.5)
                if score > best_score:
                    best_score = score
                    best_target = (fy, fx)
                    
            target_gy, target_gx = best_target
            target_x, target_y = target_gx * (arena_size / res), target_gy * (arena_size / res)
            dx, dy = target_x - env._robot_x, target_y - env._robot_y
            angle_diff = (np.arctan2(dy, dx) - env._robot_theta + np.pi) % (2 * np.pi) - np.pi
            
            lidar_scan = env.last_scan if hasattr(env, "last_scan") and env.last_scan is not None else np.ones(128) * 30.0
            if np.min(lidar_scan) < 2.5:
                angle_offset = (np.argmin(lidar_scan) / len(lidar_scan)) * 2 * np.pi
                steering = -0.6 if angle_offset < np.pi else 0.6
                speed = 0.4
            else:
                steering = np.clip(angle_diff, -0.4, 0.4)
                speed = 1.2 if abs(angle_diff) < 0.5 else 0.4
        else:
            speed, steering = 1.0, np.random.uniform(-0.3, 0.3)
            
        obs, reward, terminated, truncated, info = env.step(np.array([speed, steering], dtype=np.float32))
        step_count += 1
        gt_x, gt_y = env._robot_x, env._robot_y
        total_distance += np.hypot(gt_x - gt_trajectory[-1][0], gt_y - gt_trajectory[-1][1])
        gt_trajectory.append((gt_x, gt_y))
        coverage_history.append(info["coverage"])
        if info.get("collision", False): collisions += 1
        if terminated or truncated: break
        
    return {"name": "Stachniss (2005)", "trajectory": np.array(gt_trajectory), "coverage": coverage_history, "final_coverage": coverage_history[-1], "total_distance": total_distance, "collisions": collisions, "steps": step_count, "wall_time": time.time() - start_time}


def run_omniray(model_path, env, max_steps=300):
    """OmniRay 5-Layer System Inference."""
    model = PPO.load(model_path)
    obs, info = env.reset(seed=42)
    gt_trajectory, coverage_history = [], [info["coverage"]]
    base_env = env.env if hasattr(env, "env") else env
    gt_trajectory.append((base_env._robot_x, base_env._robot_y))
    step_count, total_distance, collisions = 0, 0.0, 0
    start_time = time.time()

    while step_count < max_steps:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        step_count += 1
        gt_x, gt_y = base_env._robot_x, base_env._robot_y
        total_distance += np.hypot(gt_x - gt_trajectory[-1][0], gt_y - gt_trajectory[-1][1])
        gt_trajectory.append((gt_x, gt_y))
        coverage_history.append(info["coverage"])
        if info.get("collision", False): collisions += 1
        if terminated or truncated: break
        
    return {"name": "OmniRay (Ours)", "trajectory": np.array(gt_trajectory), "coverage": coverage_history, "final_coverage": coverage_history[-1], "total_distance": total_distance, "collisions": collisions, "steps": step_count, "wall_time": time.time() - start_time, "slam_map": base_env.slam.map.copy()}


if __name__ == "__main__":
    print("=" * 85)
    print("  OmniRay vs Multi-Model Baseline Benchmark — Intel Research Lab Floorplan")
    print("=" * 85)

    intel_walls = get_intel_lab_walls()
    model_path = r"results\ablation_20260728_125451\full_seed42.zip"
    save_dir = r"ablation_eval_full"
    os.makedirs(save_dir, exist_ok=True)

    def create_env():
        e = ActiveSLAMEnv(backend="simd", num_rays=128, map_resolution=50, max_steps=300, use_slam=True, real_world_noise=True)
        e._walls = intel_walls
        e.raycaster.set_walls(intel_walls)
        return e

    # Execute all 5 models
    print("\n  [1/5] Running Random Walk Baseline...")
    res_rw = run_random_walk(create_env())

    print("  [2/5] Running Yamauchi (1997) Frontier Baseline...")
    res_yam = run_yamauchi(create_env())

    print("  [3/5] Running RRT-Exploration (2017) Baseline...")
    res_rrt = run_rrt_exploration(create_env())

    print("  [4/5] Running Stachniss (2005) Information-Theoretic Baseline...")
    res_stach = run_stachniss_entropy(create_env())

    print("  [5/5] Running OmniRay 5-Layer System (Ours)...")
    adaptive_env = AdaptiveActiveSLAMEnv(create_env(), enable_health=True, enable_adaptive_reward=True, enable_meta=True, enable_curriculum=True, enable_continual=True)
    res_omni = run_omniray(model_path, adaptive_env)

    results = [res_rw, res_yam, res_rrt, res_stach, res_omni]

    # Print expanded summary table
    print("\n" + "=" * 115)
    print(f"{'Model / Algorithm':<22} | {'Coverage (%)':<12} | {'Path Length':<12} | {'Cov / Meter':<14} | {'Runtime (s)':<12} | {'Step Time':<12} | {'Collisions':<10}")
    print("-" * 115)
    for r in results:
        cov_pct = r['final_coverage'] * 100
        dist = r['total_distance']
        cov_per_m = cov_pct / dist if dist > 0 else 0.0
        step_time_ms = (r['wall_time'] / r['steps']) * 1000.0 if r['steps'] > 0 else 0.0
        print(f"{r['name']:<22} | {cov_pct:.2f}%{'':<5} | {dist:.2f} m{'':<4} | {cov_per_m:.4f} %/m{'':<4} | {r['wall_time']:.2f} s{'':<5} | {step_time_ms:.2f} ms{'':<5} | {r['collisions']:<10}")
    print("=" * 115)

    # Plot Multi-Model Comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor="#0d0d1a")
    
    # Trajectories
    ax1 = axes[0]
    ax1.set_facecolor("#0d0d1a")
    for x1, y1, x2, y2 in intel_walls:
        ax1.plot([x1, x2], [y1, y2], color="#ff6b6b", linewidth=1.8, alpha=0.7)
    
    colors = ["#ff5252", "#ff9f43", "#54a0ff", "#5f27cd", "#00ff88"]
    for r, c in zip(results, colors):
        linewidth = 2.5 if r["name"] == "OmniRay (Ours)" else 1.5
        ax1.plot(r["trajectory"][:, 0], r["trajectory"][:, 1], color=c, linewidth=linewidth, label=r["name"])
    
    ax1.set_title("Exploration Trajectories (Intel Lab)", color="white", fontsize=12, fontweight="bold")
    ax1.set_xlim(-5, 105)
    ax1.set_ylim(-5, 105)
    ax1.set_aspect("equal")
    ax1.grid(color="#333355", linestyle="--", alpha=0.5)
    ax1.tick_params(colors="white")
    ax1.legend(loc="upper left", facecolor="#0d0d1a", labelcolor="white", fontsize=9)

    # Coverage progression
    ax2 = axes[1]
    ax2.set_facecolor("#0d0d1a")
    for r, c in zip(results, colors):
        linewidth = 2.5 if r["name"] == "OmniRay (Ours)" else 1.5
        ax2.plot(np.array(r["coverage"])*100, color=c, linewidth=linewidth, label=r["name"])
        
    ax2.set_title("Coverage Rate Progression (%)", color="white", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Simulation Steps", color="white")
    ax2.set_ylabel("Map Coverage (%)", color="white")
    ax2.grid(color="#333355", linestyle="--", alpha=0.5)
    ax2.tick_params(colors="white")
    ax2.legend(loc="lower right", facecolor="#0d0d1a", labelcolor="white", fontsize=9)

    plt.suptitle("Multi-Model Exploration Benchmark — Intel Research Lab", color="white", fontsize=14, fontweight="bold")
    plt.tight_layout()
    
    out_img = os.path.join(save_dir, "multi_model_intel_benchmark.png")
    plt.savefig(out_img, dpi=150, facecolor="#0d0d1a")
    plt.close()
    print(f"\n  [SUCCESS] Benchmark report saved to: {out_img}\n")
