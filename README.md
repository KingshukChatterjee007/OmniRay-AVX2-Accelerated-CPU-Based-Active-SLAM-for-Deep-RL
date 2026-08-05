# OmniRay: AVX2-Accelerated CPU-Based Active SLAM for Deep RL

**A pluggable SIMD raycasting engine, vectorized particle filter, and Gymnasium environment for training and benchmarking Active SLAM agents on consumer CPU hardware.**

<p align="center">
  <strong>Author:</strong> Kingshuk Chatterjee &nbsp;|&nbsp;
  <strong>Co-Author:</strong> Ayush Ranjan &nbsp;|&nbsp;
  <strong>Co-Author:</strong> Raghav Singh Parihar
</p>

---

## Overview & Architecture

OmniRay integrates a 256-bit C++ AVX2 SIMD raycasting backend, a vectorized NumPy particle filter (`VectorSLAM`), and a Gymnasium environment with sim-to-real noise models. It enables end-to-end Deep Reinforcement Learning (DRL) for Active SLAM on consumer CPUs without dedicated GPUs.

![OmniRay Architecture Diagram](assets/architecture_detailed_formulas.png)

---

## Hardware Disclosure

All reported performance metrics were measured on a consumer ultrabook:

| Component | Specification |
| :--- | :--- |
| **Device** | ASUS Zenbook S13 OLED |
| **CPU** | Intel Core i7-1355U (10-core / 12-thread, hybrid P+E architecture) |
| **GPU** | None (integrated Intel Iris Xe Graphics only) |
| **RAM** | 16 GB LPDDR5x, 5200 MHz |
| **SIMD ISA** | AVX2 (256-bit, 8-lane parallel raycasting) |
| **OS / Toolchain** | Python 3.11; C++ backend compiled via CMake (Release) |

---

## 5-Layer Self-Adaptive Autonomy System

OmniRay features a 5-layer hierarchical feedback control architecture for real-time autonomy adaptation:

| Layer | Module | Function |
| :---: | :--- | :--- |
| **1** | `health_monitor.py` | **System State Monitoring** — Computes health score $H_t \in [0, 1]$ from entropy rate, coverage velocity, and SLAM divergence. |
| **2** | `adaptive_reward.py` | **Dynamic Reward Adaptation** — Adjusts reward coefficients in real time based on system health. |
| **3** | `meta_policy.py` | **Meta-Policy Controller** — Neural controller optimizing reward parameters via policy gradient updates. |
| **4** | `curriculum.py` | **Curriculum Manager** — Dynamically adjusts map difficulty, noise, and step budgets. |
| **5** | `continual_learner.py` | **Continual Learner** — In-deployment replay buffer with online updates and checkpoint/rollback safety. |

---

## Quick Start

### 1. Install Dependencies & Build SIMD Core

```bash
pip install -r requirements.txt

# Compile C++ SIMD backend
cd sim
mkdir build
cd build
cmake ..
cmake --build . --config Release
cd ../..
```

### 2. Run Agent Visualizer & Profiler

```powershell
# Visualize pretrained agent
py -3.11 visualize_agent.py --model-path active_slam_ppo_robust_master.zip --episodes 3

# Benchmark backends & test environment
py -3.11 -m profiling.benchmark_bottleneck --rays 360 --iterations 500
py -3.11 test_env.py --backend simd --episodes 3
```

### 3. Adaptive Training

```powershell
py -3.11 train_rl.py --adaptive --meta-policy --curriculum --continual --total-steps 100000
```

---

## Quantitative Benchmarks

### 1. Raycasting Core Latency (360 Rays)

| Backend | Mean Scan Time | Median Scan Time | P99 Scan Time | Estimated Time (100K Steps) | Performance Profile |
| :--- | :---: | :---: | :---: | :---: | :--- |
| Pure Python | 6.139 ms | 6.469 ms | 7.636 ms | 10.2 min | Baseline |
| PyMunk (`segment_query`) | 3.009 ms | 3.447 ms | 4.785 ms | 5.0 min | Moderate |
| NumPy (vectorized) | 0.225 ms | 0.262 ms | 0.373 ms | 0.4 min (24 s) | Fast |
| **C++ SIMD (AVX2)** | **0.038 ms** | **0.038 ms** | **0.089 ms** | **0.06 min (3.8 s)** | **Ultra-Low Latency (~26× speedup)** |

### 2. Fixed Evaluation Suite & Multi-Sequence Benchmark

| Sequence / Environment | ATE RMSE (m) | Loop Closure Failures | Peak RAM (MB) | Scalar CPU Latency ($\mu\text{s}$) | AVX2 SIMD Latency ($\mu\text{s}$) | Hardware Speedup |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Intel Research Lab (Seattle)** | 0.042 | 0 | 41.8 MB | 962 $\mu\text{s}$ | **37.1 $\mu\text{s}$** | **25.9×** |
| **MIT Stata Center** | 0.058 | 0 | 54.2 MB | 1,420 $\mu\text{s}$ | **52.4 $\mu\text{s}$** | **27.1×** |
| **Freiburg Building 52** | 0.039 | 0 | 38.5 MB | 810 $\mu\text{s}$ | **31.8 $\mu\text{s}$** | **25.5×** |
| **Holdout Set: ACES (UT Austin)** | 0.064 | 1 | 62.1 MB | 1,680 $\mu\text{s}$ | **61.2 $\mu\text{s}$** | **27.5×** |

### 3. Multi-Model Baseline Comparisons

**Intel Research Lab Floorplan:**

| Model / Algorithm | Coverage (%) | Path Length (m) | Coverage / Meter (%/m) | Decision Latency (ms) | Peak RAM (MB) | Collision Count |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Walk** | 65.55 ± 12.52% | 148.57 ± 67.34 m | 0.4411 %/m | 7.16 ms | 34.2 MB | 225.7 ± 33.7 |
| **Yamauchi (1997)** | 70.85 ± 14.67% | 155.06 ± 64.90 m | 0.4570 %/m | 8.61 ms | 39.5 MB | 192.3 ± 39.9 |
| **RRT-Exploration (2017)** | 85.55 ± 13.57% | 267.64 ± 33.16 m | 0.3200 %/m | 10.49 ms | 44.8 MB | 30.7 ± 28.2 |
| **Stachniss (2005)** | 95.43 ± 0.07% | 207.65 ± 22.06 m | 0.4600 %/m | 18.51 ms | 52.1 MB | **0.0 ± 0.0** |
| **OmniRay (Ours)** | **90.87 ± 3.29%** | **418.12 ± 106.25 m** | **0.2172 %/m** | **3.09 ms** | **41.8 MB** | **62.3 ± 46.5** |

![Intel Lab Benchmark Report](ablation_eval_full/intel_lab_benchmark_report.png)

**MIT Stata Center Zero-Shot Benchmark:**

| Model / Algorithm | Coverage (%) | Path Length (m) | Coverage / Meter (%/m) | Decision Latency (ms) | Peak RAM (MB) | Collision Count |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Walk** | 59.81 ± 19.40% | 140.14 ± 96.38 m | 0.4268 %/m | 7.10 ms | 35.1 MB | 230.0 ± 48.3 |
| **Yamauchi (1997)** | 91.04 ± 4.40% | 281.62 ± 37.18 m | 0.3233 %/m | 10.29 ms | 41.2 MB | 28.3 ± 20.5 |
| **RRT-Exploration (2017)** | 95.08 ± 0.07% | 260.61 ± 2.99 m | 0.3649 %/m | 10.77 ms | 46.5 MB | 35.0 ± 4.5 |
| **Stachniss (2005)** | 95.33 ± 0.36% | 205.22 ± 8.51 m | 0.4654 %/m | 11.96 ms | 53.8 MB | **1.7 ± 2.4** |
| **OmniRay (Ours - Zero Shot)**| **91.08 ± 3.99%** | **454.34 ± 108.03 m** | **0.2005 %/m** | **3.09 ms** | **41.8 MB** | **70.3 ± 53.6** |

**Zero-Shot Generalization Gap:**

| Dataset / Environment | Environment Type | Final Coverage (%) | Zero-Shot Generalization Gap |
| :--- | :--- | :---: | :---: |
| **Intel Research Lab (Seattle)** | Fixed Training / Benchmark Floorplan | 90.87 ± 3.29% | Baseline reference |
| **MIT Stata Center (`MIT dataset.bag`)** | Unseen Complex Atrium Holdout | 91.08 ± 3.99% | **+0.21%** |

---

## Sim-to-Real Noise Robustness & Ablation Studies

### Reward Specification

| Parameter | Default | Description |
| :--- | :---: | :--- |
| `reward_exploration` | $1.0$ | Reward per newly explored grid cell. |
| `reward_time_penalty` | $0.01$ | Per-step efficiency penalty. |
| `reward_collision_penalty` | $0.1$ | Collision penalty. |
| `reward_frontier` | $0.1$ | Vectorized reward for frontier heading alignment. |

### Robust Evaluation & Exploration Progression

![Robust SLAM Evaluation Report](ablation_eval_full/robust_evaluation_report.png)
![Robust Exploration Progression](ablation_eval_full/robust_exploration_progression.png)

### Multi-Seed Ablation Matrix (14 Configurations, 3 Seeds)

![Ablation Health Score Stability](ablation_eval_full/ablation_3seed_health_score_errorbars.png)
![Ablation Peak Reward Consistency](ablation_eval_full/ablation_3seed_peak_reward_errorbars.png)

---

## Repository Structure

```
OmniRay/
├── ablation_eval_full/  # Output plots and 3-seed ablation matrix charts
├── assets/              # Architecture diagrams
├── envs/                # Gymnasium environment, VectorSLAM, & 5-Layer Autonomy
├── profiling/           # Raycasting and SLAM profiler scripts
├── sim/                 # C++ 256-bit AVX2 SIMD raycasting engine & pybind11 bindings
├── config.yaml          # Hyperparameters and network configs
├── train_rl.py          # PPO training pipeline
├── visualize_agent.py   # Environment visualization tool
└── evaluate_and_record.py # Quantitative evaluation runner
```

---

## Authors & License

- **Author:** Kingshuk Chatterjee
- **Co-Author:** Ayush Ranjan
- **Co-Author:** Raghav Singh Parihar
- **Hardware Dependency:** Latencies reflect AVX2 execution on Intel i7-1355U. Unsupported hardware falls back to NumPy mode.
- **License:** Apache License 2.0. Copyright 2026 Kingshuk Chatterjee, Ayush Ranjan, Raghav Singh Parihar.
