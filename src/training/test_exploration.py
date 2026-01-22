"""Quick test of exploration environment with cable geometry constraints"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exploration_env import ExplorationKAPSEnv

print("Creating ExplorationKAPSEnv with cable geometry constraints...")
env = ExplorationKAPSEnv()

print("Resetting...")
obs, info = env.reset()
print(f"Observation shape: {obs.shape}")
print(f"Action space: {env.action_space}")

if env.cable_detector is not None:
    print("[✓] Cable intersection detector active")
    print(f"    Sectors defined for: {list(env.cable_detector.sectors.keys())}")
else:
    print("[!] Cable geometry not available")

print("\nRunning 200 steps with random actions...")
total_reward = 0
cable_crossings = 0
forced_releases = 0

for i in range(200):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    
    # Track cable issues
    if 'cable_drag_penalty' in info:
        cable_crossings += 1
    if 'forced_releases' in env.episode_stats:
        forced_releases = env.episode_stats['forced_releases']
    
    if terminated or truncated:
        print(f"Episode ended at step {i}")
        break

print(f"\n200 steps completed!")
print(f"Total reward: {total_reward:.2f}")
print(f"Cable crossings detected: {cable_crossings}")
print(f"Forced releases: {forced_releases}")
print(f"Final info: {info}")
