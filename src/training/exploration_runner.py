"""
Exploration Training Runner
============================
Full training loop for DreamerV3 to EXPLORE the ATPS physics.

This script:
1. Runs episodes in the exploration environment
2. Collects experiences for world model training
3. Tracks physics discoveries
4. Logs intrinsic curiosity rewards
5. Optionally launches visual mode to watch

The goal is for DreamerV3 to discover:
- How to whip TABs using tether dynamics
- Optimal release angles for intercepts
- Sacrifice strategies (giving up one TAB to save another)
- Formation reconfiguration under threat
- Coordinated multi-TAB attacks
"""

import numpy as np
import time
import json
import os
from pathlib import Path
from datetime import datetime
from collections import deque
from typing import Dict, List, Optional

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.training.exploration_env import ExplorationKAPSEnv
from src.training.threat_environment import ThreatType


class ExplorationTrainer:
    """
    Manages exploration training with DreamerV3.
    
    Key metrics tracked:
    - Intrinsic curiosity rewards
    - Physics discoveries (first-time observations of capabilities)
    - Threat intercept success rates
    - Novel state visitation
    - Action entropy (diversity of exploration)
    """
    
    def __init__(self,
                 dreamer_checkpoint: Optional[str] = None,
                 log_dir: str = "exploration_logs",
                 save_every: int = 100):
        """
        Initialize exploration trainer.
        
        Args:
            dreamer_checkpoint: Path to DreamerV3 checkpoint (if training)
            log_dir: Directory for logs and discoveries
            save_every: Save frequency (episodes)
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.save_every = save_every
        
        # Training state
        self.episode_count = 0
        self.total_steps = 0
        self.best_episode_reward = float('-inf')
        
        # Metrics
        self.episode_rewards = []
        self.episode_intercepts = []
        self.episode_discoveries = []
        self.episode_novel_states = []
        
        # Discovery tracking
        self.all_discoveries = set()
        self.discovery_log = []
        
        # Load DreamerV3
        self.dreamer_agent = None
        try:
            from src.ai.dreamer_interface import DreamerBrainInterface
            self.dreamer_agent = DreamerBrainInterface()
            print("[✓] DreamerV3 agent loaded")
        except Exception as e:
            print(f"[!] DreamerV3 not available: {e}")
            print("[!] Using random exploration baseline")
        
        # Environment
        self.env = ExplorationKAPSEnv(
            episode_steps=5000,
            threat_spawn_interval=80,
            curiosity_weight=0.5
        )
        
        # Experience buffer for training
        self.experience_buffer = ExperienceBuffer(capacity=100000)
        
        print(f"[Exploration Trainer] Initialized")
        print(f"  Log dir: {self.log_dir}")
        print(f"  Agent: {'DreamerV3' if self.dreamer_agent else 'Random'}")
    
    def train(self, num_episodes: int = 1000, verbose: bool = True):
        """
        Run exploration training loop.
        
        Args:
            num_episodes: Number of episodes to run
            verbose: Print progress
        """
        print("=" * 70)
        print("EXPLORATION TRAINING")
        print("=" * 70)
        print(f"Running {num_episodes} episodes...")
        print()
        
        start_time = time.time()
        
        for ep in range(num_episodes):
            ep_reward, ep_info = self._run_episode()
            
            self.episode_count += 1
            self.episode_rewards.append(ep_reward)
            self.episode_intercepts.append(ep_info.get('intercepts', 0))
            self.episode_discoveries.append(ep_info.get('physics_discoveries', 0))
            self.episode_novel_states.append(ep_info.get('novel_states', 0))
            
            # Track best
            if ep_reward > self.best_episode_reward:
                self.best_episode_reward = ep_reward
                if verbose:
                    print(f"  [NEW BEST] Episode {self.episode_count}: {ep_reward:.1f}")
            
            # Log discoveries
            new_discoveries = ep_info.get('new_discoveries', [])
            for disc in new_discoveries:
                if disc not in self.all_discoveries:
                    self.all_discoveries.add(disc)
                    self.discovery_log.append({
                        'episode': self.episode_count,
                        'step': self.total_steps,
                        'discovery': disc,
                        'time': datetime.now().isoformat()
                    })
                    if verbose:
                        print(f"  [DISCOVERY] {disc}")
            
            # Progress
            if verbose and (ep + 1) % 10 == 0:
                recent_rewards = self.episode_rewards[-10:]
                recent_intercepts = self.episode_intercepts[-10:]
                
                print(f"Episode {self.episode_count:4d} | "
                      f"Avg Reward: {np.mean(recent_rewards):7.1f} | "
                      f"Intercepts: {np.mean(recent_intercepts):4.1f} | "
                      f"Discoveries: {len(self.all_discoveries)}")
            
            # Save periodically
            if (ep + 1) % self.save_every == 0:
                self._save_logs()
        
        # Final save
        self._save_logs()
        
        elapsed = time.time() - start_time
        print()
        print("=" * 70)
        print("TRAINING COMPLETE")
        print("=" * 70)
        print(f"Episodes: {self.episode_count}")
        print(f"Steps: {self.total_steps}")
        print(f"Time: {elapsed:.1f}s")
        print(f"Best Episode: {self.best_episode_reward:.1f}")
        print(f"Discoveries: {len(self.all_discoveries)}")
        print()
        
        return {
            'episode_rewards': self.episode_rewards,
            'discoveries': list(self.all_discoveries),
            'best_reward': self.best_episode_reward
        }
    
    def _run_episode(self) -> tuple:
        """Run single episode and collect experience."""
        obs, info = self.env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            # Get action
            if self.dreamer_agent is not None:
                obs_dict = self._obs_to_dict(obs)
                action = self.dreamer_agent.infer(obs_dict)
                action = np.clip(action, -1.0, 1.0)
                # Pad/trim to match action space
                if len(action) < 18:
                    action = np.concatenate([action, np.zeros(18 - len(action))])
                elif len(action) > 18:
                    action = action[:18]
            else:
                # Structured random exploration
                action = self._random_exploration_action()
            
            # Step
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            episode_reward += reward
            self.total_steps += 1
            
            # Store experience
            self.experience_buffer.add(obs, action, reward, next_obs, done)
            
            obs = next_obs
        
        return episode_reward, info
    
    def _random_exploration_action(self) -> np.ndarray:
        """
        Structured random exploration.
        
        More intelligent than uniform random:
        - Smooth control changes (low-pass filtered)
        - Occasional "bursts" of extreme actions
        - Random TAB releases
        """
        # Base: sample from action space
        action = self.env.action_space.sample()
        
        # 10% chance of "focused" action on single TAB
        if np.random.random() < 0.1:
            # Focus on one TAB
            focus_tab = np.random.randint(4)
            action *= 0.2  # Dampen everything
            # Strong action on focus TAB
            idx = focus_tab * 3
            action[idx:idx+3] = np.random.uniform(-1, 1, 3)
        
        # 5% chance of release action
        if np.random.random() < 0.05:
            tab_idx = np.random.randint(4)
            action[12 + tab_idx] = 1.0  # Release
        
        return action
    
    def _obs_to_dict(self, obs: np.ndarray) -> Dict:
        """Convert flat observation to dict for DreamerV3."""
        return {
            'mother_drone': {
                'position': obs[0:3] * 1000,
                'velocity': obs[3:6] * 100,
            },
            'tabs': {},
            'threats': []
        }
    
    def _save_logs(self):
        """Save training logs to disk."""
        # Metrics
        metrics = {
            'episode_count': self.episode_count,
            'total_steps': self.total_steps,
            'best_reward': self.best_episode_reward,
            'episode_rewards': self.episode_rewards[-1000:],  # Last 1000
            'discoveries': list(self.all_discoveries),
            'discovery_count': len(self.all_discoveries)
        }
        
        with open(self.log_dir / "training_metrics.json", 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Discovery log
        with open(self.log_dir / "discoveries.jsonl", 'a') as f:
            for disc in self.discovery_log[-100:]:  # Recent
                f.write(json.dumps(disc) + '\n')
        self.discovery_log.clear()
        
        print(f"  [Saved logs to {self.log_dir}]")


class ExperienceBuffer:
    """
    Simple experience buffer for collecting training data.
    
    In full DreamerV3 training, this would feed into world model updates.
    """
    
    def __init__(self, capacity: int = 100000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
    
    def add(self, obs, action, reward, next_obs, done):
        self.buffer.append((obs, action, reward, next_obs, done))
    
    def sample(self, batch_size: int):
        indices = np.random.choice(len(self.buffer), min(batch_size, len(self.buffer)), replace=False)
        batch = [self.buffer[i] for i in indices]
        
        obs = np.array([b[0] for b in batch])
        actions = np.array([b[1] for b in batch])
        rewards = np.array([b[2] for b in batch])
        next_obs = np.array([b[3] for b in batch])
        dones = np.array([b[4] for b in batch])
        
        return obs, actions, rewards, next_obs, dones
    
    def __len__(self):
        return len(self.buffer)


def main():
    """Main entry point for exploration training."""
    import argparse
    
    parser = argparse.ArgumentParser(description="ATPS Exploration Training")
    parser.add_argument("--episodes", type=int, default=100, help="Number of episodes")
    parser.add_argument("--log-dir", type=str, default="exploration_logs", help="Log directory")
    parser.add_argument("--visual", action="store_true", help="Launch visual training mode")
    args = parser.parse_args()
    
    if args.visual:
        # Visual mode
        from src.training.visual_trainer import run_visual_training
        run_visual_training(use_dreamer=True)
    else:
        # Training mode
        trainer = ExplorationTrainer(log_dir=args.log_dir)
        results = trainer.train(num_episodes=args.episodes)
        
        print("\nFinal Discoveries:")
        for disc in results['discoveries']:
            print(f"  - {disc}")


if __name__ == "__main__":
    main()
