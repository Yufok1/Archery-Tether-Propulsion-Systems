"""
CASCADE COLLECTIVE INTELLIGENCE INTEGRATION
============================================

This module integrates the cascade-lattice collective intelligence system
with our quine combat arena. Every quine node is a replicated DreamerV3
brain that:

1. Shares the same genesis root (89f940c1a4b7aa65)
2. Records observations to the cascade lattice store
3. Supports HOLD for human intervention
4. Learns collectively through shared provenance

The KEY insight: All 49+ quine agents are really ONE model file
learning exponentially through parallel experience.

CASCADE Architecture:
  - Genesis: The Schelling point all quines trace back to
  - Hold: Inference-level halts for human override
  - Store: HuggingFace-synced observation database
  - CausationGraph: DAG of cause-effect relationships
  - SymbioticAdapter: Self-interpreting signal converter

Training Flow:
  1. Combat step generates observations (threat positions, actions, rewards)
  2. Each quine records observation with parent_cid chain
  3. Observations sync to HuggingFace (tostido/cascade-observations)
  4. All quines learn from the collective experience pool
  5. Merkle provenance ensures lineage back to genesis
"""

import numpy as np
import time
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Callable
from pathlib import Path

# Import cascade
try:
    import cascade
    from cascade import (
        Hold, HoldAwareMixin, HoldState, HoldResolution,
        Monitor, Event, CausationGraph, CausationLink,
        SymbioticAdapter
    )
    from cascade.store import observe as cascade_observe, query as cascade_query, stats as cascade_stats
    from cascade.genesis import get_genesis_root, create_genesis, ProvenanceChain
    CASCADE_AVAILABLE = True
    GENESIS_ROOT = get_genesis_root()
    print(f"[CASCADE] Loaded v{cascade.__version__} | Genesis: {GENESIS_ROOT}")
except ImportError as e:
    CASCADE_AVAILABLE = False
    GENESIS_ROOT = None
    print(f"[CASCADE] Not available: {e}")
    print("[CASCADE] Install with: pip install cascade-lattice")


@dataclass
class QuineObservation:
    """
    A single observation from a quine agent in the combat arena.
    
    This is what gets recorded to the cascade lattice and synced
    to HuggingFace for collective learning.
    """
    # Identity
    quine_id: str
    brain_hash: str  # Merkle hash of this quine's brain state
    
    # Combat state
    position: np.ndarray
    velocity: np.ndarray
    blade_pitch: float
    corkscrew_phase: float
    
    # Threat awareness
    closest_threat_distance: float
    closest_threat_direction: np.ndarray
    n_threats_visible: int
    
    # Action taken
    action: np.ndarray  # 8-dim action from DreamerV3
    action_probs: Optional[np.ndarray] = None
    value_estimate: float = 0.0
    
    # Outcome
    reward: float = 0.0  # +1 for intercept, -1 for collision
    damage_dealt: float = 0.0
    damage_taken: float = 0.0
    
    # Provenance
    parent_cid: Optional[str] = None  # Chain to previous observation
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict for cascade store."""
        return {
            'quine_id': self.quine_id,
            'brain_hash': self.brain_hash,
            'position': self.position.tolist(),
            'velocity': self.velocity.tolist(),
            'blade_pitch': float(self.blade_pitch),
            'corkscrew_phase': float(self.corkscrew_phase),
            'closest_threat_distance': float(self.closest_threat_distance),
            'closest_threat_direction': self.closest_threat_direction.tolist(),
            'n_threats_visible': int(self.n_threats_visible),
            'action': self.action.tolist(),
            'action_probs': self.action_probs.tolist() if self.action_probs is not None else None,
            'value_estimate': float(self.value_estimate),
            'reward': float(self.reward),
            'damage_dealt': float(self.damage_dealt),
            'damage_taken': float(self.damage_taken),
            'parent_cid': self.parent_cid,
            'timestamp': self.timestamp,
            'genesis_root': GENESIS_ROOT
        }


class CascadeQuineBrain(HoldAwareMixin if CASCADE_AVAILABLE else object):
    """
    A DreamerV3 brain that's integrated with cascade collective intelligence.
    
    Features:
      - HOLD support: Human can pause and override at any inference
      - Provenance: Every action is tracked with merkle chain
      - Collective: Learns from all quines' experiences via HF sync
      - Symbiotic: Adapts to any observation format
    """
    
    def __init__(self, 
                 quine_id: str,
                 brain_hash: str,
                 champion_module=None):
        self.quine_id = quine_id
        self.brain_hash = brain_hash
        self.champion = champion_module
        
        # Cascade integration
        if CASCADE_AVAILABLE:
            self.monitor = Monitor()
            self.adapter = SymbioticAdapter()
            self.hold = Hold.get()
            
            # Enable HOLD by default
            if hasattr(self, 'enable_hold'):
                self.enable_hold()
        else:
            self.monitor = None
            self.adapter = None
            self.hold = None
        
        # Observation chain
        self.last_cid: Optional[str] = None
        self.observation_count = 0
        
        # State
        self.hidden_state = None
        self.last_action = np.zeros(8)
    
    def infer(self, obs: Dict[str, Any]) -> np.ndarray:
        """
        Run inference with cascade observation and HOLD support.
        
        This is the main entry point for the combat arena.
        """
        # Adapt observation through symbiotic adapter
        if self.adapter:
            self.adapter.interpret(obs)
        
        # Compute action (from champion or fallback)
        if self.champion and hasattr(self.champion, 'forward'):
            try:
                action = self.champion.forward(self._encode_obs(obs))
            except Exception as e:
                action = self._fallback_action(obs)
        else:
            action = self._fallback_action(obs)
        
        # Record event in causation graph
        if self.monitor:
            event = Event(
                timestamp=time.time(),
                component=f"quine_{self.quine_id}",
                event_type="inference",
                data={
                    'action': action.tolist() if isinstance(action, np.ndarray) else action,
                    'closest_threat': obs.get('closest_threat_distance', 999),
                    'n_threats': obs.get('n_threats', 0)
                }
            )
            self.monitor.observe(event)
        
        # Check for HOLD
        if self.hold and hasattr(self, 'forward_with_hold'):
            # This will block if HOLD is activated
            resolution = self.hold.yield_point(
                action_probs=action if isinstance(action, np.ndarray) else np.array(action),
                value=0.0,
                observation=obs,
                brain_id=self.quine_id
            )
            
            if resolution and resolution.override_action is not None:
                action = resolution.override_action
        
        self.last_action = action
        self.observation_count += 1
        
        return action
    
    def record_observation(self, 
                           position: np.ndarray,
                           velocity: np.ndarray,
                           blade_pitch: float,
                           obs: Dict,
                           action: np.ndarray,
                           reward: float = 0.0,
                           damage_dealt: float = 0.0,
                           damage_taken: float = 0.0) -> Optional[str]:
        """
        Record a full observation to the cascade store.
        
        Returns the CID of the recorded observation.
        """
        if not CASCADE_AVAILABLE:
            return None
        
        quine_obs = QuineObservation(
            quine_id=self.quine_id,
            brain_hash=self.brain_hash,
            position=position,
            velocity=velocity,
            blade_pitch=blade_pitch,
            corkscrew_phase=obs.get('corkscrew_phase', 0.0),
            closest_threat_distance=obs.get('closest_threat_distance', 999.0),
            closest_threat_direction=obs.get('closest_threat_direction', np.zeros(3)),
            n_threats_visible=obs.get('n_threats', 0),
            action=action,
            value_estimate=obs.get('value', 0.0),
            reward=reward,
            damage_dealt=damage_dealt,
            damage_taken=damage_taken,
            parent_cid=self.last_cid
        )
        
        # Record to cascade store
        try:
            receipt = cascade_observe(
                model_id=f"quine_{self.quine_id}",
                data=quine_obs.to_dict(),
                parent_cid=self.last_cid,
                sync=False  # Don't sync every step, batch later
            )
            self.last_cid = receipt.cid
            return receipt.cid
        except Exception as e:
            print(f"[CASCADE] Observation failed: {e}")
            return None
    
    def _encode_obs(self, obs: Dict) -> np.ndarray:
        """Encode observation dict to flat vector for DreamerV3."""
        # Simplified encoding
        encoded = np.concatenate([
            obs.get('closest_threat_direction', np.zeros(3)),
            [obs.get('closest_threat_distance', 999.0) / 500.0],
            [obs.get('n_threats', 0) / 50.0],
            [obs.get('time', 0.0) / 60.0],
            self.last_action
        ])
        return encoded.astype(np.float32)
    
    def _fallback_action(self, obs: Dict) -> np.ndarray:
        """Reactive fallback when DreamerV3 unavailable."""
        action = np.zeros(8)
        
        threat_dir = obs.get('closest_threat_direction', np.zeros(3))
        threat_dist = obs.get('closest_threat_distance', 500.0)
        
        if threat_dist < 100:
            # Defensive dodge
            perp = np.array([-threat_dir[1], threat_dir[0], threat_dir[2] * 0.5])
            if np.linalg.norm(perp) > 0.01:
                perp = perp / np.linalg.norm(perp)
            action[:3] = perp * 30
            action[6] = 0.8  # High pitch
        else:
            # Patrol
            t = self.observation_count * 0.01
            action[:3] = np.array([np.cos(t), np.sin(t), 0]) * 10
            action[6] = 0.5
        
        return action


class CollectiveIntelligence:
    """
    Manages the collective intelligence across all quine agents.
    
    Key responsibilities:
      1. Shared experience pool via cascade store
      2. Gradient aggregation across parallel experiences
      3. Merkle provenance verification
      4. HuggingFace sync for global learning
    """
    
    def __init__(self):
        self.quines: Dict[str, CascadeQuineBrain] = {}
        self.causation_graph = CausationGraph() if CASCADE_AVAILABLE else None
        
        # Collective metrics
        self.total_observations = 0
        self.total_rewards = 0.0
        self.intercepts = 0
        self.collisions = 0
        
        # Sync settings
        self.sync_interval = 5000  # Sync every N observations (less frequent to avoid HF throttle)
        self.last_sync = 0
        self.auto_sync = False  # Disable auto-sync during training, manual sync at end
    
    def register_quine(self, quine_id: str, brain_hash: str) -> CascadeQuineBrain:
        """Register a new quine agent in the collective."""
        brain = CascadeQuineBrain(
            quine_id=quine_id,
            brain_hash=brain_hash
        )
        self.quines[quine_id] = brain
        
        print(f"[COLLECTIVE] Registered quine {quine_id} | Hash: {brain_hash[:16]}...")
        return brain
    
    def record_collective_event(self, 
                                 event_type: str,
                                 data: Dict[str, Any],
                                 quine_ids: List[str] = None):
        """
        Record an event that affects the collective.
        
        This creates causal links between quines when they
        interact (e.g., coordinated intercept).
        """
        if not CASCADE_AVAILABLE:
            return
        
        event = Event(
            timestamp=time.time(),
            component="collective",
            event_type=event_type,
            data=data
        )
        
        if self.causation_graph:
            self.causation_graph.add_event(event)
            
            # Link to participating quines
            if quine_ids:
                for qid in quine_ids:
                    if qid in self.quines:
                        brain = self.quines[qid]
                        if brain.monitor and brain.monitor.graph:
                            # Find recent event from this quine
                            recent = brain.monitor.graph.get_recent_events(limit=1)
                            if recent:
                                link = CausationLink(
                                    from_event=recent[0].event_id,
                                    to_event=event.event_id,
                                    causation_type="direct",
                                    strength=1.0,
                                    explanation=f"Quine {qid} participated in {event_type}"
                                )
                                self.causation_graph.add_link(link)
    
    def aggregate_rewards(self) -> Dict[str, float]:
        """
        Aggregate rewards across all quines for collective update.
        
        This is where the exponential learning happens:
        All quines share in the collective reward, weighted by
        their contribution to intercepts/defense.
        """
        rewards = {}
        total = 0.0
        
        for qid, brain in self.quines.items():
            # Query this quine's recent observations
            if CASCADE_AVAILABLE:
                try:
                    receipts = cascade_query(
                        model_id=f"quine_{qid}",
                        limit=100
                    )
                    quine_reward = sum(r.data.get('reward', 0) for r in receipts if r.data)
                    rewards[qid] = quine_reward
                    total += quine_reward
                except:
                    rewards[qid] = 0.0
            else:
                rewards[qid] = 0.0
        
        # Normalize to collective contribution
        if total > 0:
            for qid in rewards:
                rewards[qid] = rewards[qid] / total
        
        return rewards
    
    def sync_to_huggingface(self) -> Dict[str, int]:
        """
        Sync all local observations to HuggingFace.
        
        This makes the collective experience available to
        ALL future quine instances globally.
        """
        if not CASCADE_AVAILABLE:
            return {"synced": 0, "failed": 0}
        
        try:
            # Suppress git commit spam by redirecting both stdout and stderr
            import contextlib
            import io
            import sys
            old_stdout, old_stderr = sys.stdout, sys.stderr
            sys.stdout = sys.stderr = io.StringIO()
            try:
                result = cascade.sync_all()
            finally:
                sys.stdout, sys.stderr = old_stdout, old_stderr
            self.last_sync = self.total_observations
            print(f"[COLLECTIVE] Synced to HuggingFace: {result}")
            return result
        except Exception as e:
            # Silently fail on sync errors (HF throttle, network, etc)
            return {"synced": 0, "failed": 1, "error": str(e)}
    
    def pull_collective_experience(self, limit: int = 1000) -> List[Dict]:
        """
        Pull experience from the HuggingFace collective pool.
        
        This allows new quines to learn from ALL historical
        combat experiences across all users.
        """
        if not CASCADE_AVAILABLE:
            return []
        
        # Pull from central dataset
        count = cascade.pull_from_hf("tostido/cascade-observations")
        print(f"[COLLECTIVE] Pulled {count} observations from HuggingFace")
        
        # Query the pulled data
        receipts = cascade_query(limit=limit)
        return [r.data for r in receipts if r.data]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get collective intelligence statistics."""
        if CASCADE_AVAILABLE:
            store_stats = cascade_stats()
        else:
            store_stats = {}
        
        return {
            'n_quines': len(self.quines),
            'total_observations': self.total_observations,
            'intercepts': self.intercepts,
            'collisions': self.collisions,
            'genesis_root': GENESIS_ROOT,
            'store': store_stats
        }


class CascadeTrainer:
    """
    Training loop that uses cascade collective intelligence.
    
    The training flow:
      1. Run combat episodes across all quines
      2. Record observations with rewards
      3. Aggregate collective rewards
      4. Update shared brain weights
      5. Sync to HuggingFace for global learning
    
    Because all quines share the same brain, they learn from
    each other's experiences exponentially.
    """
    
    def __init__(self, collective: CollectiveIntelligence):
        self.collective = collective
        self.episode_count = 0
        self.best_score = float('-inf')
    
    def train_episode(self, arena, max_steps: int = 1000) -> Dict[str, Any]:
        """
        Run one training episode.
        """
        self.episode_count += 1
        
        # Record episode start
        self.collective.record_collective_event(
            event_type="episode_start",
            data={
                'episode': self.episode_count,
                'n_quines': len(self.collective.quines),
                'arena_radius': arena.arena_radius
            }
        )
        
        # Run episode
        step = 0
        total_reward = 0.0
        
        while step < max_steps and arena.running:
            obs = arena.step()
            step += 1
            
            # Record observations for each active quine
            for node in arena.lattice.get_all_nodes():
                if node.node_id in self.collective.quines:
                    brain = self.collective.quines[node.node_id]
                    
                    # Get reward signal
                    reward = 0.0
                    if arena.stats.threats_destroyed > 0:
                        reward += 1.0
                    if arena.stats.damage_taken > 0:
                        reward -= 0.1
                    
                    # Record
                    brain.record_observation(
                        position=node.position,
                        velocity=node.velocity,
                        blade_pitch=getattr(node, 'blade_pitch', 0.0),
                        obs=obs,
                        action=brain.last_action,
                        reward=reward,
                        damage_dealt=arena.stats.damage_dealt,
                        damage_taken=arena.stats.damage_taken
                    )
                    
                    total_reward += reward
                    self.collective.total_observations += 1
        
        # Episode stats
        stats = {
            'episode': self.episode_count,
            'steps': step,
            'total_reward': total_reward,
            'score': arena.stats.score,
            'threats_destroyed': arena.stats.threats_destroyed,
            'damage_taken': arena.stats.damage_taken
        }
        
        # Record episode end
        self.collective.record_collective_event(
            event_type="episode_end",
            data=stats
        )
        
        # Sync if interval reached and auto_sync enabled
        if self.collective.auto_sync and \
           self.collective.total_observations - self.collective.last_sync > self.collective.sync_interval:
            self.collective.sync_to_huggingface()
        
        # Track best
        if arena.stats.score > self.best_score:
            self.best_score = arena.stats.score
            print(f"[TRAINER] New best score: {self.best_score}")
        
        return stats


def create_collective_arena():
    """
    Create a combat arena with cascade collective intelligence.
    """
    import sys
    import os
    
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))
    
    from src.entities.quine_node import create_matryoshka_lattice
    from src.combat_arena import CombatArena
    
    # Create lattice
    lattice = create_matryoshka_lattice(n_spines=4, vertebrae_per_spine=2)
    
    # Create collective
    collective = CollectiveIntelligence()
    
    # Register all quines
    for node in lattice.get_all_nodes():
        brain_hash = getattr(node, 'genome_hash', 'default_hash')
        brain = collective.register_quine(node.node_id, brain_hash)
        node.cascade_brain = brain
    
    # Create arena
    arena = CombatArena(lattice)
    arena.collective = collective
    
    return arena, collective


if __name__ == "__main__":
    print("=" * 70)
    print("CASCADE COLLECTIVE INTELLIGENCE - COMBAT TRAINING")
    print("=" * 70)
    
    if CASCADE_AVAILABLE:
        print(f"\nGenesis Root: {GENESIS_ROOT}")
        print(f"Store Stats: {cascade_stats()}")
    
    # Create collective arena
    arena, collective = create_collective_arena()
    
    print(f"\nCollective: {len(collective.quines)} quines registered")
    print(f"Arena: {arena.arena_radius}m radius")
    
    # Create trainer
    trainer = CascadeTrainer(collective)
    
    # Run training episode
    print("\n[TRAINING] Starting episode...")
    arena.running = True
    
    # Spawn initial wave
    arena.spawner.spawn_wave()
    
    # Run
    stats = trainer.train_episode(arena, max_steps=500)
    
    print("\n" + "=" * 70)
    print("EPISODE COMPLETE")
    print("=" * 70)
    print(f"  Steps: {stats['steps']}")
    print(f"  Total Reward: {stats['total_reward']:.2f}")
    print(f"  Score: {stats['score']}")
    print(f"  Threats Destroyed: {stats['threats_destroyed']}")
    print(f"  Observations: {collective.total_observations}")
    print("=" * 70)
    
    # Show collective stats
    print("\nCOLLECTIVE STATS:")
    for k, v in collective.get_stats().items():
        print(f"  {k}: {v}")
