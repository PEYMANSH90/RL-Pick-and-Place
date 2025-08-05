#!/usr/bin/env python3
"""
Simple TD3 Torque Control for ABB IRB1600 Robot
===============================================

Clean implementation focusing on core TD3 training without curriculum learning
or hyperparameter optimization.
"""

import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import TD3
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from collections import deque
import gymnasium as gym
from gymnasium import spaces
import os
from datetime import datetime

# Create results directory
results_dir = "results"
os.makedirs(results_dir, exist_ok=True)

class SimpleJointControlEnv(gym.Env):
    """Simple Gymnasium environment for joint torque control with curriculum learning"""
    
    def __init__(self, curriculum_level=0):
        super(SimpleJointControlEnv, self).__init__()
        
        # System parameters
        self.dt = 0.16  # Step duration: 160 ms
        self.gear_ratio = 120  # Single gear ratio for simplicity
        
        # Load profile from paper - original variable loads
        self.original_loads = [10, 75, -25, 40, 10, -10, -30, 5, 18, 10, 18, 17]  # Paper loads
        
        # Curriculum Learning: Start with simple loads, gradually increase complexity
        self.curriculum_level = curriculum_level
        if curriculum_level == 0:
            # Level 0: Very simple - 10% of original loads
            self.user_loads = [load * 0.1 for load in self.original_loads]
        elif curriculum_level == 1:
            # Level 1: Medium - 50% of original loads
            self.user_loads = [load * 0.5 for load in self.original_loads]
        elif curriculum_level == 2:
            # Level 2: Full - 100% of original loads
            self.user_loads = self.original_loads.copy()
        else:
            # Default to medium difficulty
            self.user_loads = [load * 0.5 for load in self.original_loads]
        
        print(f"📏 Load scaling for level {curriculum_level}:")
        print(f"   Original loads: {[f'{x:.1f}' for x in self.original_loads]}")
        curriculum_factor = 0.1 if curriculum_level == 0 else 0.5 if curriculum_level == 1 else 1.0
        print(f"   Curriculum loads ({curriculum_factor*100}%): {[f'{x:.1f}' for x in self.user_loads]}")
        print(f"   Max load: {max([abs(x) for x in self.user_loads]):.1f} N⋅m")
        
        # Episode parameters
        self.steps_per_episode = 100
        self.current_step = 0
        self.episode_reward = 0.0  # Track episode reward
        
        # Generate proper load profile (variable loads over time)
        self._generate_load_profile()
        
        # Action space: T* (commanded motor torque) - -2 to 2 N⋅m (allow negative torques)
        self.action_space = spaces.Box(
            low=-2, high=2, shape=(1,), dtype=np.float32
        )
        
        # Observation space: [error, prev_error, integral_error, derivative_error, current_torque, load_torque]
        self.observation_space = spaces.Box(
            low=np.array([-2.0, -2.0, -2.0, -2.0, 0.0, 0.0], dtype=np.float32), 
            high=np.array([2.0, 2.0, 2.0, 2.0, 1.0, 120.0], dtype=np.float32), 
            dtype=np.float32
        )
        
        self.reset()
    
    def _generate_load_profile(self):
        """Generate variable load profile from paper loads"""
        # Create load torque profile for 16 seconds (100 steps)
        steps_per_second = int(1 / self.dt)  # 6.25 steps per second
        seconds_per_load = 1  # Each load value lasts 1 second
        
        # Create the basic pattern by repeating each load for 1 second
        basic_pattern = np.repeat(self.user_loads, steps_per_second)
        
        # Extend to 100 steps by repeating the pattern
        total_steps_needed = self.steps_per_episode  # 100 steps
        if len(basic_pattern) < total_steps_needed:
            # Repeat the pattern to fill the remaining time
            repeats_needed = int(np.ceil(total_steps_needed / len(basic_pattern)))
            extended_pattern = np.tile(basic_pattern, repeats_needed)
            self.load_torque = extended_pattern[:total_steps_needed]
        else:
            # If basic pattern is longer, truncate it
            self.load_torque = basic_pattern[:total_steps_needed]
        
        print(f"📏 Generated load profile with {len(self.load_torque)} steps")
        print(f"   Original loads: {self.original_loads}")
        print(f"   Load range: {min(self.load_torque):.1f} to {max(self.load_torque):.1f} N⋅m")
    
    def reset(self, seed=None):
        """Reset environment to initial state"""
        super().reset(seed=seed)
        
        self.current_step = 0
        self.T = 0.0  # Current applied torque
        self.integral_error = 0.0
        self.derivative_error = 0.0
        self.prev_error = 0.0
        self.episode_reward = 0.0 # Reset episode reward
        
        # Get initial load from the generated profile
        self.current_load = self.load_torque[0]
        
        # Initial error
        joint_error = self.current_load - (self.T * self.gear_ratio)
        
        return np.array([
            joint_error / 100.0,  # Normalize error
            self.prev_error / 100.0,  # Normalize previous error
            self.integral_error / 100.0,  # Normalize integral error
            self.derivative_error / 100.0,  # Normalize derivative error
            self.T,  # Motor torque (0-1, no normalization needed)
            self.current_load,  # Joint load (variable, no normalization needed)
        ], dtype=np.float32), {}
    
    def step(self, action):
        """Execute one step in the environment"""
        T_star = float(action[0])
        
        # Apply gear ratio
        T_joint = T_star * self.gear_ratio
        
        self.T = T_star
        self.current_step += 1
        done = self.current_step >= self.steps_per_episode
        
        # Update load from the generated profile
        if self.current_step < len(self.load_torque):
            self.current_load = self.load_torque[self.current_step]
        
        # Calculate error
        joint_error = self.current_load - T_joint
        self.integral_error += joint_error * self.dt
        self.derivative_error = (joint_error - self.prev_error) / self.dt
        
        # Observation
        obs = np.array([
            joint_error / 100.0,
            self.prev_error / 100.0,
            self.integral_error / 100.0,
            self.derivative_error / 100.0,
            self.T,  # Motor torque (0-1)
            self.current_load,  # Joint load (variable)
        ], dtype=np.float32)
        
        self.prev_error = joint_error
        
        # Simple reward: penalize error and action
        error_penalty = (joint_error / 100.0) ** 2  # Scaled joint error
        action_penalty = 0.01 * T_star ** 2  # Motor action penalty (T_star is 0-1)
        reward = -(error_penalty + action_penalty)
        
        self.episode_reward += reward # Accumulate episode reward
        
        return obs, reward, done, False, {}

class SimpleRewardCallback(BaseCallback):
    """Simple callback that prints rewards directly"""
    
    def __init__(self, verbose=1):
        super().__init__(verbose)
        self.rewards = []
        self.episode_count = 0
        self.best_reward = float('-inf')
        self.paper_threshold = -493
        self.last_episode_reward = None  # Track last episode to avoid duplicates
        self.convergence_window = 5  # Check last 5 episodes for convergence (reduced from 10)
        self.convergence_threshold = 5.0  # If reward variation is less than this, consider converged (increased from 2.0)
    
    def _on_step(self) -> bool:
        return True
    
    def _on_rollout_end(self) -> None:
        # Get episode reward from the monitor
        if hasattr(self.training_env, 'envs') and len(self.training_env.envs) > 0:
            env = self.training_env.envs[0]
            if hasattr(env, 'get_episode_rewards'):
                episode_rewards = env.get_episode_rewards()
                if episode_rewards and len(episode_rewards) > 0:
                    ep_reward = episode_rewards[-1]  # Get the latest episode reward
                    
                    # Check if this is a new episode (avoid duplicates)
                    if ep_reward != self.last_episode_reward:
                        self.last_episode_reward = ep_reward
                        self.episode_count += 1
                        self.rewards.append(ep_reward)
                        
                        # Check if it's a new record
                        is_new_record = ep_reward > self.best_reward
                        if is_new_record:
                            self.best_reward = ep_reward
                            record_status = "🏆 NEW RECORD!"
                        else:
                            record_status = ""
                        
                        # Check if approaching paper threshold
                        threshold_status = ""
                        if ep_reward > self.paper_threshold:
                            threshold_status = "🎯 ABOVE PAPER THRESHOLD!"
                        elif ep_reward > self.paper_threshold + 50:
                            threshold_status = "📈 APPROACHING PAPER THRESHOLD"
                        
                        print(f"Episode {self.episode_count}: Reward = {ep_reward:.2f} {record_status} {threshold_status}")
                        
                        # Simple early stop: if we reach 50 episodes and performance is good
                        if self.episode_count >= 50:
                            print(f"✅ Early stop at 50 episodes with good performance (reward={ep_reward:.2f}). Stopping training.")
                            self.model.stop_training = True
                            return  # Force stop
                        
                        # Check for convergence (reward stability) - only if not already at 50
                        if len(self.rewards) >= self.convergence_window and self.episode_count < 50:
                            recent_rewards = self.rewards[-self.convergence_window:]
                            reward_std = np.std(recent_rewards)
                            reward_mean = np.mean(recent_rewards)
                            
                            # If reward is stable and good, stop early
                            if (reward_std < self.convergence_threshold and 
                                reward_mean > self.paper_threshold and 
                                self.episode_count >= 30):  # Minimum 30 episodes
                                print(f"✅ CONVERGED! Reward stable (std={reward_std:.2f}) for {self.convergence_window} episodes. Stopping training.")
                                self.model.stop_training = True
                                return  # Force stop
                        
                        # Stop if we reach paper threshold
                        if ep_reward > self.paper_threshold:
                            print(f"🎯 Excellent! Reached paper threshold ({self.paper_threshold}). Stopping training.")
                            self.model.stop_training = True
                            return  # Force stop
                        
                        # Hard stop after 200 episodes (paper specification)
                        if self.episode_count >= 200:
                            print(f"⏹️ Reached 200 episodes (paper specification). Stopping training.")
                            self.model.stop_training = True
                            return  # Force stop

def curriculum_training():
    """Train TD3 with curriculum learning - start simple, gradually increase complexity"""
    print("🎓 Starting Curriculum Learning Training")
    print("=" * 60)
    
    # Curriculum levels: 0=10%, 1=50%, 2=100% of original loads
    curriculum_levels = [0, 1, 2]
    episodes_per_level = 30  # 30 episodes per level
    best_model = None
    level_results = {}  # Store results for plotting
    
    for level in curriculum_levels:
        print(f"\n📚 Curriculum Level {level}: Training with {10*level if level == 0 else 50*level if level == 1 else 100}% loads")
        print("-" * 50)
        
        # Create environment for this curriculum level
        env = SimpleJointControlEnv(curriculum_level=level)
        env = Monitor(env)
        env = DummyVecEnv([lambda: env])
        
        # Create TD3 model with exact paper parameters
        if best_model is None:
            # First level - create new model
            model = TD3(
                "MlpPolicy",
                env,
                verbose=1,
                learning_rate=3e-4,  # Paper: 3 × 10^-4
                buffer_size=500000,  # Paper: 500,000
                learning_starts=1000,
                batch_size=256,  # Paper: 256
                tau=0.005,  # Paper: 0.005
                gamma=0.99,  # Paper: 0.99
                train_freq=(1, "step"),
                gradient_steps=1,
                policy_kwargs=dict(
                    net_arch=dict(pi=[256, 256], qf=[256, 256])
                ),
                policy_delay=2,
                target_policy_noise=0.2,  # Paper: 0.2
                target_noise_clip=0.5,  # Paper: 0.5
                action_noise=None
            )
        else:
            # Transfer learning - use previous model
            print(f"🔄 Transferring knowledge from previous level...")
            model = best_model
        
        # Training callback for this level
        class CurriculumCallback(BaseCallback):
            def __init__(self, level, verbose=1):
                super().__init__(verbose)
                self.level = level
                self.rewards = []
                self.episode_count = 0
                self.best_reward = float('-inf')
                self.paper_threshold = -493
                self.last_episode_reward = None
                self.convergence_window = 5
                self.convergence_threshold = 5.0
            
            def _on_step(self) -> bool:
                return True
            
            def _on_rollout_end(self) -> None:
                # Get episode reward from the monitor
                if hasattr(self.training_env, 'envs') and len(self.training_env.envs) > 0:
                    env = self.training_env.envs[0]
                    if hasattr(env, 'get_episode_rewards'):
                        episode_rewards = env.get_episode_rewards()
                        if episode_rewards and len(episode_rewards) > 0:
                            ep_reward = episode_rewards[-1]
                            
                            # Check if this is a new episode (avoid duplicates)
                            if ep_reward != self.last_episode_reward:
                                self.last_episode_reward = ep_reward
                                self.episode_count += 1
                                self.rewards.append(ep_reward)
                                
                                # Check if it's a new record
                                is_new_record = ep_reward > self.best_reward
                                if is_new_record:
                                    self.best_reward = ep_reward
                                    record_status = "🏆 NEW RECORD!"
                                else:
                                    record_status = ""
                                
                                # Check if approaching paper threshold
                                threshold_status = ""
                                if ep_reward > self.paper_threshold:
                                    threshold_status = "🎯 ABOVE PAPER THRESHOLD!"
                                elif ep_reward > self.paper_threshold + 50:
                                    threshold_status = "📈 APPROACHING PAPER THRESHOLD"
                                
                                print(f"Level {self.level}, Episode {self.episode_count}: Reward = {ep_reward:.2f} {record_status} {threshold_status}")
                                
                                # Check for convergence (reward stability)
                                if len(self.rewards) >= self.convergence_window:
                                    recent_rewards = self.rewards[-self.convergence_window:]
                                    reward_std = np.std(recent_rewards)
                                    reward_mean = np.mean(recent_rewards)
                                    
                                    # If reward is stable and good, stop early
                                    if (reward_std < self.convergence_threshold and 
                                        reward_mean > self.paper_threshold and 
                                        self.episode_count >= 20):  # Minimum 20 episodes
                                        print(f"✅ Level {self.level} converged! Moving to next level.")
                                        self.model.stop_training = True
                                        return
                                
                                # Hard stop after episodes_per_level
                                if self.episode_count >= episodes_per_level:
                                    print(f"⏹️ Reached maximum episodes for level {self.level}. Moving to next level.")
                                    self.model.stop_training = True
                                    return
        
        callback = CurriculumCallback(level=level, verbose=1)
        max_timesteps = episodes_per_level * env.envs[0].unwrapped.steps_per_episode
        
        print(f"Training level {level} for up to {max_timesteps} timesteps...")
        model.learn(total_timesteps=max_timesteps, callback=callback)
        
        # Store results for this level
        level_results[level] = {
            'rewards': callback.rewards.copy(),
            'best_reward': callback.best_reward,
            'episode_count': callback.episode_count
        }
        
        # Save model for this level
        level_model_path = os.path.join(results_dir, f"curriculum_level_{level}_model.pkl")
        model.save(level_model_path)
        print(f"Level {level} model saved to: {level_model_path}")
        
        # Store best model for next level
        best_model = model
        
        # Print level summary
        if callback.rewards:
            final_avg = np.mean(callback.rewards[-10:])  # Last 10 episodes
            print(f"📊 Level {level} Summary: Best Reward = {callback.best_reward:.2f}, Final Avg = {final_avg:.2f}")
    
    print(f"\n🎓 Curriculum Learning Complete!")
    print(f"Final model saved with best performance across all levels")
    return best_model, level_results

def plot_load_tracking_performance(model, env, episode_num=0):
    """Plot how well the agent tracks the variable loads"""
    print(f"\n📊 Plotting load tracking performance for episode {episode_num}")
    
    # Get the unwrapped environment
    unwrapped_env = env.unwrapped if hasattr(env, 'unwrapped') else env
    
    # Reset environment and run one episode
    obs, _ = env.reset()
    done = False
    step = 0
    
    # Data collection
    steps = []
    loads = []
    torques = []
    joint_torques = []
    errors = []
    rewards = []
    
    print(f"🔍 Running episode with {unwrapped_env.steps_per_episode} steps...")
    
    while not done and step < unwrapped_env.steps_per_episode:
        # Get action from model
        action, _ = model.predict(obs, deterministic=True)
        
        # Store data
        steps.append(step)
        loads.append(unwrapped_env.current_load)
        torques.append(float(action[0]))  # Motor torque T*
        joint_torques.append(float(action[0]) * unwrapped_env.gear_ratio)  # Joint torque
        errors.append(unwrapped_env.current_load - (float(action[0]) * unwrapped_env.gear_ratio))
        
        # Take step
        obs, reward, done, truncated, info = env.step(action)
        rewards.append(reward)
        
        step += 1
    
    # Debug: print action statistics
    print(f"📊 Action Statistics:")
    print(f"   Min action: {min(torques):.4f}")
    print(f"   Max action: {max(torques):.4f}")
    print(f"   Mean action: {np.mean(torques):.4f}")
    print(f"   Action std: {np.std(torques):.4f}")
    
    # Check if actions are all zero or very small
    if np.max(np.abs(torques)) < 0.01:
        print("⚠️  WARNING: Agent is producing near-zero actions! This indicates a training problem.")
        print("   Possible causes:")
        print("   - Action space too restrictive")
        print("   - Reward function not encouraging non-zero actions")
        print("   - Training not converged properly")
    
    # Explain the scaling
    print(f"📏 Torque Scaling Explanation:")
    print(f"   Agent produces motor torques T* in range [-2, 2] N⋅m")
    print(f"   Joint torques = T* × gear_ratio = T* × 120")
    print(f"   Max joint torque = ±2 × 120 = ±240 N⋅m")
    print(f"   Load range: {min(loads):.1f} to {max(loads):.1f} N⋅m")
    print(f"   Agent should produce T* ≈ {max(loads)/120:.3f} to {min(loads)/120:.3f} for perfect tracking")
    
    # Create comparison plot
    fig, axes = plt.subplots(3, 1, figsize=(15, 12))
    
    # Plot 1: Load vs Joint Torque
    axes[0].plot(steps, loads, 'b-', linewidth=2, label='Target Load (N⋅m)', alpha=0.8)
    axes[0].plot(steps, joint_torques, 'r-', linewidth=2, label='Agent Joint Torque (N⋅m)', alpha=0.8)
    axes[0].set_ylabel('Torque (N⋅m)')
    axes[0].set_title(f'Load Tracking Performance - Episode {episode_num}')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Motor Torque T*
    axes[1].plot(steps, torques, 'g-', linewidth=2, label='Motor Torque T* (N⋅m)')
    axes[1].set_ylabel('Motor Torque (N⋅m)')
    axes[1].set_title('Agent Motor Torque Commands')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Tracking Error
    axes[2].plot(steps, errors, 'm-', linewidth=2, label='Tracking Error (N⋅m)')
    axes[2].axhline(y=0, color='k', linestyle='--', alpha=0.5)
    axes[2].set_xlabel('Time Step')
    axes[2].set_ylabel('Error (N⋅m)')
    axes[2].set_title('Tracking Error Over Time')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Calculate performance metrics
    mse = np.mean(np.array(errors) ** 2)
    mae = np.mean(np.abs(errors))
    max_error = np.max(np.abs(errors))
    avg_reward = np.mean(rewards)
    
    # Add performance summary
    fig.suptitle(f'Load Tracking Analysis - MSE: {mse:.2f}, MAE: {mae:.2f}, Max Error: {max_error:.2f}, Avg Reward: {avg_reward:.2f}', 
                 fontsize=14, y=0.98)
    
    # Save plot
    plot_path = os.path.join(results_dir, f"load_tracking_episode_{episode_num}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"📈 Load tracking plot saved to: {plot_path}")
    print(f"📊 Performance Metrics:")
    print(f"   MSE: {mse:.2f}")
    print(f"   MAE: {mae:.2f}")
    print(f"   Max Error: {max_error:.2f}")
    print(f"   Average Reward: {avg_reward:.2f}")
    
    return mse, mae, max_error, avg_reward

if __name__ == "__main__":
    model, level_results = curriculum_training()
    print(f"\n🎉 Training complete! Final episode count: {level_results[2]['episode_count']}")
    
    # Create environment for testing
    test_env = SimpleJointControlEnv(curriculum_level=2) # Test on the highest level
    test_env = Monitor(test_env)
    
    # Plot load tracking performance for the final episode
    print(f"\n🔍 Analyzing load tracking performance...")
    mse, mae, max_error, avg_reward = plot_load_tracking_performance(model, test_env, episode_num=level_results[2]['episode_count'])
    
    print(f"\n📋 Final Performance Summary:")
    print(f"   Best Training Reward: {level_results[2]['best_reward']:.2f}")
    print(f"   Load Tracking MSE: {mse:.2f}")
    print(f"   Load Tracking MAE: {mae:.2f}")
    print(f"   Maximum Tracking Error: {max_error:.2f}")
    print(f"   Test Episode Reward: {avg_reward:.2f}") 