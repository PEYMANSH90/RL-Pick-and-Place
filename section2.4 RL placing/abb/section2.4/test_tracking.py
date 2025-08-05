#!/usr/bin/env python3
"""
Test TD3 Model Tracking Performance on Paper's Custom Load Profile
"""

import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import TD3
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
import gymnasium as gym
from gymnasium import spaces
import os
from datetime import datetime

class TrackingTestEnv(gym.Env):
    """Environment for testing tracking performance with paper's load profile"""
    
    def __init__(self, model_path=None):
        super(TrackingTestEnv, self).__init__()
        
        # System parameters from paper
        self.U = 48  # Voltage
        self.psi = 1.6  # Motor constant
        self.inertia = 0.0001  # Extremely low inertia
        self.b = 0.5  # Damping
        self.dt = 0.16  # Step duration: 160 ms
        
        # Gear ratios for ABB IRB1600-6/1.2
        self.gear_ratios = [120, 120, 120, 100, 100, 100]
        self.current_joint = 0
        
        # Paper's original load profile - use original loads directly
        self.original_loads = [10, 75, -25, 40, 10, -10, -30, 5, 18, 10, 18, 17]
        
        # Use the SAME loads as Level 3 training (100% of original loads)
        self.user_loads = self.original_loads.copy()  # 100% of original
        
        print(f"📏 Load scaling for tracking test (matching Level 3 training):")
        print(f"   Original loads: {[f'{x:.1f}' for x in self.original_loads]}")
        print(f"   Final loads: {[f'{x:.1f}' for x in self.user_loads]}")
        print(f"   Max load: {max([abs(x) for x in self.user_loads]):.1f} N⋅m")
        
        # Test parameters
        self.steps_per_episode = 100
        self.total_time = self.steps_per_episode * self.dt
        self.N = int(self.total_time / self.dt)
        
        # Create load torque profile for testing
        self._generate_load_profile()
        
        # Action and observation spaces (same as training)
        action_range = 30
        self.action_space = spaces.Box(
            low=-action_range, high=action_range, shape=(1,), dtype=np.float32
        )
        
        self.observation_space = spaces.Box(
            low=np.array([-2.0, -2.0, -2.0, -2.0, -1.0, -1.0, -1.0, -1.0], dtype=np.float32), 
            high=np.array([2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32), 
            dtype=np.float32
        )
        
        # Load the trained model
        self.model = None
        if model_path and os.path.exists(model_path):
            print(f"🤖 Loading trained model from: {model_path}")
            self.model = TD3.load(model_path)
        else:
            print("❌ No trained model found!")
        
        self.reset()
    
    def _generate_load_profile(self):
        """Generate load profile for testing"""
        # Create load torque profile for 16 seconds (100 steps)
        steps_per_second = int(1 / self.dt)  # 6.25 steps per second
        seconds_per_load = 1  # Each load value lasts 1 second
        
        # Create the basic 12-second pattern
        basic_pattern = np.repeat(self.user_loads, steps_per_second)
        
        # Extend to 16 seconds by repeating the pattern
        total_steps_needed = self.N  # 100 steps
        if len(basic_pattern) < total_steps_needed:
            repeats_needed = int(np.ceil(total_steps_needed / len(basic_pattern)))
            extended_pattern = np.tile(basic_pattern, repeats_needed)
            self.load_torque = extended_pattern[:total_steps_needed]
        else:
            self.load_torque = basic_pattern[:total_steps_needed]
    
    def reset(self, seed=None):
        """Reset environment to initial state"""
        super().reset(seed=seed)
        
        self.current_step = 0
        self.current_joint = 0
        self.T = 0.0  # Current applied torque
        self.integral_error = 0.0
        self.derivative_error = 0.0
        self.prev_error = 0.0
        self.prev_prev_error = 0.0
        self.prev_T_star = 0.0
        
        # Get initial load
        self.current_load = self.load_torque[0]
        
        # Initial error
        error = self.current_load - self.T
        
        return np.array([
            error / 100.0,
            self.prev_error / 100.0,
            self.prev_prev_error / 100.0,
            self.integral_error / 100.0,
            self.derivative_error / 100.0,
            self.T / 100.0,
            self.current_load / 100.0,
            self.prev_T_star / 30.0
        ], dtype=np.float32), {}
    
    def step(self, action):
        """Execute one step in the environment"""
        T_star = float(action[0]) if hasattr(action, '__len__') else float(action)
        
        # Apply gear ratio
        gear_ratio = self.gear_ratios[self.current_joint]
        T_joint = T_star * gear_ratio
        
        self.T = T_star  # Keep T as motor torque for action penalty
        
        self.current_step += 1
        done = self.current_step >= self.steps_per_episode
        truncated = False
        
        if not done and self.current_step < len(self.load_torque):
            self.current_load = self.load_torque[self.current_step]
        
        # Update current joint
        self.current_joint = (self.current_step // 16) % 6
        
        # Calculate error at JOINT level (load is joint torque, T_joint is joint torque)
        joint_error = self.current_load - T_joint
        self.integral_error += joint_error * self.dt
        self.derivative_error = (joint_error - self.prev_error) / self.dt
        
        # Update delayed errors
        obs = np.array([
            joint_error / 100.0,
            self.prev_error / 100.0,
            self.prev_prev_error / 100.0,
            self.integral_error / 100.0,
            self.derivative_error / 100.0,
            self.T / 100.0,
            self.current_load / 100.0,
            self.prev_T_star / 30.0
        ], dtype=np.float32)
        
        self.prev_prev_error = self.prev_error
        self.prev_error = joint_error
        self.prev_T_star = T_star
        
        # Paper reward function: joint error + motor action penalty
        error_penalty = joint_error ** 2
        action_penalty = 0.01 * T_star ** 2  # Use motor torque for action penalty
        reward = -(error_penalty + action_penalty)
        
        return obs, reward, done, truncated, {}

def test_tracking_performance(model_path=None):
    """Test the trained model's tracking performance"""
    print("🎯 Testing TD3 Model Tracking Performance")
    print("=" * 60)
    
    # Create test environment
    env = TrackingTestEnv(model_path=model_path)
    
    if env.model is None:
        print("❌ Cannot test without a trained model!")
        return
    
    # Test the model
    obs, _ = env.reset()
    total_reward = 0
    tracking_data = {
        'time': [],
        'desired_torque': [],
        'actual_torque': [],
        'error': [],
        'reward': [],
        'action': []
    }
    
    print("🔄 Running tracking test...")
    
    for step in range(env.steps_per_episode):
        # Get action from trained model
        action, _ = env.model.predict(obs, deterministic=True)
        
        # Take step
        obs, reward, done, truncated, _ = env.step(action)
        
        # Store data
        time = step * env.dt
        tracking_data['time'].append(time)
        tracking_data['desired_torque'].append(env.current_load)
        tracking_data['actual_torque'].append(env.T)
        tracking_data['error'].append(env.current_load - env.T)
        tracking_data['reward'].append(reward)
        tracking_data['action'].append(action[0])
        
        total_reward += reward
        
        if done:
            break
    
    # Calculate performance metrics
    avg_reward = total_reward / env.steps_per_episode
    rmse_error = np.sqrt(np.mean(np.array(tracking_data['error']) ** 2))
    max_error = np.max(np.abs(tracking_data['error']))
    mean_error = np.mean(np.abs(tracking_data['error']))
    
    print(f"\n📊 Tracking Performance Results:")
    print(f"   Average Reward: {avg_reward:.2f}")
    print(f"   RMSE Error: {rmse_error:.4f} N⋅m")
    print(f"   Max Error: {max_error:.4f} N⋅m")
    print(f"   Mean Error: {mean_error:.4f} N⋅m")
    
    # Plot results
    plot_tracking_results(tracking_data, avg_reward, rmse_error, max_error, mean_error)
    
    return tracking_data, avg_reward, rmse_error

def plot_tracking_results(tracking_data, avg_reward, rmse_error, max_error, mean_error):
    """Plot the tracking performance results"""
    plt.figure(figsize=(15, 12))
    
    # Plot 1: Torque tracking
    plt.subplot(3, 2, 1)
    plt.plot(tracking_data['time'], tracking_data['desired_torque'], 'b-', linewidth=2, label='Desired Torque')
    plt.plot(tracking_data['time'], tracking_data['actual_torque'], 'r--', linewidth=2, label='Actual Torque')
    plt.xlabel('Time (s)')
    plt.ylabel('Torque (N⋅m)')
    plt.title('Torque Tracking Performance')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Tracking error
    plt.subplot(3, 2, 2)
    plt.plot(tracking_data['time'], tracking_data['error'], 'g-', linewidth=2)
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    plt.xlabel('Time (s)')
    plt.ylabel('Error (N⋅m)')
    plt.title(f'Tracking Error (RMSE: {rmse_error:.4f})')
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Reward over time
    plt.subplot(3, 2, 3)
    plt.plot(tracking_data['time'], tracking_data['reward'], 'purple', linewidth=2)
    plt.xlabel('Time (s)')
    plt.ylabel('Reward')
    plt.title(f'Reward Over Time (Avg: {avg_reward:.2f})')
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Action commands
    plt.subplot(3, 2, 4)
    plt.plot(tracking_data['time'], tracking_data['action'], 'orange', linewidth=2)
    plt.xlabel('Time (s)')
    plt.ylabel('Action (N⋅m)')
    plt.title('Motor Commands')
    plt.grid(True, alpha=0.3)
    
    # Plot 5: Error distribution
    plt.subplot(3, 2, 5)
    plt.hist(tracking_data['error'], bins=20, alpha=0.7, color='lightblue', edgecolor='black')
    plt.xlabel('Error (N⋅m)')
    plt.ylabel('Frequency')
    plt.title('Error Distribution')
    plt.grid(True, alpha=0.3)
    
    # Plot 6: Performance summary
    plt.subplot(3, 2, 6)
    metrics = ['RMSE', 'Max Error', 'Mean Error', 'Avg Reward']
    values = [rmse_error, max_error, mean_error, abs(avg_reward)]
    colors = ['red', 'orange', 'yellow', 'green']
    
    bars = plt.bar(metrics, values, color=colors, alpha=0.7)
    plt.ylabel('Magnitude')
    plt.title('Performance Metrics')
    plt.xticks(rotation=45)
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{value:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    # Save plot
    results_dir = "results"
    plot_path = os.path.join(results_dir, f"tracking_performance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"📊 Tracking results plotted and saved to: {plot_path}")

if __name__ == "__main__":
    # Test with the Level 3 model (trained on original loads)
    model_path = "results/curriculum_level_3_model.pkl"
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        print("Available models:")
        results_dir = "results"
        for file in os.listdir(results_dir):
            if file.endswith(".pkl"):
                print(f"   - {file}")
    else:
        tracking_data, avg_reward, rmse_error = test_tracking_performance(model_path)
        
        print(f"\n🎯 TRACKING TEST COMPLETE!")
        print(f"Model: {model_path}")
        print(f"Average Reward: {avg_reward:.2f}")
        print(f"RMSE Error: {rmse_error:.4f} N⋅m")
        print(f"Paper Threshold: -493")
        print(f"Performance: {'✅ EXCELLENT' if avg_reward > -500 else '⚠️ GOOD' if avg_reward > -1000 else '❌ NEEDS IMPROVEMENT'}") 