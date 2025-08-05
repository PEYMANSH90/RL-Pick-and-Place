#!/usr/bin/env python3
"""
Simple Tracking Test - Test the model on the exact loads it was trained on
"""

import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import TD3
import gymnasium as gym
from gymnasium import spaces
import os

class SimpleTrackingEnv(gym.Env):
    """Simple environment for testing tracking on trained loads"""
    
    def __init__(self, model_path=None):
        super().__init__()
        
        # System parameters
        self.dt = 0.16  # Step duration
        self.gear_ratios = [120, 120, 120, 100, 100, 100]
        self.current_joint = 0
        
        # Use EXACTLY the same loads as Level 2 training
        # Level 2: 50% of original loads, scaled to motor range
        original_loads = [10, 75, -25, 40, 10, -10, -30, 5, 18, 10, 18, 17]
        self.user_loads = [load * 0.5 for load in original_loads]  # 50% of original
        
        # Scale to motor range (±24 N⋅m) - same as training
        max_load_abs = max(abs(max(self.user_loads)), abs(min(self.user_loads)))
        target_range = 24.0  # ±24 N⋅m
        scale_factor = target_range / max_load_abs
        self.user_loads = [load * scale_factor for load in self.user_loads]
        
        print(f"📏 Training loads (Level 2):")
        print(f"   Loads: {[f'{x:.1f}' for x in self.user_loads]}")
        print(f"   Max load: {max([abs(x) for x in self.user_loads]):.1f} N⋅m")
        
        # Create load profile for 100 steps
        steps_per_second = int(1 / self.dt)  # 6.25 steps per second
        basic_pattern = np.repeat(self.user_loads, steps_per_second)
        self.load_torque = np.tile(basic_pattern, 2)[:100]  # Repeat to get 100 steps
        
        # Spaces
        self.action_space = spaces.Box(low=-30, high=30, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=np.array([-2.0, -2.0, -2.0, -2.0, -1.0, -1.0, -1.0, -1.0], dtype=np.float32), 
            high=np.array([2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32), 
            dtype=np.float32
        )
        
        # Load model
        self.model = None
        if model_path and os.path.exists(model_path):
            print(f"🤖 Loading model: {model_path}")
            self.model = TD3.load(model_path)
        
        self.reset()
    
    def reset(self, seed=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.current_joint = 0
        self.T = 0.0
        self.integral_error = 0.0
        self.derivative_error = 0.0
        self.prev_error = 0.0
        self.prev_prev_error = 0.0
        self.prev_T_star = 0.0
        self.current_load = self.load_torque[0]
        
        error = self.current_load - self.T
        return np.array([
            error / 100.0, self.prev_error / 100.0, self.prev_prev_error / 100.0,
            self.integral_error / 100.0, self.derivative_error / 100.0,
            self.T / 100.0, self.current_load / 100.0, self.prev_T_star / 30.0
        ], dtype=np.float32), {}
    
    def step(self, action):
        T_star = float(action[0])
        
        self.T = T_star  # Motor torque
        self.current_step += 1
        done = self.current_step >= 100
        
        if not done:
            self.current_load = self.load_torque[self.current_step]
        
        # Update joint
        self.current_joint = (self.current_step // 16) % 6
        
        # Calculate error
        error = self.current_load - self.T
        self.integral_error += error * self.dt
        self.derivative_error = (error - self.prev_error) / self.dt
        
        # Observation
        obs = np.array([
            error / 100.0, self.prev_error / 100.0, self.prev_prev_error / 100.0,
            self.integral_error / 100.0, self.derivative_error / 100.0,
            self.T / 100.0, self.current_load / 100.0, self.prev_T_star / 30.0
        ], dtype=np.float32)
        
        self.prev_prev_error = self.prev_error
        self.prev_error = error
        self.prev_T_star = T_star
        
        # Reward
        error_penalty = error ** 2
        action_penalty = 0.01 * T_star ** 2
        reward = -(error_penalty + action_penalty)
        
        return obs, reward, done, False, {}

def test_simple_tracking():
    """Test tracking on the exact training loads"""
    print("🎯 Simple Tracking Test")
    print("=" * 40)
    
    model_path = "results/curriculum_level_2_model.pkl"
    env = SimpleTrackingEnv(model_path)
    
    if env.model is None:
        print("❌ No model found!")
        return
    
    # Test tracking
    obs, _ = env.reset()
    tracking_data = {
        'time': [], 'desired': [], 'actual': [], 'error': [], 'reward': []
    }
    
    total_reward = 0
    print("🔄 Testing tracking...")
    
    for step in range(100):
        action, _ = env.model.predict(obs, deterministic=True)
        obs, reward, done, _, _ = env.step(action)
        
        time = step * env.dt
        tracking_data['time'].append(time)
        tracking_data['desired'].append(env.current_load)
        tracking_data['actual'].append(env.T)
        tracking_data['error'].append(env.current_load - env.T)
        tracking_data['reward'].append(reward)
        total_reward += reward
        
        if done:
            break
    
    # Calculate metrics
    avg_reward = total_reward / 100
    rmse_error = np.sqrt(np.mean(np.array(tracking_data['error']) ** 2))
    max_error = np.max(np.abs(tracking_data['error']))
    mean_error = np.mean(np.abs(tracking_data['error']))
    
    print(f"\n📊 Results:")
    print(f"   Average Reward: {avg_reward:.2f}")
    print(f"   RMSE Error: {rmse_error:.4f} N⋅m")
    print(f"   Max Error: {max_error:.4f} N⋅m")
    print(f"   Mean Error: {mean_error:.4f} N⋅m")
    
    # Plot results
    plt.figure(figsize=(12, 8))
    
    # Plot 1: Torque tracking
    plt.subplot(2, 2, 1)
    plt.plot(tracking_data['time'], tracking_data['desired'], 'b-', linewidth=2, label='Desired')
    plt.plot(tracking_data['time'], tracking_data['actual'], 'r--', linewidth=2, label='Actual')
    plt.xlabel('Time (s)')
    plt.ylabel('Torque (N⋅m)')
    plt.title('Torque Tracking')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Error
    plt.subplot(2, 2, 2)
    plt.plot(tracking_data['time'], tracking_data['error'], 'g-', linewidth=2)
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    plt.xlabel('Time (s)')
    plt.ylabel('Error (N⋅m)')
    plt.title(f'Tracking Error (RMSE: {rmse_error:.4f})')
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Reward
    plt.subplot(2, 2, 3)
    plt.plot(tracking_data['time'], tracking_data['reward'], 'purple', linewidth=2)
    plt.xlabel('Time (s)')
    plt.ylabel('Reward')
    plt.title(f'Reward (Avg: {avg_reward:.2f})')
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Performance summary
    plt.subplot(2, 2, 4)
    metrics = ['RMSE', 'Max Error', 'Mean Error', 'Avg Reward']
    values = [rmse_error, max_error, mean_error, abs(avg_reward)]
    colors = ['red', 'orange', 'yellow', 'green']
    
    bars = plt.bar(metrics, values, color=colors, alpha=0.7)
    plt.ylabel('Magnitude')
    plt.title('Performance Metrics')
    plt.xticks(rotation=45)
    
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{value:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('results/simple_tracking_test.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n🎯 Performance: {'✅ EXCELLENT' if avg_reward > -500 else '⚠️ GOOD' if avg_reward > -1000 else '❌ POOR'}")
    print(f"📊 Plot saved to: results/simple_tracking_test.png")

if __name__ == "__main__":
    test_simple_tracking() 