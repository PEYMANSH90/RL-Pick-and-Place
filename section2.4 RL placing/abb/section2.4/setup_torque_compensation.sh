#!/bin/bash
# Setup script for Torque Compensation System

echo "🔧 Setting up Torque Compensation System..."

# Make Python files executable
chmod +x torque_compensation_node.py
chmod +x load_simulator_node.py

# Check if model exists
if [ -f "results/curriculum_level_2_model.pkl" ]; then
    echo "✅ Trained model found: results/curriculum_level_2_model.pkl"
else
    echo "❌ Trained model not found! Please run training first."
    exit 1
fi

echo "✅ Setup complete!"
echo ""
echo "🚀 Usage:"
echo "1. Start Gazebo: gz sim -v 4"
echo "2. Spawn robot: ros2 run ros_gz_sim create -file ~/ros2_ws/src/abb/abb_irb1600_support/urdf/irb1600_6_12.urdf -name abb_irb1600"
echo "3. Run torque compensation: python3 torque_compensation_node.py"
echo "4. Run load simulator: python3 load_simulator_node.py"
echo ""
echo "📊 The robot will automatically compensate for loads and maintain position!" 