The **uvms_simlab** project is an extension of the [uvms-simulator](https://github.com/edxmorgan/uvms-simulator) that adds several advanced functionalities to enhance both simulation and real hardware operations. 

---

### Key Features

- **Interactive Marker Control:**  
  Use RViz to interactively control the vehicle and its end-effector with 6-DOF markers that update in real time, which makes planning and adjustments intuitive.

- **Self Collision Avoidance:**  
  Built-in algorithms automatically adjust robot paths during both interactive and automated maneuvers to prevent self collisions.

- **Coverage Planning:**  
  Execute comprehensive coverage plans that take into account obstacles and the dynamics of the robot, ensuring complete workspace coverage.

- **PS4 Joystick Teleoperation:**  
  Integrated support for PS4 controllers allows for seamless manual control via Bluetooth, ideal for teleoperation tasks.

- **Real Hardware & Simulation Support:**  
  The project is designed to work in both simulated environments and on actual hardware, offering flexibility in testing and deployment.

---

### Dependencies

Before getting started, ensure you have the following installed:

- **pyPS4Controller:**  
  Install it via:
  ```bash
  sudo pip install pyPS4Controller
  ```
- **scipy**
- **casadi**
- **pandas**
---

### Quick Start Guide

1. **Clone the Repository:**  
   Navigate to the `src` folder of your ROS2 workspace and clone the repository:
   ```bash
   cd /absolute/path/to/your_ros2_workspace/src
   git clone https://github.com/edxmorgan/uvms_simlab.git
   ```
   Then, go back to the workspace root:
   ```bash
   cd /absolute/path/to/your_ros2_workspace
   ```

2. **Build the Workspace:**  
   Compile the project using:
   ```bash
   colcon build
   source install/setup.bash
   ```

---

### Running the Simulation

- **Interactive Markers Control (Recommended):**  
  Launch the simulation with full interactive control in RViz:
  ```bash
  ros2 launch ros2_control_blue_reach_5 robot_system_multi_interface.launch.py \
      use_manipulator_hardware:=false use_vehicle_hardware:=false \
      sim_robot_count:=1 task:=interactive
  ```

- **Manual Mode (Joystick Control):**  
  For teleoperation via a PS4 joystick:
  ```bash
  ros2 launch ros2_control_blue_reach_5 robot_system_multi_interface.launch.py \
      use_manipulator_hardware:=false use_vehicle_hardware:=false \
      sim_robot_count:=1 task:=manual
  ```

- **Coverage Execution:**  
  Execute coverage plans with integrated self collision avoidance:
  ```bash
  ros2 launch ros2_control_blue_reach_5 robot_system_multi_interface.launch.py \
      use_manipulator_hardware:=false use_vehicle_hardware:=false \
      sim_robot_count:=1 task:=coverage
  ```

---

### Additional Notes

- **Integration with uvms-simulator:**  
  This project assumes that [uvms-simulator](https://github.com/edxmorgan/uvms-simulator) is already present and built in your ROS workspace.

- **Contributions:**  
  Contributions are welcome. You can fork the repository and submit a pull request if you have improvements or fixes.
