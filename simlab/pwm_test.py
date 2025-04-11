# Copyright (C) 2025 Edward Morgan
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

#!/usr/bin/env python3
"""
A ROS 2 node that constantly publishes a neutral command on 
/forward_pwm_controller/commands and modifies individual channels 
based on keyboard key press/release events.

Mapping:
    u -> channel 0
    i -> channel 1
    o -> channel 2
    p -> channel 3
    h -> channel 4
    j -> channel 5
    k -> channel 6
    l -> channel 7

Neutral value: 1500.0
Active value:  1600.0
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from pynput import keyboard  # pip install pynput

# Mapping from key characters to channel indices
KEY_TO_CHANNEL = {
    'u': 0,
    'i': 1,
    'o': 2,
    'p': 3,
    'h': 4,
    'j': 5,
    'k': 6,
    'l': 7,
}

NEUTRAL_VALUE = 1500.0
ACTIVE_VALUE = 1700.0

class KeyMappingNeutralPublisher(Node):
    def __init__(self):
        super().__init__('key_mapping_neutral_publisher')

        # Create a publisher for the commands.
        self.publisher_ = self.create_publisher(
            Float64MultiArray,
            '/forward_pwm_controller/commands',
            10
        )

        # Define the base neutral command message.
        self.neutral_msg = [NEUTRAL_VALUE] * 8

        # Maintain the state of each key (True if pressed, False otherwise).
        self.active_keys = {key: False for key in KEY_TO_CHANNEL.keys()}

        # Set up a timer to publish the command message at 10 Hz.
        timer_period = 1/150  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.get_logger().info('Key Mapping Neutral Publisher Node has started.')

        # Start the keyboard listener in a separate thread.
        self.listener = keyboard.Listener(
            on_press=self.on_press,
            on_release=self.on_release
        )
        self.listener.start()

    def timer_callback(self):
        """
        Prepare and publish the current command message.
        Each channel is set to ACTIVE_VALUE if its corresponding key is pressed,
        otherwise it is NEUTRAL_VALUE.
        """
        # Start with a neutral command.
        current_cmd = self.neutral_msg.copy()

        # Modify channels based on which keys are active.
        for key, is_active in self.active_keys.items():
            if is_active:
                channel = KEY_TO_CHANNEL[key]
                current_cmd[channel] = ACTIVE_VALUE

        # Publish the command.
        msg = Float64MultiArray()
        msg.data = current_cmd
        self.publisher_.publish(msg)
        self.get_logger().debug(f'Published command: {msg.data}')

    def on_press(self, key):
        """
        Callback for key press events.
        If a mapped key is pressed, mark it as active.
        """
        try:
            char = key.char.lower()  # normalize to lowercase
        except AttributeError:
            # If the key does not have a char attribute (like special keys), ignore it.
            return

        if char in self.active_keys:
            if not self.active_keys[char]:
                self.active_keys[char] = True
                self.get_logger().info(f'Key pressed: {char} (channel {KEY_TO_CHANNEL[char]} active)')

    def on_release(self, key):
        """
        Callback for key release events.
        If a mapped key is released, mark it as inactive.
        Pressing the Escape key stops the listener.
        """
        try:
            char = key.char.lower()
        except AttributeError:
            # If the key does not have a char attribute, check if it is the escape key.
            if key == keyboard.Key.esc:
                self.get_logger().info("Escape pressed. Exiting keyboard listener.")
                return False
            return

        if char in self.active_keys:
            self.active_keys[char] = False
            self.get_logger().info(f'Key released: {char} (channel {KEY_TO_CHANNEL[char]} returned to neutral)')

def main(args=None):
    rclpy.init(args=args)
    node = KeyMappingNeutralPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Keyboard Interrupt (SIGINT) received. Shutting down...')
    finally:
        node.listener.stop()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
