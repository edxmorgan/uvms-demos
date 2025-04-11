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

import numpy as np

class Task:
    def __init__(self, initial_pos):
        self.initial_pos = initial_pos

    def square_velocity_uv_ref(self, t_now, T_side=10.0, speed=0.05, depth_rate=0.005):
        """
        Produces:
          1) A velocity vector (u, v, w, p, q, r, q1, q2, q3, q4) for a periodic square path
             in the XY-plane, with an additional constant descent rate along Z.
          2) The integrated state [x, y, z, roll, pitch, yaw, s1, s2, s3, s4],
             where s1..s4 correspond to q1..q4 (or any other states you want to integrate).

        In this modified version, the XY positions are computed directly from the time,
        ensuring that the length of each side is exactly fixed to:
          side_length = speed * T_side

        :param t_now: Current time (in seconds)
        :param T_side: Time (in seconds) for one side of the square.
        :param speed: Desired linear speed along a side.
        :param depth_rate: The constant rate (in m/s) at which depth (z) decreases.
        :return:
            velocity: np.array of shape (10,) with [u, v, w, p, q, r, q1, q2, q3, q4]
            state:   np.array of shape (10,) with [x, y, z, roll, pitch, yaw, s1, s2, s3, s4]
        """
        # Compute side length exactly.
        side_length = speed * T_side

        # Determine which side and the time within that side.
        cycle_time = 4.0 * T_side
        t_mod = t_now % cycle_time
        side_index = int(t_mod // T_side)
        t_in_side = t_mod % T_side

        # Initialize the velocity vector.
        velocity = np.zeros(10, dtype=float)

        # Calculate x, y offsets and corresponding velocities based on the current side.
        if side_index == 0:
            # Moving in +X direction.
            x_offset = t_in_side * speed
            y_offset = 0.0
            velocity[0] = speed
        elif side_index == 1:
            # Moving in +Y direction.
            x_offset = side_length
            y_offset = t_in_side * speed
            velocity[1] = speed
        elif side_index == 2:
            # Moving in -X direction.
            x_offset = side_length - t_in_side * speed
            y_offset = side_length
            velocity[0] = -speed
        else:
            # Moving in -Y direction.
            x_offset = 0.0
            y_offset = side_length - t_in_side * speed
            velocity[1] = -speed

        # Set constant downward velocity.
        velocity[2] = -depth_rate

        # Instead of integrating XY from the previous state, compute them directly.
        state = self.initial_pos.copy()
        state[0] = self.initial_pos[0] + x_offset
        state[1] = self.initial_pos[1] + y_offset

        # For the depth, you can either integrate or compute directly.
        # Here, we compute it directly so that the descent is exact:
        new_depth = self.initial_pos[2] - depth_rate * t_now
        state[2] = new_depth if new_depth >= 0 else 0

        # Optionally, if you have other states (e.g. orientation, s1-s4), you can update them as needed.
        # For now, we'll leave them unchanged.

        return velocity.flatten(), state.flatten()
