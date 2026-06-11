"""
Path Planning for Cortex-A Mission Layer
==========================================
Payload-adaptive waypoint generation feeding the MCU's LQR controller.
Consumes position/motion from HDC-EVIO, the environment map from HDC-SLAM,
and mission goals derived from payload sensor data (§4.3 of the outline).

Planner types:
  - Straight-line waypoint following with obstacle avoidance
  - Coverage patterns (lawnmower / spiral) for survey missions
  - Payload-reactive: adjust waypoints based on payload detection
  - RRT* for complex environments (optional, when map is dense)
"""

import numpy as np
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass, field
from collections import deque
import heapq

from ..messages import StateEstimate, Waypoint, now_us


@dataclass
class PlannerConfig:
    """Configuration for the path planner."""
    # Waypoint parameters
    default_altitude_m: float = 50.0
    default_speed_m_s: float = 5.0
    acceptance_radius_m: float = 2.0
    hold_time_s: float = 0.5

    # Safety limits
    min_altitude_m: float = 5.0
    max_altitude_m: float = 120.0
    max_speed_m_s: float = 15.0
    max_accel_m_s2: float = 5.0

    # Obstacle avoidance
    obstacle_avoidance_enabled: bool = True
    obstacle_safety_radius_m: float = 5.0   # keep-out zone around detected obstacles
    obstacle_replan_interval_s: float = 2.0  # re-check obstacles every N seconds
    max_obstacles: int = 100                 # max stored obstacles from SLAM

    # Coverage pattern
    coverage_row_spacing_m: float = 10.0  # lawnmower row-to-row distance
    coverage_turn_radius_m: float = 15.0

    # Payload adaptation
    payload_reactive: bool = True
    payload_loiter_time_s: float = 10.0  # hover time when payload triggers
    payload_approach_distance_m: float = 20.0  # distance at which to slow down

    # Mission limits
    max_waypoints: int = 500
    max_flight_time_s: float = 1800.0  # 30 minutes
    home_position: Tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass
class Obstacle:
    """Detected obstacle from HDC-SLAM map."""
    position: np.ndarray   # [3] x, y, z (world frame)
    radius_m: float = 2.0  # estimated radius
    confidence: float = 0.5
    id: int = 0


class PathPlanner:
    """
    Path planner that generates waypoints from mission goals while
    avoiding obstacles and adapting to payload sensor data.

    Implements §4.3 of the outline:
      "Path Planning consumes position/motion (HDC-EVIO), the environment
       map (HDC-SLAM), and mission goals derived from payload sensor data."

    Modes:
      - waypoint_following: Visit a list of predefined waypoints
      - coverage: Execute a lawnmower/spiral coverage pattern
      - loiter: Hold position (for payload inspection)
      - return_home: Fly back to takeoff point
      - payload_guided: Adjust flight based on payload detections
    """

    def __init__(self, config: Optional[PlannerConfig] = None):
        self.cfg = config or PlannerConfig()
        self._waypoint_queue: deque = deque()
        self._visited_waypoints: List[Waypoint] = []
        self._obstacles: List[Obstacle] = []
        self._current_target: Optional[Waypoint] = None
        self._mode: str = "idle"
        self._flight_start_time_us: int = 0
        self._last_obstacle_check_us: int = 0
        self._payload_detection_active: bool = False
        self._payload_position: Optional[np.ndarray] = None

    # ── Public interface ─────────────────────────────────────────────────────

    def set_mission_waypoints(self, waypoints: List[Tuple[float, float, float]]):
        """
        Load a list of (x, y, z) waypoints for the vehicle to visit.
        """
        self._waypoint_queue.clear()
        self._mode = "waypoint_following"

        for i, (x, y, z) in enumerate(waypoints):
            self._waypoint_queue.append(Waypoint(
                timestamp_us=now_us(),
                x=x, y=y, z=z,
                acceptance_radius_m=self.cfg.acceptance_radius_m,
                speed_m_s=self.cfg.default_speed_m_s,
            ))

    def set_coverage_mission(
        self,
        x_min: float, y_min: float,
        x_max: float, y_max: float,
        altitude: float = 50.0,
    ):
        """
        Generate a lawnmower coverage pattern for a rectangular area.
        """
        self._waypoint_queue.clear()
        self._mode = "coverage"
        spacing = self.cfg.coverage_row_spacing_m

        y = y_min
        direction = 1  # 1 = left-to-right, -1 = right-to-left
        while y <= y_max:
            if direction == 1:
                wp = Waypoint(
                    timestamp_us=now_us(),
                    x=x_min, y=y, z=altitude,
                    acceptance_radius_m=self.cfg.acceptance_radius_m,
                    speed_m_s=self.cfg.default_speed_m_s,
                )
            else:
                wp = Waypoint(
                    timestamp_us=now_us(),
                    x=x_max, y=y, z=altitude,
                    acceptance_radius_m=self.cfg.acceptance_radius_m,
                    speed_m_s=self.cfg.default_speed_m_s,
                )
            self._waypoint_queue.append(wp)

            y += spacing
            if y > y_max:
                break

            # Turn-around row
            if direction == 1:
                turn_wp = Waypoint(
                    timestamp_us=now_us(),
                    x=x_max, y=y, z=altitude,
                    acceptance_radius_m=self.cfg.acceptance_radius_m,
                    speed_m_s=self.cfg.default_speed_m_s,
                )
            else:
                turn_wp = Waypoint(
                    timestamp_us=now_us(),
                    x=x_min, y=y, z=altitude,
                    acceptance_radius_m=self.cfg.acceptance_radius_m,
                    speed_m_s=self.cfg.default_speed_m_s,
                )
            self._waypoint_queue.append(turn_wp)
            direction *= -1

    def set_loiter(self, position: np.ndarray, duration_s: float = 10.0):
        """Command the vehicle to loiter at a position."""
        self._waypoint_queue.clear()
        self._mode = "loiter"
        self._waypoint_queue.append(Waypoint(
            timestamp_us=now_us(),
            x=float(position[0]), y=float(position[1]), z=float(position[2]),
            hold_time_s=duration_s,
            acceptance_radius_m=1.0,
        ))

    def set_return_home(self, current_z: float = 50.0):
        """Command the vehicle to return to the home position."""
        self._waypoint_queue.clear()
        self._mode = "return_home"

        hx, hy, hz = self.cfg.home_position
        # First climb to safe altitude, then fly home, then descend
        self._waypoint_queue.append(Waypoint(
            timestamp_us=now_us(),
            x=0.0, y=0.0, z=current_z,  # maintain altitude
            acceptance_radius_m=1.0,
            speed_m_s=self.cfg.default_speed_m_s,
        ))
        self._waypoint_queue.append(Waypoint(
            timestamp_us=now_us(),
            x=hx, y=hy, z=current_z,
            acceptance_radius_m=self.cfg.acceptance_radius_m,
            speed_m_s=self.cfg.default_speed_m_s,
        ))
        self._waypoint_queue.append(Waypoint(
            timestamp_us=now_us(),
            x=hx, y=hy, z=hz,
            acceptance_radius_m=2.0,
            speed_m_s=2.0,  # slow descent
        ))

    # ── Payload integration ──────────────────────────────────────────────────

    def on_payload_detection(self, position: np.ndarray, object_class: str = "unknown"):
        """
        Callback when the payload sensor detects an object of interest.
        Adjusts mission: insert loiter or approach waypoint.

        Implements "payload-adaptive missions" from §1.4 of the outline:
          "the flight computer consumes payload sensor data and adjusts
           waypoints/mission goals in flight"
        """
        self._payload_detection_active = True
        self._payload_position = position.copy()

        if self.cfg.payload_reactive:
            # Insert a loiter waypoint near the detection
            # Approach to a safe distance
            approach_dist = self.cfg.payload_approach_distance_m
            dx = position[0]
            dy = position[1]
            dist = np.sqrt(dx ** 2 + dy ** 2)
            if dist > approach_dist and dist > 1e-6:
                approach_x = position[0] * (1 - approach_dist / dist)
                approach_y = position[1] * (1 - approach_dist / dist)
            else:
                approach_x = position[0]
                approach_y = position[1]

            # Clear queue and set loiter
            self._waypoint_queue.clear()
            self._mode = "payload_guided"

            self._waypoint_queue.append(Waypoint(
                timestamp_us=now_us(),
                x=approach_x, y=approach_y, z=max(30.0, position[2]),
                hold_time_s=self.cfg.payload_loiter_time_s,
                acceptance_radius_m=5.0,
                speed_m_s=3.0,  # slow approach
            ))

    def on_payload_lost(self):
        """Callback when payload detection is lost."""
        self._payload_detection_active = False
        self._payload_position = None
        self._mode = "idle"

    # ── Obstacle handling ────────────────────────────────────────────────────

    def update_obstacles(self, obstacles: List[Obstacle]):
        """Update the obstacle map from HDC-SLAM."""
        self._obstacles = obstacles[-self.cfg.max_obstacles:]

    def _check_obstacle_clearance(self, wp: Waypoint, state: StateEstimate) -> bool:
        """Check if a waypoint is safe from known obstacles."""
        if not self.cfg.obstacle_avoidance_enabled:
            return True

        wp_pos = np.array([wp.x, wp.y, wp.z])
        for obs in self._obstacles:
            dist = np.linalg.norm(wp_pos[:2] - obs.position[:2])
            min_dist = obs.radius_m + self.cfg.obstacle_safety_radius_m
            if dist < min_dist:
                return False
        return True

    def _evade_obstacle(self, wp: Waypoint, state: StateEstimate) -> Waypoint:
        """Adjust waypoint to avoid nearest obstacle (simple perpendicular offset)."""
        wp_pos = np.array([wp.x, wp.y])
        state_pos = np.array([state.x, state.y])

        # Find nearest obstacle
        nearest = None
        nearest_dist = float("inf")
        for obs in self._obstacles:
            dist = np.linalg.norm(wp_pos - obs.position[:2])
            if dist < nearest_dist:
                nearest_dist = dist
                nearest = obs

        if nearest is None:
            return wp

        # Perpendicular offset direction
        to_obs = nearest.position[:2] - wp_pos
        to_obs_norm = np.linalg.norm(to_obs)
        if to_obs_norm < 1e-6:
            return wp

        # Offset perpendicular to obstacle direction
        perp = np.array([-to_obs[1], to_obs[0]]) / to_obs_norm
        offset = self.cfg.obstacle_safety_radius_m + nearest.radius_m
        new_xy = nearest.position[:2] + perp * offset

        return Waypoint(
            timestamp_us=now_us(),
            x=float(new_xy[0]), y=float(new_xy[1]), z=wp.z,
            yaw=wp.yaw,
            hold_time_s=wp.hold_time_s,
            acceptance_radius_m=wp.acceptance_radius_m,
            speed_m_s=wp.speed_m_s,
        )

    # ── Main update ──────────────────────────────────────────────────────────

    def step(
        self,
        state: StateEstimate,
        payload_detection: Optional[Dict[str, Any]] = None,
    ) -> Optional[Waypoint]:
        """
        Produce the next waypoint target for the LQR controller.

        Called at ~10 Hz (Cortex-A planning loop).

        Args:
            state: Current state estimate
            payload_detection: Optional payload sensor data

        Returns:
            Next Waypoint, or None if idle/complete
        """
        # Start flight timer on first call
        if self._flight_start_time_us == 0:
            self._flight_start_time_us = now_us()

        # Check flight time limit
        elapsed_s = (now_us() - self._flight_start_time_us) / 1e6
        if elapsed_s > self.cfg.max_flight_time_s:
            self.set_return_home(current_z=state.z)
            self._flight_start_time_us = now_us()  # reset to avoid looping

        # Process payload detection
        if payload_detection is not None and self.cfg.payload_reactive:
            pos = payload_detection.get("position")
            cls_name = payload_detection.get("class", "unknown")
            if pos is not None:
                self.on_payload_detection(np.array(pos), cls_name)

        # Check waypoint arrival
        if self._current_target is not None:
            dx = state.x - self._current_target.x
            dy = state.y - self._current_target.y
            dz = state.z - self._current_target.z
            dist = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)

            if dist < self._current_target.acceptance_radius_m:
                # Waypoint reached
                self._visited_waypoints.append(self._current_target)
                self._current_target = None

        # Pop next waypoint if needed
        if self._current_target is None and len(self._waypoint_queue) > 0:
            self._current_target = self._waypoint_queue.popleft()

        # Obstacle avoidance
        if self._current_target is not None and self.cfg.obstacle_avoidance_enabled:
            now = now_us()
            if now - self._last_obstacle_check_us > self.cfg.obstacle_replan_interval_s * 1e6:
                self._last_obstacle_check_us = now
                if not self._check_obstacle_clearance(self._current_target, state):
                    self._current_target = self._evade_obstacle(self._current_target, state)

        # Enforce altitude limits
        if self._current_target is not None:
            z_clamped = float(np.clip(
                self._current_target.z,
                self.cfg.min_altitude_m,
                self.cfg.max_altitude_m,
            ))
            if z_clamped != self._current_target.z:
                self._current_target = Waypoint(
                    timestamp_us=now_us(),
                    x=self._current_target.x,
                    y=self._current_target.y,
                    z=z_clamped,
                    yaw=self._current_target.yaw,
                    hold_time_s=self._current_target.hold_time_s,
                    acceptance_radius_m=self._current_target.acceptance_radius_m,
                    speed_m_s=self._current_target.speed_m_s,
                )

        return self._current_target

    # ── Accessors ────────────────────────────────────────────────────────────

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def waypoints_remaining(self) -> int:
        return len(self._waypoint_queue)

    @property
    def waypoints_visited(self) -> int:
        return len(self._visited_waypoints)

    @property
    def is_complete(self) -> bool:
        return self._current_target is None and len(self._waypoint_queue) == 0

    def get_visited_trajectory(self) -> np.ndarray:
        if not self._visited_waypoints:
            return np.zeros((0, 3))
        return np.array([[wp.x, wp.y, wp.z] for wp in self._visited_waypoints])

    def reset(self):
        self._waypoint_queue.clear()
        self._visited_waypoints.clear()
        self._obstacles.clear()
        self._current_target = None
        self._mode = "idle"
        self._flight_start_time_us = 0
        self._last_obstacle_check_us = 0
        self._payload_detection_active = False
        self._payload_position = None


class MissionPlanner:
    """
    High-level mission controller that manages payload goals and
    coordinates between path planning, state estimation, and SLAM.

    Implements "Payload-adaptive missions" from §1.4 of the outline.
    """

    def __init__(self, config: Optional[PlannerConfig] = None):
        self.cfg = config or PlannerConfig()
        self.planner = PathPlanner(self.cfg)
        self._mission_phase: str = "takeoff"  # takeoff → cruise → mission → return
        self._takeoff_altitude_reached: bool = False

    def start_mission(
        self,
        waypoints: Optional[List[Tuple[float, float, float]]] = None,
        coverage_area: Optional[Tuple[float, float, float, float]] = None,
    ):
        """
        Start a new mission.

        Args:
            waypoints: Optional list of (x, y, z) waypoints
            coverage_area: Optional (x_min, y_min, x_max, y_max) for coverage
        """
        self._mission_phase = "takeoff"
        self._takeoff_altitude_reached = False

        # Queue takeoff
        alt = self.cfg.default_altitude_m
        if waypoints:
            alt = waypoints[0][2]
        elif coverage_area:
            alt = 50.0

        self.planner.set_mission_waypoints([
            (0.0, 0.0, alt),  # takeoff waypoint
        ])
        self.planner._mode = "waypoint_following"

        # Store mission waypoints for after takeoff
        self._mission_waypoints = waypoints
        self._coverage_area = coverage_area

    def step(
        self,
        state: StateEstimate,
        slam_map: Optional[np.ndarray] = None,
        payload_detection: Optional[Dict[str, Any]] = None,
    ) -> Optional[Waypoint]:
        """
        Advance the mission one planning step.

        Returns the next Waypoint for the LQR controller.
        """
        # Phase transitions
        if self._mission_phase == "takeoff":
            if abs(state.z - self.cfg.default_altitude_m) < 3.0:
                self._mission_phase = "cruise"
                self._takeoff_altitude_reached = True

                # Load actual mission
                if self._mission_waypoints:
                    self.planner.set_mission_waypoints(self._mission_waypoints)
                elif self._coverage_area:
                    x_min, y_min, x_max, y_max = self._coverage_area
                    self.planner.set_coverage_mission(
                        x_min, y_min, x_max, y_max, self.cfg.default_altitude_m
                    )
                else:
                    self.planner.set_loiter(np.array([
                        state.x, state.y, state.z
                    ]))

        elif self._mission_phase == "cruise":
            if self.planner.is_complete:
                self._mission_phase = "return"

        elif self._mission_phase == "return":
            if self.planner.is_complete:
                self._mission_phase = "complete"

        # Import obstacles from SLAM map
        if slam_map is not None and len(slam_map) > 0:
            obstacles = [
                Obstacle(
                    position=pos,
                    radius_m=2.0,
                    confidence=0.7,
                    id=i,
                )
                for i, pos in enumerate(slam_map[:50])
            ]
            self.planner.update_obstacles(obstacles)

        return self.planner.step(state, payload_detection)

    @property
    def phase(self) -> str:
        return self._mission_phase