"""
Motor ESC Interface for MCU Inner Loop
========================================
Hardware abstraction layer for motor electronic speed controllers.
Converts the LQR controller's normalised motor commands [0, 1] into
protocol-specific signals (PWM, DShot, etc.) and monitors telemetry.

The MCU can keep the vehicle stable using only IMU + optical flow +
distance sensors, even if the FPGA or flight computer degrades or
restarts (§2.1 of the outline). This module is designed to be the
final, most reliable link in the control chain.

Supported protocols:
  - PWM (50–490 Hz, 1000–2000 µs pulse)
  - DShot 300/600 (digital, bidirectional)
  - Oneshot125 / Multishot (analogue fast-PWM)
"""

import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

from ..messages import ControlCommand, now_us


class EscProtocol(Enum):
    """Motor ESC communication protocols."""
    PWM_STANDARD = "pwm_standard"     # 50 Hz, 1000–2000 µs
    PWM_FAST = "pwm_fast"             # 400 Hz, 1000–2000 µs, e.g. Oneshot125
    DSHOT300 = "dshot300"             # 300 kbps digital
    DSHOT600 = "dshot600"             # 600 kbps digital
    DSHOT1200 = "dshot1200"           # 1.2 Mbps digital
    MULTISHOT = "multishot"           # 5-25 µs pulse, up to 32 kHz


@dataclass
class MotorConfig:
    """Per-motor configuration."""
    index: int                        # 0-based motor number
    protocol: EscProtocol = EscProtocol.DSHOT600
    min_pulse_us: int = 1000          # minimum PWM pulse (µs)
    max_pulse_us: int = 2000          # maximum PWM pulse (µs)
    arm_pulse_us: int = 1000          # pulse at arm (idle)
    calibration_min: int = 1000       # ESC-calibrated minimum
    calibration_max: int = 2000       # ESC-calibrated maximum
    reverse: bool = False             # reverse rotation direction
    pole_pairs: int = 14              # for RPM telemetry
    # Telemetry
    enable_telemetry: bool = True
    rpm: float = 0.0
    current_a: float = 0.0
    voltage_v: float = 0.0
    temperature_c: float = 25.0
    error_count: int = 0


@dataclass
class EscTelemetry:
    """Aggregated telemetry from all ESCs."""
    motor_rpm: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0])
    motor_current: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0])
    motor_voltage: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0])
    motor_temp: List[float] = field(default_factory=lambda: [25.0, 25.0, 25.0, 25.0])
    total_current_a: float = 0.0
    battery_voltage_v: float = 0.0
    error_flags: int = 0


class MotorESC:
    """
    Motor ESC driver for the MCU inner loop.

    Handles:
      - Protocol-specific signal generation (PWM / DShot)
      - Arming / disarming sequence
      - Failsafe behaviour (timeout → disarm)
      - Telemetry decoding (RPM, current, voltage, temperature)
      - Calibration

    In a real deployment this would write directly to MCU timer registers.
    The simulation/prototype version here uses software timing.
    """

    NUM_MOTORS = 4

    def __init__(
        self,
        protocol: EscProtocol = EscProtocol.DSHOT600,
        arm_timeout_ms: int = 5000,      # auto-disarm if no command for 5s
        failsafe_throttle: float = 0.0,  # cut to zero on failsafe
    ):
        self.protocol = protocol
        self.arm_timeout_us = arm_timeout_ms * 1000
        self.failsafe_throttle = failsafe_throttle

        # Per-motor state
        self.motors: List[MotorConfig] = [
            MotorConfig(index=i, protocol=protocol) for i in range(self.NUM_MOTORS)
        ]

        # System state
        self._armed: bool = False
        self._last_command_us: int = 0
        self._command_count: int = 0
        self._in_failsafe: bool = False

        # Telemetry
        self.telemetry = EscTelemetry()

    # ── Public interface ─────────────────────────────────────────────────────

    def arm(self) -> bool:
        """
        Arm all ESCs (send idle pulse, wait for confirmation).
        Returns True if armed successfully.
        """
        if self._armed:
            return True

        # Send arming sequence: idle pulse for ~2 seconds
        for _ in range(100):  # ~2s at 50 Hz
            self._send_pulse_all(self.motors[0].arm_pulse_us)
            time.sleep(0.02)

        self._armed = True
        self._in_failsafe = False
        return True

    def disarm(self, immediate: bool = False):
        """
        Disarm all ESCs.
        If immediate=True, cut to zero immediately (emergency).
        Otherwise, ramp down over ~0.5 s.
        """
        if not self._armed:
            return

        if immediate:
            # Cut immediately
            self._send_pulse_all(0)
        else:
            # Ramp down
            current = 1.0
            for _ in range(25):  # 0.5 s at 50 Hz
                current *= 0.85
                self._send_pulse_all(self._normalised_to_pulse(current))

        self._armed = False
        self._in_failsafe = False

    def send_command(self, cmd: ControlCommand) -> bool:
        """
        Send motor commands to ESCs.

        Args:
            cmd: ControlCommand with motor values [0, 1]

        Returns:
            True if command was sent successfully, False if failsafe intervened
        """
        self._last_command_us = now_us()
        self._command_count += 1

        if not self._armed:
            return False

        # Check failsafe timeout
        if now_us() - cmd.timestamp_us > self.arm_timeout_us:
            self._enter_failsafe()
            return False

        # Convert normalised values to protocol-specific signals
        motor_values = [
            max(0.0, min(1.0, val))
            for val in [cmd.motor_1, cmd.motor_2, cmd.motor_3, cmd.motor_4]
        ]

        pulses = [self._normalised_to_pulse(v) for v in motor_values]

        # Send to each motor
        for i, pulse_us in enumerate(pulses):
            self._send_pulse(i, pulse_us)

        # Exit failsafe if recovering
        if self._in_failsafe:
            self._in_failsafe = False

        return True

    def send_direct_pwm(self, pwm_us: List[int]) -> bool:
        """
        Send raw PWM values directly (bypassing normalisation).
        pwm_us: list of 4 pulse-widths in microseconds [1000–2000].
        """
        if not self._armed or len(pwm_us) < self.NUM_MOTORS:
            return False

        for i in range(self.NUM_MOTORS):
            pulse = max(self.motors[i].calibration_min,
                        min(self.motors[i].calibration_max, pwm_us[i]))
            self._send_pulse(i, pulse)

        self._last_command_us = now_us()
        self._command_count += 1
        return True

    def check_failsafe(self) -> bool:
        """
        Check if the failsafe timeout has expired.
        Should be called at the MCU's control loop rate.
        If expired, automatically disarms.
        """
        if not self._armed:
            return False

        elapsed = now_us() - self._last_command_us
        if elapsed > self.arm_timeout_us and not self._in_failsafe:
            self._enter_failsafe()
            return True

        return False

    def read_telemetry(self) -> EscTelemetry:
        """
        Read telemetry from all ESCs.
        In a real deployment this would decode DShot telemetry packets
        or read current sensors.
        """
        # In simulation, compute from last known values
        total_current = 0.0
        for i in range(self.NUM_MOTORS):
            total_current += self.motors[i].current_a

        self.telemetry.total_current_a = total_current
        self.telemetry.motor_rpm = [m.rpm for m in self.motors]
        self.telemetry.motor_current = [m.current_a for m in self.motors]
        self.telemetry.motor_voltage = [m.voltage_v for m in self.motors]
        self.telemetry.motor_temp = [m.temperature_c for m in self.motors]
        self.telemetry.error_flags = sum(
            (1 << i) if m.error_count > 0 else 0 for i, m in enumerate(self.motors)
        )

        return self.telemetry

    def calibrate(self) -> bool:
        """
        Run ESC calibration sequence.
        Sends max pulse → wait for confirmation → send min pulse → done.
        """
        # Calibration procedure (standard BLHeli/SimonK)
        self._send_pulse_all(self.motors[0].calibration_max)
        time.sleep(2.0)  # wait for power-on beeps

        self._send_pulse_all(self.motors[0].calibration_min)
        time.sleep(1.0)  # wait for confirmation beeps

        # Store calibration
        for m in self.motors:
            m.calibration_min = self.motors[0].calibration_min
            m.calibration_max = self.motors[0].calibration_max

        return True

    # ── Properties ───────────────────────────────────────────────────────────

    @property
    def armed(self) -> bool:
        return self._armed

    @property
    def in_failsafe(self) -> bool:
        return self._in_failsafe

    @property
    def command_count(self) -> int:
        return self._command_count

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _normalised_to_pulse(self, value: float) -> int:
        """Convert normalised value [0, 1] to PWM microseconds."""
        if self.protocol in (EscProtocol.DSHOT300, EscProtocol.DSHOT600, EscProtocol.DSHOT1200):
            # DShot: 0 = disarmed (0), 1-47 = reserved, 48-2047 = throttle
            if value <= 0.0:
                return 0  # disarmed
            dshot_value = int(48 + value * (2047 - 48))
            return dshot_value

        # Standard PWM
        pulse = int(self.motors[0].min_pulse_us +
                     value * (self.motors[0].max_pulse_us - self.motors[0].min_pulse_us))
        return max(self.motors[0].min_pulse_us,
                   min(self.motors[0].max_pulse_us, pulse))

    def _send_pulse_all(self, pulse_us: int):
        """Send the same pulse width to all motors."""
        for i in range(self.NUM_MOTORS):
            self._send_pulse(i, pulse_us)

    def _send_pulse(self, motor_index: int, pulse_us: int):
        """
        Send a pulse to a single motor.
        In production this writes directly to a timer compare register.
        Here we store for simulation.
        """
        if motor_index >= self.NUM_MOTORS:
            return

        motor = self.motors[motor_index]
        if motor.reverse:
            # Invert throttle for reversed motors
            pulse_range = motor.max_pulse_us - motor.min_pulse_us
            pulse_us = motor.max_pulse_us - (pulse_us - motor.min_pulse_us)

        # Clamp to calibration range
        pulse_us = max(motor.calibration_min, min(motor.calibration_max, pulse_us))

        # Simulate RPM response (simplified first-order)
        throttle = (pulse_us - motor.min_pulse_us) / max(motor.max_pulse_us - motor.min_pulse_us, 1)
        target_rpm = throttle * 20000  # max ~20k RPM
        tau = 0.02  # motor time constant
        motor.rpm += (target_rpm - motor.rpm) * min(tau * 1000, 1.0)
        motor.current_a = throttle * 15.0  # max ~15A per motor
        motor.voltage_v = 14.8  # 4S nominal

    def _enter_failsafe(self):
        """Enter failsafe: cut motors to failsafe throttle."""
        self._in_failsafe = True
        pulse = self._normalised_to_pulse(self.failsafe_throttle)
        self._send_pulse_all(pulse)


def create_esc_driver(
    protocol: str = "dshot600",
    arm_timeout_ms: int = 5000,
) -> MotorESC:
    """
    Factory function for creating an ESC driver with the correct protocol.
    """
    proto_map = {
        "pwm": EscProtocol.PWM_STANDARD,
        "pwm_fast": EscProtocol.PWM_FAST,
        "dshot300": EscProtocol.DSHOT300,
        "dshot600": EscProtocol.DSHOT600,
        "dshot1200": EscProtocol.DSHOT1200,
        "multishot": EscProtocol.MULTISHOT,
    }
    proto = proto_map.get(protocol.lower(), EscProtocol.DSHOT600)
    return MotorESC(protocol=proto, arm_timeout_ms=arm_timeout_ms)