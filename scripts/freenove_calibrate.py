#!/usr/bin/env python3
"""
MH-FLOCKE — Freenove Calibration Stand
========================================
Moves all servos to the calibrated standing position.
Use this to verify servo alignment before walking.

Standing position (from IK: x=10, y=99, z=10):
  Hip yaw:   90° (centered)
  Hip pitch: 128° (FL/RL) / 52° (FR/RR)
  Knee:      91° (FL/RL) / 89° (FR/RR)

Usage:
    python3 freenove_calibrate.py           # Stand and hold
    python3 freenove_calibrate.py --sweep   # Sweep each joint for testing
    python3 freenove_calibrate.py --zero    # All servos to 90° (mechanical zero)

Author: MH-FLOCKE Project (Marc Hesse)
License: Apache 2.0
"""

import time, math, sys, os, argparse

HOME_DIR = os.path.expanduser('~')
FREENOVE_SEARCH_PATHS = [
    os.path.join(HOME_DIR, 'freenove_server'),
    os.path.join(HOME_DIR, 'Freenove_Robot_Dog_Kit_for_Raspberry_Pi', 'Code', 'Server'),
    '/home/pi/freenove_server',
]

# Channel mapping (from profile.json)
# FL: ch4=hip_yaw, ch3=hip_pitch, ch2=knee
# FR: ch11=hip_yaw, ch12=hip_pitch, ch13=knee
# RL: ch7=hip_yaw, ch6=hip_pitch, ch5=knee
# RR: ch8=hip_yaw, ch9=hip_pitch, ch10=knee
LEGS = {
    'FL': {'yaw': 4, 'pitch': 3, 'knee': 2, 'side': 'left'},
    'RL': {'yaw': 7, 'pitch': 6, 'knee': 5, 'side': 'left'},
    'RR': {'yaw': 8, 'pitch': 9, 'knee': 10, 'side': 'right'},
    'FR': {'yaw': 11, 'pitch': 12, 'knee': 13, 'side': 'right'},
}

def ik(x, y, z, l1=23, l2=55, l3=55):
    """Inverse kinematics — same as Bridge."""
    a = math.pi/2 - math.atan2(z, y)
    x4 = l1*math.sin(a); x5 = l1*math.cos(a)
    l23 = math.sqrt((z-x5)**2+(y-x4)**2+x**2)
    b = math.asin(round(x/l23,2)) - math.acos(round((l2*l2+l23*l23-l3*l3)/(2*l2*l23),2))
    c = math.pi - math.acos(round((l2**2+l3**2-l23**2)/(2*l3*l2),2))
    return round(math.degrees(a)), round(math.degrees(b)), round(math.degrees(c))


def get_standing_angles():
    """Compute standing servo angles for all 4 legs."""
    # Standing position: x=10, y=99, z=10 (from MJCF ref)
    yaw_abs, pitch_abs, knee_abs = ik(10, 99, 10)
    
    angles = {}
    for name, leg in LEGS.items():
        if leg['side'] == 'left':
            angles[leg['yaw']] = max(0, min(180, yaw_abs))
            angles[leg['pitch']] = max(0, min(180, 90 - pitch_abs))
            angles[leg['knee']] = max(0, min(180, knee_abs))
        else:
            angles[leg['yaw']] = max(0, min(180, yaw_abs))
            angles[leg['pitch']] = max(0, min(180, 90 + pitch_abs))
            angles[leg['knee']] = max(0, min(180, 180 - knee_abs))
    
    return angles


def init_servo():
    """Initialize PCA9685 servo driver."""
    for path in FREENOVE_SEARCH_PATHS:
        if os.path.exists(os.path.join(path, 'Servo.py')):
            sys.path.insert(0, path)
            try:
                from Servo import Servo
                servo = Servo()
                print('  PCA9685 initialized')
                return servo
            except Exception as e:
                print(f'  WARNING: {e}')
    print('  ERROR: Could not initialize servo driver')
    sys.exit(1)


def set_angle_slow(servo, channel, target, speed=2.0):
    """Move servo slowly to target angle (degrees per step)."""
    # Read current isn't possible on PCA9685, so just move directly
    servo.setServoAngle(channel, max(0, min(180, target)))


def main():
    parser = argparse.ArgumentParser(description='MH-FLOCKE Freenove Calibration')
    parser.add_argument('--sweep', action='store_true',
                        help='Sweep each joint ±20° for testing')
    parser.add_argument('--zero', action='store_true',
                        help='All servos to 90° (mechanical zero)')
    parser.add_argument('--hold', type=float, default=0,
                        help='Hold position for N seconds (0=forever)')
    args = parser.parse_args()

    print(f'\n{"="*50}')
    print(f'  MH-FLOCKE — Freenove Calibration')
    print(f'{"="*50}\n')

    servo = init_servo()

    if args.zero:
        # All to 90° — mechanical zero point
        print('  Setting all servos to 90° (mechanical zero)...')
        for ch in range(2, 14):
            servo.setServoAngle(ch, 90)
            time.sleep(0.05)
        print('  All servos at 90°. Check mechanical alignment.')
        print('  Press Ctrl+C to exit.')
        try:
            while True: time.sleep(1)
        except KeyboardInterrupt:
            print('\n  Done.')
        return

    # Compute standing angles
    angles = get_standing_angles()
    yaw_abs, pitch_abs, knee_abs = ik(10, 99, 10)

    print(f'  IK standing: yaw={yaw_abs}°  pitch={pitch_abs}°  knee={knee_abs}°')
    print()

    # Set standing position
    print('  Moving to standing position...')
    for name, leg in LEGS.items():
        y = angles[leg['yaw']]
        p = angles[leg['pitch']]
        k = angles[leg['knee']]
        servo.setServoAngle(leg['yaw'], y)
        servo.setServoAngle(leg['pitch'], p)
        servo.setServoAngle(leg['knee'], k)
        print(f'    {name}: yaw={y}°  pitch={p}°  knee={k}°  '
              f'(ch {leg["yaw"]},{leg["pitch"]},{leg["knee"]})')
        time.sleep(0.1)

    print(f'\n  Standing position set. Robot should be level.')

    if args.sweep:
        print(f'\n  Starting joint sweep test...\n')
        time.sleep(2.0)

        for name, leg in LEGS.items():
            for joint_name in ['yaw', 'pitch', 'knee']:
                ch = leg[joint_name]
                center = angles[ch]
                print(f'  Sweeping {name} {joint_name} (ch{ch}): '
                      f'{center-15}° → {center}° → {center+15}°')

                # Sweep down
                for a in range(int(center), int(center) - 16, -1):
                    servo.setServoAngle(ch, max(0, min(180, a)))
                    time.sleep(0.03)
                time.sleep(0.3)

                # Sweep up
                for a in range(int(center) - 15, int(center) + 16):
                    servo.setServoAngle(ch, max(0, min(180, a)))
                    time.sleep(0.03)
                time.sleep(0.3)

                # Back to center
                for a in range(int(center) + 15, int(center) - 1, -1):
                    servo.setServoAngle(ch, max(0, min(180, a)))
                    time.sleep(0.03)
                time.sleep(0.5)

        print(f'\n  Sweep complete. Back to standing.')
        # Restore standing
        for ch, angle in angles.items():
            servo.setServoAngle(ch, angle)

    if args.hold > 0:
        print(f'\n  Holding for {args.hold}s...')
        time.sleep(args.hold)
        print('  Done.')
    else:
        print(f'\n  Press Ctrl+C to release servos.')
        try:
            while True: time.sleep(1)
        except KeyboardInterrupt:
            pass

    # Release: set to 90° (neutral, less strain)
    print('\n  Releasing servos to 90°...')
    for ch in range(2, 14):
        servo.setServoAngle(ch, 90)
        time.sleep(0.02)
    print('  Done.')


if __name__ == '__main__':
    main()
