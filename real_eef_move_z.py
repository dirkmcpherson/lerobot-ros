#!/usr/bin/env python3
"""Closed-loop Cartesian world-Z servo for the real gen3_lite via the twist_controller.

The Kinova twist interface is in the TOOL frame, so we read base_link->tool_frame
each cycle, compute the desired WORLD velocity (hold x/y, drive z to target), and
rotate it into the tool frame before publishing to /twist_controller/commands.

Safety:
  - small vmax, per-move timeout
  - abort if lateral (x,y) drift exceeds LAT_ABORT
  - abort if z moves the WRONG way beyond WRONG_ABORT
  - always publishes zero twist on exit (finally)
"""
import sys, time, math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import tf2_ros

VMAX = 0.025        # m/s max commanded speed
KP = 1.2
TOL = 0.004         # m, success tolerance
RATE = 40.0         # Hz
TIMEOUT = 8.0       # s per move
LAT_ABORT = 0.020   # m lateral drift abort
WRONG_ABORT = 0.010 # m wrong-direction abort


def quat_to_R(x, y, z, w):
    return [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ]


def Rt_mul(R, v):  # R^T @ v
    return [sum(R[k][i] * v[k] for k in range(3)) for i in range(3)]


class Servo(Node):
    def __init__(self):
        super().__init__("eef_z_servo")
        self.pub = self.create_publisher(Twist, "/twist_controller/commands", 10)
        self.buf = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.buf, self)

    def lookup(self):
        t = self.buf.lookup_transform("base_link", "tool_frame", rclpy.time.Time())
        tr = t.transform.translation
        q = t.transform.rotation
        return [tr.x, tr.y, tr.z], (q.x, q.y, q.z, q.w)

    def zero(self, n=5):
        z = Twist()
        for _ in range(n):
            self.pub.publish(z)
            time.sleep(0.02)

    def wait_tf(self, timeout=5.0):
        end = time.time() + timeout
        while time.time() < end:
            rclpy.spin_once(self, timeout_sec=0.1)
            try:
                return self.lookup()
            except Exception:
                continue
        raise RuntimeError("TF base_link->tool_frame not available")

    def move_dz(self, dz):
        p0, _ = self.wait_tf()
        target = [p0[0], p0[1], p0[2] + dz]
        print(f"  start z={p0[2]:.4f} -> target z={target[2]:.4f} (dz={dz:+.3f})", flush=True)
        t_start = time.time()
        dt = 1.0 / RATE
        try:
            while True:
                rclpy.spin_once(self, timeout_sec=0.0)
                try:
                    p, q = self.lookup()
                except Exception:
                    time.sleep(dt); continue
                e = [target[i] - p[i] for i in range(3)]
                enorm = math.sqrt(sum(c * c for c in e))
                if enorm < TOL:
                    print(f"  reached z={p[2]:.4f} (err={enorm*1000:.1f} mm)", flush=True)
                    return p
                # safety checks
                lat = math.hypot(p[0] - p0[0], p[1] - p0[1])
                if lat > LAT_ABORT:
                    self.zero(); raise RuntimeError(f"ABORT: lateral drift {lat*1000:.1f} mm")
                zprog = p[2] - p0[2]              # actual z change so far
                if dz < 0 and zprog > WRONG_ABORT:
                    self.zero(); raise RuntimeError(f"ABORT: z moving UP {zprog*1000:.1f} mm, wanted DOWN")
                if dz > 0 and zprog < -WRONG_ABORT:
                    self.zero(); raise RuntimeError(f"ABORT: z moving DOWN {zprog*1000:.1f} mm, wanted UP")
                if time.time() - t_start > TIMEOUT:
                    self.zero(); raise RuntimeError(f"ABORT: timeout, z={p[2]:.4f} err={enorm*1000:.1f} mm")
                # world velocity -> clip -> tool frame
                vw = [KP * c for c in e]
                vn = math.sqrt(sum(c * c for c in vw))
                if vn > VMAX:
                    vw = [c * VMAX / vn for c in vw]
                R = quat_to_R(*q)
                vt = Rt_mul(R, vw)
                cmd = Twist()
                cmd.linear.x, cmd.linear.y, cmd.linear.z = vt
                self.pub.publish(cmd)
                time.sleep(dt)
        finally:
            self.zero()


def main():
    rclpy.init()
    node = Servo()
    try:
        print("DOWN 0.05 m ...", flush=True)
        node.move_dz(-0.05)
        time.sleep(0.5)
        print("UP 0.05 m ...", flush=True)
        node.move_dz(+0.05)
        print("DONE", flush=True)
    except Exception as ex:
        print(f"!! {ex}", flush=True)
        node.zero(10)
        sys.exit(1)
    finally:
        node.zero(10)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
