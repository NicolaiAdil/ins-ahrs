#!/usr/bin/env python3
"""
EKF Debug Plotter (ROS 2 Jazzy)

Compares:
- Yaw: EKF (Odometry) vs LIO truth (/lio/pose)
- Roll/Pitch: EKF vs LIO truth
Also shows:
- Position (N–E, NED) tracks: EKF vs LIO truth
- Z position (D in NED) vs time: EKF vs LIO truth
- Velocity panel: EKF only (truth PoseStamped has no velocity)

Each series has its own timestamps to avoid compressing the visible time window.

Also plots radar extrinsics (position and attitude of radar w.r.t. IMU/body) over time:
- /rio/radar_position  (Vector3Stamped)  [m]
- /rio/radar_attitude  (QuaternionStamped)  [rad, xyz]
with ground-truth constants:
- l_BR_B = [0.077, 0.016, -0.063]
- q_R_B  = [0.963, -0.021, -0.265, 0.021] (xyzw, converted with axes='sxyz')
"""

from collections import deque
import numpy as np
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped, Vector3Stamped, QuaternionStamped
import tf_transformations
import matplotlib.pyplot as plt
import time


def ssa(angle):
    """Wrap to (-pi, pi]."""
    a = (angle + np.pi) % (2.0 * np.pi) - np.pi
    if np.isclose(a, -np.pi):
        a = np.pi
    return a


def unwrap_append(prev_unwrapped, new_wrapped):
    if prev_unwrapped is None:
        return float(new_wrapped)
    k = round((prev_unwrapped - new_wrapped) / (2 * np.pi))
    return float(new_wrapped + 2 * np.pi * k)


def enu_pose_to_ned_euler_and_ne(qx, qy, qz, qw, px, py, pz):
    """
    Convert an ENU pose (qx,qy,qz,qw, px,py,pz) to NED roll,pitch,yaw and (N,E).
    """
    R_enu_to_ned = np.array([[0, 1, 0],
                             [1, 0, 0],
                             [0, 0, -1]], dtype=float)
    R4 = np.eye(4, dtype=float)
    R4[:3, :3] = R_enu_to_ned

    # Orientation: ENU -> NED via similarity transform
    M_enu = tf_transformations.quaternion_matrix([qx, qy, qz, qw])  # 4x4
    M_ned = R4 @ M_enu @ R4.T
    r, p, y = tf_transformations.euler_from_matrix(M_ned)
    r, p, y = ssa(r), ssa(p), ssa(y)

    # Position: x=E, y=N, z=U  ->  N=y, E=x (D=-z not used here)
    N = float(py)
    E = float(px)
    return r, p, y, N, E


class EKFDebugPlotter(Node):
    def __init__(self):
        super().__init__('ekf_debug_plotter')

        # Parameters
        self.declare_parameter('topic_state', '/rio/pose')
        self.declare_parameter('topic_truth_pose', '/lio/pose')
        self.declare_parameter('truth_frame', 'NED')  # 'ENU' or 'NED'
        self.declare_parameter('window_secs', 300.0)
        self.declare_parameter('plot_rate_hz', 5.0)
        self.declare_parameter('flip_warn_thresh_deg', 150.0)

        # Radar extrinsics topics
        self.declare_parameter('topic_radar_pos', '/rio/radar_position')
        self.declare_parameter('topic_radar_att', '/rio/radar_attitude')

        self.topic_state = self.get_parameter('topic_state').value
        self.topic_truth_pose = self.get_parameter('topic_truth_pose').value
        self.truth_frame = str(self.get_parameter('truth_frame').value).upper()
        self.window_secs = float(self.get_parameter('window_secs').value)
        self.plot_dt = 1.0 / float(self.get_parameter('plot_rate_hz').value)
        self.flip_warn_thresh = np.deg2rad(
            float(self.get_parameter('flip_warn_thresh_deg').value)
        )

        self.topic_radar_pos = self.get_parameter('topic_radar_pos').value
        self.topic_radar_att = self.get_parameter('topic_radar_att').value

        # Ground-truth radar extrinsics (body frame)
        self.radar_pos_true = np.array([0.077, 0.016, -0.063], dtype=float)
        q_R_B = [0.963, -0.021, -0.265, 0.021]  # xyzw

        self.radar_att_true_rad = np.array(
            tf_transformations.euler_from_quaternion(q_R_B, axes='sxyz'),
            dtype=float
        )

        # Time zero
        self.t0 = None

        # Per-series time/value histories (each with bounded length)
        self.t_yaw_est, self.yaw_est_hist = deque(), deque()
        self.t_yaw_truth, self.yaw_truth_hist = deque(), deque()

        self.t_roll_est, self.roll_est_hist = deque(), deque()
        self.t_pitch_est, self.pitch_est_hist = deque(), deque()
        self.t_roll_truth, self.roll_truth_hist = deque(), deque()
        self.t_pitch_truth, self.pitch_truth_hist = deque(), deque()

        # Z (D in NED) position histories
        self.t_z_est, self.z_est_hist = deque(), deque()
        self.t_z_truth, self.z_truth_hist = deque(), deque()

        # Velocities (EKF only)
        self.t_vN_est, self.vN_est = deque(), deque()
        self.t_vE_est, self.vE_est = deque(), deque()

        # NE tracks
        self.ne_est = deque()    # (N,E)
        self.ne_truth = deque()  # (N,E)

        # Flip markers (times)
        self.flip_marks_t = deque()

        # Unwrap trackers
        self._last_yaw_est = None
        self._last_yaw_truth = None

        # Radar extrinsics histories
        self.t_rpos = deque()
        self.rpos_x, self.rpos_y, self.rpos_z = deque(), deque(), deque()

        self.t_ratt = deque()
        self.ratt_roll, self.ratt_pitch, self.ratt_yaw = deque(), deque(), deque()

        # Subscribers
        self.create_subscription(Odometry, self.topic_state, self.cb_state, 10)
        self.create_subscription(PoseStamped, self.topic_truth_pose, self.cb_truth_pose, 10)
        self.create_subscription(Vector3Stamped, self.topic_radar_pos, self.cb_radar_pos, 10)
        self.create_subscription(QuaternionStamped, self.topic_radar_att, self.cb_radar_att, 10)

        # Figures + timer
        self._make_figure_main()
        self._make_figure_extrinsics()
        self._plot_timer = self.create_timer(self.plot_dt, self._on_plot_timer)

        self.get_logger().info(
            "EKF Debug Plotter started.\n"
            f" Subscribed to:\n"
            f"  state:           {self.topic_state}\n"
            f"  truthPose:       {self.topic_truth_pose}  (frame={self.truth_frame})\n"
            f"  radar_position:  {self.topic_radar_pos}\n"
            f"  radar_attitude:  {self.topic_radar_att}\n"
            f" Window = {self.window_secs}s, plot_rate = {1.0 / self.plot_dt:.1f} Hz"
        )

    # ----------------------- Callbacks -----------------------

    def _now_s(self):
        if self.t0 is None:
            self.t0 = time.time()
        return time.time() - self.t0

    def cb_state(self, msg: Odometry):
        t = self._now_s()

        # EKF orientation (RPY in NED) – uses tf default (sxyz) for consistency with Odometry
        q = msg.pose.pose.orientation
        r, p, y = tf_transformations.euler_from_quaternion([q.x, q.y, q.z, q.w], axes='sxyz')
        r, p, y = ssa(r), ssa(p), ssa(y)

        # Save yaw (unwrapped), roll, pitch
        self._append_tv(self.t_yaw_est, self.yaw_est_hist, t, y)
        self._append_tv(self.t_roll_est, self.roll_est_hist, t, r)
        self._append_tv(self.t_pitch_est, self.pitch_est_hist, t, p)

        # NE (NED) for map
        N = float(msg.pose.pose.position.x)
        E = float(msg.pose.pose.position.y)
        self._append_limited(self.ne_est, (N, E))

        # Z (D in NED) position
        D = float(msg.pose.pose.position.z)
        self._append_tv(self.t_z_est, self.z_est_hist, t, D)

        # Vel (NED)
        vN = float(msg.twist.twist.linear.x)
        vE = float(msg.twist.twist.linear.y)
        self._append_tv(self.t_vN_est, self.vN_est, t, vN)
        self._append_tv(self.t_vE_est, self.vE_est, t, vE)

    def cb_truth_pose(self, msg: PoseStamped):
        t = self._now_s()

        px, py, pz = msg.pose.position.x, msg.pose.position.y, msg.pose.position.z
        qx, qy, qz, qw = msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w

        if self.truth_frame == 'ENU':
            r, p, y, N, E = enu_pose_to_ned_euler_and_ne(qx, qy, qz, qw, px, py, pz)
            D = -float(pz)  # ENU z is Up -> NED D = -z
        else:
            # Already NED
            r, p, y = tf_transformations.euler_from_quaternion([qx, qy, qz, qw])
            r, p, y = ssa(r), ssa(p), ssa(y)
            N, E = float(px), float(py)
            D = float(pz)

        # Append orientation
        un = unwrap_append(self._last_yaw_truth, y)
        self._last_yaw_truth = un
        self._append_tv(self.t_yaw_truth, self.yaw_truth_hist, t, un)
        self._append_tv(self.t_roll_truth, self.roll_truth_hist, t, r)
        self._append_tv(self.t_pitch_truth, self.pitch_truth_hist, t, p)

        # Append position (NE, Z)
        self._append_limited(self.ne_truth, (N, E))
        self._append_tv(self.t_z_truth, self.z_truth_hist, t, D)

        # Flip detector (vs latest EKF yaw if present)
        if len(self.yaw_est_hist) > 0:
            diff = ssa(self.yaw_truth_hist[-1] - self.yaw_est_hist[-1])
            if abs(abs(diff) - np.pi) < np.deg2rad(15) or abs(diff) > self.flip_warn_thresh:
                self._append_limited(self.flip_marks_t, t)
                self.get_logger().warn(
                    f"Possible 180° flip: |Δyaw|={np.degrees(abs(diff)):.1f}° at t={t:.1f}s"
                )

    def cb_radar_pos(self, msg: Vector3Stamped):
        t = self._now_s()
        # 3-vector position in body frame
        self.t_rpos.append(float(t))
        self.rpos_x.append(float(msg.vector.x))
        self.rpos_y.append(float(msg.vector.y))
        self.rpos_z.append(float(msg.vector.z))
        if len(self.t_rpos) > 5000:
            self.t_rpos.popleft()
            self.rpos_x.popleft()
            self.rpos_y.popleft()
            self.rpos_z.popleft()

    def cb_radar_att(self, msg: QuaternionStamped):
        t = self._now_s()

        # Quaternion of radar attitude in body frame (xyzw)
        q = msg.quaternion
        q_xyzw = [q.x, q.y, q.z, q.w]

        roll_r, pitch_r, yaw_r = tf_transformations.euler_from_quaternion(
            q_xyzw, axes='sxyz'
        )

        self.t_ratt.append(float(t))
        self.ratt_roll.append(float(roll_r))
        self.ratt_pitch.append(float(pitch_r))
        self.ratt_yaw.append(float(yaw_r))

        if len(self.t_ratt) > 5000:
            self.t_ratt.popleft()
            self.ratt_roll.popleft()
            self.ratt_pitch.popleft()
            self.ratt_yaw.popleft()

    # ----------------------- Plotting -----------------------

    def _make_figure_main(self):
        plt.ion()
        self.fig = plt.figure(figsize=(12, 14))
        # Extra row for Z plot
        gs = self.fig.add_gridspec(5, 1, height_ratios=[1.2, 1.0, 1.0, 1.0, 1.0], hspace=0.35)

        # Yaw
        self.ax_head = self.fig.add_subplot(gs[0, 0])
        self.ax_head.set_title("Yaw vs Time")
        self.ax_head.set_ylabel("Yaw [deg]")
        self.ax_head.set_xlabel("Time [s]")
        # EKF thin, LIO thicker
        self.l_est_head, = self.ax_head.plot([], [], label="EKF yaw", linewidth=1.2)
        self.l_truth_head, = self.ax_head.plot([], [], label="LIO truth yaw", linewidth=2.4)
        self.flip_scatter = self.ax_head.scatter([], [], marker='x', label="Flip?")
        self.ax_head.legend(loc='best')
        self.ax_head.grid(True)

        # Roll/Pitch
        self.ax_rp = self.fig.add_subplot(gs[1, 0])
        self.ax_rp.set_title("Roll & Pitch vs Time")
        self.ax_rp.set_ylabel("Angle [deg]")
        self.ax_rp.set_xlabel("Time [s]")
        self.l_roll_est, = self.ax_rp.plot([], [], label="EKF roll", linewidth=1.2)
        self.l_pitch_est, = self.ax_rp.plot([], [], label="EKF pitch", linewidth=1.2)
        self.l_roll_truth, = self.ax_rp.plot([], [], label="LIO roll", linewidth=2.4)
        self.l_pitch_truth, = self.ax_rp.plot([], [], label="LIO pitch", linewidth=2.4)
        self.ax_rp.legend(loc='best')
        self.ax_rp.grid(True)

        # Position NE
        self.ax_ne = self.fig.add_subplot(gs[2, 0])
        self.ax_ne.set_title("Position (N–E in NED)")
        self.ax_ne.set_xlabel("E [m]")
        self.ax_ne.set_ylabel("N [m]")
        self.l_est_ne, = self.ax_ne.plot([], [], label="EKF track")
        self.l_truth_ne, = self.ax_ne.plot([], [], linestyle='None', marker='.', label="LIO truth")
        self.ax_ne.axis('equal')
        self.ax_ne.grid(True)
        self.ax_ne.legend(loc='best')

        # Z (D in NED)
        self.ax_z = self.fig.add_subplot(gs[3, 0])
        self.ax_z.set_title("Vertical Position vs Time (D in NED)")
        self.ax_z.set_ylabel("D [m] (down positive)")
        self.ax_z.set_xlabel("Time [s]")
        self.l_z_est, = self.ax_z.plot([], [], label="EKF D")
        self.l_z_truth, = self.ax_z.plot([], [], label="LIO D", linewidth=2.0)
        self.ax_z.grid(True)
        self.ax_z.legend(loc='best')

        # Velocity (EKF only)
        self.ax_vel = self.fig.add_subplot(gs[4, 0])
        self.ax_vel.set_title("Velocity Components vs Time (NED)")
        self.ax_vel.set_ylabel("v [m/s]")
        self.ax_vel.set_xlabel("Time [s]")
        self.l_vN_est, = self.ax_vel.plot([], [], label="vN EKF")
        self.l_vE_est, = self.ax_vel.plot([], [], label="vE EKF")
        self.ax_vel.grid(True)
        self.ax_vel.legend(loc='best')

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        try:
            plt.show(block=False)
        except Exception:
            pass

    def _make_figure_extrinsics(self):
        # Second figure for radar extrinsics
        self.fig_ext, (self.ax_rpos, self.ax_ratt) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        self.fig_ext.subplots_adjust(hspace=0.35)

        # Radar translation
        self.ax_rpos.set_title("Radar Translation Extrinsics vs Time (body frame)")
        self.ax_rpos.set_ylabel("Position [m]")

        # Use same colors per component: x=C0, y=C1, z=C2
        # EKF estimates (solid)
        self.l_rpos_x_est, = self.ax_rpos.plot([], [], label="EKF p_rx", color='C0', linestyle='-')
        self.l_rpos_y_est, = self.ax_rpos.plot([], [], label="EKF p_ry", color='C1', linestyle='-')
        self.l_rpos_z_est, = self.ax_rpos.plot([], [], label="EKF p_rz", color='C2', linestyle='-')
        # Ground truth (dashed, same colors)
        self.l_rpos_x_gt,  = self.ax_rpos.plot([], [], label="GT p_rx",  color='C0', linestyle='--')
        self.l_rpos_y_gt,  = self.ax_rpos.plot([], [], label="GT p_ry",  color='C1', linestyle='--')
        self.l_rpos_z_gt,  = self.ax_rpos.plot([], [], label="GT p_rz",  color='C2', linestyle='--')

        self.ax_rpos.grid(True)
        self.ax_rpos.legend(loc='best')

        # Radar attitude
        self.ax_ratt.set_title("Radar Rotation Extrinsics vs Time (body frame, xyz)")
        self.ax_ratt.set_ylabel("Angle [deg]")
        self.ax_ratt.set_xlabel("Time [s]")

        # roll=C0, pitch=C1, yaw=C2
        # EKF estimates (solid)
        self.l_ratt_r_est, = self.ax_ratt.plot([], [], label="EKF roll_r",  color='C0', linestyle='-')
        self.l_ratt_p_est, = self.ax_ratt.plot([], [], label="EKF pitch_r", color='C1', linestyle='-')
        self.l_ratt_y_est, = self.ax_ratt.plot([], [], label="EKF yaw_r",   color='C2', linestyle='-')
        # Ground truth (dashed, same colors)
        self.l_ratt_r_gt,  = self.ax_ratt.plot([], [], label="GT roll_r",   color='C0', linestyle='--')
        self.l_ratt_p_gt,  = self.ax_ratt.plot([], [], label="GT pitch_r",  color='C1', linestyle='--')
        self.l_ratt_y_gt,  = self.ax_ratt.plot([], [], label="GT yaw_r",    color='C2', linestyle='--')

        self.ax_ratt.grid(True)
        self.ax_ratt.legend(loc='best')

        self.fig_ext.canvas.draw()
        self.fig_ext.canvas.flush_events()
        try:
            plt.show(block=False)
        except Exception:
            pass


    def _on_plot_timer(self):
        try:
            self._refresh_plot()
        except Exception as e:
            self.get_logger().warn(f"Plot refresh error: {e}")

    @staticmethod
    def _finite_xy(tx, yy):
        if len(tx) == 0 or len(yy) == 0:
            return np.array([]), np.array([])
        tx = np.asarray(tx)
        yy = np.asarray(yy)
        m = np.isfinite(tx) & np.isfinite(yy)
        return tx[m], yy[m]

    def _refresh_plot(self):
        # Current visible window based on wall time
        now = self._now_s()
        tmin = max(0.0, now - self.window_secs)
        tmax = now

        # ---- Yaw (deg) ----
        te, ye = self._finite_xy(self.t_yaw_est, np.degrees(self.yaw_est_hist))
        tt, yt = self._finite_xy(self.t_yaw_truth, np.degrees(self.yaw_truth_hist))

        self.l_est_head.set_data(te, ye)
        self.l_truth_head.set_data(tt, yt)

        # Flip markers
        flips_t = np.asarray(self.flip_marks_t)
        if flips_t.size and te.size:
            flips_y = np.interp(flips_t, te, ye)
            self.flip_scatter.remove()
            self.flip_scatter = self.ax_head.scatter(flips_t, flips_y, marker='x')
        else:
            flips_y = np.array([])

        self.ax_head.set_xlim([tmin, tmax])
        self.ax_head.relim()
        if flips_t.size:
            self.ax_head.update_datalim(np.column_stack([flips_t, flips_y]))
        self.ax_head.autoscale_view(scalex=False, scaley=True)

        # ---- Roll/Pitch (deg) ----
        tr_e, rr_e = self._finite_xy(self.t_roll_est, np.degrees(self.roll_est_hist))
        tp_e, pp_e = self._finite_xy(self.t_pitch_est, np.degrees(self.pitch_est_hist))
        tr_t, rr_t = self._finite_xy(self.t_roll_truth, np.degrees(self.roll_truth_hist))
        tp_t, pp_t = self._finite_xy(self.t_pitch_truth, np.degrees(self.pitch_truth_hist))

        self.l_roll_est.set_data(tr_e, rr_e)
        self.l_pitch_est.set_data(tp_e, pp_e)
        self.l_roll_truth.set_data(tr_t, rr_t)
        self.l_pitch_truth.set_data(tp_t, pp_t)

        self.ax_rp.set_xlim([tmin, tmax])
        self.ax_rp.relim()
        self.ax_rp.autoscale_view(scalex=False, scaley=True)

        # ---- Position NE ----
        ne_est_arr = np.array(self.ne_est) if len(self.ne_est) else np.empty((0, 2))
        ne_tru_arr = np.array(self.ne_truth) if len(self.ne_truth) else np.empty((0, 2))
        if ne_est_arr.shape[0] > 0:
            self.l_est_ne.set_data(ne_est_arr[:, 1], ne_est_arr[:, 0])  # x=E, y=N
        else:
            self.l_est_ne.set_data([], [])
        if ne_tru_arr.shape[0] > 0:
            self.l_truth_ne.set_data(ne_tru_arr[:, 1], ne_tru_arr[:, 0])
        else:
            self.l_truth_ne.set_data([], [])

        if ne_est_arr.shape[0] + ne_tru_arr.shape[0] > 1:
            allE = []
            allN = []
            if ne_est_arr.shape[0] > 0:
                allE += ne_est_arr[:, 1].tolist()
                allN += ne_est_arr[:, 0].tolist()
            if ne_tru_arr.shape[0] > 0:
                allE += ne_tru_arr[:, 1].tolist()
                allN += ne_tru_arr[:, 0].tolist()
            padE = (max(allE) - min(allE)) * 0.1 + 1.0
            padN = (max(allN) - min(allN)) * 0.1 + 1.0
            self.ax_ne.set_xlim([min(allE) - padE, max(allE) + padE])
            self.ax_ne.set_ylim([min(allN) - padN, max(allN) + padN])

        # ---- Z (D in NED) ----
        t_ze, ze = self._finite_xy(self.t_z_est, self.z_est_hist)
        t_zt, zt = self._finite_xy(self.t_z_truth, self.z_truth_hist)

        self.l_z_est.set_data(t_ze, ze)
        self.l_z_truth.set_data(t_zt, zt)

        self.ax_z.set_xlim([tmin, tmax])
        self.ax_z.relim()
        self.ax_z.autoscale_view(scalex=False, scaley=True)

        # ---- Velocity (EKF only) ----
        t_vNe, vNe = self._finite_xy(self.t_vN_est, self.vN_est)
        t_vEe, vEe = self._finite_xy(self.t_vE_est, self.vE_est)

        self.l_vN_est.set_data(t_vNe, vNe)
        self.l_vE_est.set_data(t_vEe, vEe)

        self.ax_vel.set_xlim([tmin, tmax])
        self.ax_vel.relim()
        self.ax_vel.autoscale_view(scalex=False, scaley=True)

        # ---- Radar extrinsics ----
        # Translation
        if len(self.t_rpos) > 0:
            t_rpos = np.asarray(self.t_rpos, dtype=float)
            mask = (t_rpos >= tmin) & (t_rpos <= tmax)
            t_rpos_w = t_rpos[mask]
            rx = np.asarray(self.rpos_x, dtype=float)[mask]
            ry = np.asarray(self.rpos_y, dtype=float)[mask]
            rz = np.asarray(self.rpos_z, dtype=float)[mask]

            self.l_rpos_x_est.set_data(t_rpos_w, rx)
            self.l_rpos_y_est.set_data(t_rpos_w, ry)
            self.l_rpos_z_est.set_data(t_rpos_w, rz)

            # GT lines over same support
            self.l_rpos_x_gt.set_data(t_rpos_w, np.full_like(t_rpos_w, self.radar_pos_true[0]))
            self.l_rpos_y_gt.set_data(t_rpos_w, np.full_like(t_rpos_w, self.radar_pos_true[1]))
            self.l_rpos_z_gt.set_data(t_rpos_w, np.full_like(t_rpos_w, self.radar_pos_true[2]))

            self.ax_rpos.set_xlim([tmin, tmax])
            self.ax_rpos.relim()
            self.ax_rpos.autoscale_view(scalex=False, scaley=True)
        else:
            self.l_rpos_x_est.set_data([], [])
            self.l_rpos_y_est.set_data([], [])
            self.l_rpos_z_est.set_data([], [])
            self.l_rpos_x_gt.set_data([], [])
            self.l_rpos_y_gt.set_data([], [])
            self.l_rpos_z_gt.set_data([], [])

        # Attitude (deg)
        if len(self.t_ratt) > 0:
            t_ratt = np.asarray(self.t_ratt, dtype=float)
            mask = (t_ratt >= tmin) & (t_ratt <= tmax)
            t_ratt_w = t_ratt[mask]
            rr = np.degrees(np.asarray(self.ratt_roll, dtype=float)[mask])
            rp = np.degrees(np.asarray(self.ratt_pitch, dtype=float)[mask])
            ryaw = np.degrees(np.asarray(self.ratt_yaw, dtype=float)[mask])

            self.l_ratt_r_est.set_data(t_ratt_w, rr)
            self.l_ratt_p_est.set_data(t_ratt_w, rp)
            self.l_ratt_y_est.set_data(t_ratt_w, ryaw)

            gt_deg = np.degrees(self.radar_att_true_rad)
            self.l_ratt_r_gt.set_data(t_ratt_w, np.full_like(t_ratt_w, gt_deg[0]))
            self.l_ratt_p_gt.set_data(t_ratt_w, np.full_like(t_ratt_w, gt_deg[1]))
            self.l_ratt_y_gt.set_data(t_ratt_w, np.full_like(t_ratt_w, gt_deg[2]))

            self.ax_ratt.set_xlim([tmin, tmax])
            self.ax_ratt.relim()
            self.ax_ratt.autoscale_view(scalex=False, scaley=True)
        else:
            self.l_ratt_r_est.set_data([], [])
            self.l_ratt_p_est.set_data([], [])
            self.l_ratt_y_est.set_data([], [])
            self.l_ratt_r_gt.set_data([], [])
            self.l_ratt_p_gt.set_data([], [])
            self.l_ratt_y_gt.set_data([], [])

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        self.fig_ext.canvas.draw_idle()
        self.fig_ext.canvas.flush_events()

    # ----------------------- Helpers -----------------------

    def _append_limited(self, dq, value, maxlen=5000):
        dq.append(value)
        if len(dq) > maxlen:
            dq.popleft()

    def _append_tv(self, t_dq, v_dq, t, v, maxlen=5000):
        t_dq.append(float(t))
        v_dq.append(float(v))
        if len(t_dq) > maxlen:
            t_dq.popleft()
            v_dq.popleft()


def main():
    rclpy.init()
    node = EKFDebugPlotter()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
