#!/usr/bin/env python3
"""
ROS2 node that runs an Error-state Extended Kalman Filter using INS and an IMU with an AHRS system.
This is based on the Fossen 2nd edition book, chapter 14.4.2 - Error-state Kalman Filter Using Attitude Measurements.
Fuses:
 - GNSS position (/fix)
 - GNSS heading (/heading)
 - GNSS velocity (/vel)
 - IMU data (/imu/data)
    + yaw
    + yaw rate
    + linear velocity (integrated from linear acceleration)
Publishes:
 - ned → body TF
 - nav_msgs/Odometry with the state estimate
"""

import rclpy
from rclpy.node import Node
from sensor_msgs_py import point_cloud2 as pc2
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu, PointCloud2
from geometry_msgs.msg import (
    TransformStamped,
    Quaternion,
    Vector3Stamped,
    QuaternionStamped,
)
import tf_transformations
import tf2_ros
import numpy as np
import scipy.linalg

from revolt_state_estimator.es_ekf import ErrorState_ExtendedKalmanFilter
from revolt_state_estimator.revolt_sensor_transforms import Tzyx, Rzyx
from revolt_state_estimator.utils import _skew, ssa, gravity

def quat_xyzw_to_R(qx, qy, qz, qw):
    return tf_transformations.quaternion_matrix([qx, qy, qz, qw])[:3, :3]

class RevoltEKF(Node):
    # Initialize EKF system (START) ==================================================
    def __init__(self):
        # ROS2 Node Setup ----------
        super().__init__("revolt_ekf")

        # EKF Parameters
        self.declare_parameter("revolt_ekf.Q", [0.0] * 12)  # []
        self.declare_parameter(
            "revolt_ekf.radar_sigma_vr", 0.038
        )  # [v_x, v_y, v_z (m/s)]
        self.declare_parameter(
            "revolt_ekf.T_acc", 1000.0
        )  # Eq. 14.195 in Fossen 2nd edition
        self.declare_parameter(
            "revolt_ekf.T_ars", 500.0
        )  # 14.196 in Fossen 2nd edition

        # Extrinsic transformation from radar to IMU
        self.declare_parameter(
            "revolt_ekf.l_BR_B", [0.0, 0.0, 0.0]
        )
        self.declare_parameter(
            "revolt_ekf.q_R_B", [0.0, 0.0, 0.0, 1.0]
        )
        self.declare_parameter("radar_vr_sign", +1)
        # Get P_initial here
        _P_initial = self.get_initial_P()

        _Q = self.get_parameter("revolt_ekf.Q").value
        # _R_head = self.get_parameter("revolt_ekf.R_head").value
        _radar_sigma_vr = self.get_parameter("revolt_ekf.radar_sigma_vr").value
        _T_acc = self.get_parameter("revolt_ekf.T_acc").value
        _T_ars = self.get_parameter("revolt_ekf.T_ars").value
        _l_BR_B = self.get_parameter("revolt_ekf.l_BR_B").value
        _q_R_B = self.get_parameter("revolt_ekf.q_R_B").value
        _radar_vr_sign = self.get_parameter("radar_vr_sign").value

        # ROS 2 Parameters
        # Publish topics
        self.declare_parameter(
            "revolt_ekf.state_estimate_topic", "/state_estimate/revolt"
        )
        # Subscribe topics
        self.declare_parameter("revolt_ekf.imu_topic", "/imu/data")
        # self.declare_parameter("revolt_ekf.fix_topic", "/fix")
        # self.declare_parameter("revolt_ekf.heading_topic", "/heading")
        self.declare_parameter("revolt_ekf.radar_topic", "/vel")
        _state_estimate_topic = self.get_parameter(
            "revolt_ekf.state_estimate_topic"
        ).value
        _imu_topic = self.get_parameter("revolt_ekf.imu_topic").value
        _radar_topic = self.get_parameter("revolt_ekf.radar_topic").value

        l_BR_B = np.array(_l_BR_B, dtype=float).reshape(3,1)
        self.p_IR = l_BR_B
        qx,qy,qz,qw = _q_R_B
        self.q_R_B = np.array([qx, qy, qz, qw], dtype=float)
        # self.theta_IR = tf_transformations.euler_from_quaternion([qx,qy,qz,qw], axes='szyx')
        # self.R_IR = quat_xyzw_to_R(qx,qy,qz,qw)                   # Radar->IMU
        # self.R_RI = self.R_IR.T                     # IMU->Radar
        self.vr_sign = int(_radar_vr_sign)
        self.sigma_vr = float(_radar_sigma_vr)

        # EKF Setup ----------
        self.new_velocity_measurement = False

        # EKF Initialization variables
        self.initialized = False  # don’t run EKF until first GNSS fix arrives
        self.imu_last_stamp = None  # last IMU timestamp

        self.MU_R = None
        self.VR_meas = None

        # Noise models
        self.Q = np.diag(_Q)
        self.P_hat_initial = _P_initial

        # Latest measurements
        self.latest_velocity = None
        self.t_start = None

        # Error-state Extended Kalman Filter setup

        self.es_ekf = ErrorState_ExtendedKalmanFilter(
            Q=self.Q, 
            P_initial=self.P_hat_initial, 
            T_acc=_T_acc, 
            T_ars=_T_ars,
            p_IR=self.p_IR,
            q_IR=self.q_R_B
        )

        # ROS2 Interfaces Setup ----------
        # Sensor data subscribers
        self.radar_sub = self.create_subscription(
            PointCloud2, _radar_topic, self.update_radar, 1
        )
        # IMU
        self.imu_sub = self.create_subscription(Imu, _imu_topic, self.imu_callback, 1)
        # TF broadcaster
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        # TF listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        # State publisher
        self.state_pub = self.create_publisher(Odometry, _state_estimate_topic, 10)
        self.accel_bias_pub = self.create_publisher(Vector3Stamped, "/rio/accel_bias", 10)
        self.gyro_bias_pub = self.create_publisher(Vector3Stamped, "/rio/gyro_bias", 10)
        self.p_IR_pub = self.create_publisher(Vector3Stamped, "/rio/radar_position", 10)
        self.q_IR_pub = self.create_publisher(QuaternionStamped, "/rio/radar_attitude", 10)

        # Debugging ----------
        np.set_printoptions(
            linewidth=200,
            precision=6,  # adjust as you like
            suppress=True,  # so small floats don’t go to scientific notation
        )

        self.e_mean = 0.0
        self.e_std = 0.0
        self.i = 0

        self.get_logger().info(
            f"                                   \n"
            f"EKF Parameters:                    \n"
            f" Q:                                \n"
            f" {self.Q}                          \n"
            f" sigma_vr: {self.sigma_vr}          \n"
            f" T_acc: {_T_acc}               \n"
            f" T_ars: {_T_ars}               \n"
            f"                                   \n"
            f"Publisher topics:                     \n"
            f"State estimate topic: {_state_estimate_topic} \n"
            f"Subscribe topics:                  \n"
            f"IMU topic: {_imu_topic} \n"
            f"Radar topic: {_radar_topic} \n"
            f"Extrinsic transformation (l_BR_B): {_l_BR_B} \n"
            f"Extrinsic transformation (q_R_B): {self.q_R_B} \n"
            f"Radar vr sign: {_radar_vr_sign}      \n"
            f"                                   \n"
        )
        self.get_logger().info(
            "EKF waiting for IMU and radar measurements to start..."
        )

    # Initialize EKF system (STOP) ==================================================

    def _publish_state(self, x, P, stamp):

        # State is a 15‑element vector:
        # x_hat_ins = [
        #   p_x,      p_y,      p_z,        # position in navigation frame (m)
        #   v_x,      v_y,      v_z,        # velocity in navigation frame (m/s)
        #   b_acc_x,  b_acc_y,  b_acc_z,    # accelerometer biases (m/s²)
        #   ϕ,        θ,        ψ,          # attitude Euler angles: roll, pitch, yaw (rad)
        #   b_gyro_x, b_gyro_y, b_gyro_z    # gyroscope biases (rad/s)
        # ]
        # P is the covariance matrix of the full state estimate.

        x_pos, y_pos, z_pos = x[0], x[1], x[2]  # position in NED frame (m)
        v_x, v_y, v_z = x[3], x[4], x[5]  # velocity in NED frame (m/s)
        q = x[9:13]  # attitude quaternion (body → NED)
        b_acc = x[6:9]    # accelerometer biases (m/s²)
        b_gyro = x[13:16]  # gyroscope biases (rad/s)
        p_IR = x[16:19]    # radar position rel to inertial frame
        q_IR = x[19:23]    # radar attitude rel to inertial frame

        # Broadcast NED → imu TF
        t = TransformStamped()
        t.header.stamp = stamp
        t.header.frame_id = "ned"
        t.child_frame_id = "body"
        t.transform.translation.x = float(x_pos)
        t.transform.translation.y = float(y_pos)
        t.transform.translation.z = float(z_pos)
        t.transform.rotation = Quaternion(
            x=float(q[0]),
            y=float(q[1]),
            z=float(q[2]),
            w=float(q[3]),
        )
        self.tf_broadcaster.sendTransform(t)

        # 3) Publish ekf/state as Odometry 
        state = Odometry()
        state.header.stamp = stamp
        state.header.frame_id = "ned"
        state.child_frame_id = "body"

        state.pose.pose.position.x = float(x_pos)
        state.pose.pose.position.y = float(y_pos)
        state.pose.pose.position.z = float(z_pos)

        qx, qy, qz, qw = q
        state.pose.pose.orientation.x  = float(qx)
        state.pose.pose.orientation.y  = float(qy)
        state.pose.pose.orientation.z  = float(qz)
        state.pose.pose.orientation.w  = float(qw)

        # Covariance matrix
        pose_cov = np.zeros((6, 6))
        pose_cov[0:3, 0:3] = P[0:3, 0:3]
        pose_cov[3:6, 3:6] = P[9:12, 9:12]
        state.pose.covariance = pose_cov.flatten().tolist()

        state.twist.twist.linear.x = float(v_x)  # North or X velocity (m/s)
        state.twist.twist.linear.y = float(v_y)  # East or Y velocity (m/s)
        state.twist.twist.linear.z = float(v_z)  # Down or Z velocity (m/s)

        # Could feed forward the imu angular velocity, but will not do it to avoid confusion
        # state.twist.twist.angular.x = float(self.latest_roll_rate)  # Roll rate (rad/s)
        # state.twist.twist.angular.y = float(self.latest_pitch_rate)  # Pitch rate (rad/s)
        # state.twist.twist.angular.z = float(self.latest_yaw_rate)  # Yaw rate (rad/s)
        state.twist.twist.angular.x = 0.0  # Roll rate (rad/s)
        state.twist.twist.angular.y = 0.0  # Pitch rate (rad/s)
        state.twist.twist.angular.z = 0.0  # Yaw rate (rad/s)

        velocity_cov = np.zeros((6, 6))
        velocity_cov[0:3, 0:3] = P[3:6, 3:6]  # v_x variance
        state.twist.covariance = velocity_cov.flatten().tolist()

        self.state_pub.publish(state)

        # Publish accel and gyro biases
        accel_bias_msg = Vector3Stamped()
        accel_bias_msg.header.stamp = stamp
        accel_bias_msg.header.frame_id = "body"
        accel_bias_msg.vector.x = float(b_acc[0])
        accel_bias_msg.vector.y = float(b_acc[1])
        accel_bias_msg.vector.z = float(b_acc[2])
        self.accel_bias_pub.publish(accel_bias_msg)

        gyro_bias_msg = Vector3Stamped()
        gyro_bias_msg.header.stamp = stamp
        gyro_bias_msg.header.frame_id = "body"
        gyro_bias_msg.vector.x = float(b_gyro[0])
        gyro_bias_msg.vector.y = float(b_gyro[1])
        gyro_bias_msg.vector.z = float(b_gyro[2])
        self.gyro_bias_pub.publish(gyro_bias_msg)

        # Publish radar position and attitude
        p_IR_msg = Vector3Stamped()
        p_IR_msg.header.stamp = stamp
        p_IR_msg.header.frame_id = "body"
        p_IR_msg.vector.x = float(p_IR[0])
        p_IR_msg.vector.y = float(p_IR[1])
        p_IR_msg.vector.z = float(p_IR[2])
        self.p_IR_pub.publish(p_IR_msg)

        q_IR_msg = QuaternionStamped()
        q_IR_msg.header.stamp = stamp
        q_IR_msg.header.frame_id = "body"
        q_IR_msg.quaternion.x = float(q_IR[0])
        q_IR_msg.quaternion.y = float(q_IR[1])
        q_IR_msg.quaternion.z = float(q_IR[2])
        q_IR_msg.quaternion.w = float(q_IR[3])
        self.q_IR_pub.publish(q_IR_msg)

    # EKF main loop. The ES-EKF runs at the frequency of the IMU.
    def imu_callback(self, msg: Imu):
        # Ensure the value we get is NOT NaN
        if (
            np.isnan(msg.orientation.x)
            or np.isnan(msg.orientation.y)
            or np.isnan(msg.orientation.z)
            or np.isnan(msg.orientation.w)
            or np.isnan(msg.angular_velocity.z)
        ):
            return

        if not self.initialized:
            return

        t0 = self.get_clock().now()

        f_b = np.array([[msg.linear_acceleration.x],
                        [msg.linear_acceleration.y],
                        [msg.linear_acceleration.z]], dtype=float)
        w_b = np.array([[msg.angular_velocity.x],
                        [msg.angular_velocity.y],
                        [msg.angular_velocity.z]], dtype=float)

        # Initialize attitude from gravity if not yet done
        if not hasattr(self, "initialized_att") or not self.initialized_att:

            #Check if norm is roughly equal to gravity
            if np.linalg.norm(f_b) < 9.0 or np.linalg.norm(f_b) > 10.5:
                self.get_logger().warn("Drone must be level and not moving.")
                return
            
            self.get_logger().info(f"Initializing attitude from f_b: {f_b.flatten()}")

            fb = f_b / max(1e-6, np.linalg.norm(f_b))
            gb = fb
            roll  = np.arctan2(gb[1,0],  gb[2,0])
            pitch = np.arctan2(-gb[0,0], np.sqrt(gb[1,0]**2 + gb[2,0]**2))
            yaw   = 0.0  # arbitrary, no compass

            q = tf_transformations.quaternion_from_euler(roll, pitch, yaw, axes='sxyz')
            q = np.asarray(q, dtype=float).reshape(4, 1)

            self.es_ekf.q_hat_ins[:, 0] = q[:, 0]
            self.es_ekf.x_hat_ins[9:13, 0]  = q[:, 0]
            self.initialized_att = True
            
            self.get_logger().info(f"Initialized attitude from gravity: roll={roll:.3f}, pitch={pitch:.3f}")


        # Current time
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        if not hasattr(self, "imu_last_stamp") or self.imu_last_stamp is None:
            self.imu_last_stamp = t
            return

        dt = t - self.imu_last_stamp
        self.imu_last_stamp = t
        if dt <= 0.0 or dt > 0.1:
            return

        q = self.es_ekf.q_hat_ins.flatten()
        Rot = tf_transformations.quaternion_matrix(q)[:3, :3]  # body -> NED

        # System dynamics to implement the 15-state error-state model
        # ∂x_dot = A(t) * ∂x + E(t) * w (Eq. 14.188 in Fossen 2nd ed.)
        # ∂y = C * ∂x + ε (Eq. 14.189 in Fossen 2nd ed.)
        A = self.es_ekf.generate_A(
            Rot,
            f_b - self.es_ekf.b_acc_ins, 
            w_b - self.es_ekf.b_ars_ins
        )  # Eq. 14.192 in Fossen 2nd ed.
        E = self.es_ekf.generate_E(
            Rot,
        )  # Eq. 14.193 in Fossen 2nd ed.

        # Discretization according to Fossen 2nd ed. Eq. 14.201
        Ad = np.eye(self.es_ekf.num_error_states) + A * dt
        Qd = (E @ self.Q @ E.T) * dt

        # Predictor: P_hat_prior[k+1]
        self.es_ekf.predict(Ad, Qd)

        # INS propagation: x_hat_ins[k+1]
        g_w = (np.array([0.0, 0.0, -9.81])).reshape(
            3, 1
        )  # gravity vector in navigation frame
        self.es_ekf.ins_propagation(
            self.es_ekf.x_hat_ins,
            dt,
            f_b - self.es_ekf.b_acc_ins,
            w_b - self.es_ekf.b_ars_ins,
            g_w=g_w,
        )

        # Radar velocity
        if self.new_velocity_measurement:
            # e, H = self.calculate_radar_velocity_error_and_H()
            # radar_measurements = self.VR_meas
            # N = e.size

            for vr, mu_r in zip(self.VR_meas, self.MU_R):
                q = self.es_ekf.q_hat_ins.flatten()
                R_WI = tf_transformations.quaternion_matrix(q)[:3, :3]  # R->NED
                v_WI = self.es_ekf.v_hat_ins.reshape(3,1)  # WvWI

                p_IR = self.es_ekf.p_IR.reshape(3,1)
                q_IR = self.es_ekf.q_IR.flatten()
                R_IR = tf_transformations.quaternion_matrix(q_IR)[:3, :3]  # R->I

                # self.get_logger().info(f"Radar vel measurement: {vr}, bearing unit vector: {mu_r}")
                # Calculate H
                H = self.calculate_radar_H(mu_r, R_WI, v_WI, w_b, self.es_ekf.b_ars_ins, p_IR, R_IR)
                h = self.calculate_radar_h(mu_r, R_WI, v_WI, w_b, self.es_ekf.b_ars_ins, p_IR, R_IR)
                e = np.array([[self.vr_sign * vr]]) - h

                self.e_mean += e.item()
                self.e_std += e.item()**2

                # self.get_logger().info(f"H shape: {H.shape} \n e shape: {e.shape}")
                # self.get_logger().info(f"Radar vel residual e: {e}, H: {H}, \nfor mu_r: {mu_r} and vr: {vr}")
                R_meas = np.array([[self.sigma_vr**2]], dtype=np.float64)

                # Gate
                # chi2_threshold = 9.21  # 1 dof
                # S = H @ self.es_ekf.P_hat_prior @ H.T + R_meas
                # nu = e.reshape(-1,1)
                # sig = np.sqrt(S)
                # # self.get_logger().info(f"h={h:.3f} e={e}  sqrt(S)={sig}")
                # d2 = float(nu.T @ np.linalg.solve(S, nu))
                # # self.get_logger().info(f"Radar vel measurement gating d2: {d2:.2f}")
                # if d2 > chi2_threshold: 
                #     # self.get_logger().info(f"Radar vel measurement rejected by gating (d2={d2:.2f})")
                #     continue  # skip this measurement

                # Corrector: delta_x_hat[k] and P_hat[k]
                delta_x_hat_i, _ = self.es_ekf.correct(e, H, R_meas)

                # self.get_logger().info(f"Correcting by: {delta_x_hat_i.flatten()}")
                self.es_ekf.update_state_estimate(delta_x_hat_i)
                
                self.es_ekf.P_hat_prior = self.es_ekf.P_hat.copy()
                self.es_ekf.delta_x_hat_prior = self.es_ekf.delta_x_hat.copy()

            self.i += 1
            if self.i == 200:
                self.e_mean /= self.i
                self.e_std = np.sqrt(self.e_std / self.i - self.e_mean**2)
                self.get_logger().info(f"Radar vel residuals mean: {self.e_mean:.4f}, std: {self.e_std:.4f}")

            self.new_velocity_measurement = False

        else:
            # No aiding measurements
            self.es_ekf.P_hat = self.es_ekf.P_hat_prior
        # Publish the state estimate
        t1 = self.get_clock().now()
        diff_time = t1 - t0
        # self.get_logger().info(f"EKF IMU callback time: {diff_time.nanoseconds * 1e-6:.3f} ms")

        msg_stamp = rclpy.time.Time(
            seconds=msg.header.stamp.sec,
            nanoseconds=msg.header.stamp.nanosec,
        )

        stamp_time = msg_stamp + diff_time  # this is a rclpy.time.Time

        self._publish_state(self.es_ekf.x_hat_ins, self.es_ekf.P_hat, stamp_time.to_msg())

    def update_radar(self, msg: PointCloud2, min_range=1e-2):
        """Extract bearing unit vectors (in radar frame R) and per-return radial speeds."""
        U_list, vr_list = [], []
        for x, y, z, v in pc2.read_points(msg, field_names=("x", "y", "z", "velocity"), skip_nans=True):
            # print(f"Radar point r: {r}")
            r = np.linalg.norm([x, y, z])
            if r < min_range:
                continue
            # mu = np.array([y, x, -z], dtype=np.float64).reshape(3,1) / r # ENU to NED
            mu = np.array([x, y, z], dtype=np.float64).reshape(3,1) / r
            U_list.append(mu) # ENU to NED
            vr_list.append(v)
        if not U_list:
            return  # No valid points

        if (self.MU_R is None) or (self.VR_meas is None):
            self.initialized = True
            self.get_logger().info("EKF initialized with first radar measurement.")
        
        self.MU_R = np.asarray(U_list, dtype=np.float64)
        self.VR_meas = np.asarray(vr_list, dtype=np.float64)

        self.new_velocity_measurement = True

    def calculate_radar_H(self, mu_r, R_WI, v_WI, w_imu, b_ars_ins, p_IR, R_IR):
        # 2) State rotations
        R_IW = R_WI.T                      # IRW
        R_RI = R_IR.T                      # RRI
        w_imu = w_imu - b_ars_ins
        assert np.allclose(R_IR @ R_RI, np.eye(3), atol=1e-6)

        # 3) Predicted radar linear velocity per Eq. (8): v_R = R_RI( R_IW v_W + (w_I)× p_IR )
        # v_I  = R_IW @ v_WI                              # WR^T_I WvWI  == R_IW v_W
        # v_R_pred = R_RI @ ( v_I + np.cross(w_imu.flatten(), p_IR.flatten()).reshape(3,1) )

        # 5) Build Jacobian H per Eq. (10)
        # State order: [p(0:3), v(3:6), b_a(6:9), eul(9:12), b_g(12:15)]
        H = np.zeros((1, 21), dtype=np.float64)

        # d e / d v_W = - μ^T R_RI R_IW    (Eq. 10)
        H[0, 3:6] = -(mu_r.reshape(1,3) @ (R_RI @ R_IW))

        # d e / d b_g = - μ^T R_RI [p_IR]_x   (Eq. 10)
        H[0, 12:15] = -(mu_r.reshape(1,3) @ (R_RI @ _skew(p_IR.flatten())))
        # self.get_logger().info(f"Radar H b_g: {H[0,12:15]}")

        # d e / d ϕθψ = - μ^T R_RI [ R_IW v_W]_x   (Eq. 10)
        v_I = R_IW @ v_WI                     # (IRW WvWI)
        S = -(mu_r.reshape(1,3) @ (R_RI @ _skew(v_I.flatten())))   # shape (1,3)
        H[0, 9:12] = S

        # d e / d p_IR
        H[0, 15:18] = -(mu_r.reshape(1,3) @ (R_RI @ ( _skew((w_imu).flatten()) ) ))

        # d e / d theta_IR = 0
        H[0, 18:21] = -(mu_r.reshape(1,3) @ (R_RI @
                                        (
                                        _skew(R_RI @ (v_I.flatten() + np.cross(w_imu.flatten(), p_IR.flatten())))
                                        )
                                    )
                        )
        # print(f"Radar H: {H}")
        return H

    def calculate_radar_h(self, mu_r, R_WI, v_WI, w_imu, b_ars_ins, p_IR, R_IR):
        R_IW = R_WI.T
        R_RI = R_IR.T
        v_I  = R_IW @ v_WI

        w_imu = w_imu - b_ars_ins
        # print(f"v_I: {v_I.flatten()}")
        spin = np.cross((w_imu.flatten()), p_IR.flatten()).reshape(3,1)
        # print(f"spin: {spin.flatten()}")
        v_R  = R_RI @ (v_I + spin)
        return float(-(mu_r.reshape(1,3) @ v_R))
    
    def get_initial_P(self):
        self.declare_parameter("revolt_ekf.initial_sigma.attitude_deg", [6.0, 6.0, 1.0e-6])
        self.declare_parameter("revolt_ekf.initial_sigma.position",     [1.0e-6, 1.0e-6, 1.0e-6])
        self.declare_parameter("revolt_ekf.initial_sigma.velocity",     [1.0e-1, 1.0e-1, 1.0e-1])
        self.declare_parameter("revolt_ekf.initial_sigma.accel_bias",   [1.0e-2, 1.0e-2, 1.0e-2])
        self.declare_parameter("revolt_ekf.initial_sigma.gyro_bias",    [1.0e-2, 1.0e-2, 1.0e-2])
        self.declare_parameter("revolt_ekf.initial_sigma.p_IR",         [1.0e-2, 1.0e-2, 1.0e-2])
        self.declare_parameter("revolt_ekf.initial_sigma.theta_IR",     [6.0, 6.0, 6.0])

        sig_att_deg = np.array(self.get_parameter("revolt_ekf.initial_sigma.attitude_deg").value, dtype=float)
        sig_pos     = np.array(self.get_parameter("revolt_ekf.initial_sigma.position").value,     dtype=float)
        sig_vel     = np.array(self.get_parameter("revolt_ekf.initial_sigma.velocity").value,     dtype=float)
        sig_ba      = np.array(self.get_parameter("revolt_ekf.initial_sigma.accel_bias").value,   dtype=float)
        sig_bg      = np.array(self.get_parameter("revolt_ekf.initial_sigma.gyro_bias").value,    dtype=float)
        sig_p_IR    = np.array(self.get_parameter("revolt_ekf.initial_sigma.p_IR").value,         dtype=float)
        sig_theta_IR= np.array(self.get_parameter("revolt_ekf.initial_sigma.theta_IR").value,     dtype=float)

        # Convert attitude to radians
        sig_att = np.deg2rad(sig_att_deg)
        sig_theta_IR = np.deg2rad(sig_theta_IR)

        # Build 15x15 P in your state order:
        # δx = [δp(0:3), δv(3:6), δb_a(6:9), δθ(9:12), δb_g(12:15)]
        P_init = np.zeros((21, 21), dtype=np.float64)
        P_init[0:3,   0:3]   = np.diag(sig_pos**2)   # position
        P_init[3:6,   3:6]   = np.diag(sig_vel**2)   # velocity
        P_init[6:9,   6:9]   = np.diag(sig_ba**2)    # accel bias
        P_init[9:12,  9:12]  = np.diag(sig_att**2)   # attitude (rad)
        P_init[12:15,12:15]  = np.diag(sig_bg**2)    # gyro bias
        P_init[15:18,15:18]  = np.diag(sig_p_IR**2)  # radar position
        P_init[18:21,18:21]  = np.diag(sig_theta_IR**2)  # radar attitude

        return P_init


def main():
    rclpy.init()
    node = RevoltEKF()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
