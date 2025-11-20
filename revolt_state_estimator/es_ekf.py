import numpy as np
import tf_transformations
from revolt_state_estimator.revolt_sensor_transforms import Tzyx, Rzyx
from revolt_state_estimator.utils import ssa, _skew, _exp_so3, _project_to_SO3

# =============================================================================
# es_ekf.py
# =============================================================================


class ErrorState_ExtendedKalmanFilter:
    """
    Extended Kalman Filter with Euler predict + ZOH discretization + numerical Jacobians.
    """

    def __init__(self, Q, P_initial, T_acc, T_ars, p_IR, theta_IR):
        """
        Error-state model:
        δx_dot = A(t)δx + E(t)w
        δy = Cδx + ε

        δx = [(δp^n)^T, (δv^n)^T, (δb_acc^b)^T, (δΘ_nb)^T, (δb_ars^b)^T]^T
        w = [(w_acc)^T, (w_b,acc)^T, (w_ars)^T, (w_b,ars)^T]^T

        Parameters
        A : State transition matrix.
        C : The measurement matrix.
        E, ε : Process and measurement noise matrices.
        dt : timestep
        """
        self.num_states = 21  # Number of states in the error state model
        self.Q = Q
        self.T_acc = T_acc
        self.T_ars = T_ars

        self.z = np.zeros((self.num_states, 1))  # measurement placeholder

        # State estimate
        # x_hat_ins = [((p_hat_ins)^n)^T, ((v_hat_ins)^n)^T, ((b_hat_acc,ins)^b)^T, (Θ_hat_ins)^T, ((b_hat_ars,ins)^b)^T]^T (Eq. 14.213 in Fossen 2nd)
        self.x_hat_ins = np.zeros((self.num_states, 1))  # Also known as x_prior

        # INS propagation state
        self.p_hat_ins = np.zeros((3, 1))  # Position in NED frame
        self.v_hat_ins = np.zeros((3, 1))  # Velocity in NED frame
        self.b_acc_ins = np.zeros((3, 1))  # Body frame accelerometer bias
        self.theta_hat_ins = np.zeros((3, 1))  # Attitude from body to NED frame
        self.b_ars_ins = np.zeros((3, 1))  # Body frame angular rate bias
        #Convert to numpy arrays
        self.p_IR = np.array(p_IR).reshape(3,1)     # Radar pos rel to inertial frame
        self.theta_IR = np.array(theta_IR).reshape(3,1)            # Radar attitude rel to inertial frame

        self.x_hat_ins[15:18] = self.p_IR
        self.x_hat_ins[18:21] = self.theta_IR

        # Error states
        # Posteri error states
        self.delta_x_hat = np.zeros((self.num_states, 1))  # Also known as x_post
        self.P_hat = P_initial

        # Prior error states
        self.delta_x_hat_prior = np.zeros((self.num_states, 1))
        self.P_hat_prior = P_initial

    def predict(self, Ad, Qd):
        """
        Predictor update step, based on Fossen 2nd eq. 14.206, 14.207
        """

        # Predict the error state prior
        # δx_hat_prior[k+1] = Ad[k] * δx_hat[k]
        self.delta_x_hat_prior = Ad @ self.delta_x_hat

        # Predict the covariance
        # P_hat_prior[k+1] = Ad[k] * P_hat[k] * Ad[k].T + Ed[k] * Q * Ed[k].T
        self.P_hat_prior = Ad @ self.P_hat @ Ad.T + Qd

        return self.delta_x_hat_prior, self.P_hat_prior

    def correct(self, e, H, R_meas):
        """
        Update step: measurement z.
        """

        # KF gain: K[k]
        K = self.calculate_kalman_gain(H, R_meas)
        IKC = np.eye(self.num_states) - K @ H
        # innovation = e - H @ self.delta_x_hat_prior

        # print(f"K*e: {K @ e}")
        self.delta_x_hat = K @ e
        # print(f"delta_x_hat: {self.delta_x_hat.flatten()} with Kalman gain K: {K.flatten()}")
        self.P_hat = IKC @ self.P_hat_prior @ IKC.T + K @ R_meas @ K.T # Joseph form

        if self.delta_x_hat.shape != (self.num_states, 1):
            raise ValueError(
                f"Shape mismatch: delta_x_hat {self.delta_x_hat.shape} does not match expected {self.num_states, 1}"
            )

        return self.delta_x_hat, self.P_hat

    def update_state_estimate(self, delta_x_hat):
        """
        Update the state estimate based on the error state.
        """
        if delta_x_hat.shape != self.x_hat_ins.shape:
            raise ValueError(
                f"Shape mismatch: delta_x_hat {delta_x_hat.shape} does not match x_hat_ins {self.x_hat_ins.shape}"
            )

        # nominal additive parts
        self.x_hat_ins[:6]  += delta_x_hat[:6]
        self.x_hat_ins[6:9] += delta_x_hat[6:9]     # b_a
        self.x_hat_ins[12:15] += delta_x_hat[12:15] # b_g
        self.p_IR += delta_x_hat[15:18]        

        # multiplicative attitude on SO(3)
        roll, pitch, yaw = self.theta_hat_ins.flatten()
        R_nb = Rzyx(roll, pitch, yaw)
        dth = delta_x_hat[9:12, 0]
        # print("dth:", dth.flatten(), "yaw before update:", yaw)
        R_nb_next = R_nb @ _exp_so3(dth)            # compose rotation
        R_nb_next = _project_to_SO3(R_nb_next)      # re-orthogonalize
        # print("R_nb_next det:", np.linalg.det(R_nb_next))
        # r, p, y = tf_transformations.euler_from_matrix(R_nb_next, axes='szyx')
        p = -np.arcsin(np.clip(R_nb_next[2,0], -1.0, 1.0))
        r  = np.arctan2(R_nb_next[2,1], R_nb_next[2,2])
        y   = np.arctan2(R_nb_next[1,0], R_nb_next[0,0])
        # print("yaw after update:", y)
        self.theta_hat_ins[:] = np.array([ssa(r), ssa(p), ssa(y)]).reshape(3,1)

        # rebuild x_hat_ins angles
        self.x_hat_ins[9:12] = self.theta_hat_ins

        # multiplicative attitude on SO(3)
        roll_IR, pitch_IR, yaw_IR = self.theta_IR.flatten()
        R_IR = tf_transformations.euler_matrix(roll_IR, pitch_IR, yaw_IR)[:3, :3]  # R->I
        dth_IR = delta_x_hat[18:21, 0]
        # print("dth:", dth.flatten(), "yaw before update:", yaw)
        R_IR_next = R_IR @ _exp_so3(dth_IR)            # compose rotation
        R_IR_next = _project_to_SO3(R_IR_next)      # re-orthogonalize
        # print("R_nb_next det:", np.linalg.det(R_nb_next))
        # r, p, y = tf_transformations.euler_from_matrix(R_nb_next, axes='szyx')
        p = -np.arcsin(np.clip(R_IR_next[2,0], -1.0, 1.0))
        r  = np.arctan2(R_IR_next[2,1], R_IR_next[2,2])
        y   = np.arctan2(R_IR_next[1,0], R_IR_next[0,0])
        # print("yaw after update:", y)
        self.theta_IR[:] = np.array([ssa(r), ssa(p), ssa(y)]).reshape(3,1)

        # rebuild x_hat_ins angles
        self.x_hat_ins[18:21] = self.theta_IR

        G = np.eye(self.num_states)
        G[9:12,9:12] = np.eye(3) - _skew(0.5 * dth)  # attitude correction
        G[18:21,18:21] = np.eye(3) - _skew(0.5 * dth_IR)  # attitude correction
        self.P_hat = G @ self.P_hat @ G.T  # Update covariance with attitude correction

        self.delta_x_hat = np.zeros(
            (self.num_states, 1)
        )  # Reset error state after update

        return self.x_hat_ins

    def ins_propagation(self, x_hat, dt, R_bn, T_bn, f_imu_b, w_imu_b, g_w):
        """
        Propagate the INS state estimate using the error state.
        """

        self.b_acc_ins = x_hat[6:9]  # Body frame accelerometer bias
        self.b_ars_ins = x_hat[12:15]  # Body frame angular rate
        self.p_IR = x_hat[15:18]  # Radar position relative to inertial frame
        self.theta_IR = x_hat[18:21]  # Radar attitude relative to inertial frame
        # p and v is in n-frame
        # p_hat_ins[k+1] = p_hat_ins[k] + dt * v_hat_ins[k]
        # print(f"f_b: {f_imu_b}")
        # print(f"a_n: {R_bn @ (f_imu_b) + g_w}")

        self.p_hat_ins = x_hat[:3] + dt * x_hat[3:6] + 0.5 * dt**2 * (R_bn @ (f_imu_b) + g_w)  # position update

        # v_hat_ins[k+1] = v_hat_ins[k] + dt * (R_b^n[k] @ (f_imu^b[k] - b_acc,ins^b[k]) + g^n)
        self.v_hat_ins = x_hat[3:6] + dt * (
            R_bn @ (f_imu_b) + g_w
        )  # velocity update

        # Euler angles to rotation matrix
        # print(f"w_b: {-w_imu_b}")
        dR = _exp_so3(w_imu_b * dt)
        R_bn_next = R_bn @ dR 
        R_bn_next = _project_to_SO3(R_bn_next)
        
        # ZYX extraction matching Rzyx
        pitch = -np.arcsin(np.clip(R_bn_next[2,0], -1.0, 1.0))
        roll  = np.arctan2(R_bn_next[2,1], R_bn_next[2,2])
        yaw   = np.arctan2(R_bn_next[1,0], R_bn_next[0,0])
        theta_next = np.array([ssa(roll), ssa(pitch), ssa(yaw)])
        self.theta_hat_ins = theta_next.reshape(3,1)

        # print(f"yaw INS after correction: {yaw}")
        # print(f"gyro bias: {self.b_ars_ins.flatten()}")
        # print(f"w_imu_b: {w_imu_b.flatten()}")

        self.x_hat_ins = np.concatenate(
            [
                self.p_hat_ins,
                self.v_hat_ins,
                self.b_acc_ins,
                self.theta_hat_ins,
                self.b_ars_ins,
                self.p_IR,
                self.theta_IR,
            ]
        )

        return self.x_hat_ins

    def calculate_kalman_gain(self, H, R):
        """
        Compute the Kalman gain.
        """
        return self.P_hat_prior @ H.T @ np.linalg.inv(H @ self.P_hat_prior @ H.T + R)
    
    # Solas implementation

    def generate_A(self, R_nb, T_nb, f_b_nom, w_b_nom):
        O3 = np.zeros((3, 3)); I3 = np.eye(3)
        A = np.block([
            [O3,  I3,                 O3,                     O3,                 O3, O3, O3],
            [O3,  O3,              -R_nb, -R_nb @ _skew(f_b_nom),                 O3, O3, O3],
            [O3,  O3, -(1/self.T_acc)*I3,                     O3,                 O3, O3, O3],
            [O3,  O3,                 O3,        -_skew(w_b_nom),                -I3, O3, O3],
            [O3,  O3,                 O3,                     O3, -(1/self.T_ars)*I3, O3, O3],
            [O3,  O3,                 O3,                     O3,                 O3, O3, O3],
            [O3,  O3,                 O3,                     O3,                 O3, O3, O3],
        ])
        return A

    def generate_E(self, R_nb, T_bn):
        O3 = np.zeros((3, 3)); I3 = np.eye(3)
        E = np.block([
            [   O3,  O3,    O3, O3],
            [-R_nb,  O3,    O3, O3],
            [   O3,  I3,    O3, O3], 
            [   O3,  O3,   -I3, O3],
            [   O3,  O3,    O3, I3],
            [   O3,  O3,    O3, O3],
            [   O3,  O3,    O3, O3],
        ])
        return E

    #Fossens implementation

    # def generate_A(self, R_nb, T_nb, f_b_nom):
    #     """
    #     Linearized error dynamics (Fossen 2nd, §14.4.2):
    #     δv̇^n = -R_nb δb_a^b  - R_nb [f_b_nom]_x δΘ
    #     """
    #     O3 = np.zeros((3, 3)); I3 = np.eye(3)
    #     A = np.block([
    #         [O3, I3,                 O3,                  O3,               O3],
    #         [O3, O3,               -R_nb,        -R_nb @ _skew(f_b_nom),    O3],
    #         [O3, O3, -(1/self.T_acc)*I3,          O3,                       O3],
    #         [O3, O3,                 O3,          O3,                     -T_nb],
    #         [O3, O3,                 O3,          O3,     -(1/self.T_ars)*I3],
    #     ])
    #     return A

    # def generate_E(self, R_bn, T_bn):
    #     """
    #     Generate the process noise matrix E.
    #     Defined in Fossen 2nd, eq. 14.193.
    #     """
    #     O3 = np.zeros((3, 3))  # 3x3 zero matrix
    #     I3 = np.eye(3)  # 3x3 identity matrix

    #     E = np.block(
    #         [
    #             [O3, O3, O3, O3],
    #             [-R_bn, O3, O3, O3],
    #             [O3, I3, O3, O3],
    #             [O3, O3, -T_bn, O3],
    #             [O3, O3, O3, I3],
    #         ]
    #     )

    #     return E

