import numpy as np
from scipy.spatial.transform import Rotation as R
from revolt_state_estimator.utils import ssa, _skew, _exp_so3, _project_to_SO3

# =============================================================================
# es_ekf.py
# =============================================================================


class ErrorState_ExtendedKalmanFilter:
    """
    Extended Kalman Filter with Euler predict + ZOH discretization + numerical Jacobians.
    """

    def __init__(self, Q, P_initial, T_acc, T_ars, p_IR, q_IR):
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
        self.Q = Q
        self.T_acc = T_acc
        self.T_ars = T_ars

        # State estimate
        # x_hat_ins = [((p_hat_ins)^n)^T, ((v_hat_ins)^n)^T, ((b_hat_acc,ins)^b)^T, (Θ_hat_ins)^T, ((b_hat_ars,ins)^b)^T]^T (Eq. 14.213 in Fossen 2nd)
        # Also known as x_prior
        # INS propagation state
        self.p_hat_ins = np.zeros((3, 1))  # Position in NED frame
        self.v_hat_ins = np.zeros((3, 1))  # Velocity in NED frame
        self.b_acc_ins = np.zeros((3, 1))  # Body frame accelerometer bias
        self.q_hat_ins = np.array([[0.0], [0.0], [0.0], [1.0]])  # Attitude from body to NED frame quaternion
        self.b_ars_ins = np.zeros((3, 1))  # Body frame angular rate bias
        #Convert to numpy arrays
        self.p_IR = np.array(p_IR).reshape(3,1)     # Radar pos rel to inertial frame
        self.q_IR = np.array(q_IR).reshape(4,1) # Radar attitude rel to inertial frame quaternion

        self.x_hat_ins = np.concatenate(
            [
                self.p_hat_ins,
                self.v_hat_ins,
                self.b_acc_ins,
                self.q_hat_ins,
                self.b_ars_ins,
                self.p_IR,
                self.q_IR,
            ]
        )
        self.num_ins_states = self.x_hat_ins.shape[0] # Number of states in the error state model

        self.delta_p_hat = np.zeros((3,1))
        self.delta_v_hat = np.zeros((3,1))
        self.delta_b_acc_hat = np.zeros((3,1))
        self.delta_theta_hat = np.zeros((3,1))
        self.delta_b_ars_hat = np.zeros((3,1))
        self.delta_p_IR_hat = np.zeros((3,1))
        self.delta_theta_IR_hat = np.zeros((3,1))

        self.delta_x_hat = np.concatenate(
            [
                self.delta_p_hat,
                self.delta_v_hat,
                self.delta_b_acc_hat,
                self.delta_theta_hat,
                self.delta_b_ars_hat,
                self.delta_p_IR_hat,
                self.delta_theta_IR_hat,
            ]
        )
        self.num_error_states = self.delta_x_hat.shape[0]  # Total number of states

        self.delta_x_hat_prior = np.zeros((self.num_error_states, 1))
        self.P_hat = P_initial
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
        IKC = np.eye(self.num_error_states) - K @ H
        # innovation = e - H @ self.delta_x_hat_prior

        # print(f"K*e: {K @ e}")
        self.delta_x_hat = K @ e
        # print(f"delta_x_hat: {self.delta_x_hat.flatten()} with Kalman gain K: {K.flatten()}")
        self.P_hat = IKC @ self.P_hat_prior @ IKC.T + K @ R_meas @ K.T # Joseph form

        if self.delta_x_hat.shape != (self.num_error_states, 1):
            raise ValueError(
                f"Shape mismatch: delta_x_hat {self.delta_x_hat.shape} does not match expected {self.num_error_states, 1}"
            )

        return self.delta_x_hat, self.P_hat

    def update_state_estimate(self, delta_x_hat):
        """
        Update the state estimate based on the error state.
        """
        # if delta_x_hat.shape != self.x_hat_ins.shape:
        #     raise ValueError(
        #         f"Shape mismatch: delta_x_hat {delta_x_hat.shape} does not match x_hat_ins {self.x_hat_ins.shape}"
        #     )

        # nominal additive parts
        self.p_hat_ins += delta_x_hat[:3]
        self.v_hat_ins += delta_x_hat[3:6]
        self.b_acc_ins += delta_x_hat[6:9]
        self.b_ars_ins += delta_x_hat[12:15]
        self.p_IR += delta_x_hat[15:18]

        # multiplicative attitude
        q = self.x_hat_ins[9:13, 0]
        Rot = R.from_quat(q).as_matrix()

        dth = delta_x_hat[9:12, 0]
        dth_exp = _exp_so3(dth)
        Rot_next = Rot @ dth_exp
        Rot_next = _project_to_SO3(Rot_next)
        q_hat_ins = R.from_matrix(Rot_next).as_quat()
        self.q_hat_ins = q_hat_ins.reshape(4,1)


        # multiplicative attitude
        q_IR = self.x_hat_ins[19:23, 0]
        Rot_IR = R.from_quat(q_IR).as_matrix()

        dth_IR = delta_x_hat[18:21, 0]
        dth_IR_exp = _exp_so3(dth_IR)
        Rot_IR_next = Rot_IR @ dth_IR_exp
        Rot_IR_next = _project_to_SO3(Rot_IR_next)
        q_IR = R.from_matrix(Rot_IR_next).as_quat()
        self.q_IR = q_IR.reshape(4,1)

        # rebuild x_hat_ins
        self.x_hat_ins = np.concatenate(
            [
                self.p_hat_ins,
                self.v_hat_ins,
                self.b_acc_ins,
                self.q_hat_ins,
                self.b_ars_ins,
                self.p_IR,
                self.q_IR,
            ]
        )

        G = np.eye(self.num_error_states)
        G[9:12,9:12] = np.eye(3) - _skew(0.5 * dth)  # attitude correction
        G[18:21,18:21] = np.eye(3) - _skew(0.5 * dth_IR)  # attitude correction
        self.P_hat = G @ self.P_hat @ G.T  # Update covariance with attitude correction

        self.delta_x_hat = np.zeros(
            (self.num_error_states, 1)
        )  # Reset error state after update

        return self.x_hat_ins

    def ins_propagation(self, x_hat, dt, f_imu_b, w_imu_b, g_w):
        """
        Propagate the INS state estimate using the error state.
        """
        self.b_acc_ins = x_hat[6:9]  # Body frame accelerometer bias
        self.b_ars_ins = x_hat[13:16]  # Body frame angular rate

        # Radar extrinsics are not observable through ins propagation
        self.p_IR = x_hat[16:19]  # Radar position relative to inertial frame
        self.q_IR = x_hat[19:23]  # Radar attitude relative to inertial frame

        q_prior = x_hat[9:13, 0]
        Rot = R.from_quat(q_prior).as_matrix()

        # p_hat_ins[k+1] = p_hat_ins[k] + dt * v_hat_ins[k]
        self.p_hat_ins = x_hat[:3] + dt * x_hat[3:6] + 0.5 * dt**2 * (Rot @ (f_imu_b) + g_w)  # position update

        # v_hat_ins[k+1] = v_hat_ins[k] + dt * (R_b^n[k] @ (f_imu^b[k] - b_acc,ins^b[k]) + g^n)
        self.v_hat_ins = x_hat[3:6] + dt * (
            Rot @ (f_imu_b) + g_w
        )  # velocity update

        # Euler angles to rotation matrix
        # print(f"w_b: {-w_imu_b}")
        dR = _exp_so3(w_imu_b * dt)
        Rot_next = Rot @ dR 
        Rot_next = _project_to_SO3(Rot_next)
        q_hat_ins = R.from_matrix(Rot_next).as_quat()
        self.q_hat_ins = q_hat_ins.reshape(4,1)


        # print(f"yaw INS after correction: {yaw}")
        # print(f"gyro bias: {self.b_ars_ins.flatten()}")
        # print(f"w_imu_b: {w_imu_b.flatten()}")

        self.x_hat_ins = np.concatenate(
            [
                self.p_hat_ins,
                self.v_hat_ins,
                self.b_acc_ins,
                self.q_hat_ins,
                self.b_ars_ins,
                self.p_IR,
                self.q_IR,
            ]
        )

        return self.x_hat_ins

    def calculate_kalman_gain(self, H, R):
        """
        Compute the Kalman gain.
        """
        return self.P_hat_prior @ H.T @ np.linalg.inv(H @ self.P_hat_prior @ H.T + R)
    
    # Solas implementation
    def generate_A(self, R_nb, f_b_nom, w_b_nom):
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

    def generate_E(self, R_nb):
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
