import numpy as np

# ---------- minimal helpers ----------
def _skew(v):
    v = np.asarray(v).reshape(3,)
    return np.array([[    0, -v[2],  v[1]],
                     [ v[2],     0, -v[0]],
                     [-v[1],  v[0],     0]], dtype=float)

def _exp_so3(phi):
    phi = np.asarray(phi).reshape(3,)
    a = np.linalg.norm(phi)
    if a < 1e-12:
        return np.eye(3) + _skew(phi)
    A = np.sin(a)/a
    B = (1 - np.cos(a))/(a*a)
    K = _skew(phi)
    return np.eye(3) + A*K + B*(K@K)

def Rzyx(roll, pitch, yaw):
    cr, sr = np.cos(roll),  np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw),   np.sin(yaw)
    # ZYX: R = Rz(yaw) * Ry(pitch) * Rx(roll)  (body->world NED)
    return np.array([
        [cy*cp,           cy*sp*sr - sy*cr,   cy*sp*cr + sy*sr],
        [sy*cp,           sy*sp*sr + cy*cr,   sy*sp*cr - cy*sr],
        [-sp,             cp*sr,              cp*cr]
    ], dtype=float)

# ---------- your measurement model ----------
def h_pred(mu_r, R_WI, v_W, w_I, p_IR, R_RI):
    """
    Scalar: h = - mu_r^T * v_R, with
    v_R = R_RI * ( R_IW * v_W + (w_I x p_IR) ),  R_IW = R_WI^T
    """
    mu_r = np.asarray(mu_r).reshape(3,1)
    v_W  = np.asarray(v_W).reshape(3,1)
    w_I  = np.asarray(w_I).reshape(3,1)
    p_IR = np.asarray(p_IR).reshape(3,1)

    R_IW = R_WI.T
    v_I  = R_IW @ v_W
    spin = _skew(w_I) @ p_IR  # w_I x p_IR
    v_R  = R_RI @ (v_I + spin)
    return float(-(mu_r.T @ v_R))

def H_analytic(mu_r, R_WI, v_W, w_I, p_IR, R_RI):
    """
    1x15 Jacobian row in your state ordering:
    [δp(0:3), δv(3:6), δb_a(6:9), δθ(9:12), δb_g(12:15)]
    Only columns 3:6, 9:12, 12:15 are nonzero for this measurement.
    """
    mu_r = np.asarray(mu_r).reshape(3,1)
    v_W  = np.asarray(v_W).reshape(3,1)
    w_I  = np.asarray(w_I).reshape(3,1)
    p_IR = np.asarray(p_IR).reshape(3,1)

    R_IW = R_WI.T

    H = np.zeros((1,15), dtype=float)

    # d e / d v_W = - μ^T R_RI R_IW
    H[0, 3:6] = -(mu_r.T @ (R_RI @ R_IW))

    # d e / d b_g = - μ^T R_RI [p_IR]_x   (since w_I' = w_I - δb_g)
    H[0, 12:15] = -(mu_r.T @ (R_RI @ _skew(p_IR.reshape(3,)) ))

    # d e / d θ  with right-multiplicative δR = R_WI exp([δθ]x):
    # derivative shows up as - μ^T R_RI [ R_IW v_W + (w_I x p_IR) ]_x
    term = R_IW @ v_W                   # 3x1
    S = -(mu_r.T @ (R_RI @ _skew(term.reshape(3,))))
    H[0, 9:12] = S

    return H

# ---------- numerical checker ----------
def numeric_jacobian(mu_r, R_WI, v_W, w_I, p_IR, R_RI, eps=1e-6):
    """
    Finite-difference Jacobian of h w.r.t 15-dim error state:
      δx = [δp, δv, δb_a, δθ, δb_g]
    Mapping of δx -> perturbed arguments (consistent with your error-state):
      v_W'   = v_W + δv
      R_WI'  = R_WI @ exp([δθ]_x)
      w_I'   = w_I - δb_g
      (δp, δb_a) don't enter h directly -> expect near-zero columns.
    """
    base = h_pred(mu_r, R_WI, v_W, w_I, p_IR, R_RI)
    J = np.zeros((1,15), dtype=float)

    for k in range(15):
        d = np.zeros(15)
        d[k] = eps

        # Apply perturbation properly
        vWp  = v_W.copy()
        RWIp = R_WI.copy()
        wIp  = w_I.copy()

        if 3 <= k <= 5:        # δv
            vWp = v_W + d[3:6].reshape(3,1)
        elif 9 <= k <= 11:     # δθ (right-multiplicative on R_WI)
            dth = d[9:12]
            RWIp = R_WI @ _exp_so3(dth)
        elif 12 <= k <= 14:    # δb_g  (w_I' = w_I - δb_g)
            wIp = w_I - d[12:15].reshape(3,1)
        # δp, δb_a do not affect this measurement directly

        hp = h_pred(mu_r, RWIp, vWp, wIp, p_IR, R_RI)
        J[0,k] = (hp - base) / eps

    return J

# ---------- test with made-up values ----------
np.set_printoptions(precision=6, suppress=True, linewidth=200)

# Nominal attitude & states
roll, pitch, yaw = 0.10, -0.05, 0.80
R_WI = Rzyx(roll, pitch, yaw)

v_W  = np.array([[ 1.20], [-0.30], [ 0.50]])   # m/s in world
w_I  = np.array([[ 0.2], [-0.1], [ 0.5]])  # rad/s in IMU/body
p_IR = np.array([[ 0.07872], [-0.02159], [ 0.05919]])   # radar lever arm (I-frame)
R_RI = Rzyx(0.4, 0.8, 0.1)                               # IMU->Radar (identity for test)
mu_r = np.array([[0.4],[0.6],[0.7]])
mu_r = mu_r / np.linalg.norm(mu_r)

H_a = H_analytic(mu_r, R_WI, v_W, w_I, p_IR, R_RI)
H_n = numeric_jacobian(mu_r, R_WI, v_W, w_I, p_IR, R_RI, eps=1e-6)

print("Analytic H:\n", H_a)
print("Numeric  H:\n", H_n)
print("Diff      :\n", H_n - H_a)
print("Max|diff| :", np.max(np.abs(H_n - H_a)))
print("Blocks (should be ~0 except d, dθ, dbg):")
print("  d/dp   :", np.linalg.norm(H_n[0,0:3]))
print("  d/dv   :", np.linalg.norm(H_n[0,3:6] - H_a[0,3:6]))
print("  d/dba  :", np.linalg.norm(H_n[0,6:9]))
print("  d/dθ   :", np.linalg.norm(H_n[0,9:12] - H_a[0,9:12]))
print("  d/dbg  :", np.linalg.norm(H_n[0,12:15] - H_a[0,12:15]))
