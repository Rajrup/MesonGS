import torch
import sys
import os

_MESONGS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _MESONGS_ROOT not in sys.path:
    sys.path.insert(0, _MESONGS_ROOT)

from scene.gaussian_model import build_rotation_from_euler

# ---------------------------------------------------------------------------
# Euler-to-Quaternion Conversion
# ---------------------------------------------------------------------------

def rotation_matrix_to_quaternion(R):
    """Convert rotation matrices to quaternions (w, x, y, z).

    Uses the Shepperd method for numerical stability.
    R: (N, 3, 3) tensor on CUDA.
    Returns: (N, 4) tensor on CUDA with quaternions (w, x, y, z).
    """
    N = R.shape[0]
    q = torch.zeros(N, 4, device=R.device, dtype=R.dtype)

    trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]

    s = torch.zeros(N, device=R.device, dtype=R.dtype)

    # Case 1: trace > 0
    mask = trace > 0
    if mask.any():
        s[mask] = torch.sqrt(trace[mask] + 1.0) * 2  # s = 4*w
        q[mask, 0] = 0.25 * s[mask]
        q[mask, 1] = (R[mask, 2, 1] - R[mask, 1, 2]) / s[mask]
        q[mask, 2] = (R[mask, 0, 2] - R[mask, 2, 0]) / s[mask]
        q[mask, 3] = (R[mask, 1, 0] - R[mask, 0, 1]) / s[mask]

    # Case 2: R[0,0] is largest diagonal
    mask = (~(trace > 0)) & (R[:, 0, 0] > R[:, 1, 1]) & (R[:, 0, 0] > R[:, 2, 2])
    if mask.any():
        s[mask] = torch.sqrt(1.0 + R[mask, 0, 0] - R[mask, 1, 1] - R[mask, 2, 2]) * 2
        q[mask, 0] = (R[mask, 2, 1] - R[mask, 1, 2]) / s[mask]
        q[mask, 1] = 0.25 * s[mask]
        q[mask, 2] = (R[mask, 0, 1] + R[mask, 1, 0]) / s[mask]
        q[mask, 3] = (R[mask, 0, 2] + R[mask, 2, 0]) / s[mask]

    # Case 3: R[1,1] is largest diagonal
    mask = (~(trace > 0)) & (~((R[:, 0, 0] > R[:, 1, 1]) & (R[:, 0, 0] > R[:, 2, 2]))) & (R[:, 1, 1] > R[:, 2, 2])
    if mask.any():
        s[mask] = torch.sqrt(1.0 + R[mask, 1, 1] - R[mask, 0, 0] - R[mask, 2, 2]) * 2
        q[mask, 0] = (R[mask, 0, 2] - R[mask, 2, 0]) / s[mask]
        q[mask, 1] = (R[mask, 0, 1] + R[mask, 1, 0]) / s[mask]
        q[mask, 2] = 0.25 * s[mask]
        q[mask, 3] = (R[mask, 1, 2] + R[mask, 2, 1]) / s[mask]

    # Case 4: R[2,2] is largest diagonal
    mask = (~(trace > 0)) & (~((R[:, 0, 0] > R[:, 1, 1]) & (R[:, 0, 0] > R[:, 2, 2]))) & (~(R[:, 1, 1] > R[:, 2, 2]))
    if mask.any():
        s[mask] = torch.sqrt(1.0 + R[mask, 2, 2] - R[mask, 0, 0] - R[mask, 1, 1]) * 2
        q[mask, 0] = (R[mask, 1, 0] - R[mask, 0, 1]) / s[mask]
        q[mask, 1] = (R[mask, 0, 2] + R[mask, 2, 0]) / s[mask]
        q[mask, 2] = (R[mask, 1, 2] + R[mask, 2, 1]) / s[mask]
        q[mask, 3] = 0.25 * s[mask]

    return q


def euler_to_quaternion(euler):
    """Convert decoded Euler angles to quaternions using MesonGS's rotation matrix builder.

    euler: (N, 3) tensor [roll, pitch, yaw] on CUDA.
    Returns: (N, 4) tensor (w, x, y, z) on CUDA.
    """
    R = build_rotation_from_euler(euler[:, 2], euler[:, 1], euler[:, 0])
    return rotation_matrix_to_quaternion(R)