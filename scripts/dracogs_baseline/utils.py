import numpy as np
from plyfile import PlyData, PlyElement
from pathlib import Path

# ---------------------------------------------------------------------------
# PLY I/O (self-contained, no dependency on DracoGS's read/save)
# ---------------------------------------------------------------------------

def read_gs_ply(path: str, sh_degree: int = 3) -> tuple[dict, int]:
    """Read a VideoGS 3DGS PLY file into the dict format expected by encode_dracogs.

    Returns (gs_data, uncompressed_size_bytes) where gs_data has keys:
    positions, f_dc, f_rest_1, f_rest_2, f_rest_3, opacity, scale, rotation.
    """
    plydata = PlyData.read(str(path))
    v = plydata.elements[0]

    positions = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float32)
    f_dc = np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], axis=1).astype(np.float32)
    opacity = np.asarray(v["opacity"]).astype(np.float32).reshape(-1, 1)

    scale_names = sorted(
        [p.name for p in v.properties if p.name.startswith("scale_")],
        key=lambda x: int(x.split("_")[-1]),
    )
    scale = np.stack([v[n] for n in scale_names], axis=1).astype(np.float32)

    rot_names = sorted(
        [p.name for p in v.properties if p.name.startswith("rot_")],
        key=lambda x: int(x.split("_")[-1]),
    )
    rotation = np.stack([v[n] for n in rot_names], axis=1).astype(np.float32)
    
    rest_names = sorted(
        [p.name for p in v.properties if p.name.startswith("f_rest_")],
        key=lambda x: int(x.split("_")[-1]),
    )
    n_rest = len(rest_names)

    N = len(positions)
    f_rest_1 = np.empty((N, 0), dtype=np.float32)
    f_rest_2 = np.empty((N, 0), dtype=np.float32)
    f_rest_3 = np.empty((N, 0), dtype=np.float32)

    if n_rest >= 9:
        f_rest_1 = np.stack([v[f"f_rest_{i}"] for i in range(9)], axis=1).astype(np.float32)
    if n_rest >= 24:
        f_rest_2 = np.stack([v[f"f_rest_{i}"] for i in range(9, 24)], axis=1).astype(np.float32)
    if n_rest >= 45:
        f_rest_3 = np.stack([v[f"f_rest_{i}"] for i in range(24, 45)], axis=1).astype(np.float32)

    gs_data = {
        "positions": positions,
        "f_dc": f_dc,
        "f_rest_1": f_rest_1,
        "f_rest_2": f_rest_2,
        "f_rest_3": f_rest_3,
        "opacity": opacity,
        "scale": scale,
        "rotation": rotation,
    }
    uncompressed_size_bytes = sum(v.nbytes for v in gs_data.values())
    return gs_data, uncompressed_size_bytes


def save_gs_ply(gs_data: dict, path: str) -> None:
    """Write decoded 3DGS attributes to a VideoGS-compatible PLY file.

    Property order: x,y,z, nx,ny,nz, f_dc_*, f_rest_*, opacity, scale_*, rot_*
    """
    pos = gs_data["positions"]
    N = pos.shape[0]

    names = ["x", "y", "z"]
    arrays = [pos[:, 0], pos[:, 1], pos[:, 2]]

    normals = np.zeros(N, dtype=np.float32)
    for n in ("nx", "ny", "nz"):
        names.append(n)
        arrays.append(normals)

    f_dc = gs_data["f_dc"]
    for i in range(f_dc.shape[1]):
        names.append(f"f_dc_{i}")
        arrays.append(f_dc[:, i])

    for band_key, offset in [("f_rest_1", 0), ("f_rest_2", 9), ("f_rest_3", 24)]:
        band = gs_data.get(band_key, np.empty((N, 0), dtype=np.float32))
        for i in range(band.shape[1]):
            names.append(f"f_rest_{offset + i}")
            arrays.append(band[:, i])

    opacity = gs_data["opacity"]
    names.append("opacity")
    arrays.append(opacity.ravel())

    scale = gs_data["scale"]
    for i in range(scale.shape[1]):
        names.append(f"scale_{i}")
        arrays.append(scale[:, i])

    rotation = gs_data["rotation"]
    for i in range(rotation.shape[1]):
        names.append(f"rot_{i}")
        arrays.append(rotation[:, i])

    dtype_full = [(name, "f4") for name in names]
    elements = np.empty(N, dtype=dtype_full)
    for i, name in enumerate(names):
        elements[name] = arrays[i].astype(np.float32)

    el = PlyElement.describe(elements, "vertex")
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    PlyData([el]).write(str(out_path))