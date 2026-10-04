"""One RGB camera. No depth, labels, masks, or object poses reach the policy."""

import numpy as np
from scipy.spatial.transform import Rotation


def look_at_quaternion(position, target):
    """USD camera looks down local -Z, with local +Y up. Returns wxyz."""
    forward = np.asarray(target, dtype=float) - position
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0, 0, 1])
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    q = Rotation.from_matrix(np.column_stack((right, up, -forward))).as_quat()
    return q[[3, 0, 1, 2]]


class RGBCamera:
    def __init__(self, cfg):
        from isaacsim.sensors.experimental.rtx import CameraSensor, RtxCamera

        self.cfg = cfg
        self.camera = RtxCamera(
            "/World/Camera",
            tick_rate=0.0,
            positions=np.asarray(cfg["camera_position"]),
            orientations=look_at_quaternion(cfg["camera_position"], cfg["camera_target"]),
        )
        self.camera.camera.set_focal_lengths(cfg["camera_focal_length"])
        self.camera.camera.set_apertures(
            horizontal_apertures=cfg["camera_aperture"], vertical_apertures=cfg["camera_aperture"]
        )
        self.camera.camera.set_clipping_ranges(0.01, 100.0)
        self.sensor = CameraSensor(
            self.camera, resolution=(cfg["image_size"], cfg["image_size"]), annotators=["rgb"]
        )

    def reset(self, rng, condition):
        import omni.usd
        from pxr import Gf, UsdGeom

        p = np.asarray(self.cfg["camera_position"], dtype=float)
        if condition == "camera":
            p = p + rng.uniform(-0.035, 0.035, 3)
        q = look_at_quaternion(p, self.cfg["camera_target"])
        # Author USD camera transforms; avoids relying on private sensor state.
        prim = omni.usd.get_context().get_stage().GetPrimAtPath("/World/Camera")
        transform = UsdGeom.Xformable(prim)
        transform.ClearXformOpOrder()
        transform.AddTranslateOp().Set(Gf.Vec3d(*p))
        transform.AddOrientOp().Set(Gf.Quatf(float(q[0]), Gf.Vec3f(*q[1:])))

    def read(self):
        data, _ = self.sensor.get_data("rgb")
        if data is None:
            raise RuntimeError("RGB camera has not produced a frame yet.")
        data = data.numpy() if hasattr(data, "numpy") else np.asarray(data)
        rgb = np.asarray(data[..., :3], dtype=np.uint8).copy()
        expected = (self.cfg["image_size"], self.cfg["image_size"], 3)
        if rgb.shape != expected:
            raise RuntimeError(f"Camera returned {rgb.shape}; expected {expected}.")
        return rgb
