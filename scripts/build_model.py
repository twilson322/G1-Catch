"""Derive models/g1_catch.xml from the g1-arm-rtc upper-body model.

The generated file is committed. This script records how it was made: the desk scene is replaced
by a floor, a ball and a catcher cup on the right hand, and two head cameras are added.

    python scripts/build_model.py path/to/g1_upper_body.xml
"""
import math
import re
import shutil
import sys
from pathlib import Path

# Geometry of the things this project adds. Units are SI.
BALL_RADIUS_M = 0.033          # tennis-ball size
BALL_MASS_KG = 0.057
CUP_OFFSET_M = (0.13, -0.01, 0.0)   # cup base centre in the wrist-yaw frame, same as the old tool
CUP_BASE_RADIUS_M = 0.050
CUP_RIM_RADIUS_M = 0.075
CUP_DEPTH_M = 0.075
CUP_WALL_SEGMENTS = 14
CUP_WALL_HALF_THICKNESS_M = 0.004
CAMERA_BASELINE_M = 0.10
CAMERA_POS_IN_TORSO_M = (0.07, 0.0, 0.43)   # front of the head
CAMERA_PITCH_UP_RAD = math.radians(0.0)   # level, so the floor is in view beyond 1.8 m
CAMERA_FOVY_DEG = 90.0   # sees the floor beyond 1.3 m and a lob up to 45 degrees above level
# Contact: the floor returns most of the ball's speed, the cup about a fifth. solref with two
# negative numbers is (-stiffness, -damping) in MuJoCo.
FLOOR_SOLREF = "-400000 -115"
CUP_SOLREF = "-200000 -408"   # restitution about 0.2, a foam-lined cup


def cup_xml() -> str:
    ox, oy, oz = CUP_OFFSET_M
    lines = [f'<body name="catcher" pos="{ox} {oy} {oz}">',
             '  <inertial pos="0.03 0 0" mass="0.09" diaginertia="2e-4 1.5e-4 1.5e-4"/>',
             f'  <geom name="cup_base" type="cylinder" size="{CUP_BASE_RADIUS_M} 0.004" '
             f'euler="0 {math.pi / 2} 0" solref="{CUP_SOLREF}" friction="0.9" rgba="0.15 0.3 0.7 1" mass="0"/>']
    slope = math.atan2(CUP_RIM_RADIUS_M - CUP_BASE_RADIUS_M, CUP_DEPTH_M)
    wall_len = math.hypot(CUP_DEPTH_M, CUP_RIM_RADIUS_M - CUP_BASE_RADIUS_M)
    mid_r = 0.5 * (CUP_BASE_RADIUS_M + CUP_RIM_RADIUS_M)
    half_width = math.pi * CUP_RIM_RADIUS_M / CUP_WALL_SEGMENTS * 1.08
    for i in range(CUP_WALL_SEGMENTS):
        a = 2 * math.pi * i / CUP_WALL_SEGMENTS
        y, z = mid_r * math.cos(a), mid_r * math.sin(a)
        # Box long axis: along the wall (cup axis x tilted outward by `slope`), thin axis radial.
        ex = (math.cos(slope), math.sin(slope) * math.cos(a), math.sin(slope) * math.sin(a))
        ey = (0.0, -math.sin(a), math.cos(a))  # tangential
        lines.append(
            f'  <geom name="cup_wall_{i}" type="box" size="{wall_len / 2:.5f} {half_width:.5f} '
            f'{CUP_WALL_HALF_THICKNESS_M}" pos="{CUP_DEPTH_M / 2:.5f} {y:.5f} {z:.5f}" '
            f'xyaxes="{ex[0]:.6f} {ex[1]:.6f} {ex[2]:.6f} {ey[0]:.6f} {ey[1]:.6f} {ey[2]:.6f}" '
            f'solref="{CUP_SOLREF}" friction="0.9" rgba="0.15 0.3 0.7 1" mass="0"/>')
    lines += [f'  <site name="cup_center" pos="{CUP_DEPTH_M * 0.45:.4f} 0 0" size="0.006"/>',
              '  <site name="ft_site" pos="0 0 0" size="0.004"/>',
              '</body>']
    return "\n".join(lines)


def camera_xml() -> str:
    s, c = math.sin(CAMERA_PITCH_UP_RAD), math.cos(CAMERA_PITCH_UP_RAD)
    px, py, pz = CAMERA_POS_IN_TORSO_M
    out = []
    for name, y in (("cam_left", CAMERA_BASELINE_M / 2), ("cam_right", -CAMERA_BASELINE_M / 2)):
        # MuJoCo cameras look along -z with +x right and +y up. x = -Y_torso, y tilted up.
        out.append(f'<camera name="{name}" pos="{px} {py + y} {pz}" xyaxes="0 -1 0 {-s:.6f} 0 {c:.6f}" '
                   f'fovy="{CAMERA_FOVY_DEG}"/>')
    return "\n".join(out)


def main(src: str) -> None:
    src_path = Path(src)
    out_dir = Path(__file__).resolve().parents[1] / "models"
    xml = src_path.read_text()

    xml = xml.replace('model="g1_upper_body"', 'model="g1_catch"')
    # Replace the wiping tool with the catcher.
    xml = re.sub(r'<body name="tool".*?</body>', cup_xml(), xml, flags=re.S)
    # Head cameras ride on the torso link, next to the IMU site.
    xml = xml.replace('<site name="imu_in_torso"', camera_xml() + '\n<site name="imu_in_torso"', 1)
    # Remove the desk scene, add floor, backdrop and ball.
    xml = re.sub(r'<body name="table".*?</body>\s*<body name="shelf".*?</body>\s*<body name="intruder".*?</body>',
                 f'''<geom name="floor" type="plane" size="8 8 0.1" solref="{FLOOR_SOLREF}" friction="0.6" material="floor_mat"/>
    <geom name="backdrop" type="box" size="0.05 5 2" pos="6.5 0 2" contype="0" conaffinity="0" material="wall_mat"/>
    <geom name="distractor" type="sphere" size="0.09" pos="6.3 1.4 1.5" contype="0" conaffinity="0" rgba="0.85 0.9 0.15 0"/>
    <body name="ball" pos="3 0 1">
      <freejoint name="ball_free"/>
      <geom name="ball_geom" type="sphere" size="{BALL_RADIUS_M}" mass="{BALL_MASS_KG}" friction="0.9" condim="4" material="ball_mat"/>
    </body>''', xml, flags=re.S)
    xml = xml.replace('<material name="black"',
                      '<texture name="floor_tex" type="2d" builtin="checker" rgb1="0.35 0.35 0.38" rgb2="0.5 0.5 0.52" width="256" height="256"/>\n'
                      '    <texture name="wall_tex" type="2d" builtin="flat" rgb1="0.6 0.62 0.65" width="64" height="64"/>\n'
                      '    <material name="floor_mat" texture="floor_tex" texrepeat="6 6"/>\n'
                      '    <material name="wall_mat" texture="wall_tex"/>\n'
                      '    <material name="ball_mat" rgba="0.85 0.9 0.15 1"/>\n'
                      '    <material name="black"', 1)
    xml = xml.replace('<option timestep="0.0005" integrator="implicitfast"/>',
                      '<option timestep="0.0005" integrator="implicitfast"/>\n'
                      '  <visual><global offwidth="640" offheight="480"/></visual>')
    # Software rendering cost is set by triangle count (393k faces in the full robot). Only the
    # right arm can enter the head cameras' view, so every other visual mesh moves to geom group 1,
    # which the head-camera renderer leaves off. External figure renders keep both groups on.
    def regroup(match: re.Match) -> str:
        geom = match.group(0)
        if re.search(r'mesh="right_(shoulder|elbow|wrist|rubber)', geom):
            return geom
        return geom.replace('class="visual"', 'class="visual" group="1"')
    xml = re.sub(r'<geom class="visual"[^>]*/>', regroup, xml)
    xml = xml.replace('<sensor>', '<sensor>\n    <framepos name="ball_pos" objtype="body" objname="ball"/>')
    (out_dir / "g1_catch.xml").write_text(xml)

    used = set(re.findall(r'file="([^"]+)"', xml))
    (out_dir / "assets").mkdir(exist_ok=True)
    for f in used:
        shutil.copy(src_path.parent / "assets" / f, out_dir / "assets" / f)
    print(f"wrote {out_dir / 'g1_catch.xml'} with {len(used)} meshes")


if __name__ == "__main__":
    main(sys.argv[1])
