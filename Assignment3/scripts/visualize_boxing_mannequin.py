import os
import time
import numpy as np
import viser

# 1. 데이터와 표시 설정
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data", "processed")
FPS = 30
SIDES = 16
ANGLE = np.arange(SIDES) * (2 * np.pi / SIDES)
BASIC_OVERLAP_OPACITY = 0.28

LIMBS = [
    (11, 13, 0.048), (13, 15, 0.038),
    (12, 14, 0.048), (14, 16, 0.038),
    (23, 25, 0.065), (25, 27, 0.048),
    (24, 26, 0.065), (26, 28, 0.048),
    (27, 29, 0.033), (28, 30, 0.033),
]
JOINTS = {
    11: 0.058, 12: 0.058, 13: 0.048, 14: 0.048,
    15: 0.040, 16: 0.040, 23: 0.070, 24: 0.070,
    25: 0.065, 26: 0.065, 27: 0.048, 28: 0.048,
    29: 0.037, 30: 0.037,
}


def unit(vector, fallback=(0, 1, 0)):
    vector = np.asarray(vector, dtype=float)
    length = np.linalg.norm(vector)
    return vector / length if length > 1e-8 else np.array(fallback, dtype=float)


# 2. 입체 관절 연결 메시
def ring_faces(count):
    faces = []
    for row in range(count - 1):
        for j in range(SIDES):
            a = row * SIDES + j
            b = row * SIDES + (j + 1) % SIDES
            faces.extend([(a, b, b + SIDES), (a, b + SIDES, a + SIDES)])

    for j in range(1, SIDES - 1):
        faces.append((0, j + 1, j))
        k = (count - 1) * SIDES
        faces.append((k, k + j, k + j + 1))

    return np.asarray(faces, dtype=np.uint32)


def tube_vertices(a, b, radius):
    axis = unit(b - a)
    helper = np.eye(3)[np.argmin(np.abs(axis))]
    side = unit(np.cross(helper, axis))
    front = np.cross(axis, side)

    circle = radius * (
        np.cos(ANGLE)[:, None] * side
        + np.sin(ANGLE)[:, None] * front
    )
    return np.vstack((a + circle, b + circle)).astype(np.float32)


def torso_vertices(pts):
    hip = (pts[23] + pts[24]) / 2
    shoulder = (pts[11] + pts[12]) / 2
    axis = unit(shoulder - hip)
    rings = []

    for t, width_scale in [
        (0, 0.58), (0.35, 0.48), (0.8, 0.58), (1, 0.53)
    ]:
        across = (
            (1 - t) * (pts[24] - pts[23])
            + t * (pts[12] - pts[11])
        )
        side = unit(
            across - axis * np.dot(across, axis),
            (1, 0, 0)
        )
        front = unit(np.cross(axis, side), (0, 0, 1))
        center = (1 - t) * hip + t * shoulder
        width = np.linalg.norm(across) * width_scale
        depth = np.clip(width * 0.65, 0.065, 0.12)

        rings.append(
            center
            + np.cos(ANGLE)[:, None] * side * width
            + np.sin(ANGLE)[:, None] * front * depth
        )

    return np.vstack(rings).astype(np.float32)


# 3. 머리·장갑·신발용 타원체
def sphere_mesh():
    vertices = [[0, 0, -1]]

    for a in np.linspace(-np.pi / 2, np.pi / 2, 11)[1:-1]:
        vertices.extend(
            np.column_stack((
                np.cos(a) * np.cos(ANGLE),
                np.cos(a) * np.sin(ANGLE),
                np.full(SIDES, np.sin(a)),
            ))
        )

    vertices.append([0, 0, 1])
    faces = []

    for j in range(SIDES):
        faces.append((0, 1 + (j + 1) % SIDES, 1 + j))

    for row in range(8):
        for j in range(SIDES):
            a = 1 + row * SIDES + j
            b = 1 + row * SIDES + (j + 1) % SIDES
            faces.extend([
                (a, b, b + SIDES),
                (a, b + SIDES, a + SIDES),
            ])

    last = len(vertices) - 1
    for j in range(SIDES):
        faces.append((
            last,
            1 + 8 * SIDES + j,
            1 + 8 * SIDES + (j + 1) % SIDES,
        ))

    return (
        np.asarray(vertices, dtype=float),
        np.asarray(faces, dtype=np.uint32),
    )


SPHERE, SPHERE_FACES = sphere_mesh()


class Mannequin:
    def __init__(self, server, name, color, x_offset, points):
        self.offset = np.array([x_offset, 0, 0], dtype=float)
        self.parts = {}
        scene = server.scene
        dark = tuple(int(c * 0.55) for c in color)

        for j, radius in JOINTS.items():
            self.parts[f"joint{j}"] = scene.add_icosphere(
                f"/{name}/joint{j}",
                radius=radius,
                color=color,
                subdivisions=2,
            )

        for key in [
            f"limb{i}" for i in range(len(LIMBS))
        ] + ["neck"]:
            self.parts[key] = scene.add_mesh_simple(
                f"/{name}/{key}",
                vertices=np.zeros((2 * SIDES, 3), dtype=np.float32),
                faces=ring_faces(2),
                color=color,
            )

        self.parts["torso"] = scene.add_mesh_simple(
            f"/{name}/torso",
            vertices=np.zeros((4 * SIDES, 3), dtype=np.float32),
            faces=ring_faces(4),
            color=color,
        )

        for key in [
            "head", "nose", "glove15", "glove16", "shoe29", "shoe30"
        ]:
            self.parts[key] = scene.add_mesh_simple(
                f"/{name}/{key}",
                vertices=SPHERE.astype(np.float32),
                faces=SPHERE_FACES,
                color=color if key in ("head", "nose") else dark,
            )

        self.label = scene.add_label(
            f"/{name}/label",
            "BASIC | 기본 보정" if name == "basic" else "SMOOTH | 스무딩",
            position=(x_offset, 1.8, 0),
        )

        self.update(points)

    def set_offset(self, x_offset):
        self.offset = np.array([x_offset, 0, 0], dtype=float)

    def set_label_position(self, position):
        self.label.position = position

    # 겹쳐 보기 시 BASIC만 반투명 처리
    def set_opacity(self, opacity):
        for part in self.parts.values():
            part.opacity = opacity

    def ellipsoid(self, key, center, radii, basis=None):
        vertices = SPHERE * np.asarray(radii)

        if basis is not None:
            vertices = vertices @ basis.T

        self.parts[key].vertices = (vertices + center).astype(np.float32)

    # 4. 프레임별 마네킹 자세 갱신
    def update(self, points):
        pts = np.asarray(points, dtype=float) + self.offset

        for j in JOINTS:
            self.parts[f"joint{j}"].position = pts[j]

        for i, (a, b, radius) in enumerate(LIMBS):
            self.parts[f"limb{i}"].vertices = tube_vertices(
                pts[a], pts[b], radius
            )

        self.parts["torso"].vertices = torso_vertices(pts)

        shoulder = (pts[11] + pts[12]) / 2
        hip = (pts[23] + pts[24]) / 2
        up = unit(shoulder - hip)

        ear_mid = (pts[7] + pts[8]) / 2
        head = ear_mid + up * 0.025
        face = unit(pts[0] - ear_mid, (0, 0, 1))
        side = unit(np.cross(up, face), (1, 0, 0))
        face = unit(np.cross(side, up), (0, 0, 1))
        basis = np.column_stack((side, up, face))

        self.parts["neck"].vertices = tube_vertices(
            shoulder, head, 0.043
        )

        self.ellipsoid("head", head, (0.09, 0.115, 0.095), basis)

        self.ellipsoid(
            "nose", head + face * 0.093,
            (0.022, 0.024, 0.03), basis,
        )

        for wrist in (15, 16):
            self.ellipsoid(
                f"glove{wrist}", pts[wrist],
                (0.072, 0.085, 0.072),
            )

        for heel, toe in ((29, 31), (30, 32)):
            direction = pts[toe] - pts[heel]
            forward = unit(direction, (0, 0, 1))
            side = unit(
                np.cross([0, 1, 0], forward),
                (1, 0, 0),
            )
            shoe_up = unit(np.cross(forward, side))
            basis = np.column_stack((side, shoe_up, forward))

            center = (pts[heel] + pts[toe]) / 2 + shoe_up * 0.018
            length = np.linalg.norm(direction)

            self.ellipsoid(
                f"shoe{heel}",
                center,
                (0.05, 0.035, length / 2 + 0.035),
                basis,
            )


# 5. 데이터 확인
def load_motion(filename):
    path = os.path.join(DATA_DIR, filename)

    if not os.path.isfile(path):
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {path}")

    motion = np.load(path, allow_pickle=False).astype(float)

    if (
        motion.ndim != 3
        or motion.shape[1:] != (33, 3)
        or len(motion) < 2
    ):
        raise ValueError(
            f"(프레임 수, 33, 3) 형태를 확인하세요: {motion.shape}"
        )

    if not np.isfinite(motion).all():
        raise ValueError(f"NaN 또는 무한대 좌표가 있습니다: {filename}")

    return motion


def main():
    basic = load_motion("footwork_04_target.npy")
    smooth = load_motion("footwork_04_smooth.npy")

    if basic.shape != smooth.shape:
        raise ValueError("두 데이터의 크기가 다릅니다.")

    server = viser.ViserServer()
    server.scene.set_up_direction("+y")

    segments = []
    for v in np.arange(-2.5, 2.51, 0.25):
        segments.extend([
            [[v, 0, -2.5], [v, 0, 2.5]],
            [[-2.5, 0, v], [2.5, 0, v]],
        ])

    server.scene.add_line_segments(
        "/ground",
        points=np.asarray(segments, dtype=np.float32),
        colors=(190, 196, 202),
        line_width=1,
    )

    models = [
        Mannequin(server, "basic", (226, 134, 60), -0.85, basic[0]),
        Mannequin(server, "smooth", (35, 166, 153), 0.85, smooth[0]),
    ]

    # 6. 카메라와 재생 조절
    def set_camera(client):
        client.camera.up_direction = (0, 1, 0)
        client.camera.position = (2.3, 1.9, 4.8)
        client.camera.look_at = (0, 0.85, 0)
        client.camera.fov = np.deg2rad(42)

    server.on_client_connect(set_camera)

    server.gui.add_markdown(
        "**BOXING MOTION LAB**\n\n"
        "**주황: BASIC | 기본 높이 보정**\n\n"
        "**청록: SMOOTH | 스무딩 + 높이 보정**\n\n"
        "겹쳐 보기를 켜면 BASIC은 28% 불투명도, "
        "SMOOTH는 100% 불투명도로 표시됩니다."
    )

    playing = server.gui.add_checkbox(
        "재생", initial_value=False
    )

    speed = server.gui.add_slider(
        "재생 배속",
        min=0.25,
        max=1.0,
        step=0.25,
        initial_value=0.5,
    )

    overlap = server.gui.add_checkbox(
        "겹쳐 보기", initial_value=False
    )

    frame = server.gui.add_slider(
        "프레임",
        min=0,
        max=len(basic) - 1,
        step=1,
        initial_value=0,
    )

    reset = server.gui.add_button("카메라 초기화")

    @reset.on_click
    def reset_camera(event):
        if event.client is not None:
            set_camera(event.client)

    print(f"마네킹 비교 화면: http://localhost:{server.get_port()}")
    print("겹쳐 보기: BASIC 반투명 / SMOOTH 불투명")
    print("종료: Ctrl+C")

    previous_frame = -1
    previous_overlap = None

    try:
        while True:
            start = time.perf_counter()
            now = int(frame.value)
            overlap_now = bool(overlap.value)

            if overlap_now:
                basic_offset, smooth_offset = 0.0, 0.0
                basic_label = (-0.25, 1.8, -0.18)
                smooth_label = (0.25, 1.8, 0.18)
            else:
                basic_offset, smooth_offset = -0.85, 0.85
                basic_label = (-0.85, 1.8, 0.0)
                smooth_label = (0.85, 1.8, 0.0)

            if now != previous_frame or overlap_now != previous_overlap:
                with server.atomic():
                    if overlap_now != previous_overlap:
                        models[0].set_offset(basic_offset)
                        models[1].set_offset(smooth_offset)

                        models[0].set_label_position(basic_label)
                        models[1].set_label_position(smooth_label)

                        models[0].set_opacity(
                            BASIC_OVERLAP_OPACITY if overlap_now else None
                        )
                        models[1].set_opacity(None)

                    models[0].update(basic[now])
                    models[1].update(smooth[now])

                previous_frame = now
                previous_overlap = overlap_now

            interval = (
                1 / (FPS * float(speed.value))
                if playing.value else 0.03
            )

            time.sleep(
                max(
                    0.001,
                    interval - (time.perf_counter() - start)
                )
            )

            if playing.value:
                frame.value = (int(frame.value) + 1) % len(basic)

    except KeyboardInterrupt:
        print("시각화 종료")

    finally:
        server.stop()


if __name__ == "__main__":
    main()