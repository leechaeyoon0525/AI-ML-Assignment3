import os
import sys
import time
import numpy as np
import pandas as pd

# 1. 경로 설정
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_KEYPOINTS_DIR = os.path.join(ROOT, "data", "raw_keypoints")
OUT_DIR = os.path.join(ROOT, "data", "processed")
RESULT_DIR = os.path.join(ROOT, "results")

if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

NAME = "footwork_04"
FPS = 30  # 비교 화면 재생 속도
COLUMNS = ["frame", "landmark", "x", "y", "z", "visibility"]
LOWER_IDS = np.arange(25, 33)


# 2. 원본 키포인트 불러오기
def load_raw_keypoints(npz_path, return_visibility=False):
    if not os.path.isfile(npz_path):
        raise FileNotFoundError(f"파일이 없습니다: {npz_path}")

    with np.load(npz_path, allow_pickle=False) as archive:
        data = np.asarray(archive["data"], dtype=np.float64)

    if data.ndim != 2 or data.shape[1] != 6 or len(data) == 0:
        raise ValueError("키포인트 데이터는 비어 있지 않은 (N, 6) 형식이어야 합니다.")
    if not np.isfinite(data).all():
        raise ValueError("키포인트에 NaN 또는 무한대가 있습니다.")

    df = pd.DataFrame(data, columns=COLUMNS)
    frames = np.sort(df["frame"].unique())
    if not np.array_equal(frames, np.arange(len(frames))):
        raise ValueError("프레임 번호가 0부터 연속적이지 않습니다. 추출 누락을 확인하세요.")

    pts_seq = []
    visibility_seq = []
    for frame, group in df.groupby("frame", sort=True):
        group = group.sort_values("landmark")
        if not np.array_equal(group["landmark"].to_numpy(), np.arange(33)):
            raise ValueError(f"{int(frame)}번 프레임의 관절 번호를 확인하세요.")
        pts_seq.append(group[["x", "y", "z"]].to_numpy())
        visibility_seq.append(group["visibility"].to_numpy())

    seq = np.stack(pts_seq)  # (T, 33, 3)
    if return_visibility:
        return seq, np.stack(visibility_seq)
    return seq


# 3. 높이 보정
def create_height_corrected_target(raw_seq):
    seq = raw_seq.copy()
    LEFT_FOOT = 31
    RIGHT_FOOT = 32

    # 프레임별 낮은 발끝을 Y=0에 정렬
    foot_y = np.minimum(seq[:, LEFT_FOOT, 1], seq[:, RIGHT_FOOT, 1])
    frame_ground = foot_y[:, None]
    seq[:, :, 1] -= frame_ground

    # 남은 음수 Y값 제한
    seq[:, :, 1] = np.maximum(seq[:, :, 1], 0)
    return seq


# 4. 보너스: 5프레임·2차 Savitzky-Golay 스무딩
def apply_savgol_smoothing(raw_seq):
    seq = np.asarray(raw_seq, dtype=np.float64)
    if len(seq) < 5:
        raise ValueError("스무딩에는 최소 5프레임이 필요합니다.")

    smoothed = seq.copy()

    # 33개 관절의 XYZ를 시간축으로 평활화
    smoothed[2:-2] = (
        -3 * seq[:-4] + 12 * seq[1:-3] + 17 * seq[2:-2]
        + 12 * seq[3:-1] - 3 * seq[4:]
    ) / 35.0

    # 양끝 2프레임은 5점 이차식으로 처리
    x = np.arange(5, dtype=np.float64)
    design = np.column_stack((np.ones(5), x, x ** 2))
    left = np.linalg.lstsq(design, seq[:5].reshape(5, -1), rcond=None)[0]
    right = np.linalg.lstsq(design, seq[-5:].reshape(5, -1), rcond=None)[0]
    smoothed[:2] = (design[:2] @ left).reshape((2,) + seq.shape[1:])
    smoothed[-2:] = (design[-2:] @ right).reshape((2,) + seq.shape[1:])
    return smoothed


# 5. 원본 visibility를 유지한 CSV 저장
def export_unity_csv(motion, output_path, visibility):
    rows = []
    for frame in range(motion.shape[0]):
        for landmark in range(motion.shape[1]):
            x, y, z = motion[frame, landmark]
            rows.append([
                frame, landmark, float(x), float(y), float(z),
                float(visibility[frame, landmark])
            ])

    output_path = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pd.DataFrame(rows, columns=COLUMNS).to_csv(
        output_path, index=False, encoding="utf-8"
    )
    print(f"CSV 저장: {output_path}")


# 6. 시간적 좌표 변화량 계산
def second_difference_rms(seq):
    d2 = np.diff(seq, n=2, axis=0)
    return float(np.sqrt(np.mean(np.sum(d2 ** 2, axis=2))))


# 7. 기본 보정과 스무딩 결과 평가
def save_comparison(raw_seq, target_seq, smooth_seq):
    displacement = np.linalg.norm(smooth_seq - target_seq, axis=2) * 1000
    rows = []
    for label, ids in [("하체(25~32)", LOWER_IDS), ("전신(0~32)", np.arange(33))]:
        before = second_difference_rms(target_seq[:, ids, :])
        after = second_difference_rms(smooth_seq[:, ids, :])
        reduction = (1 - after / before) * 100 if before > 1e-12 else np.nan
        shift = displacement[:, ids]
        rows.append({
            "부위": label,
            "기본 RMS(mm/frame^2)": before * 1000,
            "스무딩 RMS(mm/frame^2)": after * 1000,
            "RMS 감소(%)": reduction,
            "평균 이동(mm)": float(shift.mean()),
            "이동 95백분위(mm)": float(np.quantile(shift, 0.95)),
            "최대 이동(mm)": float(shift.max())
        })

    lower_shift = displacement[:, LOWER_IDS]
    basic_ankle = np.linalg.norm(target_seq[:, 27] - target_seq[:, 28], axis=1)
    smooth_ankle = np.linalg.norm(smooth_seq[:, 27] - smooth_seq[:, 28], axis=1)
    detail = pd.DataFrame({
        "frame": np.arange(len(raw_seq)),
        "lower_mean_shift_mm": lower_shift.mean(axis=1),
        "lower_max_shift_mm": lower_shift.max(axis=1),
        "basic_ankle_distance_m": basic_ankle,
        "smooth_ankle_distance_m": smooth_ankle
    })

    os.makedirs(RESULT_DIR, exist_ok=True)
    table = pd.DataFrame(rows)
    summary_path = os.path.join(RESULT_DIR, f"{NAME}_smoothing_comparison.csv")
    detail_path = os.path.join(RESULT_DIR, f"{NAME}_smoothing_frames.csv")
    table.to_csv(summary_path, index=False, encoding="utf-8-sig")
    detail.to_csv(detail_path, index=False, encoding="utf-8-sig")

    example_frame = int(np.argmax(lower_shift.max(axis=1)))
    print("\n===== 스무딩 비교: 시간적 변화와 위치 이동 =====")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.6f}"))
    print(f"하체 최대 변화 프레임(0부터): {example_frame}")
    print(f"동일 프레임 발목 간 거리의 최대 변화: {np.max(np.abs(smooth_ankle - basic_ankle)) * 1000:.3f} mm")
    print("RMS는 2차 좌표 차분이며, 감소율은 실제 정확도 개선율이 아닙니다.")
    print("위치 이동과 발목 간 거리 변화도 함께 확인하세요.")
    print(f"비교표 저장: {summary_path}")
    print(f"프레임별 비교 저장: {detail_path}")
    return example_frame


# 8. 비교 대상 선택 및 프레임 재생
def visualize_comparison(raw_seq, target_seq, smooth_seq, example_frame):
    from utils.viser_test import PoseViser

    vis = PoseViser(fps=FPS)
    try:
        vis.server.scene.set_up_direction("+y")
        vis.pose_points.visible = False
        vis.pose_lines.visible = False
        vis._init_dual_handles(offset=1.5)

        pairs = {
            "원본 / 기본": (raw_seq, target_seq),
            "기본 / 스무딩": (target_seq, smooth_seq),
            "원본 / 스무딩": (raw_seq, smooth_seq)
        }
        mode = vis.server.gui.add_dropdown(
            "비교 대상", options=tuple(pairs), initial_value="기본 / 스무딩"
        )
        playing = vis.server.gui.add_checkbox("재생", initial_value=True)
        speed = vis.server.gui.add_slider(
            "재생 배속", min=0.25, max=1.0, step=0.25, initial_value=0.5
        )
        overlap = vis.server.gui.add_checkbox("겹쳐 보기", initial_value=False)
        frame = vis.server.gui.add_slider(
            "프레임", min=0, max=len(raw_seq) - 1,
            step=1, initial_value=0
        )
        jump = vis.server.gui.add_button("최대 변화 프레임")

        @jump.on_click
        def jump_to_frame(_):
            playing.value = False
            frame.value = example_frame

        vis.server.gui.add_markdown(
            "주황색 선: 앞 단계 / 초록색 선: 뒤 단계\n\n"
            "재생을 끄면 프레임을 선택할 수 있습니다.\n\n"
            "스무딩은 시간에 따른 움직임을 비교하세요."
        )

        print(f"\n비교 화면 주소: http://localhost:{vis.server.get_port()}")
        print("기본 / 스무딩을 0.5배속으로 재생합니다.")
        print(f"'최대 변화 프레임' 버튼을 누르면 {example_frame}번에서 정지합니다.")
        print("화면 간격·재생 배속은 저장 데이터에 영향을 주지 않습니다.")
        print("종료: 터미널에서 Ctrl+C")

        previous = None
        while True:
            state = (mode.value, int(frame.value), overlap.value)
            if state != previous:
                first, second = pairs[state[0]]
                with vis.server.atomic():
                    vis.offset = 0.0 if state[2] else 1.5
                    vis._update_dual_frame(first[state[1]], second[state[1]])
                previous = state

            time.sleep(1 / (FPS * float(speed.value)))
            if playing.value:
                frame.value = (int(frame.value) + 1) % len(raw_seq)
    except KeyboardInterrupt:
        print("\nViser 종료")
    finally:
        vis.server.stop()


# 9. 풋워크 후처리 실행
def main():
    input_path = os.path.join(RAW_KEYPOINTS_DIR, f"{NAME}.npz")
    raw_seq, visibility = load_raw_keypoints(input_path, return_visibility=True)

    # 기본: 원본 → 높이 보정
    target_seq = create_height_corrected_target(raw_seq)

    # 보너스: 원본 → 스무딩 → 동일한 높이 보정
    smoothed_raw = apply_savgol_smoothing(raw_seq)
    smooth_seq = create_height_corrected_target(smoothed_raw)

    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"데이터 크기: {raw_seq.shape}")
    for suffix, seq in [("raw", raw_seq), ("target", target_seq), ("smooth", smooth_seq)]:
        npy_path = os.path.join(OUT_DIR, f"{NAME}_{suffix}.npy")
        csv_path = os.path.join(OUT_DIR, f"{NAME}_{suffix}.csv")
        np.save(npy_path, seq)
        print(f"NPY 저장: {npy_path}")
        export_unity_csv(seq, csv_path, visibility)

    example_frame = save_comparison(raw_seq, target_seq, smooth_seq)
    print("\n파일 저장 완료. Viser 비교 화면을 시작합니다.")
    visualize_comparison(raw_seq, target_seq, smooth_seq, example_frame)


if __name__ == "__main__":
    main()