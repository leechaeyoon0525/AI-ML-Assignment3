import os
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp

# 1. 폴더 경로 설정
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_VIDEO_DIR = os.path.join(ROOT, "data", "raw_videos")
RAW_KEYPOINT_DIR = os.path.join(ROOT, "data", "raw_keypoints")


# 2. 영상 불러오기
def extract_3d_keypoints(video_path, dir_path=RAW_KEYPOINT_DIR, name="Data"):
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"영상 파일이 없습니다: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"영상을 열 수 없습니다: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    pose_rows = []
    frame_idx = 0
    detected_frames = 0

    print(f"영상: {os.path.basename(video_path)}")
    print(f"영상 정보: {total_frames}프레임, {fps:.2f} FPS")

    # 3. 3D 관절 추출
    try:
        with mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as pose_model:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = pose_model.process(rgb)

                if res.pose_world_landmarks:
                    detected_frames += 1

                    for i, lm in enumerate(res.pose_world_landmarks.landmark):
                        pose_rows.append({
                            "frame": frame_idx,
                            "landmark": i,
                            "x": lm.x,
                            "y": -lm.y,  
                            "z": lm.z,
                            "visibility": lm.visibility
                        })

                frame_idx += 1
                if frame_idx % 30 == 0:
                    print(f"처리 중: {frame_idx}/{total_frames}프레임")
    finally:
        cap.release()

    if frame_idx == 0:
        raise RuntimeError("영상에서 프레임을 읽지 못했습니다.")
    if not pose_rows:
        raise RuntimeError("3D 관절이 검출되지 않아 저장하지 않았습니다.")

    # 4. 저장
    os.makedirs(dir_path, exist_ok=True)
    columns = ["frame", "landmark", "x", "y", "z", "visibility"]
    df = pd.DataFrame(pose_rows, columns=columns)
    out_path = os.path.join(dir_path, f"{name}.npz")
    np.savez(out_path, data=df.to_numpy(dtype=np.float64))

    print("\n===== 추출 완료 =====")
    print(f"읽은 프레임: {frame_idx}")
    print(f"포즈 검출 프레임: {detected_frames}")
    print(f"포즈 검출률: {detected_frames / frame_idx:.2%}")
    print(f"데이터 크기: {df.shape}")
    print(f"저장 위치: {out_path}")

    if detected_frames < frame_idx:
        print("일부 프레임에서 포즈가 검출되지 않았습니다. 원본 영상을 확인하세요.")
    if total_frames > 0 and frame_idx != total_frames:
        print("영상 정보와 실제 읽은 프레임 수가 다릅니다. 영상 파일을 확인하세요.")

    return out_path


# 5. 풋워크 영상 추출 실행
def main():
    video_path = os.path.join(RAW_VIDEO_DIR, "footwork_04.mp4")
    extract_3d_keypoints(video_path, RAW_KEYPOINT_DIR, "footwork_04")


if __name__ == "__main__":
    main()