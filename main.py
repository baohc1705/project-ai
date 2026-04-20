import cv2
import numpy as np
import mediapipe as mp
import screen_brightness_control as sbc
from math import hypot
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL


def main():

    try:

        from pycaw.pycaw import AudioUtilities

        # Lấy trình quản lý thiết bị
        device_enumerator = AudioUtilities.GetDeviceEnumerator()
        # Lấy loa mặc định (0: eRender, 0: eMultimedia)
        speakers = device_enumerator.GetDefaultAudioEndpoint(0, 0)

        # Kích hoạt giao diện điều khiển âm lượng
        interface = speakers.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        volume = cast(interface, POINTER(IAudioEndpointVolume))

        # Lấy dải âm lượng
        volRange = volume.GetVolumeRange()
        minVol, maxVol = volRange[0], volRange[1]
        print("Đã kết nối hệ thống âm thanh!")
    except Exception as e:
        print(f"Lỗi khởi tạo âm thanh: {e}")

        return

    # --- KHỞI TẠO MEDIAPIPE ---
    mpHands = mp.solutions.hands
    hands = mpHands.Hands(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.75,
        min_tracking_confidence=0.75,
        max_num_hands=2)

    draw = mp.solutions.drawing_utils
    cap = cv2.VideoCapture(0)

    print("Chương trình đang chạy... Nhấn 'q' để thoát.")
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            processed = hands.process(frameRGB)

            left_landmark_list, right_landmark_list = get_left_right_landmarks(frame, processed, draw, mpHands)

            # 1. Điều khiển ĐỘ SÁNG (Tay TRÁI)
            if left_landmark_list:
                left_distance = get_distance(frame, left_landmark_list)
                b_level = np.interp(left_distance, [30, 200], [0, 100])
                sbc.set_brightness(int(b_level))
                cv2.putText(frame, f'Brightness: {int(b_level)}%', (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # 2. Điều khiển ÂM LƯỢNG (Tay PHẢI)
            if right_landmark_list:
                right_distance = get_distance(frame, right_landmark_list)
                
                
                vol_scalar = np.interp(right_distance, [30, 200], [0.0, 1.0])
                volume.SetMasterVolumeLevelScalar(vol_scalar, None)

                volBar = int(vol_scalar * 100)
                cv2.putText(frame, f'Volume: {volBar}%', (10, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            cv2.imshow('Control System (Press Q to quit)', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


def get_left_right_landmarks(frame, processed, draw, mpHands):
    left_landmark_list = []
    right_landmark_list = []

    if processed.multi_hand_landmarks and processed.multi_handedness:
        for idx, hand_handedness in enumerate(processed.multi_handedness):
            # MediaPipe xác định nhãn tay
            label = hand_handedness.classification[0].label
            hand_landmarks = processed.multi_hand_landmarks[idx]

            draw.draw_landmarks(frame, hand_landmarks, mpHands.HAND_CONNECTIONS)

            current_list = []
            for id, lm in enumerate(hand_landmarks.landmark):
                if id == 4 or id == 8:  # Chỉ lấy ngón cái và ngón trỏ
                    h, w, c = frame.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    current_list.append([id, cx, cy])


            if label == 'Left':
                left_landmark_list = current_list
            else:
                right_landmark_list = current_list

    return left_landmark_list, right_landmark_list


def get_distance(frame, landmark_list):
    if len(landmark_list) < 2:
        return 0
    x1, y1 = landmark_list[0][1], landmark_list[0][2]
    x2, y2 = landmark_list[1][1], landmark_list[1][2]

    cv2.circle(frame, (x1, y1), 10, (255, 0, 255), cv2.FILLED)
    cv2.circle(frame, (x2, y2), 10, (255, 0, 255), cv2.FILLED)
    cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)

    return hypot(x2 - x1, y2 - y1)


if __name__ == '__main__':
    main()