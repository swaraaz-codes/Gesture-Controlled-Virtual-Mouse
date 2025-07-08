import cv2
import mediapipe as mp
import pyautogui
import time

# --------------------------------
# ✅ Camera Setup
# --------------------------------
def setup_camera(index=1, width=640, height=480):
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    cap.set(3, width)
    cap.set(4, height)
    return cap

# --------------------------------
# ✅ Mediapipe Setup
# --------------------------------
def setup_hand_detector(max_hands=1, min_det=0.5, min_track=0.5):
    hands = mp.solutions.hands.Hands(
        max_num_hands=max_hands,
        min_detection_confidence=min_det,
        min_tracking_confidence=min_track
    )
    return hands

# --------------------------------
# ✅ Trackbars
# --------------------------------
def create_trackbars(window_name):
    def nothing(x): pass
    cv2.namedWindow(window_name)
    cv2.createTrackbar("Smoothening", window_name, 50, 100, nothing)  # 5.0 ➜ 10.0 (/10)      # Smoothening = 30 and Scale = 20 is best in my case
    cv2.createTrackbar("Scale", window_name, 10, 30, nothing)         # 1.0 ➜ 3.0 (/10)

def get_trackbar_values(window_name):
    smooth_raw = cv2.getTrackbarPos("Smoothening", window_name)
    scale_raw = cv2.getTrackbarPos("Scale", window_name)
    smoothening = smooth_raw / 10.0  # 5 ➜ 10
    scale_factor = scale_raw / 10.0  # 1.0 ➜ 3.0

    if smoothening < 1.0:
        smoothening = 1.0
    if scale_factor < 1.0:
        scale_factor = 1.0

    return smoothening, scale_factor

# --------------------------------
# ✅ Map & Scale
# --------------------------------
def map_and_scale(x, y, frame_width, frame_height, screen_width, screen_height, scale_factor):
    mapped_x = screen_width / frame_width * x
    mapped_y = screen_height / frame_height * y

    center_x = screen_width / 2
    center_y = screen_height / 2

    delta_x = mapped_x - center_x
    delta_y = mapped_y - center_y

    scaled_x = center_x + delta_x * scale_factor
    scaled_y = center_y + delta_y * scale_factor

    scaled_x = min(max(0, scaled_x), screen_width - 1)
    scaled_y = min(max(0, scaled_y), screen_height - 1)

    return scaled_x, scaled_y

# --------------------------------
# ✅ Main Loop
# --------------------------------
def run_virtual_mouse():
    window_name = "Virtual Mouse"
    cap = setup_camera()
    hand_detector = setup_hand_detector()
    screen_width, screen_height = pyautogui.size()
    create_trackbars(window_name)

    previous_x, previous_y = 0, 0
    click_time = 0

    while True:
        smoothening, scale_factor = get_trackbar_values(window_name)

        _, frame = cap.read()
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_height, frame_width, _ = frame.shape

        output = hand_detector.process(rgb_frame)
        hands = output.multi_hand_landmarks

        if hands:
            for hand in hands:
                landmarks = hand.landmark

                # Index finger
                index_finger = landmarks[8]
                x = int(index_finger.x * frame_width)
                y = int(index_finger.y * frame_height)

                cv2.circle(frame, (x, y), 10, (0, 255, 255), cv2.FILLED)

                # Thumb finger
                thumb_finger = landmarks[4]
                x_thumb = int(thumb_finger.x * frame_width)
                y_thumb = int(thumb_finger.y * frame_height)

                cv2.circle(frame, (x_thumb, y_thumb), 10, (0, 255, 0), cv2.FILLED)

                # ---------------------------
                # 🟢 Mouse Position ➜ Scaled
                # ---------------------------
                scaled_x, scaled_y = map_and_scale(
                    x, y, frame_width, frame_height,
                    screen_width, screen_height, scale_factor
                )

                current_x = previous_x + (scaled_x - previous_x) / smoothening
                current_y = previous_y + (scaled_y - previous_y) / smoothening

                pyautogui.moveTo(current_x, current_y)

                previous_x, previous_y = current_x, current_y

                # ---------------------------
                # 🟢 Click ➜ RAW landmark diff
                # ---------------------------
                if abs(y - y_thumb) < 20 and abs(x - x_thumb) < 20:
                    current_time = time.time()
                    if current_time - click_time > 1:
                        pyautogui.click()
                        click_time = current_time
                        print("Click!")

                # Display Scale & Smoothening on frame
                cv2.putText(frame, f"Scale: {scale_factor:.1f}", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                cv2.putText(frame, f"Smooth: {smoothening:.1f}", (10, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        cv2.imshow(window_name, frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# --------------------------------
# ✅ Run it!
# --------------------------------
if __name__ == "__main__":
    run_virtual_mouse()