import cv2
import numpy as np

video_path = 0  # Replace with your video path


def find_lane_lines(img):
    """
    Detecting road markings.
    Takes a color image (BGR) and returns a binary edge image.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gauss = cv2.GaussianBlur(gray, (11, 11), 0)
    thresh_low, thresh_high = 80, 135
    edges = cv2.Canny(img_gauss, thresh_low, thresh_high)
    return edges


def birdview_transform(img):
    """
    Apply bird's-eye (perspective) transform.
    """
    H, W = img.shape[:2]
    src = np.float32([[0, H], [W, H], [0, H*0.18], [W, H*0.2]])
    dst = np.float32([[240, H], [W-240, H], [-160, 0], [W+160, 0]])
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (W, H))
    return warped


def find_left_right_points(image, draw=None):
    """
    Find positions of left and right lane lines on a scan line.
    """
    h, w = image.shape[:2]
    y = int(h * 0.9)
    if draw is not None:
        cv2.line(draw, (0, y), (w, y), (0, 0, 255), 2)
    scan = image[y]

    center = w // 2
    lane_w = 100
    left_pt = -1
    right_pt = -1
    # search left
    for x in range(center, -1, -1):
        if scan[x] > 0:
            left_pt = x
            break
    # search right
    for x in range(center, w):
        if scan[x] > 0:
            right_pt = x
            break
    # estimate if missing
    if left_pt != -1 and right_pt == -1:
        right_pt = left_pt + lane_w
    if right_pt != -1 and left_pt == -1:
        left_pt = right_pt - lane_w
    # draw
    if draw is not None:
        if left_pt != -1:
            cv2.circle(draw, (left_pt, y), 7, (255, 255, 0), -1)
        if right_pt != -1:
            cv2.circle(draw, (right_pt, y), 7, (0, 255, 0), -1)
    return left_pt, right_pt


def calculate_control_signal(frame):
    """
    Given a color frame, compute throttle and steering and debug image.
    Returns (throttle, steering, debug_overlay).
    """
    edges = find_lane_lines(frame)
    bird = birdview_transform(edges)
    overlay = cv2.cvtColor(bird, cv2.COLOR_GRAY2BGR)

    left_pt, right_pt = find_left_right_points(bird, draw=overlay)
    throttle = 0.5  # normalized [0,1]
    steering = 0.0  # in radians per second approximation
    w = frame.shape[1]
    if left_pt != -1 and right_pt != -1:
        center_pt = (left_pt + right_pt) // 2
        error = (w // 2) - center_pt
        steering = -float(error * 0.01)
    return throttle, steering, overlay


def calculate_wheel_speeds(throttle, steering, wheel_base=0.42, max_linear_speed=1.0):
    """
    Compute individual wheel speeds for a differential drive robot.
    throttle: normalized forward command [0,1]
    steering: angular command (rad/s)
    wheel_base: distance between wheels (m)
    max_linear_speed: robot max forward speed (m/s)
    Returns (v_left, v_right) in m/s.
    """
    # Convert throttle command to linear velocity
    v = throttle * max_linear_speed
    # Angular velocity w approximately equals steering
    omega = steering
    # Differential drive kinematics
    v_left = v - (omega * wheel_base / 2.0)
    v_right = v + (omega * wheel_base / 2.0)
    return v_left, v_right


# --- Main processing loop --- #
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"Error: Cannot open video {video_path}")
    exit(1)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (640, 480))

    throttle, steering, debug = calculate_control_signal(frame)
    v_left, v_right = calculate_wheel_speeds(throttle, steering)

    # Overlay info
    info = f"Throttle: {throttle:.2f}, Steering: {steering:.2f}" \
           + f", v_left: {v_left:.2f} m/s, v_right: {v_right:.2f} m/s"
    cv2.putText(debug, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 255, 255), 2)

    # Show windows
    cv2.imshow('Original', frame)
    cv2.imshow('Debug Birdview', debug)
    bird_eye = birdview_transform(frame)
    cv2.imshow('Bird Eye View', bird_eye)

    key = cv2.waitKey(30) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
