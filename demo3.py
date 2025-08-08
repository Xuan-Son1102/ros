import cv2
import numpy as np
import serial
import time

video_path = (0)  # Camera mặc định (hoặc đường dẫn video)

# ================== CẤU HÌNH ESP32/Serial ================== #
SERIAL_PORT = 'COM4'
BAUDRATE = 115200

ADDR_LEFT  = 1
ADDR_RIGHT = 2
DIR_FORWARD = 0
DIR_REVERSE = 1

def open_serial(port=SERIAL_PORT, baud=BAUDRATE, timeout=1):
    try:
        ser = serial.Serial(port, baudrate=baud, timeout=timeout)
        time.sleep(2.0)
        print(f"Connected to ESP32 on {port} at {baud} baud.")
        return ser
    except Exception as e:
        print(f"Error opening serial port {port}: {e}")
        return None

def send_motor_command(ser, address, direction, speed_byte):
    if not ser or not getattr(ser, "is_open", False):
        return
    speed = int(np.clip(speed_byte, 0, 255))
    byte1 = (direction << 7) | (address & 0x7F)
    byte2 = speed
    byte3 = 0xFF
    try:
        ser.write(bytes([byte1, byte2, byte3]))
    except Exception as e:
        print("Serial write error:", e)

def forward_only_scale(vL, vR, vmax):
    # Chuyển về dạng chỉ tiến (không âm)
    minv = min(vL, vR)
    if minv < 0:
        vL -= minv
        vR -= minv
    maxv = max(vL, vR, 1e-9)
    if maxv > vmax:
        scale = vmax / maxv
        vL *= scale
        vR *= scale
    vL = float(np.clip(vL, 0.0, vmax))
    vR = float(np.clip(vR, 0.0, vmax))
    return vL, vR

def v_to_pwm_byte(v, vmax):
    v = float(np.clip(v, 0.0, vmax))
    return int(round((v / vmax) * 255.0)) if vmax > 1e-9 else 0

# ================== KALMAN FILTER (đơn giản) ================== #
class SimpleKalmanFilter:
    """
    1D Kalman filter đơn giản:
    state x: giá trị ước lượng (ở đây dùng cho steering)
    q: process variance (độ nhiễu quá trình)
    r: measurement variance (độ nhiễu đo)
    e: estimated error (P)
    """
    def __init__(self, process_variance=1e-3, measurement_variance=1e-2, estimated_error=1.0, initial_value=0.0):
        self.q = float(process_variance)
        self.r = float(measurement_variance)
        self.e = float(estimated_error)
        self.x = float(initial_value)

    def update(self, measurement):
        # Prediction step (in this simple model we assume x_k = x_{k-1})
        self.e += self.q

        # Kalman gain
        k = self.e / (self.e + self.r)

        # Update estimate with measurement
        self.x += k * (measurement - self.x)

        # Update error covariance
        self.e *= (1 - k)

        return self.x

# ================== XỬ LÝ ẢNH ================== #
def find_lane_lines(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gauss = cv2.GaussianBlur(gray, (11, 11), 0)
    edges = cv2.Canny(img_gauss, 80, 135)
    return edges

def birdview_transform(img):
    H, W = img.shape[:2]
    src = np.float32([[0, H], [W, H], [0, H*0.2], [W, H*0.15]])
    dst = np.float32([[240, H], [W-240, H], [-160, 0], [W+160, 0]])
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (W, H))
    return warped

def find_left_right_points(image, draw=None):
    h, w = image.shape[:2]
    y = int(h * 0.9)
    if draw is not None:
        cv2.line(draw, (0, y), (w, y), (0, 0, 255), 2)
    scan = image[y]
    center = w // 2
    lane_w = 180
    left_pt = -1
    right_pt = -1
    # tìm từ giữa sang trái
    for x in range(center, -1, -1):
        if scan[x] > 0:
            left_pt = x
            break
    # tìm từ giữa sang phải
    for x in range(center, w):
        if scan[x] > 0:
            right_pt = x
            break
    # nếu thiếu một bên, giả sử bề rộng là lane_w
    if left_pt != -1 and right_pt == -1:
        right_pt = left_pt + lane_w
    if right_pt != -1 and left_pt == -1:
        left_pt = right_pt - lane_w
    if draw is not None:
        if left_pt != -1:
            cv2.circle(draw, (left_pt, y), 7, (255, 255, 0), -1)
        if right_pt != -1:
            cv2.circle(draw, (right_pt, y), 7, (0, 255, 0), -1)
    return left_pt, right_pt

def calculate_control_signal(frame):
    edges = find_lane_lines(frame)
    bird = birdview_transform(edges)
    overlay = cv2.cvtColor(bird, cv2.COLOR_GRAY2BGR)
    left_pt, right_pt = find_left_right_points(bird, draw=overlay)
    throttle = 0.5  # tạm cố định throttle; bạn có thể thay bằng logic khác
    steering = 0.0
    w = frame.shape[1]
    if left_pt != -1 and right_pt != -1:
        center_pt = (left_pt + right_pt) // 2
        error = (w // 2) - center_pt
        steering = -float(error * 0.01)  # tỷ lệ chuyển error->steering (tùy chỉnh)
    return throttle, steering, overlay

def calculate_wheel_speeds(throttle, steering, wheel_base=0.42, max_linear_speed=1.0):
    v = throttle * max_linear_speed
    omega = steering
    v_left = v - (omega * wheel_base / 2.0)
    v_right = v + (omega * wheel_base / 2.0)
    return v_left, v_right

# ================== MAIN LOOP ================== #
ser = open_serial()
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"Error: Cannot open video {video_path}")
    # Không exit(1) nếu bạn muốn chương trình tiếp tục dù không có camera, nhưng ở đây exit để debug.
    exit(1)

MAX_LINEAR_SPEED = 0.09  # m/s
PWM_MIN = 10
PWM_MAX = 20

# Khởi tạo Kalman filter cho steering
# Bạn có thể điều chỉnh process_variance (q) và measurement_variance (r)
# q nhỏ => tin tưởng model (ít thay đổi), r lớn => biểu thị đo nhiễu nhiều
kf_steering = SimpleKalmanFilter(process_variance=0.0005, measurement_variance=0.02, estimated_error=1.0, initial_value=0.0)

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (640, 480))

        throttle, steering_raw, debug = calculate_control_signal(frame)

        # Lọc steering bằng Kalman filter
        steering_filtered = kf_steering.update(steering_raw)

        # Tính vận tốc bánh từ steering đã lọc
        v_left, v_right = calculate_wheel_speeds(throttle, steering_filtered, wheel_base=0.42, max_linear_speed=MAX_LINEAR_SPEED)
        vL_pos, vR_pos = forward_only_scale(v_left, v_right, MAX_LINEAR_SPEED)

        pwmL = v_to_pwm_byte(vL_pos, MAX_LINEAR_SPEED)
        pwmR = v_to_pwm_byte(vR_pos, MAX_LINEAR_SPEED)

        # Giới hạn PWM trong khoảng 10–20 (theo cấu hình cũ)
        pwmL = int(np.clip(pwmL, PWM_MIN, PWM_MAX))
        pwmR = int(np.clip(pwmR, PWM_MIN, PWM_MAX))

        # Gửi lệnh tới ESP32 (hàm send_motor_command đã tự kiểm tra serial)
        send_motor_command(ser, ADDR_LEFT,  DIR_FORWARD, pwmL)
        send_motor_command(ser, ADDR_RIGHT, DIR_REVERSE, pwmR)

        info = (f"Throttle: {throttle:.2f}, Steering(raw): {steering_raw:.3f}, Steering(kf): {steering_filtered:.3f}"
                f", vL: {v_left:.3f}->{vL_pos:.3f} m/s ({pwmL})"
                f", vR: {v_right:.3f}->{vR_pos:.3f} m/s ({pwmR})")
        cv2.putText(debug, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 255), 2)

        cv2.imshow('Original', frame)
        cv2.imshow('Debug Birdview', debug)
        bird_eye = birdview_transform(frame)
        cv2.imshow('Bird Eye View', bird_eye)

        key = cv2.waitKey(20) & 0xFF
        if key == ord('q'):
            break

finally:
    # Dừng động cơ an toàn
    send_motor_command(ser, ADDR_LEFT,  DIR_REVERSE, 0)
    send_motor_command(ser, ADDR_RIGHT, DIR_REVERSE, 0)
    cap.release()
    if ser and getattr(ser, "is_open", False):
        try:
            ser.close()
        except Exception:
            pass
    cv2.destroyAllWindows()
