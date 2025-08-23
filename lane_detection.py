# import các thư viện cần thiết

import cv2
import numpy as np
import serial
import time
import pygame

# --- CẤU HÌNH CHUNG ---
CONTROL_MODE = 'AUTO' # Chế độ khởi động: 'AUTO' hoặc 'MANUAL'

# --- CẤU HÌNH ĐIỀU KHIỂN TỰ ĐỘNG (PID) ---
# *** HIỆU CHỈNH VỚI BỘ ĐIỀU KHIỂN PID ĐỂ DI CHUYỂN MƯỢT MÀ ***
BASE_SPEED = 10
MAX_SPEED = 20

Mnaual_Speed = 35

KP = 0.015              # Proportional gain - Tăng khả năng đáp ứng
KI = 0.001              # Integral gain - Loại bỏ sai số xác lập & giúp hệ thống đạt được độ chính xác cao
KD = 0.012              # Derivative gain - Giảm vọt lố và tăng tính ổn đinh của hệ thống.
INTEGRAL_LIMIT = 300 # Giới hạn cho thành phần Tích phân để chống "tích phân bão hòa"

# --- BIẾN TOÀN CỤC CHO PID ---
integral_error = 0.0
previous_error = 0.0

# --- CẤU HÌNH BỘ LỌC KALMAN ---
kalman_Q = 0.1
kalman_R = 0.8
kalman_P = 1.0
kalman_x = None

# --- TRÍ NHỚ NGẮN HẠN CHO HƯỚNG LÁI VÀ CHIỀU RỘNG LÀN ---
last_valid_error = 0.0
lane_width_memory = 400 # Chiều rộng làn đường mặc định (pixel)

# --- Cài đặt giao tiếp Serial với ESP32 ---
try:
    ser = serial.Serial('COM7', 115200, timeout=1)      # Mở cổng kết nối Serial (UART) là cổng COM8 với tốc độ baudrate là 115200 bit/s
    time.sleep(2)                                       # Tạo trễ 2s
    print("Đã kết nối với ESP32.")                      # In ra màn hình Terminal
except serial.SerialException as e:                     # Nếu lỗi cổng giao tiếp Serial
    print(f"Lỗi: Không thể mở cổng serial. {e}")        # In ra màn hình Terminal
    ser = None                                          # Gán ser = None để đánh dấu không có kết nối Serial hợp lệ

# --- CÀI ĐẶT TAY CẦM PS2 (PYGAME) ---
pygame.init()                                               # Khởi tạo các module trong thư viện pygame
joystick = None                                             # Khai báo biến joystick là biến rỗng
if pygame.joystick.get_count() > 0:                         # Trả về số lượng joystick được kết nối và xét điều kiện
    joystick = pygame.joystick.Joystick(0)                  # Gán địa chỉ cho Joystick dược kết nối với máy tính là 0
    joystick.init()                                         # Khởi tạo tay cầm Joystick (PS2) để sử dụng nút bấm, cần analog,...
    print(f"Đã kết nối với tay cầm: {joystick.get_name()}") # In ra màn hình Terminal
else:
    print("Lỗi: Không tìm thấy tay cầm PS2.")

# --- CÁC HÀM XỬ LÝ ẢNH ---
def process_frame(frame):
    height, width, _ = frame.shape
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)          # Chuyển đổi ảnh từ không gian màu BGR sang không gian màu HSV
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 90])
    mask = cv2.inRange(hsv_frame, lower_black, upper_black)     # Tạo ra 1 ảnh nhị phân mask
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)       # Loại bỏ các "nhiễu" trắng nhỏ li ti trên ảnh.
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)      # Lấp đầy các lỗ "đen" nhỏ, bên trong vạch kẻ đường màu trắng, làm cho vạch kẻ liền mạch hơn
    
    canny_edges = cv2.Canny(mask, 50, 150)                      # Tìm cạnh biên của các vùng đen - trắng của ảnh nhị phân mask, với ngưỡng thấp là 50, ngưỡng cao là 150
    
    roi_vertices = np.array([[(0, height), (width * 0.1, height * 0.3), (width * 0.9, height * 0.3), (width, height)]], dtype=np.int32)     # Định nghĩa tọa độ các điểm của vùng ROI
                                                                                                                                            # Điểm 1 (0, height) - Góc trải bên dưới
                                                                                                                                            # Điểm 2 (width * 0.1, height * 0.3) - Góc trái bên trên
                                                                                                                                            # Điểm 3 (width * 0.9, height * 0.3) - Góc phải bên trên
                                                                                                                                            # Điểm 4 (width, height) - Góc phải bên dưới
    roi_mask = np.zeros_like(canny_edges)                       # Tạo 1 ảnh roi_mask có màu đen và cùng kích thước với ảnh canny_edges
    cv2.fillPoly(roi_mask, roi_vertices, 255)                   # Lấp đầy các đa giác được xác định bởi các đỉnh trong roi_vertices bằng màu trắng (giá trị 255) trong ảnh mask roi_mask
    masked_edges = cv2.bitwise_and(canny_edges, roi_mask)       # Giữ lại những cạnh canny_edges nằm trong ROI roi_mask
    
    lines = cv2.HoughLinesP(masked_edges, rho=1, theta=np.pi / 180, threshold=15, lines=np.array([]), minLineLength=20, maxLineGap=100)     # Sử dụng phép biến đổi Hough Transfer để phát hiện tất cả các đoạn thẳng khả dĩ trong ROI
    
    return lines, mask

def average_slope_intercept(frame, lines):                                      # Input là frame và lines
    left_fit = []
    right_fit = []
    height, width, _ = frame.shape
    if lines is not None:                                                       # Khi lines không rỗng (phát hiện được đoạn thẳng nào đó)
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            if x1 == x2: continue                                               # Nếu là 1 đường thẳng đứng hoàn toàn, phép tính độ dốc sẽ gây ra lỗi 
            parameters = np.polyfit((x1, x2), (y1, y2), 1)                      # Tìm ra đa thức bậc 1 khớp với 2 điểm (x1, y1), (x2, y2) là (slope, intercept)
            slope, intercept = parameters
            if abs(slope) < 0.3: continue                                       # Loại bỏ những đường thẳng có độ dốc quá nhỏ
            x_at_bottom = (height - intercept) / slope                          
            if slope < 0 and x_at_bottom < (width / 2):                         # Nếu 2 điều kiện thỏa mãn
                left_fit.append((slope, intercept))                             # Cặp (slope, intercept) được thêm vào left_fit
            elif slope > 0 and x_at_bottom > (width / 2):                       # Nếu 2 điều kiện thỏa mãn
                right_fit.append((slope, intercept))                            # Cặp (slope, intercept) được thêm vào right_fit
    left_fit_average = np.average(left_fit, axis=0) if left_fit else None       # Tính giá trị trung bình trong left_fit
    right_fit_average = np.average(right_fit, axis=0) if right_fit else None    # Tính giá trị trung bình trong right_fit
    return left_fit_average, right_fit_average

def make_coordinates(frame, line_parameters):                                   # Đầu vào của hàm make_cổdinates là frame và line_parameters
    if line_parameters is None: return None                                     # Kiểm tra đầu vào
    slope, intercept = line_parameters                                          # Lấy thông số slope và intercept
    y1 = frame.shape[0]                                                         # y1 là điểm dưới cùng của ảnh
    y2 = int(y1 * 0.6)                                                          # y2 là điểm có độ cao 60% từ dưới lên
    try:
        x1 = int((y1 - intercept) / slope)                                      # Dựa vào đa thức bậc 1, tìm x1
        x2 = int((y2 - intercept) / slope)                                      # Dựa vào đa thức bậc 1, tìm x2
    except ZeroDivisionError:                                                   # Bỏ qua đường thẳng nằm ngang (slope = 0)
        return None
    return np.array([x1, y1, x2, y2])

def display_lines(frame, lines, color=(255, 0, 0)):
    line_image = np.zeros_like(frame)                                           # tạo 1 ảnh đen có cùng kích thước với frame
    if lines is not None:                                                       # Kiểm tra xem có đường nào để vẽ không
        for line in lines:                                                      # Lặp qua các danh sách các đường cần vẽ
            if line is not None:
                x1, y1, x2, y2 = line                                           # Lấy tọa độ và vẽ
                cv2.line(line_image, (x1, y1), (x2, y2), color, 10)             
    return line_image

# --- CÁC HÀM TÍNH TOÁN TỐC ĐỘ ---
def calculate_auto_speeds(frame, lines):
    global kalman_x, kalman_P, integral_error, previous_error, last_valid_error, lane_width_memory  # khai báo biến
    height, width, _ = frame.shape
    left_fit, right_fit = average_slope_intercept(frame, lines)
    
    display_lanes = None
    filtered_center = None
    measurement = None
    
    # Nhớ làn, trong trường hợp robot quan sát được cả 2 làn, robot xác định tọa độ chính xác của 2 vạch kẻ và tính toán độ rộng làn đường "lane_width_memory", biến toàn cục này sẽ được cập nhật liên tục khi robot có tầm nhìn tốt.
    if left_fit is not None and right_fit is not None:
        left_line = make_coordinates(frame, left_fit)
        right_line = make_coordinates(frame, right_fit)
        if left_line is not None and right_line is not None:
            _, _, left_x2, _ = left_line
            _, _, right_x2, _ = right_line
            measurement = (left_x2 + right_x2) / 2              # xác định điểm giữa làn đường mà xe lên bám theo
            lane_width_memory = abs(right_x2 - left_x2)         # tính toán độ rộng làn đường
            display_lanes = np.array([left_line, right_line])

    # Khi robot chỉ thấy làn đường bên trái, không thấy làn bên phải, ước tính vị trí tâm làn đường.
    elif left_fit is not None:
        left_line = make_coordinates(frame, left_fit)
        if left_line is not None:
            _, _, left_x2, _ = left_line
            measurement = left_x2 + lane_width_memory / 2       # ước tính vị trí tâm làn đường bằng cách lấy vị trí làn trái + 1/2 độ rộng làn đường đã nhớ "lane_width_memory/2"
            display_lanes = np.array([left_line, None], dtype=object)

    # Khi robot chỉ thấy làn đường bên phải, không thấy làn bên trái, ước tính vị trí tâm làn đường.
    elif right_fit is not None:
        right_line = make_coordinates(frame, right_fit)
        if right_line is not None:
            _, _, right_x2, _ = right_line
            measurement = right_x2 - lane_width_memory / 2      # ước tính vị trí tâm làn đường bằng cách lấy vị trí làn phải - 1/2 độ rộng làn đường đã nhớ "lane_width_memory/2"
            display_lanes = np.array([None, right_line], dtype=object)

    # Làm mượt dữ liệu đầu vào với bộ lọc Kalman
    if measurement is not None:
        if kalman_x is None: kalman_x = measurement # Khởi tạo bộ lọc Kalman, ước tính tốt nhất ban đầu là giá trị đo đạc đầu tiên
        kalman_P += kalman_Q                        # Dự đoán sự thay đổi của độ bất định kalman_P (độ bất định cũ) được + kalman_Q (nhiễu quá trình)
        # Nếu kalman_P lớn, kalman_R nhỏ, K ~ 1 => bộ lọc ưu tiên dùng giá trị measurment mới
        # Nếu kalman_P nhỏ, kalman_R lớn, K ~ 0 => bộ lọc bỏ qua giá trị measurment mới
        K = kalman_P / (kalman_P + kalman_R)        # Tính toán Kalman_Gain
        kalman_x += K * (measurement - kalman_x)    # Cập nhật ước tính
        kalman_P = (1 - K) * kalman_P               # Cập nhật độ bất định
        
        # Tính toán sai số
        # error > 0: tâm làn đường đang ở bên phải robot => robot cần lái sang phải
        # error < 0: tâm làn đường đang ở ben trái robot => robot cần lái sang trái
        # error = 0: robot đang di chuyển chính giữa làn đường
        error = kalman_x - (width / 2)  # tính toán sai số
        last_valid_error = error        # nếu robot không nhìn thấy cả 2 làn đường, robot sẽ sử dụng dữ liệu sai số cuối cùng để điều khiển - di chuyển
        filtered_center = kalman_x
    else:
        error = last_valid_error

    # Bộ điều khiển PID
    integral_error += error                                                     # cộng dồn sai số tích lũy
    integral_error = np.clip(integral_error, -INTEGRAL_LIMIT, INTEGRAL_LIMIT)   # giới hạn giá trị integral_error để tránh gặp hiện tượng bão hòa tích phân
    
    derivative_error = error - previous_error                                   # giảm vọt lố và dao động giúp hệ thống ổn định và mượt mà
    previous_error = error                                                      # cập nhật sai số
    
    correction = (KP * error) + (KI * integral_error) + (KD * derivative_error) # lệnh lái cuối cùng, tổng hợp trọng số
    
    # *** LOGIC ĐIỀU KHIỂN ĐÃ ĐƯỢC SỬA LẠI HOÀN TOÀN ***
    # Dựa trên firmware ESP32:
    # - Bánh trái được điều khiển bởi giá trị R, không đảo ngược.
    # - Bánh phải được điều khiển bởi giá trị L, có đảo ngược.
    # Để đi thẳng: Bánh trái cần tốc độ +BASE_SPEED, Bánh phải cần tốc độ +BASE_SPEED.
    # => Gửi đi: R = +BASE_SPEED, L = -BASE_SPEED
    
    left_motor_speed = BASE_SPEED - correction
    right_motor_speed = BASE_SPEED + correction
    
    left_speed_cmd = right_motor_speed # Gửi giá trị âm cho bánh phải
    right_speed_cmd = left_motor_speed   # Gửi giá trị dương cho bánh trái
    
    left_speed_cmd = np.clip(left_speed_cmd, -MAX_SPEED, MAX_SPEED)
    right_speed_cmd = np.clip(right_speed_cmd, -MAX_SPEED, MAX_SPEED)
    
    return int(left_speed_cmd), int(right_speed_cmd), display_lanes, filtered_center

def calculate_manual_speeds():
    throttle = -joystick.get_axis(1)                                # Đọc giá trị trục dọc của cần điều khiển Analog (hướng gạt lên xuống)
    turn = joystick.get_axis(0)                                     # Đọc giá trị trục ngang của cần điều khiển Analog (hướng gạt sang trái phải)
    left_speed = throttle * Mnaual_Speed + turn * Mnaual_Speed
    right_speed = throttle * Mnaual_Speed - turn * Mnaual_Speed
    left_speed = np.clip(left_speed, -Mnaual_Speed, Mnaual_Speed)
    right_speed = np.clip(right_speed, -Mnaual_Speed, Mnaual_Speed)
    return int(left_speed), int(right_speed)

# --- VÒNG LẶP CHÍNH ---
cap = cv2.VideoCapture(0)                                                                           # Mở camera
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

running = True
while running:
    for event in pygame.event.get():                                            # Lắng nghe sự kiện từ người dùng
        if event.type == pygame.QUIT:                                           # Nếu nhấn nút thoát (Robot dừng hoạt động)
            running = False
        if joystick and event.type == pygame.JOYBUTTONDOWN:                     # Lắng nghẹ người dùng gọi chế độ
            if joystick.get_button(9):
                CONTROL_MODE = 'MANUAL' if CONTROL_MODE == 'AUTO' else 'AUTO'
                integral_error = 0.0
                previous_error = 0.0
                print(f"Chuyển sang chế độ {CONTROL_MODE}")
            if joystick.get_button(8):
                running = False

    ret, frame = cap.read()                         # Thu thập dữ liệu từ camera
    if not ret: break
    
    left_speed, right_speed = 0, 0
    display_lanes = None
    filtered_center = None
    mask_for_display = None

    if CONTROL_MODE == 'AUTO':
        # *** SỬA LỖI VALUEERROR TẠI ĐÂY ***
        detected_lines, mask_for_display = process_frame(frame)
        left_speed, right_speed, display_lanes, filtered_center = calculate_auto_speeds(frame, detected_lines)      # Lấy tốc độ 2 bánh, đường kẻ lane và tâm lane từ hàm calculate_auto_speeds()
    else: # MANUAL MODE
        if joystick:
            left_speed, right_speed = calculate_manual_speeds()                                                     # Lấy tốc độ 2 bánh từ hàm calculate_manual_speeds()

    if ser is not None:
        command = f"<{left_speed},{right_speed}>"
        ser.write(command.encode())                                                                                 # Gửi lệnh xuống động cơ
    
    display_text = f"Mode: {CONTROL_MODE}"
    text_color = (0, 255, 0) if CONTROL_MODE == 'AUTO' else (0, 255, 255)
    
    combo_image = frame
    if CONTROL_MODE == 'AUTO':
        line_image_raw = display_lines(frame, display_lanes, color=(255, 0, 0))
        combo_image = cv2.addWeighted(frame, 0.8, line_image_raw, 1, 1)
        if filtered_center is not None:
            height, width, _ = frame.shape
            start_point = (width // 2, height)
            end_point = (int(filtered_center), int(height * 0.3))
            cv2.line(combo_image, start_point, end_point, (0, 0, 255), 3)

    cv2.putText(combo_image, display_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)
    cv2.putText(combo_image, f"L:{left_speed} R:{right_speed}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    cv2.imshow("Lane Following Robot", combo_image)
    
    if mask_for_display is not None:
        cv2.imshow("Black Mask", mask_for_display)
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        running = False

cap.release()
cv2.destroyAllWindows()
pygame.quit()
if ser is not None:
    ser.write(b"<0,0>")
    ser.close()
