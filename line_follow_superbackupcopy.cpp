#include <librealsense2/rs.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
#include <thread>
#include <algorithm>
#include "LineFollow.hpp"
#include <fcntl.h>
#include <termios.h>
#include <unistd.h>

using namespace cv;

#define REALSENSE_ERROR 1
#define STD_ERROR 2
#define UNKNOWN_ERROR 3

// ==================== Ball Detection Parameters ====================
struct BallDetectionParams
{
    // HSV range for tennis balls (yellow-green)
    Scalar hsv_low = Scalar(20, 80, 120);
    Scalar hsv_high = Scalar(45, 255, 255);

    // Minimum area to consider as a ball
    int min_contour_area = 100;

    // Distance thresholds
    float detection_distance = 3.0f; // meters - distance to start intake
    float capture_distance = 0.25f;  // meters - distance when ball is captured

    // Circularity threshold for ball detection
    float min_circularity = 0.7f;
};

struct BallDetectionResult
{
    bool found = false;
    int cx = -1;
    int cy = -1;
    int radius = 0;
    float distance = 0.0f;
    float angle = 0.0f;
};

// ==================== UART Functions ====================
int open_uart(const char *dev = "/dev/ttyACM0", int baud = 115200)
{
    int fd = open(dev, O_RDWR | O_NOCTTY | O_NONBLOCK);
    if (fd < 0)
    {
        std::cerr << "Failed to open " << dev << ". Trying alternatives..." << std::endl;

        // Thử các device khác
        const char *alternatives[] = {"/dev/ttyUSB0", "/dev/ttyTHS1", "/dev/ttyAMA0"};
        for (const char *alt_dev : alternatives)
        {
            fd = open(alt_dev, O_RDWR | O_NOCTTY | O_NONBLOCK);
            if (fd >= 0)
            {
                std::cout << "Connected to " << alt_dev << std::endl;
                break;
            }
        }

        if (fd < 0)
        {
            std::cerr << "Could not open any UART device" << std::endl;
            return -1;
        }
    }

    termios tio{};
    if (tcgetattr(fd, &tio) != 0)
    {
        close(fd);
        return -1;
    }

    // Cấu hình serial port
    cfmakeraw(&tio);
    tio.c_cflag &= ~PARENB; // No parity
    tio.c_cflag &= ~CSTOPB; // 1 stop bit
    tio.c_cflag &= ~CSIZE;
    tio.c_cflag |= CS8;      // 8 data bits
    tio.c_cflag &= ~CRTSCTS; // No hardware flow control
    tio.c_cflag |= CREAD | CLOCAL;

    tio.c_cc[VMIN] = 0;
    tio.c_cc[VTIME] = 10; // Timeout in deciseconds

    // Set baud rate
    speed_t spd = B115200;
    cfsetispeed(&tio, spd);
    cfsetospeed(&tio, spd);

    if (tcsetattr(fd, TCSANOW, &tio) != 0)
    {
        close(fd);
        return -1;
    }

    // Clear buffer
    tcflush(fd, TCIOFLUSH);

    std::cout << "UART opened successfully: " << dev << std::endl;
    return fd;
}

void send_motor_command(int fd, const char *id, float speed)
{
    if (fd < 0)
        return;

    // Convert speed from [-1, 1] to PWM value [-255, 255]
    int pwm = static_cast<int>(speed * 255.0f);
    pwm = std::max(-255, std::min(255, pwm));

    char buf[32];
    int n = snprintf(buf, sizeof(buf), "%s %d\n", id, pwm);

    ssize_t bytes_written = write(fd, buf, n);
    if (bytes_written != n)
    {
        std::cerr << "UART write error: " << bytes_written << " bytes written, expected " << n << std::endl;
    }

    // Debug: In giá trị PWM được gửi
    std::cout << "UART_CMD: " << id << " " << pwm << " (from speed=" << speed << ")" << std::endl;

    // Đảm bảo dữ liệu được gửi hoàn toàn
    tcdrain(fd);
    usleep(10000); // 10ms delay
}

void drive_motors(int fd, float left_speed, float right_speed)
{
    // Nếu robot đi sai hướng, đổi M1 và M2:
    send_motor_command(fd, "M1", left_speed);  // M1 = bánh trái
    send_motor_command(fd, "M2", right_speed); // M2 = bánh phải
}

void intake_ball(int fd, int pwm)
{
    send_motor_command(fd, "M3", pwm);
}

static const int WIDTH = 640, HEIGHT = 480, FPS = 30;
static const int ROI_TOP = HEIGHT * 2 / 3;

// ==================== FSM States ====================
enum class RobotState
{
    GO_STRAIGHT_1, // First straight segment
    TURN_LEFT_90,  // Turn left 90 degrees
    GO_STRAIGHT_2, // Second straight segment
    TURN_RIGHT_90, // Turn right 90 degrees
    GO_STRAIGHT_3, // Third straight segment
    BALL_APPROACH, // Approaching ball
    BALL_INTAKE    // Intaking ball
};

// ==================== FSM Configuration ====================
struct FSMConfig
{
    // Time durations for each state (in seconds)
    float straight1_duration = 6.5f;  // First straight: 6.5 seconds
    float turn_left_duration = 0.8f;  // Turn left: 0.8 seconds
    float straight2_duration = 7.2f;  // Second straight: 7.2 seconds
    float turn_right_duration = 1.0f; // Turn right: 1 second
    float straight3_duration = 2.0f;  // Third straight: 2 seconds

    // Speed settings
    float max_speed = 0.7f;           // Max speed (maps to 255 PWM)
    float turn_speed = 0.6f;          // Turn speed
    float speed_reduction = 0.7f;     // Speed when correcting (70% of max)
    float correction_duration = 0.5f; // Correction duration in seconds

    // Line detection thresholds
    int black_threshold = 60;      // Gray values below this are considered black (0-255)
    int min_distance_to_line = 30; // Minimum distance from VML to line edge (pixels)
    int vml_left_threshold = 50;   // Pixels from left edge to trigger correction
    int vml_right_threshold = 50;  // Pixels from right edge to trigger correction

    // Ball approach settings
    float ball_approach_speed = 0.4f;
    float ball_turn_gain = 0.8f;
};

// ==================== Line Detection for Black Line Following ====================
struct LineDetection
{
    bool found_lines = false;
    int left_line_x = -1;            // Not used in centroid following
    int right_line_x = -1;           // Not used in centroid following
    int virtual_middle_line = -1;    // Now stores the centroid X position of target black line
    bool too_close_to_left = false;  // Robot is too far left (line center is left of robot center)
    bool too_close_to_right = false; // Robot is too far right (line center is right of robot center)
};

// ==================== Ball Detection Function ====================
BallDetectionResult detect_ball(const Mat &frame, const BallDetectionParams &params,
                                float fx, float ppx, float depth_scale, const rs2::depth_frame &depth_frame)
{
    BallDetectionResult result;

    // Convert to HSV
    Mat hsv;
    cvtColor(frame, hsv, COLOR_BGR2HSV);

    // Threshold for tennis ball color
    Mat mask;
    inRange(hsv, params.hsv_low, params.hsv_high, mask);

    // Morphological operations to clean up noise
    Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(5, 5));
    morphologyEx(mask, mask, MORPH_CLOSE, kernel);
    morphologyEx(mask, mask, MORPH_OPEN, kernel);

    // Find contours
    std::vector<std::vector<Point>> contours;
    findContours(mask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    // Find the best ball candidate
    double best_score = 0;
    int best_index = -1;

    for (size_t i = 0; i < contours.size(); i++)
    {
        double area = contourArea(contours[i]);
        if (area < params.min_contour_area)
            continue;

        // Calculate circularity
        double perimeter = arcLength(contours[i], true);
        double circularity = (4 * CV_PI * area) / (perimeter * perimeter);

        if (circularity < params.min_circularity)
            continue;

        // Calculate score based on area and circularity
        double score = area * circularity;

        if (score > best_score)
        {
            best_score = score;
            best_index = i;
        }
    }

    if (best_index >= 0)
    {
        result.found = true;

        // Get bounding circle
        std::vector<Point> contour = contours[best_index];
        Point2f center;
        float radius;
        minEnclosingCircle(contour, center, radius);

        result.cx = static_cast<int>(center.x);
        result.cy = static_cast<int>(center.y);
        result.radius = static_cast<int>(radius);

        // Calculate distance using depth information
        if (depth_frame)
        {
            try
            {
                float depth_value = depth_frame.get_distance(result.cx, result.cy);
                result.distance = depth_value;

                // Calculate angle
                result.angle = atan2((result.cx - ppx), fx);
            }
            catch (...)
            {
                result.distance = 0.0f;
            }
        }
    }

    return result;
}

LineDetection detect_two_white_lines(const Mat &mask, const FSMConfig &config)
{
    LineDetection result;

    // Find contours in the black line mask
    std::vector<std::vector<Point>> contours;
    findContours(mask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    // Find the best (largest) black line contour to follow
    int best_contour_idx = -1;
    double best_area = 0;
    Point best_centroid;

    for (size_t i = 0; i < contours.size(); i++)
    {
        double area = contourArea(contours[i]);
        if (area < 50)
            continue; // Filter small noise

        RotatedRect rect = minAreaRect(contours[i]);
        if (rect.center.y <= ROI_TOP)
            continue; // Must be in ROI

        // Find the largest contour (main line to follow)
        if (area > best_area)
        {
            best_area = area;
            best_contour_idx = i;

            // Calculate centroid of this contour
            Moments M = moments(contours[i]);
            if (M.m00 >= 1e-3)
            {
                best_centroid.x = int(M.m10 / M.m00);
                best_centroid.y = int(M.m01 / M.m00);
            }
        }
    }

    // If we found a good black line contour to follow
    if (best_contour_idx >= 0 && best_area > 50)
    {
        result.found_lines = true;
        result.virtual_middle_line = best_centroid.x; // Use centroid x as target

        // Calculate how far off-center we are
        int center_x = WIDTH / 2;
        int error = result.virtual_middle_line - center_x;

        // Set correction flags based on how far off-center we are
        result.too_close_to_left = (error < -config.vml_left_threshold);  // Too far left
        result.too_close_to_right = (error > config.vml_right_threshold); // Too far right

        // Store the contour info for debugging
        result.left_line_x = best_centroid.x - 20;  // For visualization
        result.right_line_x = best_centroid.x + 20; // For visualization

        // Debug output
        static int debug_counter = 0;
        if (debug_counter % 30 == 0)
        {
            std::cout << "Following black line: Centroid=(" << best_centroid.x << "," << best_centroid.y
                      << ") Area=" << best_area
                      << " Error=" << error
                      << " Center=" << center_x << std::endl;
        }
        debug_counter++;
    }

    return result;
}

int main()
try
{
    rs2::pipeline pipe;
    rs2::config cfg;
    cfg.enable_stream(RS2_STREAM_COLOR, WIDTH, HEIGHT, RS2_FORMAT_BGR8, FPS);
    cfg.enable_stream(RS2_STREAM_DEPTH, WIDTH, HEIGHT, RS2_FORMAT_Z16, FPS);
    auto profile = pipe.start(cfg);

    // Get camera intrinsics and depth scale
    auto color_stream = profile.get_stream(RS2_STREAM_COLOR).as<rs2::video_stream_profile>();
    rs2_intrinsics intrinsics = color_stream.get_intrinsics();
    float fx = intrinsics.fx;
    float ppx = intrinsics.ppx;

    auto depth_sensor = profile.get_device().first<rs2::depth_sensor>();
    float depth_scale = depth_sensor.get_depth_scale();

    int centerX = WIDTH / 2;

    // Mở kết nối UART để điều khiển động cơ
    int uart_fd = open_uart();
    if (uart_fd < 0)
    {
        std::cerr << "Warning: UART not available. Running in simulation mode." << std::endl;
    }
    else
    {
        std::cout << "UART connected successfully." << std::endl;
        sleep(2); // Đợi Arduino khởi động
    }

    const double CONTROL_HZ = 50.0;
    const auto CONTROL_DT = std::chrono::milliseconds(20); // 50Hz = 20ms

    // ==================== FSM Variables ====================
    FSMConfig config;
    BallDetectionParams ball_params;
    RobotState current_state = RobotState::GO_STRAIGHT_1;
    auto state_start_time = std::chrono::steady_clock::now();
    auto correction_start_time = std::chrono::steady_clock::now();
    bool in_correction = false;
    bool correcting_right_wheel = false; // true = reducing right wheel, false = reducing left wheel

    // Ball tracking variables
    int balls_detected = 0;
    int balls_captured = 0;
    bool ball_in_sight = false;
    auto ball_lost_time = std::chrono::steady_clock::now();
    const auto BALL_CAPTURE_DELAY = std::chrono::milliseconds(500);

    std::cout << "FSM Line Following Robot with Ball Collection Started!" << std::endl;
    std::cout << "Press [ESC] to quit.\n";

    auto next_tick = std::chrono::steady_clock::now();

    while (true)
    {
        next_tick += CONTROL_DT;
        auto current_time = std::chrono::steady_clock::now();

        // Calculate state duration
        auto state_duration = std::chrono::duration<float>(current_time - state_start_time).count();
        auto correction_duration = std::chrono::duration<float>(current_time - correction_start_time).count();

        rs2::frameset fs;
        try
        {
            fs = pipe.wait_for_frames(50);
        }
        catch (...)
        {
            std::cout << "TIMEOUT; Continuing..." << std::endl;
            std::this_thread::sleep_until(next_tick);
            continue;
        }

        rs2::video_frame color = fs.get_color_frame();
        rs2::depth_frame depth = fs.get_depth_frame();
        if (!color)
        {
            std::this_thread::sleep_until(next_tick);
            continue;
        }

        Mat bgr(Size(WIDTH, HEIGHT), CV_8UC3, (void *)color.get_data(), Mat::AUTO_STEP);

        // Draw reference lines
        cv::line(bgr, Point(WIDTH / 2, 0), Point(WIDTH / 2, HEIGHT - 1), Scalar(255, 255, 0), 1, LINE_AA);
        cv::line(bgr, Point(0, ROI_TOP), Point(WIDTH - 1, ROI_TOP), Scalar(200, 200, 200), 1, LINE_AA);

        // ==================== Image Processing ====================
        Mat gray;
        cvtColor(bgr, gray, COLOR_BGR2GRAY);

        // Direct black line detection using threshold
        Mat black_mask;
        threshold(gray, black_mask, config.black_threshold, 255, THRESH_BINARY_INV); // Detect dark/black pixels

        // ROI filtering - only process bottom third of image
        Mat roi_mask = Mat::zeros(HEIGHT, WIDTH, CV_8U);
        roi_mask(Range(ROI_TOP, HEIGHT), Range::all()) = 255;
        bitwise_and(black_mask, roi_mask, black_mask);

        // Morphology to clean up noise and connect line segments
        Mat k1 = getStructuringElement(MORPH_RECT, Size(3, 3));
        morphologyEx(black_mask, black_mask, MORPH_CLOSE, k1, Point(-1, -1), 2);

        // Remove small noise
        Mat k2 = getStructuringElement(MORPH_RECT, Size(2, 2));
        morphologyEx(black_mask, black_mask, MORPH_OPEN, k2, Point(-1, -1), 1);

        Mat mask = black_mask.clone();

        // ==================== Line Detection ====================
        LineDetection lines = detect_two_white_lines(mask, config);

        // ==================== Ball Detection ====================
        BallDetectionResult ball_result = detect_ball(bgr, ball_params, fx, ppx, depth_scale, depth);
        balls_detected = ball_result.found ? 1 : 0;

        // Check if ball was captured (disappeared from view)
        if (ball_in_sight && !ball_result.found)
        {
            auto time_since_lost = std::chrono::duration_cast<std::chrono::milliseconds>(
                current_time - ball_lost_time);

            if (time_since_lost > BALL_CAPTURE_DELAY)
            {
                balls_captured++;
                std::cout << "Ball captured! Total: " << balls_captured << std::endl;
                ball_in_sight = false;

                // If all 3 balls captured, stop intake
                if (balls_captured >= 3 && uart_fd >= 0)
                {
                    intake_ball(uart_fd, 0);
                }
            }
        }
        else if (ball_result.found)
        {
            ball_in_sight = true;
            ball_lost_time = current_time;
        }

        // ==================== FSM State Machine ====================
        float vL = 0.0f, vR = 0.0f;
        std::string state_name = "UNKNOWN";
        bool should_transition = false;

        // Check for ball detection in all states except during turns
        if ((current_state == RobotState::GO_STRAIGHT_1 ||
             current_state == RobotState::GO_STRAIGHT_2 ||
             current_state == RobotState::GO_STRAIGHT_3) &&
            ball_result.found && balls_captured < 3)
        {
            current_state = RobotState::BALL_APPROACH;
            state_start_time = current_time;
            std::cout << "BALL DETECTED! Switching to APPROACH mode" << std::endl;
        }

        switch (current_state)
        {
        case RobotState::GO_STRAIGHT_1:
            state_name = "STRAIGHT_1";
            should_transition = (state_duration >= config.straight1_duration);

            if (lines.found_lines)
            {
                int center_x = WIDTH / 2;
                int error = lines.virtual_middle_line - center_x;

                vL = config.max_speed;
                vR = config.max_speed * 0.9f;
            }
            else
            {
                vL = config.max_speed;
                vR = config.max_speed * 0.9f;
            }

            if (should_transition)
            {
                current_state = RobotState::TURN_LEFT_90;
                state_start_time = current_time;
                in_correction = false;
            }
            break;

        case RobotState::TURN_LEFT_90:
            state_name = "TURN_LEFT_90";
            should_transition = (state_duration >= config.turn_left_duration);

            vL = config.turn_speed;
            vR = -config.turn_speed;

            if (should_transition)
            {
                current_state = RobotState::GO_STRAIGHT_2;
                state_start_time = current_time;
            }
            break;

        case RobotState::GO_STRAIGHT_2:
            state_name = "STRAIGHT_2";
            should_transition = (state_duration >= config.straight2_duration);

            if (lines.found_lines)
            {
                int center_x = WIDTH / 2;
                int error = lines.virtual_middle_line - center_x;

                vL = vR = config.max_speed;

                if (error > config.vml_right_threshold)
                {
                    vL = config.max_speed * config.speed_reduction;
                }
                else if (error < -config.vml_left_threshold)
                {
                    vR = config.max_speed * config.speed_reduction;
                }
            }
            else
            {
                vL = vR = config.max_speed;
            }

            if (should_transition)
            {
                current_state = RobotState::TURN_RIGHT_90;
                state_start_time = current_time;
                in_correction = false;
            }
            break;

        case RobotState::TURN_RIGHT_90:
            state_name = "TURN_RIGHT_90";
            should_transition = (state_duration >= config.turn_right_duration);

            vL = -config.turn_speed;
            vR = config.turn_speed;

            if (should_transition)
            {
                current_state = RobotState::GO_STRAIGHT_3;
                state_start_time = current_time;
            }
            break;

        case RobotState::GO_STRAIGHT_3:
            state_name = "STRAIGHT_3";
            should_transition = (state_duration >= config.straight3_duration);

            if (lines.found_lines)
            {
                int center_x = WIDTH / 2;
                int error = lines.virtual_middle_line - center_x;

                vL = vR = config.max_speed;

                if (error > config.vml_right_threshold)
                {
                    vL = config.max_speed * config.speed_reduction;
                }
                else if (error < -config.vml_left_threshold)
                {
                    vR = config.max_speed * config.speed_reduction;
                }
            }
            else
            {
                vL = vR = config.max_speed;
            }

            if (should_transition)
            {
                vL = vR = 0.0f;
            }
            break;

        case RobotState::BALL_APPROACH:
            state_name = "BALL_APPROACH";

            if (ball_result.found)
            {
                // Calculate steering based on ball position
                float error = (ball_result.cx - centerX) / static_cast<float>(centerX);
                vL = config.ball_approach_speed - config.ball_turn_gain * error;
                vR = config.ball_approach_speed + config.ball_turn_gain * error;

                // Start intake when close enough
                if (ball_result.distance > 0 && ball_result.distance <= ball_params.detection_distance && uart_fd >= 0)
                {
                    intake_ball(uart_fd, 200);
                }

                // Check if ball is captured (very close)
                if (ball_result.distance > 0 && ball_result.distance <= ball_params.capture_distance)
                {
                    current_state = RobotState::BALL_INTAKE;
                    state_start_time = current_time;
                }
            }
            else
            {
                // Ball lost, return to line following
                current_state = RobotState::GO_STRAIGHT_2;
                state_start_time = current_time;
                if (uart_fd >= 0)
                    intake_ball(uart_fd, 0);
            }
            break;

        case RobotState::BALL_INTAKE:
            state_name = "BALL_INTAKE";

            // Keep moving forward and intake for a short time
            vL = vR = 0.3f;
            if (uart_fd >= 0)
                intake_ball(uart_fd, 200);

            // Return to line following after short intake period
            if (state_duration > 1.0f)
            {
                current_state = RobotState::GO_STRAIGHT_2;
                state_start_time = current_time;
                if (uart_fd >= 0)
                    intake_ball(uart_fd, 0);
            }
            break;
        }

        // ==================== Visual Feedback ====================
        // Draw ball detection results
        if (ball_result.found)
        {
            circle(bgr, Point(ball_result.cx, ball_result.cy), ball_result.radius, Scalar(0, 255, 0), 3);
            circle(bgr, Point(ball_result.cx, ball_result.cy), 2, Scalar(0, 0, 255), 3);

            char ball_info[64];
            snprintf(ball_info, sizeof(ball_info), "Dist: %.2fm", ball_result.distance);
            putText(bgr, ball_info, Point(ball_result.cx + 10, ball_result.cy - 10),
                    FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 2);
        }

        // Display ball count information
        char ball_count_text[64];
        snprintf(ball_count_text, sizeof(ball_count_text), "%d Ball%s Detected",
                 balls_detected, balls_detected != 1 ? "s" : "");
        putText(bgr, ball_count_text, Point(10, 180), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 255, 0), 2);

        char captured_text[64];
        snprintf(captured_text, sizeof(captured_text), "Captured: %d/3 Balls", balls_captured);
        putText(bgr, captured_text, Point(10, 210), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 255, 255), 2);

        // Draw all detected black line contours for debugging
        std::vector<std::vector<Point>> debug_contours;
        findContours(mask, debug_contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

        // ... (rest of the visualization code remains the same)

        // Display status
        char status_buf[256];
        snprintf(status_buf, sizeof(status_buf), "STATE: %s | Time: %.1fs | vL=%.2f vR=%.2f",
                 state_name.c_str(), state_duration, vL, vR);
        putText(bgr, status_buf, Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 255, 255), 2);

        // ==================== Motor Control ====================
        if (uart_fd >= 0)
        {
            drive_motors(uart_fd, vL, vR);
        }
        else
        {
            std::cout << "[" << state_name << "] MOTOR L=" << vL << " R=" << vR << std::endl;
        }

        // ==================== Display ====================
        imshow("Robot View", bgr);
        imshow("Black Line Mask", mask);
        imshow("Gray", gray);

        if (waitKey(1) == 27) // ESC key
        {
            break;
        }

        std::this_thread::sleep_until(next_tick);
    }

    // Stop motors and close UART
    if (uart_fd >= 0)
    {
        drive_motors(uart_fd, 0.0f, 0.0f);
        intake_ball(uart_fd, 0);
        close(uart_fd);
    }

    pipe.stop();
    return 0;
}
catch (const rs2::error &e)
{
    std::cerr << "RealSense error: " << e.what() << std::endl;
    return REALSENSE_ERROR;
}
catch (const std::exception &e)
{
    std::cerr << "Exception: " << e.what() << std::endl;
    return STD_ERROR;
}
catch (...)
{
    std::cerr << "Unknown exception occurred!" << std::endl;
    return UNKNOWN_ERROR;
}