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
#include <vector>

using namespace cv;

#define REALSENSE_ERROR 1
#define STD_ERROR 2
#define UNKNOWN_ERROR 3

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

    // Nếu cần đảo chiều, uncomment dòng dưới:
    // send_motor_command(fd, "M2", -left_speed);
    // send_motor_command(fd, "M1", -right_speed);
}

static const int WIDTH = 640, HEIGHT = 480, FPS = 30;
static const int ROI_TOP = HEIGHT * 2 / 3;

// ==================== Simple Ball Detection (inline) ====================
struct SimpleBallResult
{
    bool found = false;
    int cx = -1;
    int cy = -1;
    int radius_px = 0;
    float distance_m = 0.f;
};

struct SimpleBallDetectorParams
{
    cv::Scalar hsv_low{20, 80, 120};
    cv::Scalar hsv_high{45, 255, 255};
    int morph_kernel = 5;
    int open_iters = 1;
    int close_iters = 2;
    double min_area = 200.0;
};

class SimpleBallDetector
{
public:
    explicit SimpleBallDetector(const SimpleBallDetectorParams &p = {}) : p_(p) {}
    SimpleBallResult detect(const cv::Mat &bgr, const cv::Mat &depth16, float depth_scale)
    {
        SimpleBallResult r;
        if (bgr.empty())
            return r;

        cv::Mat hsv;
        cv::cvtColor(bgr, hsv, cv::COLOR_BGR2HSV);

        cv::Mat mask;
        cv::inRange(hsv, p_.hsv_low, p_.hsv_high, mask);

        cv::Mat k = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(p_.morph_kernel, p_.morph_kernel));
        if (p_.open_iters > 0)
            morphologyEx(mask, mask, cv::MORPH_OPEN, k, cv::Point(-1, -1), p_.open_iters);
        if (p_.close_iters > 0)
            morphologyEx(mask, mask, cv::MORPH_CLOSE, k, cv::Point(-1, -1), p_.close_iters);

        std::vector<std::vector<cv::Point>> cnts;
        cv::findContours(mask, cnts, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        double bestA = 0;
        int best = -1;
        cv::Point2f bc;
        float br = 0;
        for (int i = 0; i < (int)cnts.size(); ++i)
        {
            double a = cv::contourArea(cnts[i]);
            if (a < p_.min_area)
                continue;
            cv::Point2f c;
            float rad;
            cv::minEnclosingCircle(cnts[i], c, rad);
            if (rad < 5)
                continue;
            if (a > bestA)
            {
                bestA = a;
                best = i;
                bc = c;
                br = rad;
            }
        }

        if (best >= 0)
        {
            r.found = true;
            r.cx = (int)std::round(bc.x);
            r.cy = (int)std::round(bc.y);
            r.radius_px = (int)std::round(br);

            if (!depth16.empty() && r.cx >= 0 && r.cx < depth16.cols && r.cy >= 0 && r.cy < depth16.rows)
            {
                uint16_t d = depth16.at<uint16_t>(r.cy, r.cx);
                r.distance_m = d * depth_scale;
            }
        }

        last_mask_ = mask;
        return r;
    }
    const cv::Mat &lastMask() const { return last_mask_; }

private:
    SimpleBallDetectorParams p_;
    cv::Mat last_mask_;
};

// ==================== FSM States ====================
enum class RobotState
{
    GO_STRAIGHT_1, // First straight segment
    TURN_LEFT_90,  // Turn left 90 degrees
    GO_STRAIGHT_2, // Second straight segment
    TURN_RIGHT_90, // Turn right 90 degrees
    GO_STRAIGHT_3  // Third straight segment
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
    rs2::align align_to_color(RS2_STREAM_COLOR);
    auto depth_sensor = profile.get_device().first<rs2::depth_sensor>();
    float depth_scale = depth_sensor.get_depth_scale();

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

    const auto CONTROL_DT = std::chrono::milliseconds(20);
    auto next_tick = std::chrono::steady_clock::now();

    // ==================== FSM Variables ====================
    FSMConfig config;
    RobotState current_state = RobotState::GO_STRAIGHT_1;
    auto state_start_time = std::chrono::steady_clock::now();

    // Ball collection state
    SimpleBallDetector ball_detector;
    int balls_collected = 0;
    bool intake_on = false;
    const float ACTIVATE_DISTANCE = 3.0f;
    const float CAPTURE_DISTANCE = 0.40f;
    int capture_cooldown = 0;
    const int CAPTURE_COOLDOWN_FRAMES = 40; // ~0.8s

    while (true)
    {
        next_tick += CONTROL_DT;
        auto now = std::chrono::steady_clock::now();
        float state_duration = std::chrono::duration<float>(now - state_start_time).count();

        rs2::frameset fs;
        try
        {
            fs = pipe.wait_for_frames(50);
        }
        catch (...)
        {
            std::this_thread::sleep_until(next_tick);
            continue;
        }

        fs = align_to_color.process(fs);
        rs2::video_frame color = fs.get_color_frame();
        rs2::depth_frame depth = fs.get_depth_frame();
        if (!color)
        {
            std::this_thread::sleep_until(next_tick);
            continue;
        }

        Mat bgr(Size(WIDTH, HEIGHT), CV_8UC3, (void *)color.get_data(), Mat::AUTO_STEP);
        Mat depth16;
        if (depth)
            depth16 = Mat(Size(WIDTH, HEIGHT), CV_16UC1, (void *)depth.get_data(), Mat::AUTO_STEP);

        // Ball detection
        SimpleBallResult ball = ball_detector.detect(bgr, depth16, depth_scale);
        if (capture_cooldown > 0)
            capture_cooldown--;
        if (ball.found && ball.distance_m > 0 && ball.distance_m < CAPTURE_DISTANCE && capture_cooldown == 0 && balls_collected < 3)
        {
            balls_collected++;
            capture_cooldown = CAPTURE_COOLDOWN_FRAMES;
            std::cout << "Captured ball -> total=" << balls_collected << std::endl;
        }
        if (balls_collected < 3)
        {
            if (ball.found && ball.distance_m > 0 && ball.distance_m <= ACTIVATE_DISTANCE)
                intake_on = true;
        }
        else
            intake_on = false;

        // Line processing (only if still collecting)
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

        // ==================== FSM State Machine ====================
        float vL = 0.0f, vR = 0.0f;
        std::string state_name = "";
        bool handled = false;

        if (balls_collected < 3 && ball.found && ball.distance_m > 0 && ball.distance_m <= ACTIVATE_DISTANCE)
        {
            // Approach ball
            float err = (float)(ball.cx - WIDTH / 2);
            float steer = std::max(-0.4f, std::min(0.4f, err * 0.0025f));
            float base = 0.35f;
            if (ball.distance_m < 1.0f)
                base = 0.25f;
            if (ball.distance_m < 0.6f)
                base = 0.18f;
            vL = base - steer;
            vR = base + steer;
            if (ball.distance_m < 0.22f)
            {
                vL = vR = 0.f;
            }
            state_name = "APPROACH";
            handled = true;
        }
        if (!handled && balls_collected < 3)
        {
            switch (current_state)
            {
            case RobotState::GO_STRAIGHT_1:
                state_name = "S1";
                if (lines.found_lines)
                {
                    // Calculate error: positive = line is right of center, negative = line is left of center
                    int center_x = WIDTH / 2;
                    int error = lines.virtual_middle_line - center_x;

                    // Base speed
                    vL = config.max_speed;
                    vR = config.max_speed * 0.9f;

                    // Apply proportional steering correction
                    float steering_gain = 0.003f; // Adjust this value to tune response
                    float correction = error * steering_gain;

                    // // Apply correction to motor speeds
                    // if (error > config.vml_right_threshold) {
                    //     // Line is too far right - turn right (reduce left wheel)
                    //     vL = config.max_speed * config.speed_reduction;
                    //     std::cout << "TURNING RIGHT: Line at " << lines.virtual_middle_line << ", error=" << error << std::endl;
                    // }
                    // else if (error < -config.vml_left_threshold) {
                    //     // Line is too far left - turn left (reduce right wheel)
                    //     vR = config.max_speed * config.speed_reduction;
                    //     std::cout << "TURNING LEFT: Line at " << lines.virtual_middle_line << ", error=" << error << std::endl;
                    // }
                    // else {
                    //     // Line is close to center - go straight
                    //     std::cout << "GOING STRAIGHT: Line at " << lines.virtual_middle_line << ", error=" << error << std::endl;
                    // }
                }
                else
                {
                    // No lines detected - just go straight
                    vL = config.max_speed;
                    vR = config.max_speed * 0.9f;
                    std::cout << "NO LINE DETECTED - GOING STRAIGHT" << std::endl;
                }

                if (state_duration >= config.straight1_duration)
                {
                    current_state = RobotState::TURN_LEFT_90;
                    state_start_time = now;
                }
                break;

            case RobotState::TURN_LEFT_90:
                state_name = "TL";
                vL = config.turn_speed;  // Left wheel forward
                vR = -config.turn_speed; // Right wheel backward

                if (state_duration >= config.turn_left_duration)
                {
                    current_state = RobotState::GO_STRAIGHT_2;
                    state_start_time = now;
                }
                break;

            case RobotState::GO_STRAIGHT_2:
                state_name = "S2";
                if (lines.found_lines)
                {
                    int center_x = WIDTH / 2;
                    int error = lines.virtual_middle_line - center_x;

                    vL = vR = config.max_speed;

                    if (error > config.vml_right_threshold)
                    {
                        vL = config.max_speed * config.speed_reduction;
                        std::cout << "TURNING RIGHT: Line at " << lines.virtual_middle_line << ", error=" << error << std::endl;
                    }
                    else if (error < -config.vml_left_threshold)
                    {
                        vR = config.max_speed * config.speed_reduction;
                        std::cout << "TURNING LEFT: Line at " << lines.virtual_middle_line << ", error=" << error << std::endl;
                    }
                    else
                    {
                        std::cout << "GOING STRAIGHT: Line at " << lines.virtual_middle_line << ", error=" << error << std::endl;
                    }
                }
                else
                {
                    vL = vR = config.max_speed;
                    std::cout << "NO LINE DETECTED - GOING STRAIGHT" << std::endl;
                }

                if (state_duration >= config.straight2_duration)
                {
                    current_state = RobotState::TURN_RIGHT_90;
                    state_start_time = now;
                }
                break;

            case RobotState::TURN_RIGHT_90:
                state_name = "TR";
                vL = -config.turn_speed; // Left wheel backward
                vR = config.turn_speed;  // Right wheel forward

                if (state_duration >= config.turn_right_duration)
                {
                    current_state = RobotState::GO_STRAIGHT_3;
                    state_start_time = now;
                }
                break;

            case RobotState::GO_STRAIGHT_3:
                state_name = "S3";
                if (lines.found_lines)
                {
                    int center_x = WIDTH / 2;
                    int error = lines.virtual_middle_line - center_x;

                    vL = vR = config.max_speed;

                    if (error > config.vml_right_threshold)
                    {
                        vL = config.max_speed * config.speed_reduction;
                        std::cout << "TURNING RIGHT: Line at " << lines.virtual_middle_line << ", error=" << error << std::endl;
                    }
                    else if (error < -config.vml_left_threshold)
                    {
                        vR = config.max_speed * config.speed_reduction;
                        std::cout << "TURNING LEFT: Line at " << lines.virtual_middle_line << ", error=" << error << std::endl;
                    }
                    else
                    {
                        std::cout << "GOING STRAIGHT: Line at " << lines.virtual_middle_line << ", error=" << error << std::endl;
                    }
                }
                else
                {
                    vL = vR = config.max_speed;
                    std::cout << "NO LINE DETECTED - GOING STRAIGHT" << std::endl;
                }

                if (state_duration >= config.straight3_duration)
                {
                    vL = vR = 0;
                }
                break;
            }
        }
        if (balls_collected >= 3)
        {
            vL = vR = 0.f;
            state_name = "DONE";
        }

        // Draw ROI & center
        line(bgr, Point(WIDTH / 2, 0), Point(WIDTH / 2, HEIGHT - 1), Scalar(255, 255, 0), 1, LINE_AA);
        line(bgr, Point(0, ROI_TOP), Point(WIDTH - 1, ROI_TOP), Scalar(200, 200, 200), 1, LINE_AA);
        if (lines.found_lines)
            line(bgr, Point(lines.virtual_middle_line, ROI_TOP), Point(lines.virtual_middle_line, HEIGHT - 1), Scalar(0, 255, 0), 4);
        if (ball.found)
        {
            circle(bgr, Point(ball.cx, ball.cy), std::max(6, ball.radius_px), Scalar(0, 255, 0), 2);
        }

        // Two required lines
        char line1[64];
        snprintf(line1, sizeof(line1), "Balls: %d/3%s", balls_collected, balls_collected >= 3 ? " (STOP)" : "");
        char line2[64];
        if (ball.found && ball.distance_m > 0)
            snprintf(line2, sizeof(line2), "Dist: %.2fm", ball.distance_m);
        else
            snprintf(line2, sizeof(line2), "Dist: ---");
        putText(bgr, line1, Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(50, 220, 50), 2);
        putText(bgr, line2, Point(10, 65), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(255, 255, 0), 2);

        if (uart_fd >= 0)
        {
            drive_motors(uart_fd, vL, vR);
            send_motor_command(uart_fd, "M3", intake_on ? 1.0f : 0.0f);
        }
        else
        {
            std::cout << "STATE=" << state_name << " vL=" << vL << " vR=" << vR << " balls=" << balls_collected << " intake=" << intake_on << " dist=" << ball.distance_m << std::endl;
        }

        imshow("Robot View", bgr);
        if (!ball_detector.lastMask().empty())
            imshow("Ball Mask", ball_detector.lastMask());
        imshow("Black Line Mask", mask);

        if (waitKey(1) == 27)
            break;
        std::this_thread::sleep_until(next_tick);
    }

    // Stop motors and close UART
    if (uart_fd >= 0)
    {
        drive_motors(uart_fd, 0, 0);
        send_motor_command(uart_fd, "M3", 0);
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
