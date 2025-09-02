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
    send_motor_command(fd, "M2", left_speed);   // M2 = bánh trái
    send_motor_command(fd, "M1", right_speed);  // M1 = bánh phải
    
    // Nếu cần đảo chiều, uncomment dòng dưới:
    // send_motor_command(fd, "M2", -left_speed);
    // send_motor_command(fd, "M1", -right_speed);
}

static const int WIDTH = 640, HEIGHT = 480, FPS = 30;
static const int ROI_TOP = HEIGHT * 2 / 3;

// ==================== FSM States ====================
enum class RobotState {
    GO_STRAIGHT_1,      // First straight segment
    TURN_LEFT_90,       // Turn left 90 degrees
    GO_STRAIGHT_2,      // Second straight segment  
    TURN_RIGHT_90,      // Turn right 90 degrees
    GO_STRAIGHT_3       // Third straight segment
};

// ==================== FSM Configuration ====================
struct FSMConfig {
    // Time durations for each state (in seconds)
    float straight1_duration = 3.0f;    // First straight: 3 seconds
    float turn_left_duration = 1.5f;    // Turn left: 1.5 seconds
    float straight2_duration = 2.0f;    // Second straight: 2 seconds
    float turn_right_duration = 1.5f;   // Turn right: 1.5 seconds
    float straight3_duration = 3.0f;    // Third straight: 3 seconds
    
    // Speed settings
    float max_speed = 1.0f;              // Max speed (maps to 255 PWM)
    float turn_speed = 0.6f;             // Turn speed
    float speed_reduction = 0.7f;        // Speed when correcting (70% of max)
    float correction_duration = 0.5f;    // Correction duration in seconds
    
    // Line detection thresholds
    int black_threshold = 60;            // Gray values below this are considered black (0-255)
    int min_distance_to_line = 30;       // Minimum distance from VML to line edge (pixels)
    int vml_left_threshold = 50;         // Pixels from left edge to trigger correction
    int vml_right_threshold = 50;        // Pixels from right edge to trigger correction
};

// ==================== Line Detection for Two White Lines ====================
struct LineDetection {
    bool found_lines = false;
    int left_line_x = -1;
    int right_line_x = -1;
    int virtual_middle_line = -1;
    bool too_close_to_left = false;
    bool too_close_to_right = false;
};

LineDetection detect_two_white_lines(const Mat& mask, const FSMConfig& config) {
    LineDetection result;
    
    // Find contours in the black line mask
    std::vector<std::vector<Point>> contours;
    findContours(mask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    
    // Filter contours to find black line segments
    std::vector<std::pair<int, std::vector<Point>>> valid_lines;
    
    for (const auto& contour : contours) {
        double area = contourArea(contour);
        if (area < 30 || area > 5000) continue; // Filter by area for black lines
        
        RotatedRect rect = minAreaRect(contour);
        float aspect_ratio = std::max(rect.size.width, rect.size.height) / 
                           std::min(rect.size.width, rect.size.height);
        
        // Accept line-like contours (black lines should be elongated)
        if (aspect_ratio > 2.0f && rect.center.y > ROI_TOP) {
            Moments M = moments(contour);
            if (M.m00 >= 1e-3) {
                int cx = int(M.m10 / M.m00);
                valid_lines.push_back({cx, contour});
            }
        }
    }
    
    // Sort by x position
    std::sort(valid_lines.begin(), valid_lines.end(), 
              [](const auto& a, const auto& b) { return a.first < b.first; });
    
    // Look for at least 2 black lines (left and right borders)
    if (valid_lines.size() >= 2) {
        result.found_lines = true;
        result.left_line_x = valid_lines[0].first;          // Left black line
        result.right_line_x = valid_lines[valid_lines.size()-1].first; // Right black line
        
        // Calculate virtual middle line between the two black lines
        result.virtual_middle_line = (result.left_line_x + result.right_line_x) / 2;
        
        // Check if VML is too close to either black line
        int distance_to_left = result.virtual_middle_line - result.left_line_x;
        int distance_to_right = result.right_line_x - result.virtual_middle_line;
        
        result.too_close_to_left = distance_to_left < config.vml_left_threshold;
        result.too_close_to_right = distance_to_right < config.vml_right_threshold;
        
        // Debug output for black line detection
        static int debug_counter = 0;
        if (debug_counter % 30 == 0) {
            std::cout << "Black lines detected: L=" << result.left_line_x 
                      << " R=" << result.right_line_x 
                      << " VML=" << result.virtual_middle_line
                      << " DistL=" << distance_to_left
                      << " DistR=" << distance_to_right 
                      << " Total black lines=" << valid_lines.size() << std::endl;
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
    auto profile = pipe.start(cfg);

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
    RobotState current_state = RobotState::GO_STRAIGHT_1;
    auto state_start_time = std::chrono::steady_clock::now();
    auto correction_start_time = std::chrono::steady_clock::now();
    bool in_correction = false;
    bool correcting_right_wheel = false; // true = reducing right wheel, false = reducing left wheel

    std::cout << "FSM Line Following Robot Started!" << std::endl;
    std::cout << "States: STRAIGHT1 -> TURN_LEFT -> STRAIGHT2 -> TURN_RIGHT -> STRAIGHT3" << std::endl;
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

        // ==================== FSM State Machine ====================
        float vL = 0.0f, vR = 0.0f;
        std::string state_name = "UNKNOWN";
        bool should_transition = false;

        switch (current_state) {
            case RobotState::GO_STRAIGHT_1:
                state_name = "STRAIGHT_1";
                should_transition = (state_duration >= config.straight1_duration);
                
                if (lines.found_lines) {
                    // Default: max speed forward
                    vL = vR = config.max_speed;
                    
                    // Check for correction needs
                    if (!in_correction) {
                        if (lines.too_close_to_left) {
                            // VML too close to left line - reduce right wheel speed
                            in_correction = true;
                            correcting_right_wheel = true;
                            correction_start_time = current_time;
                            std::cout << "CORRECTION: VML too close to LEFT line - reducing RIGHT wheel" << std::endl;
                        }
                        else if (lines.too_close_to_right) {
                            // VML too close to right line - reduce left wheel speed
                            in_correction = true;
                            correcting_right_wheel = false;
                            correction_start_time = current_time;
                            std::cout << "CORRECTION: VML too close to RIGHT line - reducing LEFT wheel" << std::endl;
                        }
                    }
                    
                    // Apply correction if active
                    if (in_correction) {
                        if (correction_duration < config.correction_duration) {
                            if (correcting_right_wheel) {
                                vR = config.max_speed * config.speed_reduction;
                            } else {
                                vL = config.max_speed * config.speed_reduction;
                            }
                        } else {
                            // Correction time expired - return to normal
                            in_correction = false;
                            std::cout << "CORRECTION: Returning to normal speed" << std::endl;
                        }
                    }
                } else {
                    // No lines detected - just go straight
                    vL = vR = config.max_speed;
                }
                
                if (should_transition) {
                    current_state = RobotState::TURN_LEFT_90;
                    state_start_time = current_time;
                    in_correction = false;
                    std::cout << "STATE TRANSITION: STRAIGHT_1 -> TURN_LEFT_90" << std::endl;
                }
                break;

            case RobotState::TURN_LEFT_90:
                state_name = "TURN_LEFT_90";
                should_transition = (state_duration >= config.turn_left_duration);
                
                // Turn left: left wheel slower, right wheel faster
                vL = -config.turn_speed;
                vR = config.turn_speed;
                
                if (should_transition) {
                    current_state = RobotState::GO_STRAIGHT_2;
                    state_start_time = current_time;
                    std::cout << "STATE TRANSITION: TURN_LEFT_90 -> STRAIGHT_2" << std::endl;
                }
                break;

            case RobotState::GO_STRAIGHT_2:
                state_name = "STRAIGHT_2";
                should_transition = (state_duration >= config.straight2_duration);
                
                // Same line following logic as STRAIGHT_1
                if (lines.found_lines) {
                    vL = vR = config.max_speed;
                    
                    if (!in_correction) {
                        if (lines.too_close_to_left) {
                            in_correction = true;
                            correcting_right_wheel = true;
                            correction_start_time = current_time;
                            std::cout << "CORRECTION: VML too close to LEFT line - reducing RIGHT wheel" << std::endl;
                        }
                        else if (lines.too_close_to_right) {
                            in_correction = true;
                            correcting_right_wheel = false;
                            correction_start_time = current_time;
                            std::cout << "CORRECTION: VML too close to RIGHT line - reducing LEFT wheel" << std::endl;
                        }
                    }
                    
                    if (in_correction) {
                        if (correction_duration < config.correction_duration) {
                            if (correcting_right_wheel) {
                                vR = config.max_speed * config.speed_reduction;
                            } else {
                                vL = config.max_speed * config.speed_reduction;
                            }
                        } else {
                            in_correction = false;
                            std::cout << "CORRECTION: Returning to normal speed" << std::endl;
                        }
                    }
                } else {
                    vL = vR = config.max_speed;
                }
                
                if (should_transition) {
                    current_state = RobotState::TURN_RIGHT_90;
                    state_start_time = current_time;
                    in_correction = false;
                    std::cout << "STATE TRANSITION: STRAIGHT_2 -> TURN_RIGHT_90" << std::endl;
                }
                break;

            case RobotState::TURN_RIGHT_90:
                state_name = "TURN_RIGHT_90";
                should_transition = (state_duration >= config.turn_right_duration);
                
                // Turn right: left wheel faster, right wheel slower
                vL = config.turn_speed;
                vR = -config.turn_speed;
                
                if (should_transition) {
                    current_state = RobotState::GO_STRAIGHT_3;
                    state_start_time = current_time;
                    std::cout << "STATE TRANSITION: TURN_RIGHT_90 -> STRAIGHT_3" << std::endl;
                }
                break;

            case RobotState::GO_STRAIGHT_3:
                state_name = "STRAIGHT_3";
                should_transition = (state_duration >= config.straight3_duration);
                
                // Same line following logic
                if (lines.found_lines) {
                    vL = vR = config.max_speed;
                    
                    if (!in_correction) {
                        if (lines.too_close_to_left) {
                            in_correction = true;
                            correcting_right_wheel = true;
                            correction_start_time = current_time;
                            std::cout << "CORRECTION: VML too close to LEFT line - reducing RIGHT wheel" << std::endl;
                        }
                        else if (lines.too_close_to_right) {
                            in_correction = true;
                            correcting_right_wheel = false;
                            correction_start_time = current_time;
                            std::cout << "CORRECTION: VML too close to RIGHT line - reducing LEFT wheel" << std::endl;
                        }
                    }
                    
                    if (in_correction) {
                        if (correction_duration < config.correction_duration) {
                            if (correcting_right_wheel) {
                                vR = config.max_speed * config.speed_reduction;
                            } else {
                                vL = config.max_speed * config.speed_reduction;
                            }
                        } else {
                            in_correction = false;
                            std::cout << "CORRECTION: Returning to normal speed" << std::endl;
                        }
                    }
                } else {
                    vL = vR = config.max_speed;
                }
                
                if (should_transition) {
                    // Mission complete - stop or restart
                    vL = vR = 0.0f;
                    std::cout << "MISSION COMPLETE! Stopping robot." << std::endl;
                }
                break;
        }

        // ==================== Visual Feedback ====================
        
        // Draw all detected black line contours for debugging
        std::vector<std::vector<Point>> debug_contours;
        findContours(mask, debug_contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
        
        // Draw all contours in blue for visualization
        for (size_t i = 0; i < debug_contours.size(); i++) {
            double area = contourArea(debug_contours[i]);
            if (area > 30) { // Only draw significant contours
                drawContours(bgr, debug_contours, (int)i, Scalar(255, 0, 0), 2); // Blue contours
                
                // Draw centroid of each contour
                Moments M = moments(debug_contours[i]);
                if (M.m00 >= 1e-3) {
                    int cx = int(M.m10 / M.m00);
                    int cy = int(M.m01 / M.m00);
                    circle(bgr, Point(cx, cy), 5, Scalar(255, 0, 0), -1); // Blue dot
                }
            }
        }
        
        if (lines.found_lines) {
            // Draw detected black lines in green
            cv::line(bgr, Point(lines.left_line_x, ROI_TOP), Point(lines.left_line_x, HEIGHT-1), Scalar(0, 255, 0), 3);
            cv::line(bgr, Point(lines.right_line_x, ROI_TOP), Point(lines.right_line_x, HEIGHT-1), Scalar(0, 255, 0), 3);
            
            // Draw virtual middle line (between black lines) in magenta
            cv::line(bgr, Point(lines.virtual_middle_line, ROI_TOP), Point(lines.virtual_middle_line, HEIGHT-1), Scalar(255, 0, 255), 3);
            
            // Label the lines
            putText(bgr, "LEFT BLACK", Point(lines.left_line_x - 30, ROI_TOP - 10), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 2);
            putText(bgr, "RIGHT BLACK", Point(lines.right_line_x - 30, ROI_TOP - 10), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 2);
            putText(bgr, "VML", Point(lines.virtual_middle_line - 15, ROI_TOP - 30), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 0, 255), 2);
            
            // Draw warning indicators
            if (lines.too_close_to_left) {
                putText(bgr, "TOO CLOSE TO LEFT BLACK LINE", Point(10, 60), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 0, 255), 2);
            }
            if (lines.too_close_to_right) {
                putText(bgr, "TOO CLOSE TO RIGHT BLACK LINE", Point(10, 90), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 0, 255), 2);
            }
        } else {
            // No black lines detected
            putText(bgr, "NO BLACK LINES DETECTED", Point(10, 60), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 0, 255), 2);
        }

        // Display status
        char status_buf[256];
        snprintf(status_buf, sizeof(status_buf), "STATE: %s | Time: %.1fs | vL=%.2f vR=%.2f", 
                state_name.c_str(), state_duration, vL, vR);
        putText(bgr, status_buf, Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 255, 255), 2);

        if (in_correction) {
            char corr_buf[128];
            snprintf(corr_buf, sizeof(corr_buf), "CORRECTING: %s wheel for %.1fs", 
                    correcting_right_wheel ? "RIGHT" : "LEFT", correction_duration);
            putText(bgr, corr_buf, Point(10, 120), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 255), 2);
        }

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
        imshow("Black Line Mask", mask);    // Show detected black pixels
        imshow("Gray", gray);               // Show grayscale image

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
        close(uart_fd);
        std::cout << "Motors stopped and UART closed." << std::endl;
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
