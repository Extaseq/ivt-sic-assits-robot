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
    send_motor_command(fd, "M1", left_speed);   // M1 = bánh trái
    send_motor_command(fd, "M2", right_speed);  // M2 = bánh phải

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
    float straight1_duration = 8.0f;    // First straight: 8 seconds
    float turn_left_duration = 1.0f;    // Turn left: 1 second
    float straight2_duration = 10.0f;    // Second straight: 10 seconds
    float turn_right_duration = 1.0f;   // Turn right: 1 second
    float straight3_duration = 2.0f;    // Third straight: 30 seconds

    // Speed settings
    float max_speed = 0.7f;              // Max speed (maps to 255 PWM)
    float turn_speed = 0.6f;             // Turn speed
    float speed_reduction = 0.7f;        // Speed when correcting (70% of max)
    float correction_duration = 0.5f;    // Correction duration in seconds
    
    // Line detection thresholds
    int black_threshold = 60;            // Gray values below this are considered black (0-255)
    int min_distance_to_line = 30;       // Minimum distance from VML to line edge (pixels)
    int vml_left_threshold = 50;         // Pixels from left edge to trigger correction
    int vml_right_threshold = 50;        // Pixels from right edge to trigger correction
};

// ==================== Line Detection for Black Line Following ====================
struct LineDetection {
    bool found_lines = false;
    int left_line_x = -1;           // Not used in centroid following
    int right_line_x = -1;          // Not used in centroid following  
    int virtual_middle_line = -1;   // Now stores the centroid X position of target black line
    bool too_close_to_left = false; // Robot is too far left (line center is left of robot center)
    bool too_close_to_right = false;// Robot is too far right (line center is right of robot center)
};

LineDetection detect_two_white_lines(const Mat& mask, const FSMConfig& config) {
    LineDetection result;
    
    // Find contours in the black line mask
    std::vector<std::vector<Point>> contours;
    findContours(mask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    
    // Find the best (largest) black line contour to follow
    int best_contour_idx = -1;
    double best_area = 0;
    Point best_centroid;
    
    for (size_t i = 0; i < contours.size(); i++) {
        double area = contourArea(contours[i]);
        if (area < 50) continue; // Filter small noise
        
        RotatedRect rect = minAreaRect(contours[i]);
        if (rect.center.y <= ROI_TOP) continue; // Must be in ROI
        
        // Find the largest contour (main line to follow)
        if (area > best_area) {
            best_area = area;
            best_contour_idx = i;
            
            // Calculate centroid of this contour
            Moments M = moments(contours[i]);
            if (M.m00 >= 1e-3) {
                best_centroid.x = int(M.m10 / M.m00);
                best_centroid.y = int(M.m01 / M.m00);
            }
        }
    }
    
    // If we found a good black line contour to follow
    if (best_contour_idx >= 0 && best_area > 50) {
        result.found_lines = true;
        result.virtual_middle_line = best_centroid.x; // Use centroid x as target
        
        // Calculate how far off-center we are
        int center_x = WIDTH / 2;
        int error = result.virtual_middle_line - center_x;
        
        // Set correction flags based on how far off-center we are
        result.too_close_to_left = (error < -config.vml_left_threshold);   // Too far left
        result.too_close_to_right = (error > config.vml_right_threshold);  // Too far right
        
        // Store the contour info for debugging
        result.left_line_x = best_centroid.x - 20;   // For visualization
        result.right_line_x = best_centroid.x + 20;  // For visualization
        
        // Debug output
        static int debug_counter = 0;
        if (debug_counter % 30 == 0) {
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
                    // Calculate error: positive = line is right of center, negative = line is left of center
                    int center_x = WIDTH / 2;
                    int error = lines.virtual_middle_line - center_x;
                    
                    // Base speed
                    vL = vR = config.max_speed;
                    
                    // Apply proportional steering correction
                    float steering_gain = 0.003f; // Adjust this value to tune response
                    float correction = error * steering_gain;
                    
                    // Apply correction to motor speeds
                    if (error > config.vml_right_threshold) {
                        // Line is too far right - turn right (reduce left wheel)
                        vL = config.max_speed * config.speed_reduction;
                        std::cout << "TURNING RIGHT: Line at " << lines.virtual_middle_line << ", error=" << error << std::endl;
                    }
                    else if (error < -config.vml_left_threshold) {
                        // Line is too far left - turn left (reduce right wheel)  
                        vR = config.max_speed * config.speed_reduction;
                        std::cout << "TURNING LEFT: Line at " << lines.virtual_middle_line << ", error=" << error << std::endl;
                    }
                    else {
                        // Line is close to center - go straight
                        std::cout << "GOING STRAIGHT: Line at " << lines.virtual_middle_line << ", error=" << error << std::endl;
                    }
                } else {
                    // No lines detected - just go straight
                    vL = vR = config.max_speed;
                    std::cout << "NO LINE DETECTED - GOING STRAIGHT" << std::endl;
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
                
                // FIXED: Swap the motor speeds to actually turn left
                vL = config.turn_speed;   // Left wheel forward
                vR = -config.turn_speed;  // Right wheel backward
                
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
                    int center_x = WIDTH / 2;
                    int error = lines.virtual_middle_line - center_x;
                    
                    vL = vR = config.max_speed;
                    
                    if (error > config.vml_right_threshold) {
                        vL = config.max_speed * config.speed_reduction;
                        std::cout << "TURNING RIGHT: Line at " << lines.virtual_middle_line << ", error=" << error << std::endl;
                    }
                    else if (error < -config.vml_left_threshold) {
                        vR = config.max_speed * config.speed_reduction;
                        std::cout << "TURNING LEFT: Line at " << lines.virtual_middle_line << ", error=" << error << std::endl;
                    }
                    else {
                        std::cout << "GOING STRAIGHT: Line at " << lines.virtual_middle_line << ", error=" << error << std::endl;
                    }
                } else {
                    vL = vR = config.max_speed;
                    std::cout << "NO LINE DETECTED - GOING STRAIGHT" << std::endl;
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
                
                // FIXED: Swap the motor speeds to actually turn right
                vL = -config.turn_speed;  // Left wheel backward
                vR = config.turn_speed;   // Right wheel forward
                
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
                    int center_x = WIDTH / 2;
                    int error = lines.virtual_middle_line - center_x;
                    
                    vL = vR = config.max_speed;
                    
                    if (error > config.vml_right_threshold) {
                        vL = config.max_speed * config.speed_reduction;
                        std::cout << "TURNING RIGHT: Line at " << lines.virtual_middle_line << ", error=" << error << std::endl;
                    }
                    else if (error < -config.vml_left_threshold) {
                        vR = config.max_speed * config.speed_reduction;
                        std::cout << "TURNING LEFT: Line at " << lines.virtual_middle_line << ", error=" << error << std::endl;
                    }
                    else {
                        std::cout << "GOING STRAIGHT: Line at " << lines.virtual_middle_line << ", error=" << error << std::endl;
                    }
                } else {
                    vL = vR = config.max_speed;
                    std::cout << "NO LINE DETECTED - GOING STRAIGHT" << std::endl;
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
        
        // Find the best contour again for visualization
        int best_idx = -1;
        double best_area = 0;
        Point best_centroid;
        
        // Draw all contours in blue for visualization
        for (size_t i = 0; i < debug_contours.size(); i++) {
            double area = contourArea(debug_contours[i]);
            if (area > 30) { // Only draw significant contours
                bool is_best = false;
                
                // Check if this is the best (largest) contour
                if (area > best_area) {
                    RotatedRect rect = minAreaRect(debug_contours[i]);
                    if (rect.center.y > ROI_TOP) { // Must be in ROI
                        best_area = area;
                        best_idx = i;
                        
                        Moments M = moments(debug_contours[i]);
                        if (M.m00 >= 1e-3) {
                            best_centroid.x = int(M.m10 / M.m00);
                            best_centroid.y = int(M.m01 / M.m00);
                            is_best = true;
                        }
                    }
                }
                
                // Draw contour - red if it's the target, blue otherwise
                Scalar contour_color = is_best ? Scalar(0, 0, 255) : Scalar(255, 0, 0); // Red for target, blue for others
                drawContours(bgr, debug_contours, (int)i, contour_color, 2);
                
                // Draw centroid of each contour
                Moments M = moments(debug_contours[i]);
                if (M.m00 >= 1e-3) {
                    int cx = int(M.m10 / M.m00);
                    int cy = int(M.m01 / M.m00);
                    
                    // Larger red circle for target, smaller blue for others
                    if (is_best) {
                        circle(bgr, Point(cx, cy), 8, Scalar(0, 0, 255), -1); // Large red dot for target
                        putText(bgr, "TARGET", Point(cx + 15, cy - 10), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255), 2);
                    } else {
                        circle(bgr, Point(cx, cy), 5, Scalar(255, 0, 0), -1); // Small blue dot
                    }
                    
                    // Label with area
                    char area_text[32];
                    snprintf(area_text, sizeof(area_text), "%.0f", area);
                    putText(bgr, area_text, Point(cx + 10, cy + 15), FONT_HERSHEY_SIMPLEX, 0.4, Scalar(255, 255, 0), 1);
                }
            }
        }
        
        if (lines.found_lines) {
            // Draw the target line that robot is following (vertical line through centroid)
            cv::line(bgr, Point(lines.virtual_middle_line, ROI_TOP), Point(lines.virtual_middle_line, HEIGHT-1), Scalar(0, 255, 0), 4);
            
            // Label the target line
            putText(bgr, "FOLLOWING LINE", Point(lines.virtual_middle_line - 50, ROI_TOP - 10), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 255, 0), 2);
            
            // Draw warning indicators based on how far off-center the target is
            int center_x = WIDTH / 2;
            int error = lines.virtual_middle_line - center_x;
            
            if (lines.too_close_to_left) {
                putText(bgr, "ROBOT TOO FAR LEFT - TURN RIGHT", Point(10, 60), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 0, 255), 2);
            }
            if (lines.too_close_to_right) {
                putText(bgr, "ROBOT TOO FAR RIGHT - TURN LEFT", Point(10, 90), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 0, 255), 2);
            }
            
            // Show error value
            char error_text[64];
            snprintf(error_text, sizeof(error_text), "Error: %d pixels", error);
            putText(bgr, error_text, Point(10, 150), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 255, 0), 2);
            
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
