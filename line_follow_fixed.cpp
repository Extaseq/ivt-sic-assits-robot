#include <librealsense2/rs.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
#include <thread>
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

    // Convert speed from [-1, 1] to PWM value [0, 255]
    int pwm = static_cast<int>((speed + 1.0f) * 127.5f);
    pwm = std::max(0, std::min(255, pwm));

    char buf[32];
    int n = snprintf(buf, sizeof(buf), "%s %d\n", id, pwm);

    ssize_t bytes_written = write(fd, buf, n);
    if (bytes_written != n)
    {
        std::cerr << "UART write error: " << bytes_written << " bytes written, expected " << n << std::endl;
    }

    // Đảm bảo dữ liệu được gửi hoàn toàn
    tcdrain(fd);
    usleep(10000); // 10ms delay
}

void drive_motors(int fd, float left_speed, float right_speed)
{
    send_motor_command(fd, "M1", left_speed);
    send_motor_command(fd, "M2", right_speed);
}

static const int WIDTH = 640, HEIGHT = 480, FPS = 30;
static const int ROI_TOP = HEIGHT * 2 / 3;

int main()
try
{
    rs2::pipeline pipe;
    rs2::config cfg;
    cfg.enable_stream(RS2_STREAM_COLOR, WIDTH, HEIGHT, RS2_FORMAT_BGR8, FPS);
    auto profile = pipe.start(cfg);

    int centerX = WIDTH / 2;
    LineFollower follower(0.02f, 0.001f, 0.008f, 0.2f, 0.02f); // Improved PID

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
    const auto CONTROL_DT = std::chrono::milliseconds(100);

    // Line following state variables
    int line_lost_counter = 0;
    const int TURN_THRESHOLD = 15; // 15 frames = 0.3 seconds at 50Hz
    const float TURN_SPEED = 0.2f;
    int turn_direction = 1; // 1 = left, -1 = right (start with left turn)
    const int MAX_TURN_TIME = 100; // 100 frames = 2 seconds at 50Hz
    int turn_timer = 0;

    std::cout << "Press [ESC] to quit.\n";
    auto next_tick = std::chrono::steady_clock::now();

    while (true)
    {
        next_tick += CONTROL_DT;

        rs2::frameset fs;
        try
        {
            fs = pipe.wait_for_frames(50);
        }
        catch (...)
        {
            float vL = -0.12f, vR = 0.12f;
            std::cout << "TIMEOUT; MOTOR " << vL << " " << vR << "\n";
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
        
        // Đường tham chiếu: đường giữa ảnh & ranh giới ROI
        cv::line(bgr, Point(WIDTH / 2, 0), Point(WIDTH / 2, HEIGHT - 1), Scalar(255, 255, 0), 1, LINE_AA);
        cv::line(bgr, Point(0, ROI_TOP), Point(WIDTH - 1, ROI_TOP), Scalar(200, 200, 200), 1, LINE_AA);

        // ==== EDGE-BASED DETECTION thay vì color-based ====
        // Chuyển sang grayscale cho edge detection
        Mat gray;
        cvtColor(bgr, gray, COLOR_BGR2GRAY);
        
        // Method 1: Adaptive threshold để tìm vùng tối
        Mat thresh;
        adaptiveThreshold(gray, thresh, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY_INV, 15, 10);
        
        // Method 2: Canny edge detection
        Mat blurred;
        GaussianBlur(gray, blurred, Size(5, 5), 1.5);
        Mat edges;
        Canny(blurred, edges, 30, 90, 3);
        
        // Kết hợp adaptive threshold và edges
        Mat mask_combined;
        bitwise_or(thresh, edges, mask_combined);
        
        // Chỉ xét vùng ROI
        Mat roi_mask = Mat::zeros(HEIGHT, WIDTH, CV_8U);
        roi_mask(Range(ROI_TOP, HEIGHT), Range::all()) = 255;
        bitwise_and(mask_combined, roi_mask, mask_combined);
        
        // Morphology để tạo line liên tục
        Mat k1 = getStructuringElement(MORPH_RECT, Size(3, 3));
        morphologyEx(mask_combined, mask_combined, MORPH_CLOSE, k1, Point(-1, -1), 1);
        
        Mat k2 = getStructuringElement(MORPH_ELLIPSE, Size(5, 1)); // Kernel ngang để nối line
        morphologyEx(mask_combined, mask_combined, MORPH_OPEN, k2, Point(-1, -1), 1);
        
        Mat mask = mask_combined.clone();

        // DEBUG: In thông tin về detection methods
        static int debug_counter = 0;
        if (debug_counter % 30 == 0) {
            std::cout << "Adaptive thresh pixels: " << countNonZero(thresh) 
                      << " | Edge pixels: " << countNonZero(edges)
                      << " | Combined mask: " << countNonZero(mask) << std::endl;
        }
        debug_counter++;

        std::vector<std::vector<Point>> contours;
        findContours(mask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

        // Lọc contours theo hình dạng line và vị trí
        std::vector<std::vector<Point>> valid_contours;
        for (const auto& contour : contours) {
            double area = contourArea(contour);
            if (area < 30 || area > 3000) continue; // Mở rộng range cho edge detection
            
            RotatedRect rect = minAreaRect(contour);
            float aspect_ratio = std::max(rect.size.width, rect.size.height) / 
                               std::min(rect.size.width, rect.size.height);
            
            // Giảm yêu cầu aspect ratio vì edge có thể không liên tục
            if (aspect_ratio > 2.0f && rect.center.y > ROI_TOP) {
                // Thêm filter theo vị trí: ưu tiên contour gần center
                float distance_from_center = abs(rect.center.x - WIDTH/2);
                if (distance_from_center < WIDTH/3) { // Trong 1/3 giữa ảnh
                    valid_contours.push_back(contour);
                }
            }
        }
        
        contours = valid_contours;

        float vL = 0.0f, vR = 0.0f;
        
        if (contours.empty())
        { 
            // Lost line - turn logic
            line_lost_counter++;
            if (line_lost_counter >= TURN_THRESHOLD)
            {
                // Turn to search for line
                turn_timer++;
                if (turn_timer >= MAX_TURN_TIME)
                {
                    // Switch turn direction after max turn time
                    turn_direction = -turn_direction;
                    turn_timer = 0;
                    std::cout << "SWITCHING TURN DIRECTION TO " << (turn_direction > 0 ? "LEFT" : "RIGHT") << std::endl;
                }
                
                vL = -turn_direction * TURN_SPEED;
                vR = turn_direction * TURN_SPEED;
                std::cout << "SEARCHING; TURN=" << (turn_direction > 0 ? "LEFT" : "RIGHT")
                          << " TIMER=" << turn_timer << "/" << MAX_TURN_TIME
                          << " | MOTOR " << vL << " " << vR << std::endl;
            }
            else
            {
                // Brief pause before turning
                vL = 0.0f;
                vR = 0.0f;
                std::cout << "MISS; COUNTER=" << line_lost_counter << "/" << TURN_THRESHOLD
                          << " | MOTOR " << vL << " " << vR << std::endl;
            }
        }
        else
        {
            // Tìm contour tốt nhất (gần center và lớn)
            std::vector<Point> best_contour;
            float best_score = -1;
            
            for (const auto& contour : contours) {
                RotatedRect rect = minAreaRect(contour);
                float distance_from_center = abs(rect.center.x - centerX);
                float area = contourArea(contour);
                
                // Score = area / distance_from_center (ưu tiên gần center và lớn)
                float score = area / (distance_from_center + 1);
                
                if (score > best_score) {
                    best_score = score;
                    best_contour = contour;
                }
            }
            
            if (!best_contour.empty()) {
                Moments M = moments(best_contour);
                if (M.m00 >= 1e-3)
                {
                    int cx = int(M.m10 / M.m00);
                    int cy = int(M.m01 / M.m00);
                    
                    // ====== VẼ OVERLAY KHI BẮT ĐƯỢC LINE ======
                    float err = float(cx - centerX);

                    // Reset line lost counter when line is found
                    line_lost_counter = 0;
                    turn_timer = 0;

                    // vẽ contour được chọn
                    drawContours(bgr, std::vector<std::vector<Point>>{best_contour}, -1, Scalar(0, 255, 0), 2, LINE_AA);

                    // vẽ đường dọc qua centroid trong ROI
                    cv::line(bgr, Point(cx, ROI_TOP), Point(cx, HEIGHT - 1), Scalar(0, 255, 255), 2, LINE_AA);

                    // đánh dấu centroid
                    cv::circle(bgr, Point(cx, cy), 6, Scalar(0, 0, 255), -1, LINE_AA);
                    
                    // Vẽ text hiển thị error
                    char error_text[64];
                    snprintf(error_text, sizeof(error_text), "ERR: %d", (int)err);
                    putText(bgr, error_text, Point(cx + 10, cy - 10), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 0), 1, LINE_AA);

                    // Tính motor speeds từ PID controller
                    std::pair<float, float> motor_speeds = follower.update(cx, centerX);
                    vL = motor_speeds.first;
                    vR = motor_speeds.second;
                    
                    // Giới hạn tốc độ để robot không đi quá nhanh
                    float max_speed = 0.3f;
                    vL = std::max(-max_speed, std::min(max_speed, vL));
                    vR = std::max(-max_speed, std::min(max_speed, vR));

                    // in thông tin lỗi & tốc độ
                    char buf[128];
                    snprintf(buf, sizeof(buf), "cx=%d cy=%d err=%.1f vL=%.2f vR=%.2f", cx, cy, err, vL, vR);
                    putText(bgr, buf, Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(50, 220, 50), 2, LINE_AA);

                    std::cout << "FOLLOW; CX=" << cx << " ERR=" << int(err)
                              << " | MOTOR L=" << vL << " R=" << vR << std::endl;
                }
                else {
                    // Contour quá nhỏ
                    line_lost_counter++;
                    std::cout << "CONTOUR_SMALL; COUNTER=" << line_lost_counter << "/" << TURN_THRESHOLD << std::endl;
                }
            }
            else {
                // Không có contour phù hợp
                line_lost_counter++;
                std::cout << "NO_VALID_CONTOUR; COUNTER=" << line_lost_counter << "/" << TURN_THRESHOLD << std::endl;
            }
        }

        // Send vL and vR to motors
        if (uart_fd >= 0)
        {
            drive_motors(uart_fd, vL, vR);
        }
        else
        {
            // Simulation mode - just print values
            std::cout << "[SIMULATION] MOTOR " << vL << " " << vR << std::endl;
        }

        imshow("mask", mask);
        imshow("color", bgr);
        imshow("gray", gray);           // Để xem grayscale
        imshow("adaptive", thresh);     // Để xem adaptive threshold
        imshow("edges", edges);         // Để xem edge detection

        if (waitKey(1) == 27)
        {
            break;
        }

        std::this_thread::sleep_until(next_tick);
    }

    // Stop motors and close UART
    if (uart_fd >= 0)
    {
        drive_motors(uart_fd, 0.0f, 0.0f); // Stop motors
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
