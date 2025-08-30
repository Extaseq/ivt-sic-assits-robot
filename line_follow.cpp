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

// Line #214ea3 ~ HSV(109, 203, 163)
Scalar LINE_LOW(100, 120, 80);
Scalar LINE_HIGH(120, 255, 255);

int main()
try
{
    rs2::pipeline pipe;
    rs2::config cfg;
    cfg.enable_stream(RS2_STREAM_COLOR, WIDTH, HEIGHT, RS2_FORMAT_BGR8, FPS);
    auto profile = pipe.start(cfg);

    int centerX = WIDTH / 2;
    LineFollower follower(0.015f, 0.0f, 0.004f, 0.25f, 0.02f); // dt = 20ms = 0.02s

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

        // ==== BGR-based mask: Blue-dominant ====
        const int T = 30;    // biên chênh: B phải lớn hơn R/G ít nhất T
        const int Vmin = 60; // ngưỡng “độ sáng” tối thiểu (max(B,G,R))

        Mat mask = Mat::zeros(HEIGHT, WIDTH, CV_8U);

        // Chỉ quét vùng ROI 1/3 dưới ảnh để giảm nhiễu & tiết kiệm CPU
        for (int y = ROI_TOP; y < HEIGHT; ++y)
        {
            const Vec3b *row = bgr.ptr<Vec3b>(y); // B,G,R
            uchar *mrow = mask.ptr<uchar>(y);
            for (int x = 0; x < WIDTH; ++x)
            {
                int B = row[x][0], G = row[x][1], R = row[x][2];
                int V = std::max({B, G, R});
                if (B > G + T && B > R + T && V >= Vmin)
                {
                    mrow[x] = 255;
                }
            }
        }

        // Lọc morphology như cũ (có thể giữ nguyên kernel 5x5 và 2 lần closing)
        Mat k = getStructuringElement(MORPH_RECT, Size(5, 5));
        morphologyEx(mask, mask, MORPH_CLOSE, k, Point(-1, -1), 2);

        std::vector<std::vector<Point>> contours;
        findContours(mask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

        float vL = 0.0f, vR = 0.0f;
        if (contours.empty())
        { // Lost line
            vL = -0.15f;
            vR = 0.15f;
            std::cout << "MISS; MOTOR " << vL << " " << vR << "\n";
        }
        else
        {
            auto cmax = *std::max_element(contours.begin(), contours.end(),
                                          [](auto &a, auto &b)
                                          { return contourArea(a) < contourArea(b); });
            Moments M = moments(cmax);
            if (M.m00 >= 1e-3)
            {
                int cx = int(M.m10 / M.m00);
                // ====== VẼ OVERLAY KHI BẮT ĐƯỢC LINE ======
                float err = float(cx - centerX);

                // vẽ contour lớn nhất
                drawContours(bgr, std::vector<std::vector<Point>>{cmax}, -1, Scalar(0, 255, 0), 2, LINE_AA);

                // vẽ đường dọc qua centroid trong ROI
                cv::line(bgr, Point(cx, ROI_TOP), Point(cx, HEIGHT - 1), Scalar(0, 255, 255), 2, LINE_AA);

                // đánh dấu centroid
                cv::circle(bgr, Point(cx, std::max(ROI_TOP + 1, HEIGHT - 5)), 4, Scalar(0, 0, 255), -1, LINE_AA);

                // vẽ minAreaRect (giúp nhìn hướng của vạch)
                RotatedRect rr = minAreaRect(cmax);
                Point2f verts[4];
                rr.points(verts);
                for (int i = 0; i < 4; ++i)
                    line(bgr, verts[i], verts[(i + 1) % 4], Scalar(0, 180, 255), 2, LINE_AA);

                // in thông tin lỗi & tốc độ
                char buf[128];
                snprintf(buf, sizeof(buf), "cx=%d  err=%d  vL=%.2f vR=%.2f", cx, int(err), vL, vR);
                putText(bgr, buf, Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(50, 220, 50), 2, LINE_AA);

                std::pair<float, float> motor_speeds = follower.update(cx, centerX);
                vL = motor_speeds.first;
                vR = motor_speeds.second;

                std::cout << "OK; ERR=" << int(err)
                          << " | MOTOR " << vL << " " << vR << "\n";
            }
            else
            {
                vL = -0.15f;
                vR = 0.15f;
                std::cout << "MISS(m00); MOTOR " << vL << " " << vR << "\n";
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