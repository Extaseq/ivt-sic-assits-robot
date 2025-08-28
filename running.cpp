#include <librealsense2/rs.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
#include <thread>
#include <fcntl.h>
#include <termios.h>
#include <unistd.h>
#include <vector>
#include <algorithm>

using namespace cv;
using namespace std;

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

void send_motor_command(int fd, const char *id, int speed)
{
    if (fd < 0)
        return;

    speed = std::max(-255, std::min(255, speed));

    char buf[32];
    int n = snprintf(buf, sizeof(buf), "%s %d\n", id, speed);

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
    send_motor_command(fd, "M1", static_cast<int>(left_speed * 255));
    send_motor_command(fd, "M2", static_cast<int>(right_speed * 255));
}

void set_intake(int fd, int power)
{
    send_motor_command(fd, "M3", power);
}

// ==================== Line Follower ====================
class LineFollower
{
public:
    LineFollower(float kp = 0.015f, float ki = 0.0f, float kd = 0.004f,
                 float base_speed = 0.25f, float dt = 0.02f)
        : kp_(kp), ki_(ki), kd_(kd), base_speed_(base_speed), dt_(dt) {}

    pair<float, float> update(int line_center_x, int image_center_x)
    {
        float error = static_cast<float>(line_center_x - image_center_x);
        integral_ += error * dt_;
        float derivative = (error - prev_error_) / dt_;

        float steering = kp_ * error + ki_ * integral_ + kd_ * derivative;
        steering = clamp(steering, -0.6f, 0.6f);

        float left_speed = base_speed_ - steering;
        float right_speed = base_speed_ + steering;

        left_speed = clamp(left_speed, -1.0f, 1.0f);
        right_speed = clamp(right_speed, -1.0f, 1.0f);

        prev_error_ = error;
        return {left_speed, right_speed};
    }

    void reset()
    {
        integral_ = 0.0f;
        prev_error_ = 0.0f;
    }

private:
    float clamp(float value, float min_val, float max_val)
    {
        return max(min_val, min(max_val, value));
    }

    float kp_, ki_, kd_;
    float base_speed_;
    float dt_;
    float integral_ = 0.0f;
    float prev_error_ = 0.0f;
};

// ==================== Ball Detection ====================
struct BallResult
{
    bool found = false;
    int center_x = 0;
    int center_y = 0;
    int radius = 0;
    float distance = 0.0f;
    float angle = 0.0f;
    float score = 0.0f;
};

BallResult detect_ball(const Mat &frame, const Mat &depth_frame,
                       float fx, float ppx, float depth_scale)
{
    BallResult result;

    // Convert to HSV for color detection
    Mat hsv;
    cvtColor(frame, hsv, COLOR_BGR2HSV);

    // Tennis ball color range (yellow-green)
    Scalar lower_bound(20, 80, 120);
    Scalar upper_bound(45, 255, 255);

    Mat mask;
    inRange(hsv, lower_bound, upper_bound, mask);

    // Morphological operations
    Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(5, 5));
    morphologyEx(mask, mask, MORPH_CLOSE, kernel);
    morphologyEx(mask, mask, MORPH_OPEN, kernel);

    // Find contours
    vector<vector<Point>> contours;
    findContours(mask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    if (!contours.empty())
    {
        // Find largest contour
        auto largest_contour = *max_element(contours.begin(), contours.end(),
                                            [](const vector<Point> &a, const vector<Point> &b)
                                            {
                                                return contourArea(a) < contourArea(b);
                                            });

        // Fit circle
        Point2f center;
        float radius;
        minEnclosingCircle(largest_contour, center, radius);

        if (radius > 10)
        {
            result.found = true;
            result.center_x = static_cast<int>(center.x);
            result.center_y = static_cast<int>(center.y);
            result.radius = static_cast<int>(radius);

            // Calculate distance from depth
            if (depth_frame.empty() == false)
            {
                uint16_t depth_value = depth_frame.at<uint16_t>(
                    static_cast<int>(center.y), static_cast<int>(center.x));
                result.distance = depth_value * depth_scale;

                // Calculate angle
                result.angle = atan2(center.x - ppx, fx);
            }

            // Simple scoring based on circularity
            double area = contourArea(largest_contour);
            double circularity = (4 * CV_PI * area) / (arcLength(largest_contour, true) * arcLength(largest_contour, true));
            result.score = static_cast<float>(circularity);
        }
    }

    return result;
}

// ==================== Main Application ====================
int main()
{
    // UART setup
    int uart_fd = open_uart("/dev/ttyACM0", 115200);

    if (uart_fd < 0)
    {
        std::cerr << "Warning: UART not available. Running in simulation mode." << std::endl;
    }
    else
    {
        std::cout << "UART connected successfully. Waiting for Arduino ready message..." << std::endl;

        // Đợi Arduino khởi động
        sleep(2);

        // Đọc message từ Arduino
        char buffer[256];
        ssize_t bytes_read = read(uart_fd, buffer, sizeof(buffer) - 1);
        if (bytes_read > 0)
        {
            buffer[bytes_read] = '\0';
            std::cout << "Arduino: " << buffer;
        }
    }

    // RealSense setup
    rs2::pipeline pipe;
    rs2::config cfg;
    const int WIDTH = 640, HEIGHT = 480, FPS = 30;

    cfg.enable_stream(RS2_STREAM_COLOR, WIDTH, HEIGHT, RS2_FORMAT_BGR8, FPS);
    cfg.enable_stream(RS2_STREAM_DEPTH, WIDTH, HEIGHT, RS2_FORMAT_Z16, FPS);

    auto profile = pipe.start(cfg);
    auto depth_sensor = profile.get_device().first<rs2::depth_sensor>();
    float depth_scale = depth_sensor.get_depth_scale();

    auto color_profile = profile.get_stream(RS2_STREAM_COLOR).as<rs2::video_stream_profile>();
    rs2_intrinsics intrinsics = color_profile.get_intrinsics();
    float fx = intrinsics.fx;
    float ppx = intrinsics.ppx;

    // Initialize components
    LineFollower follower(0.015f, 0.0f, 0.004f, 0.25f, 0.02f);
    const int ROI_TOP = HEIGHT * 2 / 3;
    const int IMAGE_CENTER_X = WIDTH / 2;

    // ==================== State Machine Parameters ====================
    enum State
    {
        FOLLOW_LINE,
        BALL_SEEN,
        APPROACH_BALL,
        CAPTURE_BALL, // Thêm state mới cho giai đoạn cuốn bóng
        INTAKE_STATE,
        RELOCK_LINE
    };

    // Thêm các parameters mới
    static const float APPROACH_SPEED = 0.4f;   // Tốc độ tiếp cận nhanh hơn
    static const float CAPTURE_SPEED = 0.3f;    // Tốc độ cuốn bóng
    static const float CAPTURE_DURATION = 1.5f; // Thời gian cuốn bóng (giây)
    static const float ROTATE_SPEED = 0.25f;    // Tốc độ quay khi mất bóng
    static const int CAPTURE_FRAMES = 30;
    State current_state = FOLLOW_LINE;

    int ball_detection_count = 0;
    const int BALL_CONFIRMATION_COUNT = 5;
    const float BALL_DETECTION_THRESHOLD = 0.6f;
    const float CAPTURE_DISTANCE = 0.3f;

    cout << "Starting robot application..." << endl;

    while (true)
    {
        rs2::frameset frames = pipe.wait_for_frames();
        rs2::frame color_frame = frames.get_color_frame();
        rs2::frame depth_frame = frames.get_depth_frame();

        Mat color_image(Size(WIDTH, HEIGHT), CV_8UC3, (void *)color_frame.get_data());
        Mat depth_image(Size(WIDTH, HEIGHT), CV_16UC1, (void *)depth_frame.get_data());

        // Line detection (blue line)
        Mat line_mask = Mat::zeros(HEIGHT, WIDTH, CV_8U);
        const int BLUE_THRESHOLD = 30;
        const int VALUE_THRESHOLD = 60;

        for (int y = ROI_TOP; y < HEIGHT; y++)
        {
            Vec3b *row = color_image.ptr<Vec3b>(y);
            uchar *mask_row = line_mask.ptr<uchar>(y);

            for (int x = 0; x < WIDTH; x++)
            {
                int blue = row[x][0], green = row[x][1], red = row[x][2];
                int max_value = max({blue, green, red});

                if (blue > green + BLUE_THRESHOLD && blue > red + BLUE_THRESHOLD &&
                    max_value >= VALUE_THRESHOLD)
                {
                    mask_row[x] = 255;
                }
            }
        }

        // Process line mask
        Mat kernel = getStructuringElement(MORPH_RECT, Size(5, 5));
        morphologyEx(line_mask, line_mask, MORPH_CLOSE, kernel, Point(-1, -1), 2);

        vector<vector<Point>> contours;
        findContours(line_mask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

        int line_center_x = -1;
        if (!contours.empty())
        {
            auto largest_contour = *max_element(contours.begin(), contours.end(),
                                                [](const vector<Point> &a, const vector<Point> &b)
                                                {
                                                    return contourArea(a) < contourArea(b);
                                                });

            Moments m = moments(largest_contour);
            if (m.m00 > 0)
            {
                line_center_x = static_cast<int>(m.m10 / m.m00);
            }
        }

        // Ball detection
        BallResult ball = detect_ball(color_image, depth_image, fx, ppx, depth_scale);

        // State machine logic
        float left_speed = 0.0f, right_speed = 0.0f;
        int intake_power = 0;

        float vL = 0.0f, vR = 0.0f;
        int intake_pwm = 0;
        static int capture_counter = 0;

        switch (st)
        {
        case FOLLOW_LINE:
        {
            if (cx_line >= 0)
            {
                line_ok_hold = std::min(LINE_OK_HOLD, line_ok_hold + 1);
                auto lr = follower.update(cx_line, centerX);
                vL = lr.first;
                vR = lr.second;
                intake_pwm = 0; // Tắt intake khi follow line
            }
            else
            {
                line_ok_hold = 0;
                vL = -0.18f;
                vR = 0.18f;
            }

            if (r.found && r.score > Smin && r.distance_m > 0.0f && r.distance_m < D_detect_max)
            {
                if (++seen_hold >= N_detect_hold)
                {
                    st = BALL_SEEN;
                    seen_hold = 0;
                }
            }
            else
            {
                seen_hold = 0;
            }
            break;
        }

        case BALL_SEEN:
        {
            // Chuyển ngay sang approach
            st = APPROACH_BALL;
            std::cout << "Ball detected! Switching to APPROACH_BALL" << std::endl;
            break;
        }

        case APPROACH_BALL:
        {
            if (r.found)
            {
                // Tiếp cận nhanh hơn với tốc độ cao hơn
                float omega = std::clamp(r.angle_rad / (25.0f * (float)M_PI / 180.0f), -1.0f, 1.0f);
                float fwd = std::clamp((r.distance_m - 0.2f) * 2.0f, 0.3f, APPROACH_SPEED);

                vL = std::clamp(fwd - 0.7f * omega, -0.8f, 0.8f);
                vR = std::clamp(fwd + 0.7f * omega, -0.8f, 0.8f);

                // Bật intake sớm hơn khi gần bóng
                if (r.distance_m < 0.5f)
                {
                    intake_pwm = INTAKE_PWM;
                }

                // Khi rất gần bóng, chuyển sang trạng thái cuốn
                if (r.distance_m < D_captured || r.very_close)
                {
                    st = CAPTURE_BALL;
                    capture_counter = 0;
                    std::cout << "Very close to ball! Switching to CAPTURE_BALL" << std::endl;
                }
            }
            else
            {
                // Quay tìm bóng với tốc độ cao hơn
                vL = -ROTATE_SPEED;
                vR = ROTATE_SPEED;
                intake_pwm = 0;

                // Nếu mất bóng quá lâu, quay lại tìm line
                if (++seen_hold > 45)
                {
                    st = RELOCK_LINE;
                    seen_hold = 0;
                }
            }
            break;
        }

        case CAPTURE_BALL:
        {
            // Di chuyển thẳng với tốc độ ổn định để cuốn bóng
            vL = CAPTURE_SPEED;
            vR = CAPTURE_SPEED;
            intake_pwm = INTAKE_PWM; // Bật intake mạnh

            capture_counter++;

            // Tiếp tục cuốn trong CAPTURE_FRAMES frames
            if (capture_counter >= CAPTURE_FRAMES)
            {
                st = INTAKE_STATE;
                std::cout << "Capture complete! Switching to INTAKE_STATE" << std::endl;
            }
            break;
        }

        case INTAKE_STATE:
        {
            // Tiếp tục cuốn thêm một chút
            vL = 0.2f;
            vR = 0.2f;
            intake_pwm = INTAKE_PWM;

            static auto capture_start = std::chrono::steady_clock::now();
            auto now = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - capture_start);

            if (elapsed.count() > CAPTURE_DURATION * 1000)
            {
                st = RELOCK_LINE;
                std::cout << "Intake complete! Returning to line." << std::endl;
            }
            break;
        }

        case RELOCK_LINE:
        {
            if (cx_line >= 0)
            {
                auto lr = follower.update(cx_line, centerX);
                vL = lr.first;
                vR = lr.second;

                if (++line_ok_hold >= LINE_OK_HOLD)
                {
                    st = FOLLOW_LINE;
                    line_ok_hold = 0;
                    std::cout << "Line reacquired! Switching to FOLLOW_LINE" << std::endl;
                }
            }
            else
            {
                vL = -0.2f;
                vR = 0.2f;
                line_ok_hold = 0;
            }
            intake_pwm = 0; // Tắt intake khi tìm line
            break;
        }
        }

        // Send commands to motors
        drive_motors(uart_fd, left_speed, right_speed);
        set_intake(uart_fd, intake_power);

        // Display for debugging
        Mat display = color_image.clone();
        line(display, Point(IMAGE_CENTER_X, 0), Point(IMAGE_CENTER_X, HEIGHT), Scalar(255, 255, 0), 2);
        line(display, Point(0, ROI_TOP), Point(WIDTH, ROI_TOP), Scalar(200, 200, 200), 2);

        if (line_center_x >= 0)
        {
            line(display, Point(line_center_x, ROI_TOP), Point(line_center_x, HEIGHT), Scalar(0, 255, 255), 3);
        }

        if (ball.found)
        {
            circle(display, Point(ball.center_x, ball.center_y), ball.radius, Scalar(0, 255, 0), 3);
            putText(display, format("Dist: %.2fm", ball.distance),
                    Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 255, 0), 2);
        }

        string state_names[] = {"FOLLOW_LINE", "APPROACH_BALL", "INTAKE_BALL", "RETURN_TO_LINE"};
        putText(display, "State: " + state_names[current_state],
                Point(10, 60), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(50, 220, 50), 2);

        putText(display, format("Motors: L=%.2f R=%.2f", left_speed, right_speed),
                Point(10, 90), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(50, 220, 50), 2);

        putText(display, format("Intake: %d", intake_power),
                Point(10, 120), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(50, 220, 50), 2);

        imshow("Robot View", display);

        if (waitKey(1) == 27)
        {
            break;
        }
    }

    // Cleanup
    drive_motors(uart_fd, 0.0f, 0.0f);
    set_intake(uart_fd, 0);

    if (uart_fd >= 0)
    {
        close(uart_fd);
    }

    pipe.stop();
    return 0;
}