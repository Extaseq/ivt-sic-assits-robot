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
#include <cmath>

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
    LineFollower(float kp = 0.010f, float ki = 0.0f, float kd = 0.002f,
                 float base_speed = 0.25f, float dt = 0.02f, float max_steer_rate = 0.1f)
        : kp_(kp), ki_(ki), kd_(kd), base_speed_(base_speed), dt_(dt), max_steer_rate_(max_steer_rate) {}

    pair<float, float> update(float error_norm, bool line_detected)
    {
        if (!line_detected)
        {
            prev_steering_ = 0.0f;
            return {0.0f, 0.0f}; // No control when line not detected
        }

        // Reset PID when line is newly detected
        if (!prev_line_detected_)
        {
            integral_ = 0.0f;
            prev_error_ = error_norm;
            prev_line_detected_ = true;
        }

        integral_ += error_norm * dt_;
        float derivative = (error_norm - prev_error_) / dt_;

        float steering = kp_ * error_norm + ki_ * integral_ + kd_ * derivative;
        steering = clamp(steering, -0.6f, 0.6f);

        // Apply slew-rate limiting
        float steer_delta = steering - prev_steering_;
        steer_delta = clamp(steer_delta, -max_steer_rate_, max_steer_rate_);
        steering = prev_steering_ + steer_delta;
        steering = clamp(steering, -0.6f, 0.6f);

        float left_speed = base_speed_ - steering;
        float right_speed = base_speed_ + steering;

        left_speed = clamp(left_speed, -1.0f, 1.0f);
        right_speed = clamp(right_speed, -1.0f, 1.0f);

        prev_error_ = error_norm;
        prev_steering_ = steering;
        return {left_speed, right_speed};
    }

    void reset()
    {
        integral_ = 0.0f;
        prev_error_ = 0.0f;
        prev_steering_ = 0.0f;
        prev_line_detected_ = false;
    }

private:
    float clamp(float value, float min_val, float max_val)
    {
        return max(min_val, min(max_val, value));
    }

    float kp_, ki_, kd_;
    float base_speed_;
    float dt_;
    float max_steer_rate_;
    float integral_ = 0.0f;
    float prev_error_ = 0.0f;
    float prev_steering_ = 0.0f;
    bool prev_line_detected_ = false;
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

// ... (giữ nguyên phần include, UART, LineFollower, BallResult, detect_ball như bạn có)

int main()
{
    int uart_fd = open_uart("/dev/ttyACM0", 115200);
    if (uart_fd < 0)
    {
        std::cerr << "Warning: UART not available. Running in simulation mode." << std::endl;
    }
    else
    {
        std::cout << "UART connected successfully." << std::endl;
        sleep(2);
    }

    // RealSense
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
    float fx = intrinsics.fx, ppx = intrinsics.ppx;

    LineFollower follower(0.010f, 0.0f, 0.002f, 0.25f, 0.02f, 0.1f);
    const int ROI_TOP = HEIGHT * 2 / 3;
    const int IMAGE_CENTER_X = WIDTH / 2;
    const int ROI_NEAR_H = 60;
    const int ROI_FAR_H = 60;
    const int ROI_NEAR_Y = HEIGHT - ROI_NEAR_H;
    const int ROI_FAR_Y = HEIGHT - ROI_NEAR_H - ROI_FAR_H;

    const int T_NEAR = 9000;
    const int T_FAR = 2000;
    const int MARGIN = 30;
    const int BAND_CX = 40;
    const int CORNER_HOLD = 3;
    const float TURN_SPEED = 0.3f;
    const float TURN_TIMEOUT = 0.8f;

    // --- State machine
    enum State
    {
        FOLLOW_LINE,
        TURN_LEFT_90,
        TURN_RIGHT_90,
        BALL_SEEN,
        APPROACH_BALL,
        CAPTURE_BALL,
        INTAKE_STATE,
        RELOCK_LINE
    };
    State current_state = FOLLOW_LINE;

    int seen_hold = 0, line_ok_hold = 0, capture_counter = 0;
    int corner_left_hold = 0, corner_right_hold = 0;
    const int LINE_OK_HOLD = 4;
    const int N_detect_hold = 3;
    const float Smin = 0.6f, D_detect_max = 2.5f, D_captured = 0.3f;
    const int INTAKE_PWM = 220;
    chrono::steady_clock::time_point turn_start;

    std::cout << "Starting robot application..." << std::endl;

    while (true)
    {
        rs2::frameset frames = pipe.wait_for_frames();
        rs2::frame color_frame = frames.get_color_frame();
        rs2::frame depth_frame = frames.get_depth_frame();
        Mat color_image(Size(WIDTH, HEIGHT), CV_8UC3, (void *)color_frame.get_data());
        Mat depth_image(Size(WIDTH, HEIGHT), CV_16UC1, (void *)depth_frame.get_data());

        // --- Line detection using Edge Detection
        Mat gray_image;
        cvtColor(color_image, gray_image, COLOR_BGR2GRAY);

        // Apply Bilateral filter to reduce noise while preserving edges
        Mat filtered;
        bilateralFilter(gray_image, filtered, 9, 75, 75);

        // Define ROI_near for gradient calculation
        Rect roi_near_rect(0, ROI_NEAR_Y, WIDTH, ROI_NEAR_H);
        Mat filtered_near = filtered(roi_near_rect);

        // Compute gradient magnitude for auto-Canny threshold on ROI_near
        Mat grad_x_near, grad_y_near, grad_mag_near;
        Sobel(filtered_near, grad_x_near, CV_16S, 1, 0, 3);
        Sobel(filtered_near, grad_y_near, CV_16S, 0, 1, 3);
        convertScaleAbs(grad_x_near, grad_x_near);
        convertScaleAbs(grad_y_near, grad_y_near);
        addWeighted(grad_x_near, 0.5, grad_y_near, 0.5, 0, grad_mag_near);

        // Calculate median of gradient magnitude for auto-threshold on ROI_near
        vector<uchar> grad_values_near;
        grad_values_near.reserve(grad_mag_near.rows * grad_mag_near.cols);
        for (int i = 0; i < grad_mag_near.rows; ++i) {
            for (int j = 0; j < grad_mag_near.cols; ++j) {
                grad_values_near.push_back(grad_mag_near.at<uchar>(i, j));
            }
        }
        sort(grad_values_near.begin(), grad_values_near.end());
        double median_near = grad_values_near[grad_values_near.size() / 2];

        // Auto-Canny thresholds based on median
        double sigma = 0.33;
        double lower_threshold = max(0.0, (1.0 - sigma) * median_near);
        double upper_threshold = min(255.0, (1.0 + sigma) * median_near);

        // Apply Canny edge detection with auto-thresholds
        Mat edges;
        Canny(filtered, edges, lower_threshold, upper_threshold);

        // Create ROI mask for bottom part of image
        Mat roi_mask = Mat::zeros(HEIGHT, WIDTH, CV_8U);
        roi_mask(Range(ROI_TOP, HEIGHT), Range::all()) = 255;

        // Apply ROI mask to edges
        Mat line_mask;
        bitwise_and(edges, roi_mask, line_mask);

        // Morphological operations to clean up edges
        Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
        morphologyEx(line_mask, line_mask, MORPH_CLOSE, kernel);
        morphologyEx(line_mask, line_mask, MORPH_OPEN, kernel);

        // --- Calculate cx_near using fitLine on ROI_near
        Mat mask_near_full = line_mask(roi_near_rect);
        vector<vector<Point>> c_near_full;
        findContours(mask_near_full, c_near_full, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

        int cx_near = -1;
        bool line_confident = false;
        if (!c_near_full.empty())
        {
            auto largest_near = *max_element(c_near_full.begin(), c_near_full.end(),
                                           [](auto &a, auto &b) { return contourArea(a) < contourArea(b); });

            if (largest_near.size() >= 2)
            {
                Vec4f line_params;
                fitLine(largest_near, line_params, DIST_L2, 0, 0.01, 0.01);

                // Calculate intersection with bottom of ROI_near (y = ROI_NEAR_Y + ROI_NEAR_H)
                float vx = line_params[0], vy = line_params[1];
                float x0 = line_params[2], y0 = line_params[3];

                if (abs(vx) > 1e-6) // Avoid division by zero
                {
                    float t = (ROI_NEAR_Y + ROI_NEAR_H - y0) / vy;
                    cx_near = static_cast<int>(x0 + t * vx);
                    cx_near = max(0, min(WIDTH - 1, cx_near)); // Clamp to image bounds
                }
                else
                {
                    cx_near = static_cast<int>(x0); // Vertical line
                }

                // Check confidence: enough pixels and nearly vertical
                float theta_near = atan2(vy, vx) * 180.0f / CV_PI;
                line_confident = (near_pixels > T_NEAR) && (fabs(theta_near) < 30.0f);
            }
        }

        // Use cx_near for PID control (no need for line_center_x anymore)

        // --- ROI analysis for corner detection
        Rect roi_near(0, ROI_NEAR_Y, WIDTH, ROI_NEAR_H);
        Rect roi_far(0, ROI_FAR_Y, WIDTH, ROI_FAR_H);
        Mat mask_near = line_mask(roi_near);
        Mat mask_far = line_mask(roi_far);
        int near_pixels = countNonZero(mask_near);
        int far_pixels = countNonZero(mask_far);

        vector<vector<Point>> c_near;
        findContours(mask_near, c_near, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
        Rect near_bbox;
        float near_theta = 0.0f;
        if (!c_near.empty())
        {
            auto ln = *max_element(c_near.begin(), c_near.end(),
                                   [](auto &a, auto &b)
                                   { return contourArea(a) < contourArea(b); });
            near_bbox = boundingRect(ln);
            if (ln.size() >= 2)
            {
                Vec4f line;
                fitLine(ln, line, DIST_L2, 0, 0.01, 0.01);
                near_theta = atan2(line[1], line[0]) * 180.0f / CV_PI;
            }
        }

        vector<vector<Point>> c_far;
        findContours(mask_far, c_far, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
        float far_theta = 0.0f;
        int far_cx = -1;
        if (!c_far.empty())
        {
            auto lf = *max_element(c_far.begin(), c_far.end(),
                                   [](auto &a, auto &b)
                                   { return contourArea(a) < contourArea(b); });
            Moments m = moments(lf);
            if (m.m00 > 0)
                far_cx = int(m.m10 / m.m00);
            if (lf.size() >= 2)
            {
                Vec4f line;
                fitLine(lf, line, DIST_L2, 0, 0.01, 0.01);
                far_theta = atan2(line[1], line[0]) * 180.0f / CV_PI;
            }
        }

        bool near_big = near_pixels > T_NEAR;
        bool far_small = far_pixels < T_FAR;
        bool near_left = near_bbox.x <= MARGIN;
        bool near_right = near_bbox.x + near_bbox.width >= WIDTH - MARGIN;
        bool near_flat = fabs(near_theta) < 30.0f;

        if (current_state == FOLLOW_LINE)
        {
            if (near_big && far_small && near_flat && near_left)
                corner_left_hold = min(CORNER_HOLD, corner_left_hold + 1);
            else
                corner_left_hold = 0;

            if (near_big && far_small && near_flat && near_right)
                corner_right_hold = min(CORNER_HOLD, corner_right_hold + 1);
            else
                corner_right_hold = 0;
        }

        // --- Ball detection
        BallResult ball = detect_ball(color_image, depth_image, fx, ppx, depth_scale);

        // --- State logic
        float vL = 0, vR = 0;
        int intake_pwm = 0;
        switch (current_state)
        {
        case FOLLOW_LINE:
            if (line_confident)
            {
                line_ok_hold = min(LINE_OK_HOLD, line_ok_hold + 1);
                // Normalized error: (cx_near - IMAGE_CENTER_X) / (WIDTH/2) ∈ [-1,1]
                float error_norm = static_cast<float>(cx_near - IMAGE_CENTER_X) / (WIDTH / 2.0f);
                auto [l, r] = follower.update(error_norm, true);
                vL = l;
                vR = r;
            }
            else
            {
                line_ok_hold = 0;
                // No confident line detection - gentle search
                auto [l, r] = follower.update(0.0f, false);
                vL = -0.15f; // Gentle left turn to search
                vR = 0.15f;
            }
            if (ball.found && ball.score > Smin && ball.distance > 0 && ball.distance < D_detect_max)
            {
                if (++seen_hold >= N_detect_hold)
                {
                    current_state = BALL_SEEN;
                    seen_hold = 0;
                }
            }
            else
                seen_hold = 0;

            if (corner_left_hold >= CORNER_HOLD)
            {
                current_state = TURN_LEFT_90;
                follower.reset();
                turn_start = chrono::steady_clock::now();
                corner_left_hold = corner_right_hold = 0;
            }
            else if (corner_right_hold >= CORNER_HOLD)
            {
                current_state = TURN_RIGHT_90;
                follower.reset();
                turn_start = chrono::steady_clock::now();
                corner_left_hold = corner_right_hold = 0;
            }
            break;

        case TURN_LEFT_90:
            vL = -TURN_SPEED;
            vR = TURN_SPEED;
            if ((far_pixels > T_NEAR && fabs(fabs(far_theta) - 90.0f) < 20.0f &&
                 far_cx >= 0 && abs(far_cx - IMAGE_CENTER_X) < BAND_CX) ||
                chrono::steady_clock::now() - turn_start > chrono::duration<float>(TURN_TIMEOUT))
            {
                current_state = RELOCK_LINE;
                follower.reset();
                line_ok_hold = 0;
            }
            break;

        case TURN_RIGHT_90:
            vL = TURN_SPEED;
            vR = -TURN_SPEED;
            if ((far_pixels > T_NEAR && fabs(fabs(far_theta) - 90.0f) < 20.0f &&
                 far_cx >= 0 && abs(far_cx - IMAGE_CENTER_X) < BAND_CX) ||
                chrono::steady_clock::now() - turn_start > chrono::duration<float>(TURN_TIMEOUT))
            {
                current_state = RELOCK_LINE;
                follower.reset();
                line_ok_hold = 0;
            }
            break;

        case BALL_SEEN:
            current_state = APPROACH_BALL;
            std::cout << "Ball detected -> APPROACH" << std::endl;
            break;

        case APPROACH_BALL:
            if (ball.found)
            {
                float omega = std::clamp(
                    static_cast<float>(ball.angle / (25.0f * M_PI / 180.0f)), -1.0f, 1.0f);
                float fwd = clamp((ball.distance - 0.2f) * 2.0f, 0.3f, 0.5f);
                vL = clamp(fwd - 0.7f * omega, -0.8f, 0.8f);
                vR = clamp(fwd + 0.7f * omega, -0.8f, 0.8f);
                if (ball.distance < 0.5f)
                    intake_pwm = INTAKE_PWM;
                if (ball.distance < D_captured)
                {
                    current_state = CAPTURE_BALL;
                    capture_counter = 0;
                }
            }
            else
            {
                vL = -0.2f;
                vR = 0.2f;
            }
            break;

        case CAPTURE_BALL:
            vL = vR = 0.3f;
            intake_pwm = INTAKE_PWM;
            if (++capture_counter >= 30)
                current_state = INTAKE_STATE;
            break;

        case INTAKE_STATE:
        {
            static auto t0 = chrono::steady_clock::now();
            static bool started = false;
            if (!started)
            {
                t0 = chrono::steady_clock::now();
                started = true;
            }
            vL = vR = 0.2f;
            intake_pwm = INTAKE_PWM;
            if (chrono::steady_clock::now() - t0 > chrono::seconds(2))
            {
                started = false;
                current_state = RELOCK_LINE;
            }
            break;
        }

        case RELOCK_LINE:
            if (line_confident)
            {
                auto [l, r] = follower.update(
                    static_cast<float>(cx_near - IMAGE_CENTER_X) / (WIDTH / 2.0f), true);
                vL = l;
                vR = r;
                if (++line_ok_hold >= LINE_OK_HOLD)
                {
                    current_state = FOLLOW_LINE;
                    line_ok_hold = 0;
                }
            }
            else
            {
                // Gentle search when no confident line
                auto [l, r] = follower.update(0.0f, false);
                vL = -0.15f;
                vR = 0.15f;
            }
            intake_pwm = 0;
            break;
        }

        // --- Send motor
        drive_motors(uart_fd, vL, vR);
        set_intake(uart_fd, intake_pwm);

        // --- Overlay
        Mat disp = color_image.clone();
        line(disp, {IMAGE_CENTER_X, 0}, {IMAGE_CENTER_X, HEIGHT}, {255, 255, 0}, 2);
        line(disp, {0, ROI_TOP}, {WIDTH, ROI_TOP}, {200, 200, 200}, 2);
        rectangle(disp, roi_near, {100, 100, 255}, 2);
        rectangle(disp, roi_far, {100, 255, 100}, 2);
        if (cx_near >= 0)
            line(disp, {cx_near, ROI_TOP}, {cx_near, HEIGHT}, {0, 255, 255}, 3);
        if (ball.found)
            circle(disp, {ball.center_x, ball.center_y}, ball.radius, {0, 255, 0}, 2);

        string names[] = {"FOLLOW_LINE", "TURN_L90", "TURN_R90", "BALL_SEEN", "APPROACH_BALL", "CAPTURE_BALL", "INTAKE", "RELOCK"};
        putText(disp, "State:" + names[current_state], {10, 30}, FONT_HERSHEY_SIMPLEX, 0.7, {50, 220, 50}, 2);
        putText(disp, "Confident: " + string(line_confident ? "YES" : "NO"), {10, 60}, FONT_HERSHEY_SIMPLEX, 0.7, {50, 220, 50}, 2);
        imshow("Robot", disp);
        if (waitKey(1) == 27)
            break;
    }

    drive_motors(uart_fd, 0, 0);
    set_intake(uart_fd, 0);
    if (uart_fd >= 0)
        close(uart_fd);
    pipe.stop();
    return 0;
}
