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

// ==================== UART Communication ====================
int open_uart(const char *dev = "/dev/ttyTHS1", int baud = 115200)
{
    int fd = open(dev, O_RDWR | O_NOCTTY | O_NONBLOCK);
    if (fd < 0)
        return -1;

    termios tio{};
    if (tcgetattr(fd, &tio) != 0)
        return -1;
    cfmakeraw(&tio);

    tio.c_cflag &= ~PARENB;
    tio.c_cflag &= ~CSTOPB;
    tio.c_cflag &= ~CSIZE;
    tio.c_cflag |= CS8;
    tio.c_cflag |= (CLOCAL | CREAD);

    speed_t spd = B115200;
    cfsetispeed(&tio, spd);
    cfsetospeed(&tio, spd);

    if (tcsetattr(fd, TCSANOW, &tio) != 0)
        return -1;
    return fd;
}

void send_motor_command(int fd, const char *id, int speed)
{
    if (fd < 0)
        return;
    if (speed > 255)
        speed = 255;
    if (speed < -255)
        speed = -255;

    char buf[32];
    int n = snprintf(buf, sizeof(buf), "%s %d\n", id, speed);
    write(fd, buf, n);
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
    int uart_fd = open_uart("/dev/ttyTHS1", 115200);
    if (uart_fd < 0)
    {
        cerr << "Failed to open UART. Using USB fallback..." << endl;
        uart_fd = open_uart("/dev/ttyUSB0", 115200);
    }

    if (uart_fd < 0)
    {
        cerr << "Warning: Could not open UART. Motor commands will be simulated." << endl;
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

    // State machine
    enum State
    {
        FOLLOW_LINE,
        APPROACH_BALL,
        INTAKE_BALL,
        RETURN_TO_LINE
    };
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

        switch (current_state)
        {
        case FOLLOW_LINE:
            if (line_center_x >= 0)
            {
                tie(left_speed, right_speed) = follower.update(line_center_x, IMAGE_CENTER_X);
            }
            else
            {
                // Search for line
                left_speed = -0.15f;
                right_speed = 0.15f;
            }

            if (ball.found && ball.score > BALL_DETECTION_THRESHOLD)
            {
                ball_detection_count++;
                if (ball_detection_count >= BALL_CONFIRMATION_COUNT)
                {
                    current_state = APPROACH_BALL;
                    ball_detection_count = 0;
                    cout << "Ball detected! Switching to APPROACH_BALL" << endl;
                }
            }
            else
            {
                ball_detection_count = 0;
            }
            break;

        case APPROACH_BALL:
            if (ball.found)
            {
                // Approach ball with steering
                float steering = clamp(ball.angle * 2.0f, -0.8f, 0.8f);
                float forward_speed = clamp((ball.distance - 0.2f) / 1.0f, 0.2f, 0.6f);

                left_speed = forward_speed - steering * 0.6f;
                right_speed = forward_speed + steering * 0.6f;

                if (ball.distance <= CAPTURE_DISTANCE)
                {
                    current_state = INTAKE_BALL;
                    cout << "Ball captured! Switching to INTAKE_BALL" << endl;
                }
            }
            else
            {
                // Lost ball, search
                left_speed = -0.1f;
                right_speed = 0.1f;
                ball_detection_count++;

                if (ball_detection_count > 30)
                {
                    current_state = RETURN_TO_LINE;
                    cout << "Ball lost. Returning to line." << endl;
                }
            }
            break;

        case INTAKE_BALL:
            // Run intake and move forward slightly
            intake_power = 200;
            left_speed = 0.2f;
            right_speed = 0.2f;

            // After 1 second, return to line following
            this_thread::sleep_for(chrono::seconds(1));
            current_state = RETURN_TO_LINE;
            cout << "Intake complete. Returning to line." << endl;
            break;

        case RETURN_TO_LINE:
            intake_power = 0;

            if (line_center_x >= 0)
            {
                tie(left_speed, right_speed) = follower.update(line_center_x, IMAGE_CENTER_X);

                // Stay in line following for a while before looking for balls again
                ball_detection_count++;
                if (ball_detection_count > 50)
                {
                    current_state = FOLLOW_LINE;
                    ball_detection_count = 0;
                    cout << "Line reacquired. Switching to FOLLOW_LINE" << endl;
                }
            }
            else
            {
                // Search for line
                left_speed = -0.15f;
                right_speed = 0.15f;
            }
            break;
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