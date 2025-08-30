#include <librealsense2/rs.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
#include <thread>
#include <vector>
#include <algorithm>
#include <cmath>

using namespace cv;
using namespace std;

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

int main()
{
    // RealSense setup
    rs2::pipeline pipe;
    rs2::config cfg;
    const int WIDTH = 640, HEIGHT = 480, FPS = 30;
    cfg.enable_stream(RS2_STREAM_COLOR, WIDTH, HEIGHT, RS2_FORMAT_BGR8, FPS);
    auto profile = pipe.start(cfg);

    LineFollower follower(0.010f, 0.0f, 0.002f, 0.25f, 0.02f, 0.1f);
    const int ROI_TOP = HEIGHT * 2 / 3;
    const int IMAGE_CENTER_X = WIDTH / 2;
    const int ROI_NEAR_H = 60;
    const int ROI_NEAR_Y = HEIGHT - ROI_NEAR_H;

    const int T_NEAR = 9000;

    std::cout << "Line Follow Test - Press ESC to quit" << std::endl;
    std::cout << "Color Detection: #214ea3 Blue Line" << std::endl;
    std::cout << "PID Parameters: Kp=0.010, Kd=0.002, Base Speed=0.25" << std::endl;

    while (true)
    {
        rs2::frameset frames = pipe.wait_for_frames();
        rs2::frame color_frame = frames.get_color_frame();
        Mat color_image(Size(WIDTH, HEIGHT), CV_8UC3, (void *)color_frame.get_data());

        // --- Line detection using Color Detection (for #214ea3 blue line)
        Mat hsv_image;
        cvtColor(color_image, hsv_image, COLOR_BGR2HSV);

        // Define color range for #214ea3 blue line
        // #214ea3 = RGB(33, 78, 163) ≈ HSV(210, 204, 163)
        Scalar lower_blue(190, 150, 100);  // Lower bound for blue
        Scalar upper_blue(230, 255, 200);  // Upper bound for blue

        Mat color_mask;
        inRange(hsv_image, lower_blue, upper_blue, color_mask);

        // Create ROI mask for bottom part of image
        Mat roi_mask = Mat::zeros(HEIGHT, WIDTH, CV_8U);
        roi_mask(Range(ROI_TOP, HEIGHT), Range::all()) = 255;

        // Apply ROI mask to color detection
        Mat line_mask;
        bitwise_and(color_mask, roi_mask, line_mask);

        // Morphological operations to clean up the mask
        Mat kernel = getStructuringElement(MORPH_RECT, Size(5, 5));
        morphologyEx(line_mask, line_mask, MORPH_CLOSE, kernel, Point(-1, -1), 2);
        morphologyEx(line_mask, line_mask, MORPH_OPEN, kernel, Point(-1, -1), 1);

        // --- ROI analysis for corner detection
        Rect roi_near(0, ROI_NEAR_Y, WIDTH, ROI_NEAR_H);
        Mat mask_near = line_mask(roi_near);
        int near_pixels = countNonZero(mask_near);

        // --- Calculate cx_near using moments on color mask
        Mat mask_near_full = line_mask(roi_near);
        vector<vector<Point>> c_near_full;
        findContours(mask_near_full, c_near_full, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

        int cx_near = -1;
        bool line_confident = false;
        float theta_near = 0.0f;

        if (!c_near_full.empty())
        {
            auto largest_near = *max_element(c_near_full.begin(), c_near_full.end(),
                                           [](auto &a, auto &b) { return contourArea(a) < contourArea(b); });

            if (contourArea(largest_near) > 50) // Minimum area threshold
            {
                // Use moments to find centroid
                Moments M = moments(largest_near);
                if (M.m00 > 0)
                {
                    cx_near = static_cast<int>(M.m10 / M.m00);
                    cx_near = max(0, min(WIDTH - 1, cx_near)); // Clamp to image bounds

                    // Calculate orientation using fitLine for theta
                    if (largest_near.size() >= 2)
                    {
                        Vec4f line_params;
                        fitLine(largest_near, line_params, DIST_L2, 0, 0.01, 0.01);
                        float vx = line_params[0], vy = line_params[1];
                        theta_near = atan2(vy, vx) * 180.0f / CV_PI;
                    }

                    // Check confidence: enough pixels and reasonable area
                    line_confident = (near_pixels > T_NEAR * 0.3) && (fabs(theta_near) < 45.0f);
                }
            }
        }

        // --- Line Following Control
        float vL = 0.0f, vR = 0.0f;
        if (line_confident && cx_near >= 0)
        {
            // Normalized error: (cx_near - IMAGE_CENTER_X) / (WIDTH/2) ∈ [-1,1]
            float error_norm = static_cast<float>(cx_near - IMAGE_CENTER_X) / (WIDTH / 2.0f);
            auto [l, r] = follower.update(error_norm, true);
            vL = l;
            vR = r;
        }
        else
        {
            // No confident line detection - gentle search
            auto [l, r] = follower.update(0.0f, false);
            vL = -0.15f; // Gentle left turn to search
            vR = 0.15f;
        }

        // --- Display results on camera view
        Mat display = color_image.clone();

        // Draw reference lines
        line(display, {IMAGE_CENTER_X, 0}, {IMAGE_CENTER_X, HEIGHT}, {255, 255, 0}, 2); // Yellow center line
        line(display, {0, ROI_TOP}, {WIDTH, ROI_TOP}, {200, 200, 200}, 2); // Gray ROI top line
        rectangle(display, roi_near, {100, 100, 255}, 2); // Blue ROI rectangle

        // Draw detected line position
        if (cx_near >= 0)
        {
            Scalar line_color = line_confident ? Scalar(0, 255, 0) : Scalar(0, 0, 255); // Green if confident, Red if not
            line(display, {cx_near, ROI_TOP}, {cx_near, HEIGHT}, line_color, 3);
        }

        // Display engine control values
        char motor_text[128];
        sprintf(motor_text, "MOTOR: L=%.3f R=%.3f", vL, vR);
        putText(display, motor_text, {10, 30}, FONT_HERSHEY_SIMPLEX, 0.7, {0, 255, 255}, 2);

        // Display line detection info
        char line_info[256];
        sprintf(line_info, "cx=%d err=%.2f conf=%s theta=%.1f",
                cx_near,
                cx_near >= 0 ? static_cast<float>(cx_near - IMAGE_CENTER_X) / (WIDTH / 2.0f) : 0.0f,
                line_confident ? "YES" : "NO",
                theta_near);
        putText(display, line_info, {10, 60}, FONT_HERSHEY_SIMPLEX, 0.6, {255, 255, 255}, 2);

        // Display pixel count
        char pixel_info[64];
        sprintf(pixel_info, "Near pixels: %d/%d", near_pixels, T_NEAR);
        putText(display, pixel_info, {10, 90}, FONT_HERSHEY_SIMPLEX, 0.6, {200, 200, 200}, 2);

        // Display PID parameters
        putText(display, "Color Detection: #214ea3 Blue Line", {10, HEIGHT - 50},
                FONT_HERSHEY_SIMPLEX, 0.5, {150, 150, 150}, 1);
        putText(display, "PID: Kp=0.010 Kd=0.002 Base=0.25", {10, HEIGHT - 30},
                FONT_HERSHEY_SIMPLEX, 0.5, {150, 150, 150}, 1);

        imshow("Line Follow Test", display);

        if (waitKey(1) == 27) // ESC key
            break;
    }

    pipe.stop();
    return 0;
}
