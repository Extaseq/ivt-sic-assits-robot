#include <librealsense2/rs.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
#include <thread>
#include "LineFollow.hpp"

using namespace cv;

#define REALSENSE_ERROR 1
#define STD_ERROR 2
#define UNKNOWN_ERROR 3

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

            std::pair<float, float> motor_speeds = follower.update(cx, centerX);
            vL = motor_speeds.first;
            vR = motor_speeds.second;

            float err = float(cx - centerX);
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

    // TODO: Send vL and vR to motors here
    (void)vL; // Suppress unused variable warning
    (void)vR; // Suppress unused variable warning

    imshow("mask", mask);
    imshow("color", bgr);

    if (waitKey(1) == 27)
    {
        break;
    }

    std::this_thread::sleep_until(next_tick);
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