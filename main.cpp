#include <librealsense2/rs.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>

static cv::Mat frame_to_mat(const rs2::video_frame& f) {
    int width = f.get_width();
    int height = f.get_height();
    if (f.get_profile().format() == RS2_FORMAT_BGR8) {
        return cv::Mat(cv::Size(width, height), CV_8UC3, (void*)f.get_data(), cv::Mat::AUTO_STEP);
    } else if (f.get_profile().format() == RS2_FORMAT_RGB8) {
        cv::Mat rgb(cv::Size(width, height), CV_8UC3, (void*)f.get_data(), cv::Mat::AUTO_STEP);
        cv::Mat bgr; cv::cvtColor(rgb, bgr, cv::COLOR_RGB2BGR);
        return bgr;
    } else if (f.get_profile().format() == RS2_FORMAT_Y8) {
        return cv::Mat(cv::Size(width, height), CV_8UC1, (void*)f.get_data(), cv::Mat::AUTO_STEP);
    }
    throw std::runtime_error("Unsupported frame format!");
}

int main() try {
    const int W = 640, H = 480, FPS = 60;

    rs2::pipeline pipe;
    rs2::config cfg;
    cfg.enable_stream(RS2_STREAM_COLOR, W, H, RS2_FORMAT_BGR8, FPS);
    cfg.enable_stream(RS2_STREAM_DEPTH, W, H, RS2_FORMAT_Z16, FPS);

    auto profile = pipe.start(cfg);

    rs2::align align_to_color(RS2_STREAM_COLOR);

    float depth_scale = profile.get_device().first<rs2::depth_sensor>().get_depth_scale();

    rs2::colorizer depth_colorizer;

    std::cout << "Press [ESC] to quit.\n";

    while (true) {
        rs2::frameset fs = pipe.wait_for_frames();
        fs = align_to_color.process(fs);

        rs2::video_frame color = fs.get_color_frame();
        rs2::depth_frame depth = fs.get_depth_frame();
        rs2::video_frame depth_color = depth_colorizer.process(depth);

        cv::Mat color_mat = frame_to_mat(color);
        cv::Mat depth_color_mat = frame_to_mat(depth_color);

        int cx = color_mat.cols / 2, cy = color_mat.rows / 2;
        float dist_m = depth.get_distance(cx, cy);

        cv::circle(color_mat, {cx, cy}, 4,  {0, 255, 0}, -1);
        char text[64];
        std::snprintf(text, sizeof(text), "%.2f m", dist_m);
        cv::putText(color_mat, text, {10, 30}, cv::FONT_HERSHEY_SIMPLEX, 1.0, {0, 255, 0}, 2);

        cv::imshow("Color (BGR)", color_mat);
        cv::imshow("Depth (colorized)", depth_color_mat);

        int k = cv::waitKey(1);
        if (k == 27) break;
    }

    pipe.stop();
    return 0;
} catch (const rs2::error& e) {
    std::cerr << "RealSense error: " << e.what() << "\n";
    return 1;
} catch (const std::exception& e) {
    std::cerr << "Std error: " << e.what() << "\n";
    return 2;
}