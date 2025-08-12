#include <librealsense2/rs.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include "BallDetection.hpp"

int main() {
    try {
        // 1) RealSense
        rs2::pipeline pipe;
        rs2::config cfg;
        cfg.enable_stream(RS2_STREAM_COLOR, 640, 480, RS2_FORMAT_BGR8, 30);
        cfg.enable_stream(RS2_STREAM_DEPTH, 640, 480, RS2_FORMAT_Z16, 30);
        pipe.start(cfg);

        // 2) Intrinsics & depth scale
        auto profile = pipe.get_active_profile();
        auto color_vsp = profile.get_stream(RS2_STREAM_COLOR).as<rs2::video_stream_profile>();
        rs2_intrinsics intr = color_vsp.get_intrinsics();
        float fx  = intr.fx;
        float ppx = intr.ppx;

        auto dev = profile.get_device();
        auto depth_sensor = dev.first<rs2::depth_sensor>();
        float depth_scale = depth_sensor.get_depth_scale();

        // 3) Detector params
        ball::Params params;
        params.roi_top_frac = 0.2f;          // bỏ 20% phía trên ảnh
        params.topk_candidates = 5;          // vẽ tối đa 5 ứng viên
        params.enable_lock = true;           // khóa mục tiêu
        ball::Detector det(params);

        // 4) Loop
        while (true) {
            rs2::frameset frames = pipe.wait_for_frames();
            rs2::frame color_frame = frames.get_color_frame();
            rs2::frame depth_frame = frames.get_depth_frame();

            cv::Mat color(cv::Size(intr.width, intr.height), CV_8UC3,
                          (void*)color_frame.get_data(), cv::Mat::AUTO_STEP);
            cv::Mat depth(cv::Size(intr.width, intr.height), CV_16UC1,
                          (void*)depth_frame.get_data(), cv::Mat::AUTO_STEP);

            // Lấy primary + danh sách ứng viên
            std::vector<ball::Candidate> cands;
            ball::Result r = det.detect(color, depth, fx, ppx, depth_scale, &cands);

            // Vẽ
            cv::Mat disp = color.clone();

            // Vẽ tất cả ứng viên (màu vàng), ghi điểm
            for (size_t i = 0; i < cands.size(); ++i) {
                const auto& cr = cands[i].r;
                cv::Scalar col(0, 255, 255); // vàng
                cv::circle(disp, {cr.cx, cr.cy}, std::max(6, cr.radius_px), col, 2);
                std::string t = cv::format("#%zu Z=%.2f a=%.1f sc=%.2f",
                                           i+1, cr.distance_m, cr.angle_rad*180.0/CV_PI, cands[i].score);
                cv::putText(disp, t, {10, 30 + 20*(int)i}, cv::FONT_HERSHEY_SIMPLEX, 0.55, col, 2);
            }

            // Vẽ primary (màu xanh lá, đè lên)
            if (r.found) {
                cv::circle(disp, {r.cx, r.cy}, std::max(8, r.radius_px), {0,255,0}, 3);
                std::string info = cv::format("PRIMARY  Z=%.2fm  Angle=%.1fdeg  sc=%.2f",
                                              r.distance_m, r.angle_rad*180.0/CV_PI, r.score);
                cv::putText(disp, info, {10, (int)(30 + 20*cands.size() + 10)},
                            cv::FONT_HERSHEY_SIMPLEX, 0.65, {0,255,0}, 2);

                if (r.in_capture_win)
                    cv::putText(disp, "CAPTURE!", {10, (int)(30 + 20*cands.size() + 35)},
                                cv::FONT_HERSHEY_SIMPLEX, 0.75, {0,0,255}, 2);

                if (r.very_close)
                    cv::putText(disp, "VERY CLOSE!", {10, (int)(30 + 20*cands.size() + 60)},
                                cv::FONT_HERSHEY_SIMPLEX, 0.75, {255,0,0}, 2);
            } else {
                cv::putText(disp, "No ball", {10, 30}, cv::FONT_HERSHEY_SIMPLEX, 0.7, {0,0,255}, 2);
            }

            cv::imshow("Ball Detection (multi)", disp);

            // In điều khiển giả lập
            if (r.found) {
                if (r.in_capture_win)
                    std::cout << "[INTAKE ON] Z=" << r.distance_m << " angle=" << r.angle_rad
                              << " score=" << r.score << "\n";
                else
                    std::cout << "[APPROACH] Z=" << r.distance_m << " angle=" << r.angle_rad
                              << " score=" << r.score << "\n";
            } else {
                std::cout << "[SEARCH] candidates=" << cands.size() << "\n";
            }

            if (cv::waitKey(1) == 27) break; // ESC
        }
    } catch (const rs2::error& e) {
        std::cerr << "RealSense error: " << e.what() << "\n"; return 1;
    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << "\n"; return 1;
    }
}
