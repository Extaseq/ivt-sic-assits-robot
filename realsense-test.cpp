#include <librealsense2/rs.hpp>
#include <iostream>

#define WIDTH 640
#define HEIGHT 480
#define FPS 30

int main() {
    try {
        rs2::pipeline pipe;
        rs2::config cfg;

        cfg.enable_stream(RS2_STREAM_COLOR, WIDTH, HEIGHT, RS2_FORMAT_BGR8, FPS);
        cfg.enable_stream(RS2_STREAM_DEPTH, WIDTH, HEIGHT, RS2_FORMAT_Z16, FPS);

        auto profile = pipe.start(cfg);

        rs2::align align_to_color(RS2_STREAM_COLOR);

        while (true) {
            rs2::frameset fs = pipe.wait_for_frames();
            fs = align_to_color.process(fs);
            rs2::depth_frame d = fs.get_depth_frame();
            float dist = d.get_distance(320, 240);
            std::cout << "\rCenter distance: " << dist << " m" << std::flush;
        }
   } catch (const rs2::error& e) {
        std::cerr << "RealSense error: " << e.what() << "\n";
        return 1;
   } catch (const std::exception& e) {
        std::cerr << "Std error: " << e.what() << "\n";
        return 2;
   }
}