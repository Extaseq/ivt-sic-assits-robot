// running.cpp
#include <librealsense2/rs.hpp>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <thread>
#include <unistd.h> // open, write, close
#include <fcntl.h>
#include <termios.h>
#include <cstdio>    // snprintf
#include <algorithm> // std::clamp, std::max, std::min
#include <cmath>     // std::round, M_PI
#include <vector>

#include "LineFollow.hpp"
#include "BallDetection.hpp"

using namespace cv;

//==================== UART (Jetson Nano) ====================
// J41 UART (3.3V): "/dev/ttyTHS1";  USB-UART: "/dev/ttyUSB0"
int open_uart(const char *dev = "/dev/ttyUSB0", int baud = 115200)
{
    int fd = open(dev, O_RDWR | O_NOCTTY | O_NONBLOCK);
    if (fd < 0)
        return -1;

    termios tio{};
    if (tcgetattr(fd, &tio) != 0)
        return -1;
    cfmakeraw(&tio);

    // 8N1
    tio.c_cflag &= ~PARENB;
    tio.c_cflag &= ~CSTOPB;
    tio.c_cflag &= ~CSIZE;
    tio.c_cflag |= CS8;
    tio.c_cflag |= (CLOCAL | CREAD);

    // Baud
    speed_t spd = B115200;
    cfsetispeed(&tio, spd);
    cfsetospeed(&tio, spd);

    if (tcsetattr(fd, TCSANOW, &tio) != 0)
        return -1;
    return fd;
}

static inline int clamp255(float x)
{
    if (x > 1)
        x = 1;
    if (x < -1)
        x = -1;
    return int(std::round(x * 255.0f));
}

static inline void sendMotorCmd(int fd, const char *id, int spd)
{
    if (fd < 0)
        return;
    if (spd > 255)
        spd = 255;
    if (spd < -255)
        spd = -255;
    char buf[32];
    int n = std::snprintf(buf, sizeof(buf), "%s %d\n", id, spd);
    (void)write(fd, buf, n);
}

static inline void driveLR(int fd, float vL_norm, float vR_norm)
{
    sendMotorCmd(fd, "M1", clamp255(vL_norm));
    sendMotorCmd(fd, "M2", clamp255(vR_norm));
}

static inline void intake(int fd, int pwm)
{ // pwm ∈ [-255..255]
    sendMotorCmd(fd, "M3", pwm);
}

//==================== State machine ====================
enum State
{
    FOLLOW_LINE,
    BALL_SEEN,
    APPROACH_BALL,
    INTAKE_STATE,
    RELOCK_LINE
};

static const int WIDTH = 640, HEIGHT = 480, FPS = 30;
static const int ROI_TOP = HEIGHT * 2 / 3;
static const float DT = 0.02f; // 20ms
static const auto CONTROL_DT = std::chrono::milliseconds(20);

// Ngưỡng phát hiện bóng / điều khiển
static const float Smin = 0.55f;
static const float D_detect_max = 2.5f; // m
static const int N_detect_hold = 3;     // thấy liên tiếp

static const float D_captured = 0.25f; // coi như đã vào miệng
static const int INTAKE_PWM = 220;     // lực cuốn
static const int LINE_OK_HOLD = 4;

// helper
static inline float rad2deg(float r) { return r * 180.0f / (float)M_PI; }

int main()
{
    // --- Serial to Arduino ---
    int uart = open_uart("/dev/ttyTHS1", 115200); // đổi sang "/dev/ttyUSB0" nếu dùng USB-UART
    if (uart < 0)
        perror("UART open failed");

    // --- RealSense ---
    rs2::pipeline pipe;
    rs2::config cfg;
    cfg.enable_stream(RS2_STREAM_COLOR, WIDTH, HEIGHT, RS2_FORMAT_BGR8, FPS);
    cfg.enable_stream(RS2_STREAM_DEPTH, WIDTH, HEIGHT, RS2_FORMAT_Z16, FPS);
    auto profile = pipe.start(cfg);

    // === Intrinsics & depth scale ===
    auto color_vsp = profile.get_stream(RS2_STREAM_COLOR).as<rs2::video_stream_profile>();
    rs2_intrinsics K = color_vsp.get_intrinsics();
    float fx = K.fx, ppx = K.ppx;

    rs2::device dev = profile.get_device();
    float depth_scale = 0.001f;
    if (dev.first<rs2::depth_sensor>())
    {
        auto ds = dev.first<rs2::depth_sensor>();
        depth_scale = ds.get_depth_scale(); // mét / tick
    }

    // --- Detector params & init ---
    ball::Params bp;
    // HSV bóng tennis (vàng-xanh)
    bp.hsv_low = Scalar(20, 80, 120);
    bp.hsv_high = Scalar(45, 255, 255);

    // Bỏ phần trên ảnh để bớt nhiễu (30%)
    bp.roi_top_frac = 0.30f;

    // Hysteresis cho cửa sổ bắt bóng (khớp với logic intake)
    bp.z_cap_on = 0.45f;
    bp.z_cap_off = 0.55f;
    bp.th_cap_on_rad = 8.0f * (float)M_PI / 180.0f;
    bp.th_cap_off_rad = 10.0f * (float)M_PI / 180.0f;
    bp.roi_cap_on_frac = 0.60f;
    bp.roi_cap_off_frac = 0.55f;
    bp.debounce_on_frames = 3;
    bp.debounce_off_frames = 2;

    // Scoring / lock giữ nguyên mặc định hợp lý
    ball::Detector det(bp);

    // --- Line follower ---
    LineFollower follower(0.015f, 0.0f, 0.004f, 0.25f, DT);

    State st = FOLLOW_LINE;
    int seen_hold = 0, line_ok_hold = 0;
    auto next_tick = std::chrono::steady_clock::now();

    while (true)
    {
        next_tick += CONTROL_DT;

        rs2::frameset fs;
        if (!pipe.poll_for_frames(&fs))
        {
            std::this_thread::sleep_until(next_tick);
            continue;
        }

        rs2::video_frame color = fs.get_color_frame();
        rs2::depth_frame depth = fs.get_depth_frame();
        if (!color)
        {
            std::this_thread::sleep_until(next_tick);
            continue;
        }

        Mat bgr(Size(WIDTH, HEIGHT), CV_8UC3, (void *)color.get_data(), Mat::AUTO_STEP);

        // ---- LINE MASK (Blue-dominant) ----
        Mat mask = Mat::zeros(HEIGHT, WIDTH, CV_8U);
        const int T = 30, Vmin = 60;
        for (int y = ROI_TOP; y < HEIGHT; ++y)
        {
            const Vec3b *row = bgr.ptr<Vec3b>(y);
            uchar *mrow = mask.ptr<uchar>(y);
            for (int x = 0; x < WIDTH; ++x)
            {
                int B = row[x][0], G = row[x][1], R = row[x][2];
                int V = std::max({B, G, R});
                if (B > G + T && B > R + T && V >= Vmin)
                    mrow[x] = 255;
            }
        }
        morphologyEx(mask, mask, MORPH_CLOSE,
                     getStructuringElement(MORPH_RECT, {5, 5}), Point(-1, -1), 2);

        // ---- Centroid line ----
        std::vector<std::vector<Point>> contours;
        findContours(mask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
        int centerX = WIDTH / 2;
        int cx_line = -1;
        if (!contours.empty())
        {
            auto cmax = *std::max_element(
                contours.begin(), contours.end(),
                [](auto &a, auto &b)
                { return contourArea(a) < contourArea(b); });
            Moments M = moments(cmax);
            if (M.m00 >= 1e-3)
                cx_line = int(M.m10 / M.m00);
        }

        // ---- Ball detection ----
        std::vector<ball::Candidate> cands; // optional, để vẽ debug
        ball::Result r = det.detect(bgr, depth, fx, ppx, depth_scale, &cands);

        // ---- State machine ----
        float vL = 0.0f, vR = 0.0f; // [-1..1]
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
            }
            else
            {
                line_ok_hold = 0;
                vL = -0.18f;
                vR = 0.18f; // quay tìm line
            }
            if (r.found && r.score > Smin && r.distance_m > 0.0f && r.distance_m < D_detect_max)
            {
                if (++seen_hold >= N_detect_hold)
                    st = BALL_SEEN;
            }
            else
                seen_hold = 0;
            break;
        }
        case BALL_SEEN:
        {
            seen_hold = 0;
            st = APPROACH_BALL;
            break;
        }
        case APPROACH_BALL:
        {
            if (r.found)
            {
                float omega = std::clamp(r.angle_rad / (30.0f * (float)M_PI / 180.0f), -0.8f, 0.8f); // 30°
                float fwd = std::clamp((r.distance_m - 0.3f) / 0.8f, 0.2f, 0.6f);
                vL = std::clamp(fwd - 0.6f * omega, -0.8f, 0.8f);
                vR = std::clamp(fwd + 0.6f * omega, -0.8f, 0.8f);

                // Bật intake dựa trên cửa sổ capture có hysteresis
                if (uart >= 0 && r.in_capture_win)
                    intake(uart, +INTAKE_PWM);

                if (r.very_close || (r.distance_m > 0.f && r.distance_m < D_captured))
                {
                    st = INTAKE_STATE;
                }
            }
            else
            {
                vL = -0.15f;
                vR = 0.15f; // quét tìm lại bóng
            }
            break;
        }
        case INTAKE_STATE:
        {
            if (uart >= 0)
                intake(uart, +INTAKE_PWM);
            static auto t0 = std::chrono::steady_clock::now();
            static bool started = false;
            if (!started)
            {
                t0 = std::chrono::steady_clock::now();
                started = true;
            }
            vL = vR = 0.25f; // “nuốt” thêm chút rồi quay lại line
            if (std::chrono::steady_clock::now() - t0 > std::chrono::milliseconds(500))
            {
                started = false;
                st = RELOCK_LINE;
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
                }
            }
            else
            {
                line_ok_hold = 0;
                vL = -0.18f;
                vR = 0.18f;
            }
            if (uart >= 0)
                intake(uart, 0); // tắt cuốn khi đang tìm lại line
            break;
        }
        }

        // ---- Gửi động cơ ----
        if (uart >= 0)
            driveLR(uart, vL, vR);

        // ---- Overlay debug ----
        line(bgr, {WIDTH / 2, 0}, {WIDTH / 2, HEIGHT - 1}, {255, 255, 0}, 1, LINE_AA);
        line(bgr, {0, ROI_TOP}, {WIDTH - 1, ROI_TOP}, {200, 200, 200}, 1, LINE_AA);
        if (cx_line >= 0)
            line(bgr, {cx_line, ROI_TOP}, {cx_line, HEIGHT - 1}, {0, 255, 255}, 2, LINE_AA);

        // Vẽ top candidates (debug)
        for (int i = 0; i < (int)cands.size(); ++i)
        {
            auto &cr = cands[i].r;
            if (!cr.found)
                continue;
            circle(bgr, {cr.cx, cr.cy}, cr.radius_px, (i == 0 ? Scalar(0, 255, 0) : Scalar(0, 255, 255)), 2, LINE_AA);
            char buf[64];
            std::snprintf(buf, sizeof(buf), "#%d d=%.2f ang=%.1f sc=%.2f", i, cr.distance_m, rad2deg(cr.angle_rad), cands[i].score);
            putText(bgr, buf, {cr.cx + 6, cr.cy - 6}, FONT_HERSHEY_SIMPLEX, 0.45, (i == 0 ? Scalar(0, 255, 0) : Scalar(0, 255, 255)), 1, LINE_AA);
        }

        char txt[180];
        std::snprintf(txt, sizeof(txt),
                      "ST=%d vL=%.2f vR=%.2f  ball:%s d=%.2f ang=%.1f sc=%.2f cap:%d close:%d",
                      st, vL, vR, r.found ? "Y" : "N", r.distance_m,
                      rad2deg(r.angle_rad), r.score, r.in_capture_win ? 1 : 0, r.very_close ? 1 : 0);
        putText(bgr, txt, {10, 30}, FONT_HERSHEY_SIMPLEX, 0.55, {50, 220, 50}, 2, LINE_AA);

        imshow("color", bgr);
        if (waitKey(1) == 27)
            break;
        std::this_thread::sleep_until(next_tick);
    }

    if (uart >= 0)
    {
        driveLR(uart, 0, 0);
        intake(uart, 0);
        close(uart);
    }
    pipe.stop();
    return 0;
}
