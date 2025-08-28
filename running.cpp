#include <librealsense2/rs.hpp>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <thread>
#include <unistd.h> // write(), close() (Linux). Trên Windows thay bằng Win32 serial.
#include <fcntl.h>
#include <termios.h>

// --- UART open trên Jetson ---
int open_uart(const char *dev = "/dev/ttyTHS1", int baud = 115200)
{
    int fd = open(dev, O_RDWR | O_NOCTTY | O_NONBLOCK);
    if (fd < 0)
        return -1;

    termios tio{};
    tcgetattr(fd, &tio);
    cfmakeraw(&tio);
    // set baud
    speed_t spd = B115200; // bạn có thể map baud khác nếu cần
    cfsetispeed(&tio, spd);
    cfsetospeed(&tio, spd);
    tio.c_cflag |= (CLOCAL | CREAD);
    // 8N1
    tio.c_cflag &= ~PARENB;
    tio.c_cflag &= ~CSTOPB;
    tio.c_cflag &= ~CSIZE;
    tio.c_cflag |= CS8;

    tcsetattr(fd, TCSANOW, &tio);
    return fd;
}

// Map tốc độ [-1..1] -> [-255..255] và gửi theo giao thức UART
int clamp255(float x)
{
    if (x > 1)
        x = 1;
    if (x < -1)
        x = -1;
    return int(std::round(x * 255));
}

void sendMotorCmd(int fd, const char *id, int spd)
{
    if (fd < 0)
        return; // nếu chưa mở UART thì bỏ qua
    char buf[32];
    int n = snprintf(buf, sizeof(buf), "%s %d\n", id, spd);
    (void)write(fd, buf, n);
}

void driveLR(int fd, float vL_norm, float vR_norm)
{
    sendMotorCmd(fd, "M1", clamp255(vL_norm));
    sendMotorCmd(fd, "M2", clamp255(vR_norm));
}
void intake(int fd, int pwm)
{ // pwm in [-255..255]
    if (pwm > 255)
        pwm = 255;
    if (pwm < -255)
        pwm = -255;
    sendMotorCmd(fd, "M3", pwm);
}

// Map tốc độ [-1..1] -> [-255..255] và gửi theo giao thức UART
int clamp255(float x)
{
    if (x > 1)
        x = 1;
    if (x < -1)
        x = -1;
    return int(std::round(x * 255));
}

void sendMotorCmd(int fd, const char *id, int spd)
{
    if (fd < 0)
        return; // nếu chưa mở UART thì bỏ qua
    char buf[32];
    int n = snprintf(buf, sizeof(buf), "%s %d\n", id, spd);
    (void)write(fd, buf, n);
}

#include "LineFollow.hpp"
#include "BallDetection.hpp" // lớp bạn có: trả về primary ball (angle, dist, score,...)

using namespace cv;

enum State
{
    FOLLOW_LINE,
    BALL_SEEN,
    APPROACH_BALL,
    INTAKE,
    RELOCK_LINE
};

static const int WIDTH = 640, HEIGHT = 480, FPS = 30;
static const int ROI_TOP = HEIGHT * 2 / 3;
static const float DT = 0.02f; // 20ms
static const auto CONTROL_DT = std::chrono::milliseconds(20);

// Ngưỡng cho bóng (tùy bộ detect của bạn)
static const float Smin = 0.55f;
static const float D_detect_max = 2.5f; // m
static const int N_detect_hold = 3;     // cần thấy liên tiếp

// Áp sát & intake
static const float D_intake_on = 0.45f; // m: bật M3
static const float D_captured = 0.25f;  // m: coi như đã vào miệng
static const int INTAKE_PWM = 220;      // lực cuốn

// Tìm lại line
static const int LINE_OK_HOLD = 4; // số khung có line để coi là relock ok

// UART helpers (Linux/Jetson; Windows đổi API)
int open_uart(const char *dev = "/dev/ttyUSB0", int baud = 115200)
{
    int fd = open(dev, O_RDWR | O_NOCTTY | O_NONBLOCK);
    if (fd < 0)
        return -1;
    termios tio{};
    tcgetattr(fd, &tio);
    cfmakeraw(&tio);
    cfsetspeed(&tio, B115200);
    tio.c_cflag |= (CLOCAL | CREAD);
    tcsetattr(fd, TCSANOW, &tio);
    return fd;
}
int clamp255(float x)
{
    if (x > 1)
        x = 1;
    if (x < -1)
        x = -1;
    return int(std::round(x * 255));
}
void sendMotorCmd(int fd, const char *id, int spd)
{
    char buf[32];
    int n = snprintf(buf, sizeof(buf), "%s %d\n", id, spd);
    write(fd, buf, n);
}
void driveLR(int fd, float vL, float vR)
{
    sendMotorCmd(fd, "M1", clamp255(vL));
    sendMotorCmd(fd, "M2", clamp255(vR));
}
void intake(int fd, int pwm) { sendMotorCmd(fd, "M3", std::max(-255, std::min(255, pwm))); }

int main()
{
    // --- Serial to Arduino ---
    int uart = open_uart("/dev/ttyUSB0", 115200); // sửa port cho đúng máy bạn
    if (uart < 0)
    {
        perror("UART"); /* vẫn cho chạy demo hình ảnh */
    }

    // --- RealSense ---
    rs2::pipeline pipe;
    rs2::config cfg;
    cfg.enable_stream(RS2_STREAM_COLOR, WIDTH, HEIGHT, RS2_FORMAT_BGR8, FPS);
    cfg.enable_stream(RS2_STREAM_DEPTH, WIDTH, HEIGHT, RS2_FORMAT_Z16, FPS);
    auto profile = pipe.start(cfg);

    // Detector & follower
    LineFollower follower(0.015f, 0.0f, 0.004f, 0.25f, DT);
    ball::Detector det(/*tham số của bạn*/);

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

        // ---- LINE MASK (B-dominant, đơn giản) ----
        Mat mask = Mat::zeros(HEIGHT, WIDTH, CV_8U);
        const int T = 30, Vmin = 60;
        for (int y = ROI_TOP; y < HEIGHT; ++y)
        {
            const Vec3b *row = bgr.ptr<Vec3b>(y);
            uchar *mrow = mask.ptr<uchar>(y);
            for (int x = 0; x < WIDTH; ++x)
            {
                int B = row[x][0], G = row[x][1], R = row[x][2], V = std::max({B, G, R});
                if (B > G + T && B > R + T && V >= Vmin)
                    mrow[x] = 255;
            }
        }
        morphologyEx(mask, mask, MORPH_CLOSE, getStructuringElement(MORPH_RECT, {5, 5}), Point(-1, -1), 2);

        // ---- Find centroid of largest contour (line) ----
        std::vector<std::vector<Point>> contours;
        findContours(mask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
        int centerX = WIDTH / 2;
        int cx_line = -1;
        if (!contours.empty())
        {
            auto cmax = *std::max_element(contours.begin(), contours.end(), [](auto &a, auto &b)
                                          { return contourArea(a) < contourArea(b); });
            Moments M = moments(cmax);
            if (M.m00 >= 1e-3)
                cx_line = int(M.m10 / M.m00);
        }

        // ---- Ball detection (dùng API trong BallDetection.hpp của bạn) ----
        ball::Result r = det.detect(bgr, depth); // giả định: r.valid, r.dist_m, r.angle_deg, r.score

        // ---- State machine ----
        float vL = 0, vR = 0; // tốc độ chuẩn hoá [-1..1]
        switch (st)
        {
        case FOLLOW_LINE:
        {
            if (cx_line >= 0)
            {
                line_ok_hold = std::min(LINE_OK_HOLD, line_ok_hold + 1);
                auto [l, r] = follower.update(cx_line, centerX);
                vL = l;
                vR = r;
            }
            else
            {
                line_ok_hold = 0;
                vL = -0.18f;
                vR = 0.18f; // quay tìm line
            }
            // thấy bóng đủ tốt?
            if (r.valid && r.score > Smin && r.dist_m < D_detect_max)
            {
                if (++seen_hold >= N_detect_hold)
                {
                    st = BALL_SEEN;
                }
            }
            else
                seen_hold = 0;
            break;
        }
        case BALL_SEEN:
        {
            // chốt chuyển ngay sang tiếp cận, dừng intake (chưa bật)
            seen_hold = 0;
            st = APPROACH_BALL;
            break;
        }
        case APPROACH_BALL:
        {
            if (r.valid)
            {
                // điều khiển “góc-lái, tiến-chậm theo khoảng cách”
                float ang = r.angle_deg;                                      // >0 nghĩa là bóng nằm bên phải
                float omega = std::clamp(ang / 30.0f, -0.8f, 0.8f);           // quay theo góc
                float fwd = std::clamp((r.dist_m - 0.3f) / 0.8f, 0.2f, 0.6f); // tiến tới, chậm dần
                vL = std::clamp(fwd - 0.6f * omega, -0.8f, 0.8f);
                vR = std::clamp(fwd + 0.6f * omega, -0.8f, 0.8f);
                if (uart >= 0 && r.dist_m < D_intake_on)
                    intake(uart, +INTAKE_PWM);
                if (r.dist_m < D_captured)
                {
                    st = INTAKE;
                }
            }
            else
            {
                // mất bóng: quét tại chỗ
                vL = -0.15f;
                vR = 0.15f;
            }
            break;
        }
        case INTAKE:
        {
            if (uart >= 0)
                intake(uart, +INTAKE_PWM);
            // tiến nhẹ thêm 0.5s rồi sang relock line
            static auto t0 = std::chrono::steady_clock::now();
            static bool started = false;
            if (!started)
            {
                t0 = std::chrono::steady_clock::now();
                started = true;
            }
            vL = vR = 0.25f;
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
                auto [l, r] = follower.update(cx_line, centerX);
                vL = l;
                vR = r;
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
                vR = 0.18f; // quay tìm line
            }
            // tắt intake khi đã bắt đầu relock
            if (uart >= 0)
                intake(uart, 0);
            break;
        }
        }

        // ---- Gửi motor xuống Arduino ----
        if (uart >= 0)
        {
            driveLR(uart, vL, vR);
        }

        // ---- Overlay debug ----
        line(bgr, {WIDTH / 2, 0}, {WIDTH / 2, HEIGHT - 1}, {255, 255, 0}, 1, LINE_AA);
        line(bgr, {0, ROI_TOP}, {WIDTH - 1, ROI_TOP}, {200, 200, 200}, 1, LINE_AA);
        if (cx_line >= 0)
        {
            line(bgr, {cx_line, ROI_TOP}, {cx_line, HEIGHT - 1}, {0, 255, 255}, 2, LINE_AA);
        }
        char txt[128];
        snprintf(txt, sizeof(txt), "ST=%d  vL=%.2f vR=%.2f  ball:%s d=%.2fm ang=%.1f sc=%.2f",
                 st, vL, vR, r.valid ? "Y" : "N", r.dist_m, r.angle_deg, r.score);
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
