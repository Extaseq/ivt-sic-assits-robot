#pragma once
#include <opencv2/opencv.hpp>
#include <cmath>
#include <cstdint>
#include <utility>
#include <algorithm>
#include <vector>

// ====== BallDetection.hpp (multi-candidate + target lock) ======
// - HSV + morphology + contour filters + depth-radius sanity
// - Scoring nhiều ứng viên, ưu tiên gần/giữa/ở thấp ảnh
// - Target lock (nearest-neighbor qua thời gian) để không nhảy mục tiêu
// - Hysteresis + debounce cho capture window
// Yêu cầu: OpenCV, depth Z16 (uint16) + depth_scale (m/tick)

namespace ball {

    struct Result {
        bool  found = false;
        float distance_m = 0.0f;   // Z (m), median quanh (cx,cy)
        float angle_rad = 0.0f;    // góc lệch ngang
        int   cx = -1, cy = -1;    // toạ độ pixel
        int   radius_px = 0;       // bán kính ảnh
        bool  in_capture_win = false;
        bool  very_close = false;
        float score = 0.0f;        // điểm chấm cho ứng viên được chọn
    };

    struct Candidate {
        Result r;
        float  score = 0.0f;
    };

    struct Params {
        // HSV cho tennis ball (OpenCV H[0..179])
        cv::Scalar hsv_low  {20, 80, 120};
        cv::Scalar hsv_high {45, 255, 255};

        // Morphology
        int morph_kernel = 5;
        int morph_open_iters = 2;
        int morph_close_iters = 2;

        // Contour filters
        double min_area = 300.0;       // px^2
        double min_circularity = 0.6;  // 4πA/P^2

        // Depth sanity (D ≈ 0.067 m)
        float ball_diameter_m = 0.067f;
        float radius_ratio_lo = 0.6f;  // cho phép [0.6..1.6] × kỳ vọng
        float radius_ratio_hi = 1.6f;

        // Capture window (hysteresis + debounce)
        float z_cap_on  = 0.45f;              // m
        float z_cap_off = 0.55f;              // m
        float th_cap_on_rad  = 8.0f * (float)M_PI/180.0f;
        float th_cap_off_rad =10.0f * (float)M_PI/180.0f;
        float roi_cap_on_frac  = 0.60f;       // cy >= H*on
        float roi_cap_off_frac = 0.55f;       // OFF khi cy < H*off
        int   debounce_on_frames  = 3;
        int   debounce_off_frames = 2;

        // Very close
        float z_close = 0.28f;
        float th_close_rad = 5.0f * (float)M_PI/180.0f;

        // ROI detection (bỏ phần trên ảnh)
        float roi_top_frac = 0.0f;

        // Depth sampling median window (odd)
        int depth_kernel = 7;

        // --- Scoring nhiều ứng viên ---
        // score = w_dist*(1/Z) + w_ang*(1 - |ang|/th_max) + w_row*(cy/H) + w_lock*lock_bonus
        float w_dist = 1.0f;
        float w_ang  = 0.7f;
        float w_row  = 0.3f;                                  // ưu tiên bóng ở thấp ảnh (gần mũi)
        float th_score_max_rad = 15.0f * (float)M_PI/180.0f;  // chuẩn hoá góc

        // --- Target lock ---
        bool  enable_lock = true;
        float lock_max_px = 80.0f;   // bán kính ưu tiên NN theo pixel
        int   lock_forget_frames = 5;
        float w_lock = 0.5f;         // trọng số thưởng cho ứng viên gần lock

        // Trả về tối đa N ứng viên (để debug/vẽ)
        int topk_candidates = 3;
    };

    class Detector {
    public:
        explicit Detector(const Params& p = Params{}) : p_(p) {}

        void setHSV(const cv::Scalar& low, const cv::Scalar& high) { p_.hsv_low=low; p_.hsv_high=high; }

        // detect: trả về ứng viên được chọn (primary).
        // Nếu cần danh sách ứng viên (đã chấm điểm, sắp xếp giảm dần), truyền con trỏ out_vec (tuỳ chọn).
        Result detect(const cv::Mat& bgr,
                    const cv::Mat& depth_z16,
                    float fx, float ppx, float depth_scale,
                    std::vector<Candidate>* out_candidates = nullptr)
        {
            Result primary;
            if (bgr.empty()) return finalize(primary);

            // 1) HSV mask + ROI
            cv::Mat hsv; cv::cvtColor(bgr, hsv, cv::COLOR_BGR2HSV);
            cv::Mat mask; cv::inRange(hsv, p_.hsv_low, p_.hsv_high, mask);
            if (p_.roi_top_frac > 0.f) {
                int cut = std::clamp((int)std::round(bgr.rows * p_.roi_top_frac), 0, bgr.rows);
                if (cut > 0) mask(cv::Rect(0,0, bgr.cols, cut)).setTo(0);
            }

            // 2) Morphology: OPEN -> CLOSE
            cv::Mat k = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(p_.morph_kernel, p_.morph_kernel));
            if (p_.morph_open_iters  > 0) cv::morphologyEx(mask, mask, cv::MORPH_OPEN,  k, {-1,-1}, p_.morph_open_iters);
            if (p_.morph_close_iters > 0) cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, k, {-1,-1}, p_.morph_close_iters);

            // 3) Contours -> duyệt tất cả
            std::vector<std::vector<cv::Point>> cnts;
            cv::findContours(mask, cnts, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

            std::vector<Candidate> cands;
            cands.reserve(cnts.size());

            for (const auto& c : cnts) {
                double area = cv::contourArea(c);
                if (area < p_.min_area) continue;

                double peri = cv::arcLength(c, true);
                double circ = (peri > 1e-6) ? 4.0 * M_PI * area / (peri*peri) : 0.0;
                if (circ < p_.min_circularity) continue;

                cv::Point2f center; float radius=0.f;
                cv::minEnclosingCircle(c, center, radius);
                cv::Moments M = cv::moments(c);
                if (M.m00 < 1e-6) continue;
                int cx = int(M.m10 / M.m00);
                int cy = int(M.m01 / M.m00);

                // Depth median quanh (cx,cy)
                float Z = 0.0f;
                if (!depth_z16.empty() && depth_z16.type()==CV_16UC1 && depth_scale>0.f) {
                    Z = medianDepth(depth_z16, cx, cy, p_.depth_kernel, depth_scale);
                }

                // Góc
                float angle = 0.0f;
                if (fx > 1e-3f) angle = std::atan( ( (float)cx - ppx ) / fx );

                // Sanity radius-depth nếu có Z
                if (Z > 0.f && fx > 1e-3f) {
                    float r_expected = (fx * (p_.ball_diameter_m * 0.5f)) / Z;
                    if (!(radius >= p_.radius_ratio_lo * r_expected && radius <= p_.radius_ratio_hi * r_expected)) {
                        continue;
                    }
                }

                // Tạo ứng viên
                Candidate cand;
                cand.r.found = true;
                cand.r.distance_m = Z;
                cand.r.angle_rad  = angle;
                cand.r.cx = cx; cand.r.cy = cy; cand.r.radius_px = (int)std::round(radius);

                // Scoring
                float s = 0.f;
                // khoảng cách: 1/Z (nếu có Z)
                if (Z > 0.f) s += p_.w_dist * (1.0f / (Z + 1e-6f));
                // góc: chuẩn hoá về [0..1]
                {
                    float th = p_.th_score_max_rad;
                    float ang_term = 1.0f - std::min(std::fabs(angle), th) / th;
                    s += p_.w_ang * ang_term;
                }
                // vị trí dọc ảnh (cy/H): càng thấp càng tốt
                s += p_.w_row * ( (float)cy / std::max(1, bgr.rows) );

                // lock bonus nếu enable
                if (p_.enable_lock && has_lock_) {
                    float dx = (float)cx - lock_cx_;
                    float dy = (float)cy - lock_cy_;
                    float d  = std::sqrt(dx*dx + dy*dy);
                    float bonus = 1.0f - std::min(d, p_.lock_max_px) / p_.lock_max_px; // [0..1]
                    s += p_.w_lock * bonus;
                }

                cand.score = s;
                cand.r.score = s;
                cands.push_back(std::move(cand));
            }

            if (cands.empty()) {
                // mất lock nếu không thấy quá nhiều khung
                if (has_lock_) {
                    if (++lock_miss_ >= p_.lock_forget_frames) { has_lock_ = false; lock_miss_ = 0; }
                }
                return finalize(primary);
            }

            // Sắp xếp theo score giảm dần
            std::sort(cands.begin(), cands.end(),
                    [](const Candidate& a, const Candidate& b){ return a.score > b.score; });

            // Nếu đang lock, ưu tiên NN trong bán kính lock_max_px
            int best_idx = 0;
            if (p_.enable_lock && has_lock_) {
                float best_d = p_.lock_max_px;
                bool  found_nn = false;
                for (int i=0; i<(int)cands.size(); ++i) {
                    float dx = (float)cands[i].r.cx - lock_cx_;
                    float dy = (float)cands[i].r.cy - lock_cy_;
                    float d  = std::sqrt(dx*dx + dy*dy);
                    if (d <= best_d) { best_d = d; best_idx = i; found_nn = true; }
                }
                // nếu không có NN trong bán kính → vẫn lấy theo score
                (void)found_nn;
            }

            // Chọn primary
            primary = cands[best_idx].r;

            // Cập nhật lock
            if (p_.enable_lock) {
                has_lock_ = true;
                lock_cx_ = (float)primary.cx;
                lock_cy_ = (float)primary.cy;
                lock_miss_ = 0;
            }

            // Capture window + very close
            applyCaptureWindows(primary, bgr.rows);

            // Trả về danh sách ứng viên top-k nếu cần
            if (out_candidates) {
                int K = std::min<int>(p_.topk_candidates, (int)cands.size());
                out_candidates->assign(cands.begin(), cands.begin()+K);
            }

            return finalize(primary);
        }

        // Reset lock/hysteresis (khi chuyển trạng thái hệ thống)
        void reset() {
            in_capture_ = false;
            on_count_ = off_count_ = 0;
            has_lock_ = false;
            lock_miss_ = 0;
            lock_cx_ = lock_cy_ = 0.f;
        }

    private:
        Params p_;

        // Hysteresis/debounce cho capture window
        bool in_capture_ = false;
        int  on_count_ = 0;
        int  off_count_ = 0;

        // Target lock state
        bool  has_lock_ = false;
        float lock_cx_ = 0.f, lock_cy_ = 0.f;
        int   lock_miss_ = 0;

        static float medianDepth(const cv::Mat& z16, int x, int y, int k, float scale) {
            int r = std::max(1, k/2);
            int x0 = std::clamp(x - r, 0, z16.cols-1);
            int y0 = std::clamp(y - r, 0, z16.rows-1);
            int x1 = std::clamp(x + r, 0, z16.cols-1);
            int y1 = std::clamp(y + r, 0, z16.rows-1);
            std::vector<uint16_t> v;
            v.reserve((x1-x0+1)*(y1-y0+1));
            for (int j=y0; j<=y1; ++j) {
                const uint16_t* row = z16.ptr<uint16_t>(j);
                for (int i=x0; i<=x1; ++i) {
                    uint16_t d = row[i];
                    if (d > 0) v.push_back(d);
                }
            }
            if (v.empty()) return 0.0f;
            size_t m = v.size()/2;
            std::nth_element(v.begin(), v.begin()+m, v.end());
            uint16_t med = v[m];
            return med * scale;
        }

        void applyCaptureWindows(Result& r, int H) {
            // Hysteresis ON/OFF conditions
            bool cond_on =
                r.found &&
                (r.distance_m > 0.f ? (r.distance_m <= p_.z_cap_on) : false) &&
                (std::fabs(r.angle_rad) <= p_.th_cap_on_rad) &&
                (r.cy >= (int)std::round(H * p_.roi_cap_on_frac));

            bool cond_off =
                (!r.found) ||
                (r.distance_m <= 0.f ? true : (r.distance_m >= p_.z_cap_off)) ||
                (std::fabs(r.angle_rad) >= p_.th_cap_off_rad) ||
                (r.cy < (int)std::round(H * p_.roi_cap_off_frac));

            if (cond_on)  ++on_count_;  else on_count_ = 0;
            if (cond_off) ++off_count_; else off_count_ = 0;

            if (!in_capture_) {
                if (on_count_ >= p_.debounce_on_frames) {
                    in_capture_ = true;
                    off_count_ = 0;
                }
            } else {
                if (off_count_ >= p_.debounce_off_frames) {
                    in_capture_ = false;
                    on_count_ = 0;
                }
            }

            r.in_capture_win = in_capture_;
            r.very_close = r.found &&
                        r.distance_m > 0.f &&
                        r.distance_m <= p_.z_close &&
                        std::fabs(r.angle_rad) <= p_.th_close_rad;
        }

        static Result finalize(Result r) { return r; }
    };

} // namespace ball