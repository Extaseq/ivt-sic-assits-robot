#pragma once

#include <algorithm>
#include <utility>

#define WHEEL_CMD std::pair<float, float>

class LineFollower {
public:
    explicit LineFollower(float Kp = 0.015f, float Ki = 0.0f, float Kd = 0.004f,
                          float base = 0.25f, float dt = 0.1f)
        : Kp_(Kp), Ki_(Ki), Kd_(Kd), BASE_(base), dt_(dt) {}

    void setBase(float base)                   { BASE_ = base; }
    void setGains(float Kp,float Ki,float Kd)  { Kp_=Kp; Ki_=Ki; Kd_=Kd; }
    void setDt(float dt)                       { dt_ = dt; }
    void setSteerLimit(float s)                { steer_limit_ = std::max(0.f, s); }
    void setSpeedLimit(float s)                { speed_limit_ = std::max(0.f, s); }
    void setSlewRate(float s)                  { slew_rate_ = std::max(0.f, s); }

    void reset() { prev_error_ = 0.0f; integ_ = 0.0f; prev_vL_ = prev_vR_ = 0.0f; }

    WHEEL_CMD update(int cx, int centerX) {
        float err = static_cast<float>(cx - centerX);
        return updateByError(err);
    }

    WHEEL_CMD updateByError(float error) {
        integ_ += error * dt_;
        float derr = (error - prev_error_) / (dt_ > 1e-6 ? dt_ : 1.0f);
        float steer = Kp_ * error + Ki_ * integ_ + Kd_ * derr;

        steer = clamp(steer, -steer_limit_, steer_limit_);

        float vL = BASE_ - steer, vR = BASE_ + steer;

        vL = clamp(vL, -speed_limit_, speed_limit_);
        vR = clamp(vR, -speed_limit_, speed_limit_);

        if (slew_rate_ > 0.0f) {
            vL = clamp(vL, prev_vL_ - slew_rate_, prev_vL_ + slew_rate_);
            vR = clamp(vR, prev_vR_ - slew_rate_, prev_vR_ + slew_rate_);
        }

        prev_error_ = error;
        prev_vL_ = vL;
        prev_vR_ = vR;
        return {vL, vR};
    }

private:
    static float clamp(float x, float lo, float hi) {
        return std::max(lo, std::min(hi, x));
    }

    float Kp_{0.015f}, Ki_{0.0f}, Kd_{0.004f};
    float BASE_{0.25f};             // Base speed (-1, 1)
    float dt_{0.1f};                // Control cycle time in seconds (10Hz -> 0.1s)
    float steer_limit_{0.6f};       // Maximum steering angle (-0.6, 0.6)
    float speed_limit_{1.0f};       // Maximum speed (-1, 1)
    float slew_rate_{0.15f};        // Maximum change in speed per second

    // State
    float prev_error_{0.0f};
    float integ_{0.0f};
    float prev_vL_{0.0f}, prev_vR_{0.0f};
};