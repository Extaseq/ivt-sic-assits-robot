#include "../LineFollow.hpp"
#include <cassert>
#include <cmath>
#include <iostream>

int main() {
    LineFollower lf(1.0f, 0.0f, 0.0f, 0.0f, 0.1f); // dt=0.1s
    lf.setSpeedLimit(1.0f);
    lf.setSteerLimit(1.0f);
    lf.setSlewRate(0.5f); // 0.5 units per second

    auto cmd1 = lf.update(100, 0); // large error to saturate steer
    assert(std::fabs(cmd1.first + 0.05f) < 1e-5f);
    assert(std::fabs(cmd1.second - 0.05f) < 1e-5f);

    auto cmd2 = lf.update(100, 0);
    assert(std::fabs(cmd2.first + 0.10f) < 1e-5f);
    assert(std::fabs(cmd2.second - 0.10f) < 1e-5f);

    std::cout << "Slew rate test passed\n";
    return 0;
}
