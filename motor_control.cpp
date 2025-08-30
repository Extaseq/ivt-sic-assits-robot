#include <iostream>
#include <string>
#include <fcntl.h>
#include <termios.h>
#include <unistd.h>
#include <cstring>
#include <thread>
#include <chrono>

class MotorController {
private:
    int uart_fd;
    bool left_turning;
    bool right_turning;
    float current_speed;
    
public:
    MotorController() : uart_fd(-1), left_turning(false), right_turning(false), current_speed(0.0f) {}
    
    ~MotorController() {
        if (uart_fd >= 0) {
            stop();
            close(uart_fd);
            std::cout << "UART connection closed." << std::endl;
        }
    }
    
    // Mở kết nối UART
    bool open_uart(const char *dev = "/dev/ttyACM0", int baud = 115200) {
        uart_fd = open(dev, O_RDWR | O_NOCTTY | O_NONBLOCK);
        if (uart_fd < 0) {
            std::cerr << "Failed to open " << dev << ". Trying alternatives..." << std::endl;

            // Thử các device khác
            const char *alternatives[] = {"/dev/ttyUSB0", "/dev/ttyTHS1", "/dev/ttyAMA0"};
            for (const char *alt_dev : alternatives) {
                uart_fd = open(alt_dev, O_RDWR | O_NOCTTY | O_NONBLOCK);
                if (uart_fd >= 0) {
                    std::cout << "Connected to " << alt_dev << std::endl;
                    break;
                }
            }

            if (uart_fd < 0) {
                std::cerr << "Could not open any UART device" << std::endl;
                return false;
            }
        }

        termios tio{};
        if (tcgetattr(uart_fd, &tio) != 0) {
            close(uart_fd);
            uart_fd = -1;
            return false;
        }

        // Cấu hình serial port
        cfmakeraw(&tio);
        tio.c_cflag &= ~PARENB; // No parity
        tio.c_cflag &= ~CSTOPB; // 1 stop bit
        tio.c_cflag &= ~CSIZE;
        tio.c_cflag |= CS8;      // 8 data bits
        tio.c_cflag &= ~CRTSCTS; // No hardware flow control
        tio.c_cflag |= CREAD | CLOCAL;

        tio.c_cc[VMIN] = 0;
        tio.c_cc[VTIME] = 10; // Timeout in deciseconds

        // Set baud rate
        speed_t spd = B115200;
        cfsetispeed(&tio, spd);
        cfsetospeed(&tio, spd);

        if (tcsetattr(uart_fd, TCSANOW, &tio) != 0) {
            close(uart_fd);
            uart_fd = -1;
            return false;
        }

        // Clear buffer
        tcflush(uart_fd, TCIOFLUSH);
        
        std::cout << "UART opened successfully: " << dev << std::endl;
        return true;
    }
    
    // Gửi command qua UART
    void send_command(const std::string& command) {
        if (uart_fd < 0) {
            std::cout << "[SIMULATION] " << command << std::endl;
            return;
        }
        
        std::string cmd = command + "\n";
        ssize_t bytes_written = write(uart_fd, cmd.c_str(), cmd.length());
        if (bytes_written != (ssize_t)cmd.length()) {
            std::cerr << "UART write error: " << bytes_written << " bytes written, expected " << cmd.length() << std::endl;
        }
        
        // Đảm bảo dữ liệu được gửi hoàn toàn
        tcdrain(uart_fd);
        usleep(10000); // 10ms delay
        
        std::cout << "Sent: " << command << std::endl;
    }
    
    // Điều khiển motor riêng lẻ - M1, M2, hoặc M3
    void control_motor(int motor_id, int speed) {
        if (motor_id < 1 || motor_id > 3) {
            std::cout << "Invalid motor ID. Use 1, 2, or 3" << std::endl;
            return;
        }
        
        if (speed < 0) speed = 0;
        if (speed > 255) speed = 255;
        
        std::string command = "M" + std::to_string(motor_id) + " " + std::to_string(speed);
        send_command(command);
    }
    
    // M <speed> - Di chuyển với tốc độ (0-255) - gửi lệnh cho cả M1 và M2
    void move(float speed) {
        if (speed < 0) speed = 0;
        if (speed > 255) speed = 255;
        
        current_speed = speed;
        left_turning = false;
        right_turning = false;
        
        // Gửi lệnh cho cả 2 motor để đi thẳng
        std::string command1 = "M1 " + std::to_string((int)speed);
        std::string command2 = "M2 " + std::to_string((int)speed);
        send_command(command1);
        send_command(command2);
    }
    
    // L - Toggle turn left (M1 chậm, M2 nhanh)
    void turn_left() {
        left_turning = !left_turning;
        right_turning = false; // Tắt turn right nếu đang bật
        
        if (left_turning) {
            // Turn left: M1 chậm (hoặc dừng), M2 nhanh
            send_command("M1 100");  // Motor trái chậm
            send_command("M2 200");  // Motor phải nhanh
            std::cout << "Turn LEFT started" << std::endl;
        } else {
            // Dừng turn, trở về tốc độ bình thường
            int speed = (int)current_speed;
            std::string command1 = "M1 " + std::to_string(speed);
            std::string command2 = "M2 " + std::to_string(speed);
            send_command(command1);
            send_command(command2);
            std::cout << "Turn LEFT stopped" << std::endl;
        }
    }
    
    // R - Toggle turn right (M1 nhanh, M2 chậm)
    void turn_right() {
        right_turning = !right_turning;
        left_turning = false; // Tắt turn left nếu đang bật
        
        if (right_turning) {
            // Turn right: M1 nhanh, M2 chậm (hoặc dừng)
            send_command("M1 200");  // Motor trái nhanh
            send_command("M2 100");  // Motor phải chậm
            std::cout << "Turn RIGHT started" << std::endl;
        } else {
            // Dừng turn, trở về tốc độ bình thường
            int speed = (int)current_speed;
            std::string command1 = "M1 " + std::to_string(speed);
            std::string command2 = "M2 " + std::to_string(speed);
            send_command(command1);
            send_command(command2);
            std::cout << "Turn RIGHT stopped" << std::endl;
        }
    }
    
    // S - Stop all motors
    void stop() {
        current_speed = 0.0f;
        left_turning = false;
        right_turning = false;
        
        send_command("M1 127"); // PWM 127 = stop (middle value)
        send_command("M2 127"); // PWM 127 = stop (middle value)
        std::cout << "All motors STOPPED" << std::endl;
    }
    
    // Hiển thị trạng thái hiện tại
    void show_status() {
        std::cout << "\n=== MOTOR STATUS ===" << std::endl;
        std::cout << "Speed: " << current_speed << std::endl;
        std::cout << "Left Turn: " << (left_turning ? "ON" : "OFF") << std::endl;
        std::cout << "Right Turn: " << (right_turning ? "ON" : "OFF") << std::endl;
        std::cout << "UART: " << (uart_fd >= 0 ? "Connected" : "Simulation") << std::endl;
        std::cout << "===================" << std::endl;
    }
};

void print_help() {
    std::cout << "\n=== MOTOR CONTROL COMMANDS ===" << std::endl;
    std::cout << "M <speed>  - Move forward with speed (0-255), sends M1 & M2" << std::endl;
    std::cout << "M1 <speed> - Control Motor 1 directly (0-255)" << std::endl;
    std::cout << "M2 <speed> - Control Motor 2 directly (0-255)" << std::endl;
    std::cout << "M3 <speed> - Control Motor 3 directly (0-255)" << std::endl;
    std::cout << "L          - Toggle turn left (M1 slow, M2 fast)" << std::endl;
    std::cout << "R          - Toggle turn right (M1 fast, M2 slow)" << std::endl;
    std::cout << "S          - Stop all motors (M1 & M2 = 127)" << std::endl;
    std::cout << "status     - Show current motor status" << std::endl;
    std::cout << "help       - Show this help" << std::endl;
    std::cout << "quit       - Exit program" << std::endl;
    std::cout << "\nArduino Protocol:" << std::endl;
    std::cout << "- PWM 127 = Stop, <127 = Reverse, >127 = Forward" << std::endl;
    std::cout << "- Range: 0-255 (0=full reverse, 127=stop, 255=full forward)" << std::endl;
    std::cout << "===============================" << std::endl;
}

int main() {
    MotorController motor;
    
    std::cout << "Motor Control System Starting..." << std::endl;
    
    // Thử kết nối UART
    if (!motor.open_uart()) {
        std::cout << "Running in SIMULATION mode (no UART connection)" << std::endl;
    }
    
    print_help();
    
    std::string input;
    while (true) {
        std::cout << "\nmotor> ";
        std::getline(std::cin, input);
        
        // Loại bỏ khoảng trắng đầu cuối
        size_t start = input.find_first_not_of(" \t");
        if (start == std::string::npos) continue; // Empty line
        size_t end = input.find_last_not_of(" \t");
        input = input.substr(start, end - start + 1);
        
        if (input.empty()) continue;
        
        // Parse commands
        if (input == "quit" || input == "q" || input == "exit") {
            std::cout << "Stopping motors and exiting..." << std::endl;
            motor.stop();
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            break;
        }
        else if (input == "help" || input == "h") {
            print_help();
        }
        else if (input == "status") {
            motor.show_status();
        }
        else if (input == "S" || input == "s") {
            motor.stop();
        }
        else if (input == "L" || input == "l") {
            motor.turn_left();
        }
        else if (input == "R" || input == "r") {
            motor.turn_right();
        }
        else if (input.length() >= 2 && (input[0] == 'M' || input[0] == 'm') && input[1] == ' ') {
            // Parse M <speed> - điều khiển cả M1 và M2
            try {
                float speed = std::stof(input.substr(2));
                motor.move(speed);
            } catch (const std::exception& e) {
                std::cout << "Invalid speed format. Use: M <speed>" << std::endl;
            }
        }
        else if (input.length() >= 3 && (input[0] == 'M' || input[0] == 'm') && 
                 (input[1] >= '1' && input[1] <= '3') && input[2] == ' ') {
            // Parse M1 <speed>, M2 <speed>, M3 <speed>
            try {
                int motor_id = input[1] - '0';
                int speed = std::stoi(input.substr(3));
                motor.control_motor(motor_id, speed);
            } catch (const std::exception& e) {
                std::cout << "Invalid format. Use: M1 <speed>, M2 <speed>, or M3 <speed>" << std::endl;
            }
        }
        else {
            std::cout << "Unknown command: " << input << std::endl;
            std::cout << "Type 'help' for available commands." << std::endl;
        }
    }
    
    return 0;
}
