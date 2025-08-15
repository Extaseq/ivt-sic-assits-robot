#include <iostream>
#include <fcntl.h>      // open
#include <termios.h>    // struct termios
#include <unistd.h>     // write, read, sleep
#include <cstring>      // strlen

int main() {
    const char* port = "/dev/ttyTHS1";   // Cổng UART Jetson
    int serial_port = open(port, O_RDWR);

    if (serial_port < 0) {
        std::cerr << "Không mở được " << port << std::endl;
        return 1;
    }

    // Cấu hình UART
    struct termios tty;
    if (tcgetattr(serial_port, &tty) != 0) {
        std::cerr << "Lỗi: Không lấy được cấu hình UART\n";
        close(serial_port);
        return 1;
    }

    // Thiết lập baudrate = 9600
    cfsetospeed(&tty, B9600);
    cfsetispeed(&tty, B9600);

    // 8N1 (8 data bits, no parity, 1 stop bit)
    tty.c_cflag &= ~PARENB; // tắt parity
    tty.c_cflag &= ~CSTOPB; // 1 stop bit
    tty.c_cflag &= ~CSIZE;
    tty.c_cflag |= CS8;     // 8 data bits

    tty.c_cflag &= ~CRTSCTS;        // tắt flow control
    tty.c_cflag |= CREAD | CLOCAL;  // bật receiver, bỏ modem control

    tty.c_lflag &= ~ICANON;
    tty.c_lflag &= ~ECHO;
    tty.c_lflag &= ~ECHOE;
    tty.c_lflag &= ~ECHONL;
    tty.c_lflag &= ~ISIG; // tắt Ctrl-C/Z

    tty.c_iflag &= ~(IXON | IXOFF | IXANY); // tắt software flow control
    tty.c_iflag &= ~(IGNBRK | BRKINT | PARMRK | ISTRIP | INLCR | IGNCR | ICRNL);

    tty.c_oflag &= ~OPOST;
    tty.c_oflag &= ~ONLCR;

    // Áp dụng cấu hình
    if (tcsetattr(serial_port, TCSANOW, &tty) != 0) {
        std::cerr << "Lỗi: Không set được UART config\n";
        close(serial_port);
        return 1;
    }

    std::cout << "Đang gửi dữ liệu tới Arduino qua UART..." << std::endl;

    while (true) {
        const char* msg = "m\n";
        write(serial_port, msg, strlen(msg));
        sleep(1);
    }

    close(serial_port);
    return 0;
}
