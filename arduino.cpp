#include <Arduino.h>

// Motor pin definitions - ĐIỀU CHỈNH THEO PHẦN CỨNG THỰC TẾ
const int M1_IN1 = 2, M1_IN2 = 4, M1_PWM = 5;   // Left wheel
const int M2_IN1 = 7, M2_IN2 = 8, M2_PWM = 6;   // Right wheel
const int M3_IN1 = 10, M3_IN2 = 11, M3_PWM = 9; // Intake motor

void setup()
{
    Serial.begin(115200);

    // Chờ kết nối Serial - QUAN TRỌNG cho ttyACM0
    while (!Serial)
    {
        delay(10);
    }

    // Setup motor pins
    pinMode(M1_IN1, OUTPUT);
    pinMode(M1_IN2, OUTPUT);
    pinMode(M1_PWM, OUTPUT);

    pinMode(M2_IN1, OUTPUT);
    pinMode(M2_IN2, OUTPUT);
    pinMode(M2_PWM, OUTPUT);

    pinMode(M3_IN1, OUTPUT);
    pinMode(M3_IN2, OUTPUT);
    pinMode(M3_PWM, OUTPUT);

    // Dừng tất cả motor ban đầu
    digitalWrite(M1_IN1, LOW);
    digitalWrite(M1_IN2, LOW);
    analogWrite(M1_PWM, 0);

    digitalWrite(M2_IN1, LOW);
    digitalWrite(M2_IN2, LOW);
    analogWrite(M2_PWM, 0);

    digitalWrite(M3_IN1, LOW);
    digitalWrite(M3_IN2, LOW);
    analogWrite(M3_PWM, 0);

    Serial.println("READY: Arduino connected via ttyACM0");
    Serial.println("Send commands: M1 <speed>, M2 <speed>, M3 <speed>");
    Serial.println("Speed range: -255 to 255");
}

void setMotor(int in1, int in2, int pwmPin, int speed)
{
    // Giới hạn tốc độ
    if (speed > 255)
        speed = 255;
    if (speed < -255)
        speed = -255;

    if (speed > 0)
    {
        // Chạy thuận
        digitalWrite(in1, HIGH);
        digitalWrite(in2, LOW);
        analogWrite(pwmPin, speed);
    }
    else if (speed < 0)
    {
        // Chạy nghịch
        digitalWrite(in1, LOW);
        digitalWrite(in2, HIGH);
        analogWrite(pwmPin, -speed);
    }
    else
    {
        // Dừng
        digitalWrite(in1, LOW);
        digitalWrite(in2, LOW);
        analogWrite(pwmPin, 0);
    }
}

void loop()
{
    if (Serial.available() > 0)
    {
        String command = Serial.readStringUntil('\n');
        command.trim();

        // Debug: in lệnh nhận được
        Serial.print("Received: ");
        Serial.println(command);

        int spaceIndex = command.indexOf(' ');
        if (spaceIndex > 0)
        {
            String motorId = command.substring(0, spaceIndex);
            String speedStr = command.substring(spaceIndex + 1);
            int speed = speedStr.toInt();

            Serial.print("Executing: ");
            Serial.print(motorId);
            Serial.print(" = ");
            Serial.println(speed);

            if (motorId == "M1")
            {
                setMotor(M1_IN1, M1_IN2, M1_PWM, speed);
                Serial.print("OK M1 ");
                Serial.println(speed);
            }
            else if (motorId == "M2")
            {
                setMotor(M2_IN1, M2_IN2, M2_PWM, speed);
                Serial.print("OK M2 ");
                Serial.println(speed);
            }
            else if (motorId == "M3")
            {
                setMotor(M3_IN1, M3_IN2, M3_PWM, speed);
                Serial.print("OK M3 ");
                Serial.println(speed);
            }
            else
            {
                Serial.println("ERROR: Invalid motor ID. Use M1, M2, or M3");
            }
        }
        else
        {
            Serial.println("ERROR: Invalid format. Use 'M1 100' or 'M2 -150'");
        }
    }

    delay(10); // Tránh flood serial
}