#include <Arduino.h>

// Motor pin definitions
const int M1_IN1 = 2, M1_IN2 = 4, M1_PWM = 5;   // Left wheel
const int M2_IN1 = 7, M2_IN2 = 8, M2_PWM = 6;   // Right wheel
const int M3_IN1 = 10, M3_IN2 = 11, M3_PWM = 9; // Intake motor

void setup()
{
    Serial.begin(115200);

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

    // Stop all motors initially
    digitalWrite(M1_IN1, LOW);
    digitalWrite(M1_IN2, LOW);
    analogWrite(M1_PWM, 0);

    digitalWrite(M2_IN1, LOW);
    digitalWrite(M2_IN2, LOW);
    analogWrite(M2_PWM, 0);

    digitalWrite(M3_IN1, LOW);
    digitalWrite(M3_IN2, LOW);
    analogWrite(M3_PWM, 0);

    Serial.println("READY: Send 'M1 <speed>', 'M2 <speed>', 'M3 <speed>'");
}

void setMotor(int in1, int in2, int pwm, int speed)
{
    if (speed > 255)
        speed = 255;
    if (speed < -255)
        speed = -255;

    if (speed > 0)
    {
        digitalWrite(in1, HIGH);
        digitalWrite(in2, LOW);
        analogWrite(pwm, speed);
    }
    else if (speed < 0)
    {
        digitalWrite(in1, LOW);
        digitalWrite(in2, HIGH);
        analogWrite(pwm, -speed);
    }
    else
    {
        digitalWrite(in1, LOW);
        digitalWrite(in2, LOW);
        analogWrite(pwm, 0);
    }
}

void loop()
{
    if (Serial.available())
    {
        String command = Serial.readStringUntil('\n');
        command.trim();

        // Parse command: "M1 100", "M2 -150", "M3 0"
        int spaceIndex = command.indexOf(' ');
        if (spaceIndex > 0)
        {
            String motorId = command.substring(0, spaceIndex);
            int speed = command.substring(spaceIndex + 1).toInt();

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
                Serial.println("ERR Invalid motor ID");
            }
        }
        else
        {
            Serial.println("ERR Invalid format");
        }
    }
}