#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>
#include <Arduino.h>

Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver();

const float freq1 = 1.0; 
const float freq2 = 0.5;
const float freq3 = 2.0; 
const float freq4 = 1.5; 

const int pwmMax = 4095; 

void setup() {
  Serial.begin(9600);
  pwm.begin();
  pwm.setPWMFreq(1600);
}

void loop() {
  unsigned long currentMillis = millis();
  float t = currentMillis / 1000.0; 

  int brightness1 = (pwmMax / 2) + (pwmMax / 2) * sin(2 * PI * freq1 * t);  // LED 1 (PWM0)
  int brightness2 = (pwmMax / 2) + (pwmMax / 2) * sin(2 * PI * freq2 * t);  // LED 2 (PWM1)
  int brightness3 = (pwmMax / 2) + (pwmMax / 2) * sin(2 * PI * freq3 * t);  // LED 3 (PWM2)
  int brightness4 = (pwmMax / 2) + (pwmMax / 2) * sin(2 * PI * freq4 * t);  // LED 4 (PWM3)

  pwm.setPWM(0, 0, brightness1);
  pwm.setPWM(1, 0, brightness2);
  pwm.setPWM(2, 0, brightness3); 
  pwm.setPWM(3, 0, brightness4);

  Serial.print(brightness1);
  Serial.print(",");          
  Serial.print(brightness2); 
  Serial.print(",");         
  Serial.print(brightness3); 
  Serial.print(",");          
  Serial.println(brightness4);

  delay(10);
}
