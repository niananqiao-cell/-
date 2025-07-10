// attitude_calculation.h
#ifndef ATTITUDE_CALCULATION_H
#define ATTITUDE_CALCULATION_H

#include <math.h>

typedef struct {
    float roll;
    float pitch;
    float yaw;
} Attitude;

/**
 * @brief 使用加速度计和磁力计数据计算姿态
 * @param accel 加速度计数据 [ax, ay, az]
 * @param mag 磁力计数据 [mx, my, mz]
 * @return 姿态结构体 (roll, pitch, yaw in degrees)
 */
Attitude calculate_attitude(const float accel[3], const float mag[3]) {
    Attitude result;

    // 1. 归一化加速度计数据
    float norm = sqrt(accel[0]*accel[0] + accel[1]*accel[1] + accel[2]*accel[2]);
    float ax = accel[0] / norm;
    float ay = accel[1] / norm;
    float az = accel[2] / norm;

    // 2. 计算俯仰和滚转 (使用加速度计)
    result.pitch = asin(-ax) * 180.0f / M_PI;
    result.roll = atan2(ay, az) * 180.0f / M_PI;

    // 3. 使用磁力计计算偏航角
    // 先补偿滚转和俯仰
    float cos_roll = cos(result.roll * M_PI / 180.0f);
    float sin_roll = sin(result.roll * M_PI / 180.0f);
    float cos_pitch = cos(result.pitch * M_PI / 180.0f);
    float sin_pitch = sin(result.pitch * M_PI / 180.0f);

    // 补偿后的磁力计数据
    float mx = mag[0] * cos_pitch + mag[2] * sin_pitch;
    float my = mag[0] * sin_roll * sin_pitch + mag[1] * cos_roll - mag[2] * sin_roll * cos_pitch;

    // 计算偏航角
    result.yaw = atan2(-my, mx) * 180.0f / M_PI;

    // 确保偏航角在0-360度范围内
    if (result.yaw < 0) {
        result.yaw += 360.0f;
    }

    return result;
}

#endif // ATTITUDE_CALCULATION_H
