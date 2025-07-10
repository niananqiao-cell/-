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
 * @brief ʹ�ü��ٶȼƺʹ��������ݼ�����̬
 * @param accel ���ٶȼ����� [ax, ay, az]
 * @param mag ���������� [mx, my, mz]
 * @return ��̬�ṹ�� (roll, pitch, yaw in degrees)
 */
Attitude calculate_attitude(const float accel[3], const float mag[3]) {
    Attitude result;

    // 1. ��һ�����ٶȼ�����
    float norm = sqrt(accel[0]*accel[0] + accel[1]*accel[1] + accel[2]*accel[2]);
    float ax = accel[0] / norm;
    float ay = accel[1] / norm;
    float az = accel[2] / norm;

    // 2. ���㸩���͹�ת (ʹ�ü��ٶȼ�)
    result.pitch = asin(-ax) * 180.0f / M_PI;
    result.roll = atan2(ay, az) * 180.0f / M_PI;

    // 3. ʹ�ô����Ƽ���ƫ����
    // �Ȳ�����ת�͸���
    float cos_roll = cos(result.roll * M_PI / 180.0f);
    float sin_roll = sin(result.roll * M_PI / 180.0f);
    float cos_pitch = cos(result.pitch * M_PI / 180.0f);
    float sin_pitch = sin(result.pitch * M_PI / 180.0f);

    // ������Ĵ���������
    float mx = mag[0] * cos_pitch + mag[2] * sin_pitch;
    float my = mag[0] * sin_roll * sin_pitch + mag[1] * cos_roll - mag[2] * sin_roll * cos_pitch;

    // ����ƫ����
    result.yaw = atan2(-my, mx) * 180.0f / M_PI;

    // ȷ��ƫ������0-360�ȷ�Χ��
    if (result.yaw < 0) {
        result.yaw += 360.0f;
    }

    return result;
}

#endif // ATTITUDE_CALCULATION_H
