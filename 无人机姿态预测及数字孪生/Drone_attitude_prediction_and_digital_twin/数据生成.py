import numpy as np
import pandas as pd


# 生成模拟传感器数据
def generate_sensor_data(num_samples=5000):
    # 加速度计数据（模拟重力+运动）
    accel_z = 9.81 + np.random.normal(0, 0.1, num_samples)
    accel_x = np.sin(np.linspace(0, 10 * np.pi, num_samples)) * 0.5 + np.random.normal(0, 0.05, num_samples)
    accel_y = np.cos(np.linspace(0, 8 * np.pi, num_samples)) * 0.3 + np.random.normal(0, 0.05, num_samples)

    # 陀螺仪数据（模拟旋转）
    gyro_x = np.sin(np.linspace(0, 12 * np.pi, num_samples)) * 0.8 + np.random.normal(0, 0.02, num_samples)
    gyro_y = np.cos(np.linspace(0, 10 * np.pi, num_samples)) * 0.6 + np.random.normal(0, 0.02, num_samples)
    gyro_z = np.sin(np.linspace(0, 15 * np.pi, num_samples)) * 0.4 + np.random.normal(0, 0.01, num_samples)

    # 磁力计数据（模拟磁场变化）
    mag_x = 35.0 + np.sin(np.linspace(0, 5 * np.pi, num_samples)) * 3 + np.random.normal(0, 0.1, num_samples)
    mag_y = 42.0 + np.cos(np.linspace(0, 4 * np.pi, num_samples)) * 2.5 + np.random.normal(0, 0.1, num_samples)
    mag_z = -12.0 + np.sin(np.linspace(0, 6 * np.pi, num_samples)) * 1.5 + np.random.normal(0, 0.1, num_samples)

    # 创建DataFrame
    data = pd.DataFrame({
        'accel_x': accel_x,
        'accel_y': accel_y,
        'accel_z': accel_z,
        'gyro_x': gyro_x,
        'gyro_y': gyro_y,
        'gyro_z': gyro_z,
        'mag_x': mag_x,
        'mag_y': mag_y,
        'mag_z': mag_z
    })

    return data


# 生成并保存数据
sensor_data = generate_sensor_data(5000)
sensor_data.to_csv('sensor_data.csv', index=False)
print("模拟数据已保存为 sensor_data.csv")