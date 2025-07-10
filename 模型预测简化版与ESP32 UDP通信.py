import numpy as np
import pandas as pd
import socket
import threading
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import load_model
import time
from collections import deque


class RealTimePredictor:
    def __init__(self, model_path, sequence_length=20):
        # 加载预训练模型
        self.model = load_model(model_path)
        self.sequence_length = sequence_length
        self.scaler = StandardScaler()

        # 初始化数据缓冲区
        self.data_buffer = deque(maxlen=sequence_length + 1)
        self.feature_names = ['accel_x', 'accel_y', 'accel_z',
                              'gyro_x', 'gyro_y', 'gyro_z',
                              'mag_x', 'mag_y', 'mag_z']

        # 伪拟合scaler (实际使用时应该用训练数据的统计量)
        dummy_data = np.zeros((100, 9))
        self.scaler.fit(dummy_data)

        # 预测统计
        self.prediction_count = 0
        self.total_prediction_time = 0

    def update_scaler_stats(self, mean, var):
        """更新标准化器的统计量"""
        self.scaler.mean_ = mean
        self.scaler.var_ = var
        self.scaler.scale_ = np.sqrt(var)

    def add_data(self, new_data):
        """添加新的传感器数据"""
        if len(new_data) == 9:  # 确保是9轴数据
            self.data_buffer.append(new_data)

    def make_prediction(self):
        """基于当前缓冲区数据进行预测"""
        if len(self.data_buffer) < self.sequence_length:
            return None

        # 准备输入数据
        sequence = list(self.data_buffer)[-self.sequence_length:]
        input_data = np.array(sequence)

        # 标准化
        scaled_data = self.scaler.transform(input_data)
        scaled_data = scaled_data.reshape(1, self.sequence_length, 9)

        # 预测
        start_time = time.time()
        prediction = self.model.predict(scaled_data)
        elapsed = time.time() - start_time

        # 更新预测统计
        self.prediction_count += 1
        self.total_prediction_time += elapsed

        # 反标准化
        prediction = self.scaler.inverse_transform(prediction)

        return prediction[0]  # 返回一维数组


class UDPServer:
    def __init__(self, host='0.0.0.0', port=12345):
        self.host = host
        self.port = port
        self.running = False
        self.predictor = None

    def start(self, predictor):
        """启动UDP服务器"""
        self.predictor = predictor
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.bind((self.host, self.port))
        self.socket.settimeout(1.0)
        self.running = True

        print(f"UDP服务器启动，监听 {self.host}:{self.port}")

        # 启动接收线程
        self.thread = threading.Thread(target=self._receive_loop)
        self.thread.start()

    def stop(self):
        """停止服务器"""
        self.running = False
        self.thread.join()
        self.socket.close()
        print("UDP服务器已停止")

    def _receive_loop(self):
        """接收数据的循环"""
        while self.running:
            try:
                data, addr = self.socket.recvfrom(1024)
                try:
                    # 解析数据 (格式: "accel_x,accel_y,accel_z,gyro_x,gyro_y,gyro_z,mag_x,mag_y,mag_z")
                    sensor_values = list(map(float, data.decode().strip().split(',')))

                    if len(sensor_values) == 9:
                        # 添加到预测器
                        self.predictor.add_data(sensor_values)

                        # 当有足够数据时进行预测
                        if len(self.predictor.data_buffer) >= self.predictor.sequence_length:
                            prediction = self.predictor.make_prediction()
                            if prediction is not None:
                                # 这里可以处理预测结果 (发送到前端、保存到文件等)
                                print(f"预测结果: {prediction}")

                                # 可选: 将预测结果发回ESP32
                                # self._send_prediction(addr, prediction)
                except ValueError as e:
                    print(f"数据解析错误: {e}")
            except socket.timeout:
                continue

    def _send_prediction(self, addr, prediction):
        """将预测结果发送回客户端"""
        try:
            response = ",".join([f"{x:.2f}" for x in prediction])
            self.socket.sendto(response.encode(), addr)
        except Exception as e:
            print(f"发送预测结果失败: {e}")


def main():
    # 1. 初始化预测器
    predictor = RealTimePredictor("simple_prediction_model.h5")

    # 2. 从训练数据加载scaler统计量 (实际使用时应该用训练数据的统计量)
    # 这里假设您已经保存了训练时的scaler参数
    # predictor.update_scaler_stats(mean, var)

    # 3. 启动UDP服务器
    server = UDPServer()
    server.start(predictor)

    try:
        # 主线程可以在这里做其他事情
        while True:
            time.sleep(1)

            # 打印预测统计
            if predictor.prediction_count > 0:
                avg_time = predictor.total_prediction_time / predictor.prediction_count
                print(f"平均预测时间: {avg_time * 1000:.2f}ms")
    except KeyboardInterrupt:
        print("\n正在停止服务器...")
    finally:
        server.stop()


if __name__ == "__main__":
    main()