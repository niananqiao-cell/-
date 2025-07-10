import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Input
from tensorflow.keras.losses import MeanSquaredError
from sklearn.preprocessing import StandardScaler
import joblib

# =============================================
# 1. 配置路径和参数
# =============================================
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 减少TensorFlow日志输出
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # 禁用oneDNN警告

# 文件路径配置
DATA_FILE = r"F:\无人机姿态预测及数字孪生\sensor_data.csv"
MODEL_PATH = r"F:\无人机姿态预测及数字孪生\Drone_attitude_prediction_and_digital_twin\simple_prediction_model.h5"
OUTPUT_DIR = r"F:\无人机姿态预测及数字孪生\Drone_attitude_prediction_and_digital_twin"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 模型参数
SEQUENCE_LENGTH = 20
PREDICT_STEPS = 1
SENSOR_COLS = ['accel_x', 'accel_y', 'accel_z',
               'gyro_x', 'gyro_y', 'gyro_z',
               'mag_x', 'mag_y', 'mag_z']

# =============================================
# 2. 数据加载与标准化
# =============================================
print("加载并预处理传感器数据...")
df = pd.read_csv(DATA_FILE)

# 验证数据列
missing_cols = [col for col in SENSOR_COLS if col not in df.columns]
if missing_cols:
    raise ValueError(f"数据文件缺少必要的列: {missing_cols}")

# 提取传感器数据
sensor_data = df[SENSOR_COLS].values.astype(np.float32)

# 创建时间序列数据集
X, y = [], []
for i in range(len(sensor_data) - SEQUENCE_LENGTH - PREDICT_STEPS + 1):
    X.append(sensor_data[i:i + SEQUENCE_LENGTH])
    y.append(sensor_data[i + SEQUENCE_LENGTH:i + SEQUENCE_LENGTH + PREDICT_STEPS].flatten())

X = np.array(X)
y = np.array(y)

# 标准化数据
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
y_scaled = scaler.transform(y)

# 保存标准化器
scaler_path = os.path.join(OUTPUT_DIR, 'scaler.save')
joblib.dump(scaler, scaler_path)
print(f"标准化器已保存到: {scaler_path}")

# =============================================
# 3. 模型加载与修复
# =============================================
print("\n加载并修复预训练模型...")
try:
    # 尝试使用自定义对象加载
    model = load_model(MODEL_PATH, compile=False)
    model.compile(optimizer='adam', loss=MeanSquaredError(), metrics=['mae'])
    print("模型加载成功 (使用自定义对象)")
except Exception as e:
    print(f"模型加载失败: {e}")
    print("尝试重建模型结构...")

    # 重建模型结构 (根据您的模型代码)
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout

    model = Sequential()
    model.add(LSTM(64, input_shape=(SEQUENCE_LENGTH, len(SENSOR_COLS))))
    model.add(Dropout(0.3))
    model.add(Dense(len(SENSOR_COLS) * PREDICT_STEPS))

    # 编译模型
    model.compile(optimizer='adam', loss=MeanSquaredError(), metrics=['mae'])

    # 尝试加载权重
    try:
        model.load_weights(MODEL_PATH)
        print("模型权重加载成功!")
    except:
        print("警告: 无法加载模型权重，请确保模型文件格式正确")
        print("您可能需要重新训练模型")
        exit(1)

# 验证模型输入输出
print(f"输入形状: {model.input_shape}")
print(f"输出形状: {model.output_shape}")

# =============================================
# 4. 模型量化 (解决 LSTM 问题)
# =============================================
print("\n应用训练后量化...")

# 方法1: 尝试使用 TFLiteConverter 的改进设置
try:
    # 创建转换器
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    # 尝试1: 使用动态范围量化
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,  # 支持所有内置操作
        tf.lite.OpsSet.SELECT_TF_OPS  # 支持 TensorFlow 操作
    ]

    # 设置输入形状为固定大小
    converter.experimental_new_converter = True
    converter.experimental_new_quantizer = True


    # 代表性数据集函数
    def representative_dataset():
        # 使用部分训练数据
        for i in range(min(100, len(X_scaled))):
            yield [X_scaled[i:i + 1].astype(np.float32)]


    converter.representative_dataset = representative_dataset

    # 转换模型
    quant_tflite = converter.convert()
    print("量化成功完成 (使用 SELECT_TF_OPS)!")

except Exception as e:
    print(f"方法1失败: {e}")
    print("尝试方法2: 使用旧版转换器...")

    try:
        # 创建转换器
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

        # 尝试2: 使用旧版转换器
        converter.experimental_new_converter = False
        converter.experimental_new_quantizer = False


        # 代表性数据集函数
        def representative_dataset():
            # 使用部分训练数据
            for i in range(min(100, len(X_scaled))):
                yield [X_scaled[i:i + 1].astype(np.float32)]


        converter.representative_dataset = representative_dataset

        # 转换模型
        quant_tflite = converter.convert()
        print("量化成功完成 (使用旧版转换器)!")

    except Exception as e2:
        print(f"方法2失败: {e2}")
        print("尝试方法3: 不带代表性数据集...")

        try:
            # 创建转换器
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
            converter.representative_dataset = None

            # 转换模型
            quant_tflite = converter.convert()
            print("量化成功完成 (不带代表性数据集)!")

        except Exception as e3:
            print(f"方法3失败: {e3}")
            print("尝试方法4: 仅优化模型大小...")

            # 创建转换器
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
            converter.representative_dataset = None

            # 转换模型
            quant_tflite = converter.convert()
            print("量化成功完成 (仅优化模型大小)!")

# 保存量化模型
tflite_path = os.path.join(OUTPUT_DIR, 'quant_model.tflite')
with open(tflite_path, 'wb') as f:
    f.write(quant_tflite)
print(f"量化模型已保存到: {tflite_path}")

# =============================================
# 5. 获取量化参数
# =============================================
interpreter = tf.lite.Interpreter(model_content=quant_tflite)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]

input_scale = input_details['quantization'][0]
input_zero_point = input_details['quantization'][1]
output_scale = output_details['quantization'][0]
output_zero_point = output_details['quantization'][1]

print(f"\n输入量化参数: scale={input_scale}, zero_point={input_zero_point}")
print(f"输出量化参数: scale={output_scale}, zero_point={output_zero_point}")


# =============================================
# 6. 生成C代码文件
# =============================================
def generate_c_header():
    # 读取量化模型二进制数据
    with open(tflite_path, 'rb') as f:
        model_data = f.read()

    # 生成十六进制数组
    hex_array = ',\n    '.join([f'0x{byte:02x}' for byte in model_data])

    # 生成C头文件
    c_code = f"""#ifndef MODEL_DATA_H
#define MODEL_DATA_H

// 模型参数
#define SEQUENCE_LENGTH {SEQUENCE_LENGTH}
#define SENSOR_FEATURES {len(SENSOR_COLS)}
#define PREDICT_STEPS {PREDICT_STEPS}
#define OUTPUT_FEATURES (SENSOR_FEATURES * PREDICT_STEPS)

// 量化参数
const float input_scale = {input_scale}f;
const int input_zero_point = {input_zero_point};
const float output_scale = {output_scale}f;
const int output_zero_point = {output_zero_point};

// 模型数据
const unsigned char model_data[] = {{
    {hex_array}
}};

const unsigned int model_len = {len(model_data)};

#endif // MODEL_DATA_H
"""

    c_path = os.path.join(OUTPUT_DIR, 'model_data.h')
    with open(c_path, 'w') as f:
        f.write(c_code)
    print(f"C头文件已保存到: {c_path}")


generate_c_header()


# =============================================
# 7. 姿态解算集成函数
# =============================================
def generate_attitude_calculation():
    attitude_code = """// attitude_calculation.h
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
"""

    attitude_path = os.path.join(OUTPUT_DIR, 'attitude_calculation.h')
    with open(attitude_path, 'w') as f:
        f.write(attitude_code)
    print(f"姿态解算头文件已保存到: {attitude_path}")


generate_attitude_calculation()


# =============================================
# 8. 标准化参数生成
# =============================================
def generate_normalization_params():
    # 计算均值和标准差
    input_mean = np.mean(X, axis=(0, 1))
    input_std = np.std(X, axis=(0, 1))
    output_mean = np.mean(y, axis=0)
    output_std = np.std(y, axis=0)

    # 生成C头文件
    norm_code = """#ifndef NORMALIZATION_PARAMS_H
#define NORMALIZATION_PARAMS_H

// 输入标准化参数 (用于预处理)
const float input_mean[SENSOR_FEATURES] = {
    %s
};

const float input_std[SENSOR_FEATURES] = {
    %s
};

// 输出标准化参数 (用于后处理)
const float output_mean[OUTPUT_FEATURES] = {
    %s
};

const float output_std[OUTPUT_FEATURES] = {
    %s
};

#endif // NORMALIZATION_PARAMS_H
""" % (
        ',\n    '.join([f'{val:.6f}f' for val in input_mean]),
        ',\n    '.join([f'{val:.6f}f' for val in input_std]),
        ',\n    '.join([f'{val:.6f}f' for val in output_mean]),
        ',\n    '.join([f'{val:.6f}f' for val in output_std])
    )

    norm_path = os.path.join(OUTPUT_DIR, 'normalization_params.h')
    with open(norm_path, 'w') as f:
        f.write(norm_code)
    print(f"标准化参数头文件已保存到: {norm_path}")


generate_normalization_params()

# =============================================
# 9. 部署说明
# =============================================
print("\n" + "=" * 60)
print("量化部署完成!")
print("=" * 60)
print("下一步操作:")
print("1. 将以下文件添加到STM32工程:")
print(f"   - {os.path.join(OUTPUT_DIR, 'model_data.h')}")
print(f"   - {os.path.join(OUTPUT_DIR, 'attitude_calculation.h')}")
print(f"   - {os.path.join(OUTPUT_DIR, 'normalization_params.h')}")
print("2. 在STM32代码中实现以下功能:")
print("   a. 传感器数据缓冲 (保留最近的SEQUENCE_LENGTH个样本)")
print("   b. 使用标准化参数对输入数据进行预处理")
print("   c. 调用TFLite Micro进行预测")
print("   d. 对输出数据进行后处理")
print("   e. 使用calculate_attitude函数计算姿态角")
print("   f. 通过UART/USB将实时姿态和预测姿态发送到Unity3D")
print("3. 在Unity3D中:")
print("   a. 接收来自STM32的实时姿态和预测姿态数据")
print("   b. 创建两个无人机模型: 一个显示实时姿态，一个显示预测姿态")
print("   c. 实现数据可视化界面")