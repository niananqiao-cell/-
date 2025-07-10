import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential, save_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, MaxPooling1D, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import os
import time
import warnings

warnings.filterwarnings('ignore')

# 设置随机种子确保可重复性
np.random.seed(42)
tf.random.set_seed(42)


# 1. 数据加载与预处理 ========================================================

def load_and_preprocess_data(file_path, sequence_length=20, predict_steps=1):
    """
    加载并预处理9轴传感器数据
    :param file_path: 数据文件路径
    :param sequence_length: 输入序列长度
    :param predict_steps: 预测步数
    :return: 处理后的数据集
    """
    # 加载数据
    print(f"\n加载数据: {file_path}")
    try:
        df = pd.read_csv(file_path)
        print(f"数据维度: {df.shape}")
        print(f"前5行数据:\n{df.head()}")
    except Exception as e:
        print(f"加载数据失败: {e}")
        return None, None, None, None

    # 检查数据列
    sensor_cols = ['accel_x', 'accel_y', 'accel_z',
                   'gyro_x', 'gyro_y', 'gyro_z',
                   'mag_x', 'mag_y', 'mag_z']

    # 验证数据列是否存在
    missing_cols = [col for col in sensor_cols if col not in df.columns]
    if missing_cols:
        print(f"警告: 缺少必要的列: {missing_cols}")
        return None, None, None, None

    # 提取传感器数据
    sensor_data = df[sensor_cols].values

    # 数据标准化
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(sensor_data)

    # 创建时间序列数据集
    X, y = [], []
    for i in range(len(scaled_data) - sequence_length - predict_steps + 1):
        X.append(scaled_data[i:i + sequence_length])
        y.append(scaled_data[i + sequence_length:i + sequence_length + predict_steps])

    X = np.array(X)
    y = np.array(y)

    # 如果只预测一步，调整y的形状
    if predict_steps == 1:
        y = y.reshape(y.shape[0], y.shape[2])

    print(f"X形状: {X.shape}, y形状: {y.shape}")

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False)

    return X_train, X_test, y_train, y_test, scaler


# 2. 简化模型构建 ===============================================================

def build_simple_model(input_shape, output_dim):
    """
    构建简化但有效的预测模型
    :param input_shape: 输入形状 (sequence_length, features)
    :param output_dim: 输出维度
    :return: 编译后的模型
    """
    model = Sequential()

    # 单层LSTM作为核心
    model.add(LSTM(64, input_shape=input_shape))

    # 添加Dropout防止过拟合
    model.add(Dropout(0.3))

    # 输出层
    model.add(Dense(output_dim))

    # 编译模型
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mse',
                  metrics=['mae'])

    model.summary()
    return model


# 3. 模型训练 ===============================================================

def train_model(model, X_train, y_train, X_test, y_test, epochs=100, batch_size=64):
    """
    训练模型并返回训练历史
    :param model: 要训练的模型
    :param X_train: 训练数据
    :param y_train: 训练标签
    :param X_test: 测试数据
    :param y_test: 测试标签
    :param epochs: 训练轮数
    :param batch_size: 批大小
    :return: 训练历史
    """
    # 回调函数
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)
    ]

    # 训练模型
    print("\n开始训练模型...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )

    return history


# 4. 模型评估与可视化 =======================================================

def evaluate_model(model, X_test, y_test, scaler, predict_steps=1):
    """
    评估模型性能并可视化结果
    :param model: 训练好的模型
    :param X_test: 测试数据
    :param y_test: 测试标签
    :param scaler: 数据标准化器
    :param predict_steps: 预测步数
    """
    print("\n评估模型性能...")
    start_time = time.time()

    # 预测
    y_pred = model.predict(X_test)

    # 计算预测时间
    pred_time = time.time() - start_time
    print(f"预测完成! 耗时: {pred_time:.2f}秒")
    print(f"平均每个样本预测时间: {pred_time / len(X_test) * 1000:.3f}毫秒")

    # 反标准化
    if predict_steps > 1:
        # 多步预测需要特殊处理
        y_test_original = np.zeros((y_test.shape[0], y_test.shape[1], y_test.shape[2]))
        y_pred_original = np.zeros_like(y_test_original)

        for i in range(y_test.shape[1]):
            y_test_original[:, i, :] = scaler.inverse_transform(y_test[:, i, :])
            y_pred_original[:, i, :] = scaler.inverse_transform(y_pred[:, i, :])

        # 只取第一个预测步进行评估
        y_test_eval = y_test_original[:, 0, :]
        y_pred_eval = y_pred_original[:, 0, :]
    else:
        y_test_original = scaler.inverse_transform(y_test)
        y_pred_original = scaler.inverse_transform(y_pred)
        y_test_eval = y_test_original
        y_pred_eval = y_pred_original

    # 计算评估指标
    mse = mean_squared_error(y_test_eval, y_pred_eval)
    mae = mean_absolute_error(y_test_eval, y_pred_eval)
    r2 = r2_score(y_test_eval, y_pred_eval)

    print("\n测试集评估结果:")
    print(f"均方误差 (MSE): {mse:.6f}")
    print(f"平均绝对误差 (MAE): {mae:.6f}")
    print(f"决定系数 (R²): {r2:.6f}")

    # 可视化结果
    plot_results(y_test_eval, y_pred_eval)

    return y_test_original, y_pred_original


def plot_results(y_true, y_pred, feature_names=None):
    """
    结果可视化
    :param y_true: 真实值
    :param y_pred: 预测值
    :param feature_names: 特征名称列表
    """
    if feature_names is None:
        feature_names = ['accel_x', 'accel_y', 'accel_z',
                         'gyro_x', 'gyro_y', 'gyro_z',
                         'mag_x', 'mag_y', 'mag_z']

    # 1. 误差分布分析
    errors = np.abs(y_true - y_pred)
    plt.figure(figsize=(14, 8))
    sns.boxplot(data=pd.DataFrame(errors, columns=feature_names), palette="viridis")
    plt.title('各特征绝对误差分布', fontsize=16)
    plt.ylabel('绝对误差', fontsize=12)
    plt.xticks(rotation=45, fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig('error_distribution.png', bbox_inches='tight', dpi=150)
    plt.close()

    # 2. 整体预测性能散点图
    plt.figure(figsize=(10, 8))
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.scatter(y_true[:, i], y_pred[:, i], alpha=0.3, s=10)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
        plt.title(feature_names[i])
        plt.xlabel('真实值')
        plt.ylabel('预测值')
        plt.grid(True)

    plt.tight_layout()
    plt.suptitle('预测值 vs 真实值', fontsize=16, y=1.02)
    plt.savefig('prediction_scatter.png', bbox_inches='tight', dpi=150)
    plt.close()

    # 3. 随机样本对比
    sample_idx = np.random.randint(0, len(y_true))
    plt.figure(figsize=(15, 10))
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.plot([0], [y_true[sample_idx, i]], 'bo', markersize=8, label='真实值')
        plt.plot([0], [y_pred[sample_idx, i]], 'rx', markersize=8, label='预测值')
        plt.title(f"{feature_names[i]} (样本 {sample_idx})")
        plt.ylabel('值')
        plt.grid(True)
        plt.legend()

    plt.tight_layout()
    plt.suptitle('单点预测结果对比', fontsize=16, y=1.02)
    plt.savefig('single_point_comparison.png', bbox_inches='tight', dpi=150)
    plt.close()


# 5. 完整流程 ===============================================================

def main():
    # 配置参数
    DATA_FILE = "sensor_data.csv"  # 替换为你的数据文件路径
    SEQUENCE_LENGTH = 20  # 输入序列长度（历史时间步数）
    PREDICT_STEPS = 1  # 预测未来时间步数
    EPOCHS = 100  # 训练轮数
    BATCH_SIZE = 64  # 批大小

    print("=" * 60)
    print("简化版9轴传感器数据预测模型")
    print("=" * 60)
    print(f"参数配置: 序列长度={SEQUENCE_LENGTH}, 预测步数={PREDICT_STEPS}")
    print(f"         训练轮数={EPOCHS}, 批大小={BATCH_SIZE}")

    # 1. 加载并预处理数据
    print("\n步骤1: 加载并预处理数据...")
    X_train, X_test, y_train, y_test, scaler = load_and_preprocess_data(
        DATA_FILE,
        sequence_length=SEQUENCE_LENGTH,
        predict_steps=PREDICT_STEPS
    )

    if X_train is None:
        print("数据加载失败，程序退出。")
        return

    # 2. 构建模型
    print("\n步骤2: 构建预测模型...")
    input_shape = (SEQUENCE_LENGTH, 9)  # 输入形状：时间步长 × 特征数
    output_dim = 9 * PREDICT_STEPS  # 输出维度：特征数 × 预测步数

    model = build_simple_model(input_shape, output_dim)

    # 3. 训练模型
    print("\n步骤3: 训练模型...")
    history = train_model(
        model,
        X_train, y_train,
        X_test, y_test,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE
    )

    # 4. 可视化训练过程
    plt.figure(figsize=(14, 6))
    plt.plot(history.history['loss'], 'b-', label='训练损失')
    plt.plot(history.history['val_loss'], 'r-', label='验证损失')
    plt.title('模型训练损失', fontsize=16)
    plt.ylabel('损失 (MSE)', fontsize=12)
    plt.xlabel('训练轮次', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig('training_history.png', bbox_inches='tight', dpi=150)
    plt.show()

    # 5. 评估模型
    print("\n步骤4: 评估模型性能...")
    y_test_original, y_pred_original = evaluate_model(
        model,
        X_test, y_test,
        scaler,
        predict_steps=PREDICT_STEPS
    )

    # 6. 保存最终模型
    print("\n步骤5: 保存最终模型...")
    model.save("simple_prediction_model.h5")
    print(f"最终模型已保存为 simple_prediction_model.h5")

    # 7. 保存预测结果
    print("\n步骤6: 保存预测结果...")
    results_df = pd.DataFrame({
        'true_accel_x': y_test_original[:, 0],
        'pred_accel_x': y_pred_original[:, 0],
        'true_accel_y': y_test_original[:, 1],
        'pred_accel_y': y_pred_original[:, 1],
        'true_accel_z': y_test_original[:, 2],
        'pred_accel_z': y_pred_original[:, 2],
        'true_gyro_x': y_test_original[:, 3],
        'pred_gyro_x': y_pred_original[:, 3],
        'true_gyro_y': y_test_original[:, 4],
        'pred_gyro_y': y_pred_original[:, 4],
        'true_gyro_z': y_test_original[:, 5],
        'pred_gyro_z': y_pred_original[:, 5],
        'true_mag_x': y_test_original[:, 6],
        'pred_mag_x': y_pred_original[:, 6],
        'true_mag_y': y_test_original[:, 7],
        'pred_mag_y': y_pred_original[:, 7],
        'true_mag_z': y_test_original[:, 8],
        'pred_mag_z': y_pred_original[:, 8],
    })
    results_df.to_csv('prediction_results.csv', index=False)
    print("预测结果已保存为 prediction_results.csv")

    # 8. 训练分析
    best_epoch = np.argmin(history.history['val_loss'])
    print("\n训练分析:")
    print(f"最佳验证损失在 epoch {best_epoch + 1}: {history.history['val_loss'][best_epoch]:.4f}")
    print(f"最终训练损失: {history.history['loss'][-1]:.4f}")
    print(f"最终验证损失: {history.history['val_loss'][-1]:.4f}")

    # 计算过拟合程度
    overfitting_ratio = history.history['val_loss'][-1] / history.history['loss'][-1]
    print(f"过拟合程度: {overfitting_ratio:.2f} (越接近1越好)")

    print("\n" + "=" * 60)
    print("模型训练和评估完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()