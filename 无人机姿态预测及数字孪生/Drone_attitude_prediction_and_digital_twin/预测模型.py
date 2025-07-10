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
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import l2
import os
import time
import warnings

warnings.filterwarnings('ignore')

# 设置随机种子确保可重复性
np.random.seed(42)
tf.random.set_seed(42)


# 1. 数据加载与预处理 ========================================================

def load_and_preprocess_data(file_path, sequence_length=20, predict_steps=1, augment=True):
    """
    加载并预处理9轴传感器数据
    :param file_path: 数据文件路径
    :param sequence_length: 输入序列长度
    :param predict_steps: 预测步数
    :param augment: 是否进行数据增强
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
        seq = scaled_data[i:i + sequence_length]
        target = scaled_data[i + sequence_length:i + sequence_length + predict_steps]

        # 添加到数据集
        X.append(seq)
        y.append(target)

        # 数据增强 - 添加噪声
        if augment and np.random.rand() > 0.7:
            noise = np.random.normal(0, 0.05, seq.shape)
            X.append(seq + noise)
            y.append(target)

        # 数据增强 - 时间扭曲
        if augment and np.random.rand() > 0.9:
            warp_factor = np.random.uniform(0.9, 1.1)
            warp_idx = np.clip(np.round(np.arange(sequence_length) * warp_factor).astype(int), 0, sequence_length - 1)
            X.append(seq[warp_idx])
            y.append(target)

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


# 2. 模型构建 ===============================================================

def build_improved_model(input_shape, output_dim):
    """
    构建改进的预测模型
    :param input_shape: 输入形状 (sequence_length, features)
    :param output_dim: 输出维度
    :return: 编译后的模型
    """
    model = Sequential()

    # 输入层
    model.add(Conv1D(filters=48, kernel_size=3,
                     activation='relu',
                     input_shape=input_shape,
                     padding='same',
                     kernel_regularizer=l2(0.001)))

    # 批归一化
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))

    # LSTM层
    model.add(LSTM(64, return_sequences=True,
                   kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.3))
    model.add(BatchNormalization())

    model.add(LSTM(32, return_sequences=False,
                   kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.3))
    model.add(BatchNormalization())

    # 输出层
    model.add(Dense(32, activation='relu'))
    model.add(Dense(output_dim))

    # 编译模型
    optimizer = Adam(learning_rate=0.0008)
    model.compile(optimizer=optimizer, loss='mse',
                  metrics=['mae', 'mse'])

    model.summary()
    return model


# 3. 模型训练 ===============================================================

def train_model(model, X_train, y_train, X_test, y_test, epochs=150, batch_size=64):
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
    # 创建保存模型的目录
    os.makedirs('saved_models', exist_ok=True)
    model_name = f"model_{int(time.time())}"

    # 回调函数
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, min_lr=1e-6, verbose=1),
        ModelCheckpoint(
            filepath=f'saved_models/{model_name}.h5',
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        )
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

    return history, model_name


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
    detailed_analysis(y_test_eval, y_pred_eval)

    return y_test_original, y_pred_original


def detailed_analysis(y_true, y_pred, feature_names=None, n_samples=3):
    """
    详细的预测结果可视化分析 - 修复版本
    :param y_true: 真实值
    :param y_pred: 预测值
    :param feature_names: 特征名称列表
    :param n_samples: 要可视化的样本数量
    """
    if feature_names is None:
        feature_names = ['accel_x', 'accel_y', 'accel_z',
                         'gyro_x', 'gyro_y', 'gyro_z',
                         'mag_x', 'mag_y', 'mag_z']

    # 创建可视化目录
    os.makedirs('visualizations', exist_ok=True)

    # 1. 随机选择几个样本进行详细可视化
    sample_indices = np.random.choice(len(y_true), n_samples, replace=False)

    # 创建特征图
    fig, axes = plt.subplots(n_samples, 9, figsize=(30, 3 * n_samples))
    if n_samples == 1:
        axes = [axes]  # 确保单样本情况也能正确处理

    for idx, sample_idx in enumerate(sample_indices):
        for i in range(9):
            ax = axes[idx][i]
            ax.plot([0], [y_true[sample_idx, i]], 'bo', markersize=8, label='真实值')
            ax.plot([0], [y_pred[sample_idx, i]], 'rx', markersize=8, label='预测值')
            ax.set_title(f"{feature_names[i]} (样本 {sample_idx})")
            ax.set_ylabel('值')
            ax.grid(True)
            ax.legend()

    plt.tight_layout()
    plt.suptitle('单点预测结果对比', fontsize=16, y=1.02)
    plt.savefig('visualizations/single_point_comparison.png', bbox_inches='tight', dpi=150)
    plt.close()

    # 2. 特征分布对比
    plt.figure(figsize=(15, 12))
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        sns.kdeplot(y_true[:, i], label='真实值', fill=True)
        sns.kdeplot(y_pred[:, i], label='预测值', fill=True)
        plt.title(f'{feature_names[i]} 分布')
        plt.xlabel('值')
        plt.ylabel('密度')
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.suptitle('特征值分布对比', fontsize=16, y=1.02)
    plt.savefig('visualizations/feature_distributions.png', bbox_inches='tight', dpi=150)
    plt.close()

    # 3. 误差分布分析
    errors = np.abs(y_true - y_pred)
    plt.figure(figsize=(14, 8))
    sns.boxplot(data=pd.DataFrame(errors, columns=feature_names), palette="viridis")
    plt.title('各特征绝对误差分布', fontsize=16)
    plt.ylabel('绝对误差', fontsize=12)
    plt.xticks(rotation=45, fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig('visualizations/error_distribution.png', bbox_inches='tight', dpi=150)
    plt.close()

    # 4. 相关矩阵分析
    residuals = y_true - y_pred
    corr_matrix = np.corrcoef(residuals, rowvar=False)

    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm",
                xticklabels=feature_names, yticklabels=feature_names)
    plt.title('预测误差相关矩阵', fontsize=16)
    plt.savefig('visualizations/error_correlation.png', bbox_inches='tight', dpi=150)
    plt.close()

    # 5. 整体预测性能散点图
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
    plt.savefig('visualizations/prediction_scatter.png', bbox_inches='tight', dpi=150)
    plt.close()


# 5. 完整流程 ===============================================================

def main():
    # 配置参数
    DATA_FILE = "sensor_data.csv"  # 替换为你的数据文件路径
    SEQUENCE_LENGTH = 25  # 输入序列长度（历史时间步数）
    PREDICT_STEPS = 1  # 预测未来时间步数
    EPOCHS = 150  # 训练轮数
    BATCH_SIZE = 72  # 批大小
    AUGMENT_DATA = True  # 是否使用数据增强

    print("=" * 60)
    print("优化版9轴传感器数据预测模型")
    print("=" * 60)
    print(f"参数配置: 序列长度={SEQUENCE_LENGTH}, 预测步数={PREDICT_STEPS}")
    print(f"         训练轮数={EPOCHS}, 批大小={BATCH_SIZE}, 数据增强={AUGMENT_DATA}")

    # 1. 加载并预处理数据
    print("\n步骤1: 加载并预处理数据...")
    X_train, X_test, y_train, y_test, scaler = load_and_preprocess_data(
        DATA_FILE,
        sequence_length=SEQUENCE_LENGTH,
        predict_steps=PREDICT_STEPS,
        augment=AUGMENT_DATA
    )

    if X_train is None:
        print("数据加载失败，程序退出。")
        return

    # 2. 构建模型
    print("\n步骤2: 构建预测模型...")
    input_shape = (SEQUENCE_LENGTH, 9)  # 输入形状：时间步长 × 特征数
    output_dim = 9 * PREDICT_STEPS  # 输出维度：特征数 × 预测步数

    model = build_improved_model(input_shape, output_dim)

    # 3. 训练模型
    print("\n步骤3: 训练模型...")
    history, model_name = train_model(
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
    final_model_path = f"final_model_{model_name}.h5"
    save_model(model, final_model_path, save_format='h5')
    print(f"最终模型已保存为 {final_model_path}")

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

    # 8. 分析训练过程
    best_epoch = np.argmin(history.history['val_loss'])
    print("\n训练过程分析:")
    print(f"最佳验证损失在 epoch {best_epoch + 1}: {history.history['val_loss'][best_epoch]:.4f}")
    print(f"最终训练损失: {history.history['loss'][-1]:.4f}")
    print(f"最终验证损失: {history.history['val_loss'][-1]:.4f}")

    # 9. 模型改进建议
    print("\n模型改进建议:")
    if history.history['val_loss'][-1] > history.history['loss'][-1]:
        print("- 模型存在过拟合现象，建议增加Dropout比例或减少模型复杂度")
    else:
        print("- 模型欠拟合，建议增加模型容量或训练轮数")

    print("- 考虑增加序列长度以捕获更多时间依赖性")
    print("- 尝试添加注意力机制提高模型性能")
    print("- 对于嵌入式部署，使用模型量化技术减少大小")

    print("\n" + "=" * 60)
    print("模型训练和评估完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()