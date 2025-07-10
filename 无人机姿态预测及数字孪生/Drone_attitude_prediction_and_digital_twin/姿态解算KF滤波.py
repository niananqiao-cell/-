import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# —— 1. 读取并预处理数据 ——
file_path = r'F:\无人机姿态预测及数字孪生\姿态数据.xlsx'
df = pd.read_excel(file_path)
for c in ['t','ax','ay','az','gx','gy','gz','mx','my','mz']:
    df[c] = pd.to_numeric(df[c], errors='coerce')
df.ffill(inplace=True)
df['dt'] = df['t'].diff().fillna(0.01).values

# —— 2. 工具函数 ——
def quat_to_rotmat(q):
    q0,q1,q2,q3 = q
    return np.array([
        [1-2*(q2*q2+q3*q3),   2*(q1*q2-q0*q3),   2*(q1*q3+q0*q2)],
        [  2*(q1*q2+q0*q3), 1-2*(q1*q1+q3*q3),   2*(q2*q3-q0*q1)],
        [  2*(q1*q3-q0*q2),   2*(q2*q3+q0*q1), 1-2*(q1*q1+q2*q2)]
    ])

# —— 3. EKF 参数与初始化 ——
n = 7  # 状态维度
x = np.zeros(n)
x[:4] = np.array([1,0,0,0])   # 初始单位四元数
P = np.eye(n) * 0.01          # 初始协方差
Q = np.eye(n) * 1e-6          # 过程噪声
R = np.eye(6) * 0.05          # 测量噪声

# —— 4. 迭代 EKF ——
euler_list = []
bias_list  = []

for idx, row in df.iterrows():
    dt = row['dt']
    # 控制输入：角速度 + dt
    omega = np.array([row['gx'], row['gy'], row['gz']])
    # 测量：加速度计 + 磁力计
    z_acc = np.array([row['ax'], row['ay'], row['az']])
    z_mag = np.array([row['mx'], row['my'], row['mz']])
    z = np.hstack((z_acc, z_mag))

    # ——— 4.1 预测步骤 ———
    # 状态转移
    q = x[:4]
    b = x[4:]
    omega_corr = omega - b
    Ω = np.array([
        [    0,      -omega_corr[0], -omega_corr[1], -omega_corr[2]],
        [ omega_corr[0],       0,    omega_corr[2], -omega_corr[1]],
        [ omega_corr[1], -omega_corr[2],       0,    omega_corr[0]],
        [ omega_corr[2],  omega_corr[1], -omega_corr[0],      0   ]
    ])
    q_dot = 0.5 * Ω.dot(q)
    q_pred = q + q_dot * dt
    q_pred /= np.linalg.norm(q_pred)
    x_pred = np.hstack((q_pred, b))

    # 雅可比 F 数值计算
    F = np.zeros((n,n))
    eps = 1e-6
    for i in range(n):
        dx = np.zeros(n); dx[i] = eps
        # fx(x+dx)
        q_i = x + dx
        # predict for perturbed
        q_i0, b_i0 = q_i[:4], q_i[4:]
        omega_i = omega - b_i0
        Ωi = np.array([
            [    0,      -omega_i[0], -omega_i[1], -omega_i[2]],
            [ omega_i[0],        0,   omega_i[2], -omega_i[1]],
            [ omega_i[1], -omega_i[2],        0,   omega_i[0]],
            [ omega_i[2],  omega_i[1], -omega_i[0],       0    ]
        ])
        qid = q_i0 + 0.5*Ωi.dot(q_i0)*dt
        qid /= np.linalg.norm(qid)
        x_i = np.hstack((qid, b_i0))
        F[:,i] = (x_i - x_pred)/eps

    # 协方差预测
    P = F.dot(P).dot(F.T) + Q
    x = x_pred

    # ——— 4.2 更新步骤 ———
    # 构造测量预测 h(x)
    Rm = quat_to_rotmat(x[:4])
    h_acc = Rm.dot(np.array([0,0,1.0]))
    h_mag = Rm.dot(np.array([30.2,-5.1,40.0]))
    h = np.hstack((h_acc, h_mag))

    # 雅可比 H 数值计算
    H = np.zeros((6,n))
    for i in range(n):
        dx = np.zeros(n); dx[i] = eps
        x_i = x + dx
        Ri = quat_to_rotmat(x_i[:4])
        hi_acc = Ri.dot(np.array([0,0,1.0]))
        hi_mag = Ri.dot(np.array([30.2,-5.1,40.0]))
        hi = np.hstack((hi_acc, hi_mag))
        H[:,i] = (hi - h)/eps

    # 卡尔曼增益
    S = H.dot(P).dot(H.T) + R
    K = P.dot(H.T).dot(np.linalg.inv(S))

    # 更新状态与协方差
    y = z - h
    x = x + K.dot(y)
    P = (np.eye(n) - K.dot(H)).dot(P)

    # 归一化四元数
    x[:4] /= np.linalg.norm(x[:4])

    # 保存结果
    # 四元数转欧拉
    q0,q1,q2,q3 = x[:4]
    roll  = np.arctan2(2*(q0*q1+q2*q3), 1-2*(q1*q1+q2*q2))
    pitch = np.arcsin(2*(q0*q2-q3*q1))
    yaw   = np.arctan2(2*(q0*q3+q1*q2), 1-2*(q2*q2+q3*q3))
    euler_list.append((roll,pitch,yaw))

    # 偏置部分
    bias_list.append(tuple(x[4:]))

print("points:", len(euler_list), " first 3 eulers:", euler_list[:3])
print("biases:",  len(bias_list),  " first 3 biases:", bias_list[:3])




# —— 5. 可视化 ——
euler_arr = np.array(euler_list)
bias_arr  = np.array(bias_list)
t = df['t'].values

plt.figure(figsize=(10,6))
plt.subplot(2,1,1)
plt.plot(t, euler_arr[:,0], label='Roll')
plt.plot(t, euler_arr[:,1], label='Pitch')
plt.plot(t, euler_arr[:,2], label='Yaw')
plt.legend(); plt.title('Attitude'); plt.grid()

plt.subplot(2,1,2)
plt.plot(t, bias_arr[:,0], label='bx')
plt.plot(t, bias_arr[:,1], label='by')
plt.plot(t, bias_arr[:,2], label='bz')
plt.legend(); plt.title('Gyro Bias'); plt.xlabel('Time (s)'); plt.grid()

plt.tight_layout()
plt.show()
