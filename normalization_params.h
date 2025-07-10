#ifndef NORMALIZATION_PARAMS_H
#define NORMALIZATION_PARAMS_H

// 输入标准化参数 (用于预处理)
const float input_mean[SENSOR_FEATURES] = {
    -0.000118f,
    -0.001736f,
    9.808640f,
    -0.000008f,
    -0.001820f,
    0.017080f,
    35.381779f,
    41.989613f,
    -12.001609f
};

const float input_std[SENSOR_FEATURES] = {
    0.357130f,
    0.218359f,
    0.099653f,
    0.567542f,
    0.423666f,
    0.283078f,
    2.092169f,
    1.766229f,
    1.070025f
};

// 输出标准化参数 (用于后处理)
const float output_mean[OUTPUT_FEATURES] = {
    -0.000260f,
    -0.001721f,
    9.808356f,
    -0.000262f,
    -0.001831f,
    0.017024f,
    35.381638f,
    41.990059f,
    -12.001823f
};

const float output_std[OUTPUT_FEATURES] = {
    0.357128f,
    0.218368f,
    0.099637f,
    0.567535f,
    0.423651f,
    0.283074f,
    2.092191f,
    1.766424f,
    1.070029f
};

#endif // NORMALIZATION_PARAMS_H
