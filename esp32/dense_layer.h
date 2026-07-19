// dense_layer.h — versi ESP32/Arduino
#pragma once
#include "matrix_ops.h"
#include <Arduino.h>

enum class ActivationType {
    RELU,
    SIGMOID,
    TANH,
    LINEAR,
    SOFTMAX
};

class DenseLayer {
public:
    DenseLayer() {}

    DenseLayer(size_t in_dim, size_t out_dim, ActivationType act)
        : activation_(act),
          weights_(Matrix::random_uniform(in_dim, out_dim, sqrtf(2.0f / in_dim))),
          bias_(Matrix(1, out_dim, 0.0f)),
          weight_grad_(in_dim, out_dim, 0.0f),
          bias_grad_(1, out_dim, 0.0f) {}

    ActivationType activation_type() const { return activation_; }
    Matrix& weights() { return weights_; }
    Matrix& bias() { return bias_; }
    const Matrix& weight_grad() const { return weight_grad_; }
    const Matrix& bias_grad() const { return bias_grad_; }

    // input: (batch, in_dim) -> output: (batch, out_dim)
    bool forward(const Matrix& input, Matrix& output) {
        last_input_ = input; // dipakai backward

        Matrix z;
        if (!input.matmul(weights_, z)) return false;
        if (!z.add_bias_broadcast(bias_)) return false;

        last_z_ = z;
        output = apply_activation(z);
        last_output_ = output;
        return true;
    }

    // grad_output: dL/d(output). combined_with_loss=true artinya grad_output
    // sudah termasuk turunan activation (mis. Softmax+CrossEntropy digabung),
    // jadi di sini tidak perlu dikalikan turunan activation lagi.
    bool backward(const Matrix& grad_output, bool combined_with_loss, Matrix& grad_input) {
        Matrix delta;
        if (combined_with_loss) {
            delta = grad_output;
        } else {
            delta = multiply_activation_derivative(grad_output);
        }

        // weight_grad_ = input^T * delta
        Matrix input_T = last_input_.transpose();
        if (!input_T.matmul(delta, weight_grad_)) return false;

        // bias_grad_ = sum(delta, axis=0)  -> (1, out_dim)
        bias_grad_ = Matrix(1, delta.cols(), 0.0f);
        for (size_t i = 0; i < delta.rows(); ++i)
            for (size_t j = 0; j < delta.cols(); ++j)
                bias_grad_(0, j) += delta(i, j);

        // grad_input = delta * weights_^T
        Matrix weights_T = weights_.transpose();
        if (!delta.matmul(weights_T, grad_input)) return false;

        return true;
    }

    // Plain SGD, sama seperti versi Pi — optimizer lebih canggih dipegang trainer
    void update(float lr) {
        Matrix wg_scaled = weight_grad_;
        wg_scaled.scale_inplace(lr);
        weights_.add_inplace(negate(wg_scaled));

        Matrix bg_scaled = bias_grad_;
        bg_scaled.scale_inplace(lr);
        bias_.add_inplace(negate(bg_scaled));
    }

private:
    ActivationType activation_ = ActivationType::LINEAR;
    Matrix weights_;
    Matrix bias_;
    Matrix weight_grad_;
    Matrix bias_grad_;

    Matrix last_input_;
    Matrix last_z_;
    Matrix last_output_;

    static Matrix negate(Matrix m) {
        m.scale_inplace(-1.0f);
        return m;
    }

    Matrix apply_activation(const Matrix& z) {
        Matrix out(z.rows(), z.cols());
        switch (activation_) {
            case ActivationType::RELU:
                for (size_t i = 0; i < z.size(); ++i)
                    out.data()[i] = z.data()[i] > 0.0f ? z.data()[i] : 0.0f;
                break;
            case ActivationType::SIGMOID:
                for (size_t i = 0; i < z.size(); ++i)
                    out.data()[i] = 1.0f / (1.0f + expf(-z.data()[i]));
                break;
            case ActivationType::TANH:
                for (size_t i = 0; i < z.size(); ++i)
                    out.data()[i] = tanhf(z.data()[i]);
                break;
            case ActivationType::LINEAR:
                out = z;
                break;
            case ActivationType::SOFTMAX:
                for (size_t r = 0; r < z.rows(); ++r) {
                    float max_val = z(r, 0);
                    for (size_t c = 1; c < z.cols(); ++c)
                        if (z(r, c) > max_val) max_val = z(r, c);

                    float sum = 0.0f;
                    for (size_t c = 0; c < z.cols(); ++c) {
                        float e = expf(z(r, c) - max_val);
                        out(r, c) = e;
                        sum += e;
                    }
                    for (size_t c = 0; c < z.cols(); ++c)
                        out(r, c) /= sum;
                }
                break;
        }
        return out;
    }

    // Turunan activation, dikalikan elementwise dengan grad_output (kasus non-combined)
    Matrix multiply_activation_derivative(const Matrix& grad_output) {
        Matrix out(grad_output.rows(), grad_output.cols());
        switch (activation_) {
            case ActivationType::RELU:
                for (size_t i = 0; i < grad_output.size(); ++i)
                    out.data()[i] = last_z_.data()[i] > 0.0f ? grad_output.data()[i] : 0.0f;
                break;
            case ActivationType::SIGMOID:
                for (size_t i = 0; i < grad_output.size(); ++i) {
                    float s = last_output_.data()[i];
                    out.data()[i] = grad_output.data()[i] * s * (1.0f - s);
                }
                break;
            case ActivationType::TANH:
                for (size_t i = 0; i < grad_output.size(); ++i) {
                    float t = last_output_.data()[i];
                    out.data()[i] = grad_output.data()[i] * (1.0f - t * t);
                }
                break;
            case ActivationType::LINEAR:
                out = grad_output;
                break;
            case ActivationType::SOFTMAX:
                // Softmax murni (tanpa combined_with_loss) jarang dipakai sendiri
                // di sini disederhanakan jadi passthrough — pakai combined_with_loss=true
                // saat Softmax dipasangkan CrossEntropy (kasus umum kita)
                out = grad_output;
                break;
        }
        return out;
    }
};