// losses.h — versi ESP32/Arduino
#pragma once
#include "matrix_ops.h"
#include <Arduino.h>
#include <cmath>

class Losses {
public:
    // Sparse Softmax + CrossEntropy digabung (dipakai untuk output layer klasifikasi
    // char-level: prediksi karakter berikutnya dari vocab kecil).
    //
    // probs        : (batch, num_classes) — hasil forward() DenseLayer dengan
    //                activation SOFTMAX
    // target_ids   : id kelas benar per sample, panjang = batch
    // grad_out     : (batch, num_classes) — dL/d(logits), langsung dipakai sebagai
    //                grad_output ke DenseLayer::backward() dengan combined_with_loss=true
    //
    // Return: nilai loss (rata-rata per batch)
    static float sparse_softmax_cross_entropy(
        const Matrix& probs,
        const uint16_t* target_ids,
        Matrix& grad_out
    ) {
        size_t batch = probs.rows();
        size_t num_classes = probs.cols();

        grad_out = Matrix(batch, num_classes, 0.0f);
        float total_loss = 0.0f;

        for (size_t i = 0; i < batch; ++i) {
            uint16_t target = target_ids[i];

            float p_target = probs(i, target);
            // clamp kecil biar tidak log(0)
            if (p_target < 1e-7f) p_target = 1e-7f;
            total_loss += -logf(p_target);

            // Gradien Softmax+CE gabungan: grad = probs - one_hot(target)
            for (size_t c = 0; c < num_classes; ++c) {
                grad_out(i, c) = probs(i, c);
            }
            grad_out(i, target) -= 1.0f;

            // Rata-rata terhadap batch (biar konsisten dgn skala lr)
            for (size_t c = 0; c < num_classes; ++c) {
                grad_out(i, c) /= static_cast<float>(batch);
            }
        }

        return total_loss / static_cast<float>(batch);
    }

    // MSE — disediakan untuk keperluan debug/eksperimen ringan, bukan jalur utama
    static float mse(const Matrix& pred, const Matrix& target, Matrix& grad_out) {
        if (pred.rows() != target.rows() || pred.cols() != target.cols()) {
            Serial.println("[Losses] mse: dimensi tidak cocok");
            return -1.0f;
        }
        grad_out = Matrix(pred.rows(), pred.cols(), 0.0f);
        float total = 0.0f;
        size_t n = pred.size();

        for (size_t i = 0; i < n; ++i) {
            float diff = pred.data()[i] - target.data()[i];
            total += diff * diff;
            grad_out.data()[i] = 2.0f * diff / static_cast<float>(n);
        }
        return total / static_cast<float>(n);
    }
};