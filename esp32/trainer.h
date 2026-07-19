// trainer.h — versi ESP32/Arduino
#pragma once
#include "matrix_ops.h"
#include "dense_layer.h"
#include "losses.h"
#include <Arduino.h>
#include <vector>

// Model MLP sederhana: stack DenseLayer, forward berurutan.
// Tidak pakai LayerBase/unique_ptr (beda dari versi Pi) — cukup vector<DenseLayer>
// karena arsitektur ESP32 ini tetap MLP polos, tanpa BatchNorm/Attention.
class SimpleMLP {
public:
    std::vector<DenseLayer> layers;

    bool forward(const Matrix& input, Matrix& output) {
        Matrix current = input;
        for (auto& layer : layers) {
            Matrix next;
            if (!layer.forward(current, next)) return false;
            current = next;
        }
        output = current;
        return true;
    }

    // grad_output = dL/d(logits) dari Losses (sudah combined dengan Softmax)
    bool backward(const Matrix& grad_output) {
        Matrix grad = grad_output;
        for (int i = (int)layers.size() - 1; i >= 0; --i) {
            bool is_last = (i == (int)layers.size() - 1);
            Matrix grad_input;
            if (!layers[i].backward(grad, is_last, grad_input)) return false;
            grad = grad_input;
        }
        return true;
    }

    void update(float lr) {
        for (auto& layer : layers) layer.update(lr);
    }

    // Total jumlah float (weight + bias) di seluruh layer — dipakai untuk alokasi
    // buffer flat sebelum kirim ke Pi
    size_t total_param_count() const {
        size_t total = 0;
        for (const auto& layer : layers) {
            // const_cast aman di sini karena cuma baca ukuran, bukan isi
            total += const_cast<DenseLayer&>(layer).weights().size();
            total += const_cast<DenseLayer&>(layer).bias().size();
        }
        return total;
    }

    // Serialize semua weight+bias jadi satu buffer float flat, urutan:
    // [layer0.weights, layer0.bias, layer1.weights, layer1.bias, ...]
    // Dipakai untuk payload HTTP POST ke Pi.
    bool get_weights_flat(std::vector<float>& out) const {
        out.clear();
        out.reserve(total_param_count());
        for (const auto& layer : layers) {
            auto& w = const_cast<DenseLayer&>(layer).weights().data();
            auto& b = const_cast<DenseLayer&>(layer).bias().data();
            out.insert(out.end(), w.begin(), w.end());
            out.insert(out.end(), b.begin(), b.end());
        }
        return true;
    }

    // Load weight dari buffer flat (hasil broadcast global model dari Pi).
    // Urutan harus sama persis dengan get_weights_flat().
    bool set_weights_flat(const std::vector<float>& in) {
        if (in.size() != total_param_count()) {
            Serial.println("[SimpleMLP] set_weights_flat: ukuran tidak cocok");
            return false;
        }
        size_t offset = 0;
        for (auto& layer : layers) {
            auto& w = layer.weights().data();
            for (size_t i = 0; i < w.size(); ++i) w[i] = in[offset++];
            auto& b = layer.bias().data();
            for (size_t i = 0; i < b.size(); ++i) b[i] = in[offset++];
        }
        return true;
    }
};

// Trainer: jalankan beberapa epoch training lokal di atas data node ini.
// Dipanggil sekali per "ronde federated" setelah model menerima weight global dari Pi.
class Trainer {
public:
    // inputs      : (num_samples, input_dim) — context char (mis. one-hot/id ternormalisasi)
    // target_ids  : id kelas benar per sample, panjang = num_samples
    // Return: rata-rata loss di epoch terakhir (buat logging/Serial print)
    static float train_local_epochs(
        SimpleMLP& model,
        const Matrix& inputs,
        const uint16_t* target_ids,
        size_t num_samples,
        int epochs,
        size_t batch_size,
        float lr
    ) {
        std::vector<size_t> indices(num_samples);
        for (size_t i = 0; i < num_samples; ++i) indices[i] = i;

        float last_epoch_loss = 0.0f;

        for (int epoch = 0; epoch < epochs; ++epoch) {
            // Fisher-Yates shuffle in-place (hemat memori, tanpa alokasi baru)
            for (size_t i = num_samples - 1; i > 0; --i) {
                size_t j = random(0, i + 1);
                size_t tmp = indices[i];
                indices[i] = indices[j];
                indices[j] = tmp;
            }

            float epoch_loss = 0.0f;
            size_t num_batches = 0;

            for (size_t start = 0; start < num_samples; start += batch_size) {
                size_t end = min(start + batch_size, num_samples);
                size_t cur_batch = end - start;

                Matrix batch_input(cur_batch, inputs.cols());
                std::vector<uint16_t> batch_targets(cur_batch);

                for (size_t bi = 0; bi < cur_batch; ++bi) {
                    size_t src_row = indices[start + bi];
                    for (size_t c = 0; c < inputs.cols(); ++c) {
                        batch_input(bi, c) = inputs(src_row, c);
                    }
                    batch_targets[bi] = target_ids[src_row];
                }

                Matrix probs;
                if (!model.forward(batch_input, probs)) {
                    Serial.println("[Trainer] forward gagal");
                    continue;
                }

                Matrix grad_out;
                float loss = Losses::sparse_softmax_cross_entropy(
                    probs, batch_targets.data(), grad_out
                );

                if (!model.backward(grad_out)) {
                    Serial.println("[Trainer] backward gagal");
                    continue;
                }
                model.update(lr);

                epoch_loss += loss;
                num_batches++;
            }

            last_epoch_loss = (num_batches > 0) ? (epoch_loss / num_batches) : 0.0f;
            Serial.printf("[Trainer] epoch %d/%d, loss=%.4f\n", epoch + 1, epochs, last_epoch_loss);
        }

        return last_epoch_loss;
    }
};