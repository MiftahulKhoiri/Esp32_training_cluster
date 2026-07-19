// matrix_ops.h — versi ESP32/Arduino (ringan, tanpa OpenBLAS, tanpa exception)
#pragma once
#include <vector>
#include <cstdint>
#include <cmath>
#include <Arduino.h>

class Matrix {
public:
    using Scalar = float;

    Matrix() : rows_(0), cols_(0) {}

    Matrix(size_t rows, size_t cols, Scalar init_val = 0.0f)
        : rows_(rows), cols_(cols), data_(rows * cols, init_val) {}

    size_t rows() const { return rows_; }
    size_t cols() const { return cols_; }
    size_t size() const { return data_.size(); }

    Scalar& operator()(size_t r, size_t c) {
        return data_[r * cols_ + c];
    }
    Scalar operator()(size_t r, size_t c) const {
        return data_[r * cols_ + c];
    }

    // Akses raw buffer — dipakai nanti untuk serialize weight ke Pi (FedAvg)
    const std::vector<Scalar>& data() const { return data_; }
    std::vector<Scalar>& data() { return data_; }

    static Matrix random_uniform(size_t rows, size_t cols, Scalar scale) {
        Matrix m(rows, cols);
        for (size_t i = 0; i < m.data_.size(); ++i) {
            // random(-1000,1000)/1000.0 -> [-1,1], lalu diskalakan
            float r = (static_cast<float>(random(-1000, 1000)) / 1000.0f);
            m.data_[i] = r * scale;
        }
        return m;
    }

    // out = this * other  (matmul biasa, tanpa cache-blocking — matriks kecil di ESP32)
    bool matmul(const Matrix& other, Matrix& out) const {
        if (cols_ != other.rows_) {
            Serial.println("[Matrix] matmul: dimensi tidak cocok");
            return false;
        }
        out = Matrix(rows_, other.cols_, 0.0f);
        for (size_t i = 0; i < rows_; ++i) {
            for (size_t k = 0; k < cols_; ++k) {
                Scalar a_ik = (*this)(i, k);
                if (a_ik == 0.0f) continue; // sedikit hemat compute
                for (size_t j = 0; j < other.cols_; ++j) {
                    out(i, j) += a_ik * other(k, j);
                }
            }
        }
        return true;
    }

    // Transpose
    Matrix transpose() const {
        Matrix out(cols_, rows_);
        for (size_t i = 0; i < rows_; ++i)
            for (size_t j = 0; j < cols_; ++j)
                out(j, i) = (*this)(i, j);
        return out;
    }

    // Elementwise add (in-place ke this)
    bool add_inplace(const Matrix& other) {
        if (rows_ != other.rows_ || cols_ != other.cols_) {
            Serial.println("[Matrix] add_inplace: dimensi tidak cocok");
            return false;
        }
        for (size_t i = 0; i < data_.size(); ++i) data_[i] += other.data_[i];
        return true;
    }

    // Scale in-place (dipakai buat SGD update: w -= lr * grad)
    void scale_inplace(Scalar s) {
        for (auto& v : data_) v *= s;
    }

    // Tambah bias per-baris (broadcast 1 x cols ke rows x cols)
    bool add_bias_broadcast(const Matrix& bias) {
        if (bias.rows_ != 1 || bias.cols_ != cols_) {
            Serial.println("[Matrix] add_bias_broadcast: dimensi bias salah");
            return false;
        }
        for (size_t i = 0; i < rows_; ++i)
            for (size_t j = 0; j < cols_; ++j)
                (*this)(i, j) += bias(0, j);
        return true;
    }

private:
    size_t rows_;
    size_t cols_;
    std::vector<Scalar> data_;
};