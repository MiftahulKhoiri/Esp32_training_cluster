// matrix_ops.h — versi ESP32/Arduino (ringan, tanpa OpenBLAS, tanpa exception)
#pragma once
#include <vector>
#include <cstdint>
#include <cmath>
#include <algorithm> // Tambahan untuk std::fill
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

    const std::vector<Scalar>& data() const { return data_; }
    std::vector<Scalar>& data() { return data_; }

    static Matrix random_uniform(size_t rows, size_t cols, Scalar scale) {
        Matrix m(rows, cols);
        for (size_t i = 0; i < m.data_.size(); ++i) {
            float r = (static_cast<float>(random(-1000, 1000)) / 1000.0f);
            m.data_[i] = r * scale;
        }
        return m;
    }

    bool matmul(const Matrix& other, Matrix& out) const {
        if (cols_ != other.rows_) {
            Serial.println("[Matrix] matmul: dimensi tidak cocok");
            return false;
        }
        
        // PERBAIKAN: Jangan buat objek Matrix baru.
        // Pastikan dimensi output sesuai dengan alokasi awal di main.ino
        if (out.rows() != rows_ || out.cols() != other.cols_) {
            Serial.println("[Matrix] matmul: dimensi output tidak sesuai alokasi awal");
            return false;
        }

        // PERBAIKAN: Bersihkan nilai sebelumnya menjadi 0 tanpa membuang alokasi memori
        std::fill(out.data().begin(), out.data().end(), 0.0f);

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

    Matrix transpose() const {
        Matrix out(cols_, rows_);
        for (size_t i = 0; i < rows_; ++i)
            for (size_t j = 0; j < cols_; ++j)
                out(j, i) = (*this)(i, j);
        return out;
    }

    bool add_inplace(const Matrix& other) {
        if (rows_ != other.rows_ || cols_ != other.cols_) {
            Serial.println("[Matrix] add_inplace: dimensi tidak cocok");
            return false;
        }
        for (size_t i = 0; i < data_.size(); ++i) data_[i] += other.data_[i];
        return true;
    }

    void scale_inplace(Scalar s) {
        for (auto& v : data_) v *= s;
    }

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
