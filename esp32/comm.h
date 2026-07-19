// comm.h — WiFi + HTTP komunikasi ke Pi (versi ESP32/Arduino)
#pragma once
#include <WiFi.h>
#include <HTTPClient.h>
#include <vector>

class Comm {
public:
    static bool connect_wifi(const char* ssid, const char* password, uint32_t timeout_ms = 15000) {
        WiFi.begin(ssid, password);
        Serial.print("[Comm] Menyambung WiFi");

        uint32_t start = millis();
        while (WiFi.status() != WL_CONNECTED) {
            if (millis() - start > timeout_ms) {
                Serial.println("\n[Comm] Gagal konek WiFi (timeout)");
                return false;
            }
            delay(300);
            Serial.print(".");
        }
        Serial.println();
        Serial.print("[Comm] Terhubung, IP: ");
        Serial.println(WiFi.localIP());
        return true;
    }

    static bool send_weights(const char* server_url, int node_id, const std::vector<float>& weights) {
        if (WiFi.status() != WL_CONNECTED) {
            Serial.println("[Comm] send_weights: WiFi belum konek");
            return false;
        }

        HTTPClient http;
        String url = String(server_url) + "?node_id=" + String(node_id);
        http.begin(url);
        http.addHeader("Content-Type", "application/octet-stream");

        int code = http.POST(
            reinterpret_cast<const uint8_t*>(weights.data()),
            weights.size() * sizeof(float)
        );

        bool ok = (code == 200);
        if (!ok) {
            Serial.print("[Comm] send_weights gagal, HTTP code: ");
            Serial.println(code);
        } else {
            Serial.println("[Comm] send_weights sukses");
        }

        http.end();
        return ok;
    }

    static bool receive_global_weights(const char* server_url, std::vector<float>& out, size_t expected_count) {
        if (WiFi.status() != WL_CONNECTED) {
            Serial.println("[Comm] receive_global_weights: WiFi belum konek");
            return false;
        }

        HTTPClient http;
        http.begin(server_url);
        int code = http.GET();

        if (code != 200) {
            Serial.print("[Comm] receive_global_weights gagal, HTTP code: ");
            Serial.println(code);
            http.end();
            return false;
        }

        WiFiClient* stream = http.getStreamPtr();
        int content_len = http.getSize();

        size_t expected_bytes = expected_count * sizeof(float);
        if (content_len != (int)expected_bytes) {
            Serial.print("[Comm] receive_global_weights: ukuran payload tidak cocok, dapat ");
            Serial.print(content_len);
            Serial.print(" byte, harusnya ");
            Serial.println(expected_bytes);
            http.end();
            return false;
        }

        out.resize(expected_count);
        size_t bytes_read = stream->readBytes(
            reinterpret_cast<uint8_t*>(out.data()),
            expected_bytes
        );

        http.end();

        if (bytes_read != expected_bytes) {
            Serial.println("[Comm] receive_global_weights: baca stream tidak lengkap");
            return false;
        }

        Serial.println("[Comm] receive_global_weights sukses");
        return true;
    }

    // Ambil training data node ini dari Pi.
    // Format payload: [uint32 num_samples][context_ids uint8, num_samples*context_len]
    //                 [target_ids uint16, num_samples]
    // max_samples: batas aman kapasitas buffer ESP32 — kalau server kirim lebih, ditolak.
    static bool receive_training_data(
        const char* server_url,
        int node_id,
        size_t context_len,
        size_t max_samples,
        size_t& out_num_samples,
        std::vector<uint8_t>& out_context_ids,
        std::vector<uint16_t>& out_target_ids
    ) {
        if (WiFi.status() != WL_CONNECTED) {
            Serial.println("[Comm] receive_training_data: WiFi belum konek");
            return false;
        }

        HTTPClient http;
        String url = String(server_url) + "?node_id=" + String(node_id);
        http.begin(url);
        int code = http.GET();

        if (code != 200) {
            Serial.print("[Comm] receive_training_data gagal, HTTP code: ");
            Serial.println(code);
            http.end();
            return false;
        }

        WiFiClient* stream = http.getStreamPtr();

        // 1. Baca header: uint32 num_samples (4 byte, little-endian)
        uint8_t header_buf[4];
        size_t header_read = stream->readBytes(header_buf, 4);
        if (header_read != 4) {
            Serial.println("[Comm] receive_training_data: gagal baca header");
            http.end();
            return false;
        }
        uint32_t num_samples = 0;
        memcpy(&num_samples, header_buf, 4);

        if (num_samples == 0 || num_samples > max_samples) {
            Serial.print("[Comm] receive_training_data: num_samples tidak valid: ");
            Serial.println(num_samples);
            http.end();
            return false;
        }

        // 2. Baca context_ids (num_samples * context_len byte)
        size_t context_bytes = num_samples * context_len;
        out_context_ids.resize(context_bytes);
        size_t ctx_read = stream->readBytes(out_context_ids.data(), context_bytes);
        if (ctx_read != context_bytes) {
            Serial.println("[Comm] receive_training_data: baca context_ids tidak lengkap");
            http.end();
            return false;
        }

        // 3. Baca target_ids (num_samples * 2 byte, uint16)
        size_t target_bytes = num_samples * sizeof(uint16_t);
        out_target_ids.resize(num_samples);
        size_t tgt_read = stream->readBytes(
            reinterpret_cast<uint8_t*>(out_target_ids.data()), target_bytes
        );
        if (tgt_read != target_bytes) {
            Serial.println("[Comm] receive_training_data: baca target_ids tidak lengkap");
            http.end();
            return false;
        }

        http.end();
        out_num_samples = num_samples;
        Serial.print("[Comm] receive_training_data sukses, jumlah sample: ");
        Serial.println(num_samples);
        return true;
    }
};