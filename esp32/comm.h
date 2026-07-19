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

    // Kirim weight lokal (hasil training) ke Pi.
    // server_url contoh: "http://192.168.1.10:5000/upload_weights"
    // node_id dipakai Pi buat tahu weight ini dari node mana.
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

    // Ambil global weight terbaru dari Pi (setelah FedAvg ronde sebelumnya).
    // server_url contoh: "http://192.168.1.10:5000/get_global_weights"
    // expected_count harus sama dengan total_param_count() model lokal.
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
};