// comm.h — WiFi + HTTP komunikasi ke Pi (versi komputasi paralel matmul + Dinamis)
#pragma once
#include <WiFi.h>
#include <HTTPClient.h>
#include <vector>
#include "matrix_ops.h"

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

    // Fungsi Baru: Ambil N dan NUM_NODES dari Server
    static bool fetch_config(const char* url, size_t& out_N, int& out_num_nodes) {
        if (WiFi.status() != WL_CONNECTED) return false;

        HTTPClient http;
        http.begin(url);
        int code = http.GET();

        if (code == 200) {
            String payload = http.getString();
            int comma_idx = payload.indexOf(',');
            
            if (comma_idx > 0) {
                out_N = payload.substring(0, comma_idx).toInt();
                out_num_nodes = payload.substring(comma_idx + 1).toInt();
                http.end();
                return true;
            }
        }
        
        http.end();
        return false;
    }

    static bool fetch_matrix(const char* url, Matrix& out) {
        if (WiFi.status() != WL_CONNECTED) {
            Serial.println("[Comm] fetch_matrix: WiFi belum konek");
            return false;
        }

        HTTPClient http;
        http.begin(url);
        int code = http.GET();

        if (code != 200) {
            Serial.print("[Comm] fetch_matrix gagal, HTTP code: ");
            Serial.println(code);
            http.end();
            return false;
        }

        WiFiClient* stream = http.getStreamPtr();
        int content_len = http.getSize();
        size_t expected_bytes = out.size() * sizeof(float);

        if (content_len != (int)expected_bytes) {
            Serial.print("[Comm] fetch_matrix: ukuran payload tidak cocok, dapat ");
            Serial.print(content_len);
            Serial.print(" byte, harusnya ");
            Serial.println(expected_bytes);
            http.end();
            return false;
        }

        size_t bytes_read = stream->readBytes(
            reinterpret_cast<uint8_t*>(out.data().data()), expected_bytes
        );
        http.end();

        if (bytes_read != expected_bytes) {
            Serial.println("[Comm] fetch_matrix: baca stream tidak lengkap");
            return false;
        }

        Serial.println("[Comm] fetch_matrix sukses");
        return true;
    }

    static bool fetch_row_block(const char* server_url, int node_id, Matrix& out) {
        String url = String(server_url) + "?node_id=" + String(node_id);
        return fetch_matrix(url.c_str(), out);
    }

    static bool send_result(const char* server_url, int node_id, Matrix& result) {
        if (WiFi.status() != WL_CONNECTED) {
            Serial.println("[Comm] send_result: WiFi belum konek");
            return false;
        }

        HTTPClient http;
        String url = String(server_url) + "?node_id=" + String(node_id);
        http.begin(url);
        http.addHeader("Content-Type", "application/octet-stream");

        size_t payload_bytes = result.size() * sizeof(float);
        int code = http.POST(
            reinterpret_cast<uint8_t*>(result.data().data()), payload_bytes
        );

        bool ok = (code == 200);
        if (!ok) {
            Serial.print("[Comm] send_result gagal, HTTP code: ");
            Serial.println(code);
        } else {
            Serial.println("[Comm] send_result sukses");
        }

        http.end();
        return ok;
    }
};
