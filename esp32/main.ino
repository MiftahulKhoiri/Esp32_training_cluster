// matmul_node.ino — node komputasi paralel: hitung blok baris A x B
#include <WiFi.h>
#include <HTTPClient.h>
#include "esp_heap_caps.h"
#include "matrix_ops.h"

const char* WIFI_SSID     = "wifi server";
const char* WIFI_PASSWORD = "1234rewq";
const char* PI_SERVER_MATRIX_B  = "http://192.168.1.2:5000/get_matrix_b";
const char* PI_SERVER_ROW_BLOCK = "http://192.168.1.2:5000/get_row_block";
const char* PI_SERVER_RESULT    = "http://192.168.1.2:5000/submit_result";
const int   NODE_ID    = 1; // GANTI: 1-5, beda tiap board
const int   NUM_NODES  = 5;

// ===== UBAH CUMA INI UNTUK NAIKKAN UKURAN UJI =====
const size_t N = 10; // ukuran matriks persegi — naikkan bertahap: 10, 25, 50, 75, 100, 150, 200...

Matrix matrix_b;
Matrix row_block_a;
Matrix result_block;

void heap_checkpoint(const char* label) {
    Serial.print("[HEAP][");
    Serial.print(label);
    Serial.print("] free=");
    Serial.print(ESP.getFreeHeap());
    Serial.print(" largest_block=");
    Serial.print(heap_caps_get_largest_free_block(MALLOC_CAP_8BIT));
    Serial.print(" integrity=");
    bool ok = heap_caps_check_integrity_all(true);
    Serial.println(ok ? "OK" : "CORRUPT!!");
}

bool connect_wifi() {
    WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
    Serial.print("[node] Connecting WiFi");
    int retry = 0;
    while (WiFi.status() != WL_CONNECTED && retry < 30) {
        delay(500);
        Serial.print(".");
        retry++;
    }
    Serial.println();
    if (WiFi.status() != WL_CONNECTED) return false;
    Serial.print("[node] WiFi connected, IP: ");
    Serial.println(WiFi.localIP());
    return true;
}

bool fetch_matrix(const char* url, Matrix& out) {
    HTTPClient http;
    http.begin(url);
    int code = http.GET();
    if (code != 200) {
        Serial.printf("[node] GET %s gagal, kode=%d\n", url, code);
        http.end();
        return false;
    }
    WiFiClient* stream = http.getStreamPtr();
    size_t expected_bytes = out.size() * sizeof(float);
    size_t total_len = http.getSize();
    if (total_len != expected_bytes) {
        Serial.printf("[node] Ukuran payload tidak cocok: expected=%u got=%u\n",
                      (unsigned)expected_bytes, (unsigned)total_len);
        http.end();
        return false;
    }
    size_t read_bytes = stream->readBytes(
        reinterpret_cast<uint8_t*>(out.data().data()), expected_bytes
    );
    http.end();
    if (read_bytes != expected_bytes) {
        Serial.println("[node] Baca stream tidak lengkap");
        return false;
    }
    return true;
}

bool fetch_row_block() {
    String url = String(PI_SERVER_ROW_BLOCK) + "?node_id=" + NODE_ID;
    return fetch_matrix(url.c_str(), row_block_a);
}

bool send_result() {
    HTTPClient http;
    String url = String(PI_SERVER_RESULT) + "?node_id=" + NODE_ID;
    http.begin(url);
    http.addHeader("Content-Type", "application/octet-stream");
    size_t payload_bytes = result_block.size() * sizeof(float);
    int code = http.POST(
        reinterpret_cast<uint8_t*>(result_block.data().data()), payload_bytes
    );
    http.end();
    if (code != 200) {
        Serial.printf("[node] POST hasil gagal, kode=%d\n", code);
        return false;
    }
    return true;
}

void setup() {
    Serial.begin(115200);
    delay(1000);
    Serial.printf("[node] Boot node matmul paralel, N=%u\n", (unsigned)N);

    heap_checkpoint("awal boot, sebelum alokasi apapun");

    // Alokasi matriks kerja SEBELUM WiFi connect — heap masih bersih
    size_t rows_per_node = N / NUM_NODES;
    matrix_b     = Matrix(N, N, 0.0f);
    row_block_a  = Matrix(rows_per_node, N, 0.0f);
    result_block = Matrix(rows_per_node, N, 0.0f);

    heap_checkpoint("setelah alokasi matrix_b + row_block_a + result_block");

    if (!connect_wifi()) {
        Serial.println("[node] WiFi gagal, restart...");
        delay(3000);
        ESP.restart();
    }

    heap_checkpoint("setelah WiFi connect");

    while (!fetch_matrix(PI_SERVER_MATRIX_B, matrix_b)) {
        Serial.println("[node] Retry ambil matrix B dalam 3 detik...");
        delay(3000);
    }
    Serial.println("[node] Matrix B diterima");
    heap_checkpoint("setelah fetch matrix B");

    while (!fetch_row_block()) {
        Serial.println("[node] Retry ambil row block A dalam 3 detik...");
        delay(3000);
    }
    Serial.println("[node] Row block A diterima");
    heap_checkpoint("setelah fetch row block A");

    unsigned long t0 = millis();
    if (!row_block_a.matmul(matrix_b, result_block)) {
        Serial.println("[node] matmul gagal!");
        return;
    }
    unsigned long t1 = millis();
    Serial.printf("[node] Matmul selesai dalam %lu ms\n", t1 - t0);
    heap_checkpoint("setelah matmul selesai");

    while (!send_result()) {
        Serial.println("[node] Retry kirim hasil dalam 3 detik...");
        delay(3000);
    }
    Serial.println("[node] Hasil terkirim, node selesai");
    heap_checkpoint("akhir, setelah kirim hasil");
}

void loop() {
    delay(10000);
}