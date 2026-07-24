// main.ino — node komputasi paralel matmul (Pi sebagai koordinator)
#include <WiFi.h>
#include <HTTPClient.h>
#include "esp_heap_caps.h"
#include "matrix_ops.h"
#include "comm.h"
#include "status_led.h"

// ===== KONFIGURASI — SESUAIKAN PER NODE =====
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

void setup() {
    Serial.begin(115200);
    delay(1000);
    Serial.printf("[main] Boot node matmul paralel, N=%u, NODE_ID=%d\n", (unsigned)N, NODE_ID);

    StatusLed::init();
    heap_checkpoint("awal boot, sebelum alokasi apapun");

    // Alokasi matriks kerja SEBELUM WiFi connect — heap masih bersih
    size_t rows_per_node = N / NUM_NODES;
    matrix_b     = Matrix(N, N, 0.0f);
    row_block_a  = Matrix(rows_per_node, N, 0.0f);
    result_block = Matrix(rows_per_node, N, 0.0f);

    heap_checkpoint("setelah alokasi matrix_b + row_block_a + result_block");

    StatusLed::working();
    if (!Comm::connect_wifi(WIFI_SSID, WIFI_PASSWORD)) {
        Serial.println("[main] WiFi gagal, restart ESP32...");
        StatusLed::blink_error();
        delay(3000);
        ESP.restart();
    }

    heap_checkpoint("setelah WiFi connect");

    while (!Comm::fetch_matrix(PI_SERVER_MATRIX_B, matrix_b)) {
        Serial.println("[main] Retry ambil matrix B dalam 3 detik...");
        StatusLed::blink_error();
        delay(3000);
    }
    Serial.println("[main] Matrix B diterima");
    heap_checkpoint("setelah fetch matrix B");

    while (!Comm::fetch_row_block(PI_SERVER_ROW_BLOCK, NODE_ID, row_block_a)) {
        Serial.println("[main] Retry ambil row block A dalam 3 detik...");
        StatusLed::blink_error();
        delay(3000);
    }
    Serial.println("[main] Row block A diterima");
    heap_checkpoint("setelah fetch row block A");

    unsigned long t0 = millis();
    if (!row_block_a.matmul(matrix_b, result_block)) {
        Serial.println("[main] matmul gagal!");
        StatusLed::blink_error();
        return;
    }
    unsigned long t1 = millis();
    Serial.printf("[main] Matmul selesai dalam %lu ms\n", t1 - t0);
    heap_checkpoint("setelah matmul selesai");

    while (!Comm::send_result(PI_SERVER_RESULT, NODE_ID, result_block)) {
        Serial.println("[main] Retry kirim hasil dalam 3 detik...");
        StatusLed::blink_error();
        delay(3000);
    }
    Serial.println("[main] Hasil terkirim, node selesai");
    heap_checkpoint("akhir, setelah kirim hasil");

    StatusLed::training_done(); // dipakai sebagai pola kedip lambat 3x = tanda tugas selesai
    StatusLed::idle();
}

void loop() {
    delay(10000); // tugas selesai di setup(), loop kosong
}