// main.ino — node komputasi paralel matmul (Dynamic Config Auto-Restart)
#include <WiFi.h>
#include <HTTPClient.h>
#include "esp_heap_caps.h"
#include "matrix_ops.h"
#include "comm.h"
#include "status_led.h"

// ===== KONFIGURASI JARINGAN =====
const char* WIFI_SSID           = "wifi server";
const char* WIFI_PASSWORD       = "1234rewq";
const char* PI_SERVER_CONFIG    = "http://192.168.1.2:5000/get_config";
const char* PI_SERVER_MATRIX_B  = "http://192.168.1.2:5000/get_matrix_b";
const char* PI_SERVER_ROW_BLOCK = "http://192.168.1.2:5000/get_row_block";
const char* PI_SERVER_RESULT    = "http://192.168.1.2:5000/submit_result";

const int NODE_ID = 1; // GANTI: 1-5, beda tiap board ESP32

// ===== VARIABEL GLOBAL DINAMIS =====
size_t current_N = 0;
int current_NUM_NODES = 0;

Matrix matrix_b;
Matrix row_block_a;
Matrix result_block;

void heap_checkpoint(const char* label) {
    Serial.print("[HEAP][");
    Serial.print(label);
    Serial.print("] free=");
    Serial.print(ESP.getFreeHeap());
    Serial.print(" largest_block=");
    Serial.println(heap_caps_get_largest_free_block(MALLOC_CAP_8BIT));
}

void setup() {
    Serial.begin(115200);
    delay(1000);
    Serial.printf("[main] Boot node matmul paralel, NODE_ID=%d\n", NODE_ID);

    StatusLed::init();
    StatusLed::working();
    heap_checkpoint("awal boot (RAM kosong)");

    // 1. Konek WiFi
    if (!Comm::connect_wifi(WIFI_SSID, WIFI_PASSWORD)) {
        Serial.println("[main] WiFi gagal, restart ESP32...");
        StatusLed::blink_error();
        delay(3000);
        ESP.restart();
    }
    heap_checkpoint("setelah WiFi connect");

    // 2. Ambil konfigurasi (N dan NUM_NODES) dari Server Pi
    while (!Comm::fetch_config(PI_SERVER_CONFIG, current_N, current_NUM_NODES)) {
        Serial.println("[main] Menunggu server Pi hidup (Retry 3s)...");
        StatusLed::blink_error(2);
        delay(3000);
    }
    Serial.printf("[main] Config diterima: N=%u, NUM_NODES=%d\n", current_N, current_NUM_NODES);

    // 3. Hitung beban baris (menangani sisa bagi untuk node terakhir)
    size_t base_rows = current_N / current_NUM_NODES;
    size_t remainder = current_N % current_NUM_NODES;
    size_t rows_per_node = (NODE_ID == current_NUM_NODES) ? (base_rows + remainder) : base_rows;

    // 4. Alokasikan memori berdasarkan konfigurasi dinamis (hanya sekali)
    matrix_b     = Matrix(current_N, current_N, 0.0f);
    row_block_a  = Matrix(rows_per_node, current_N, 0.0f);
    result_block = Matrix(rows_per_node, current_N, 0.0f);
    heap_checkpoint("setelah alokasi RAM matriks");

    // 5. Fetch Matriks B & Blok A
    while (!Comm::fetch_matrix(PI_SERVER_MATRIX_B, matrix_b)) { 
        Serial.println("[main] Retry Matrix B...");
        delay(3000); 
    }
    Serial.println("[main] Matrix B diterima");

    while (!Comm::fetch_row_block(PI_SERVER_ROW_BLOCK, NODE_ID, row_block_a)) { 
        Serial.println("[main] Retry Row Block A...");
        delay(3000); 
    }
    Serial.println("[main] Row block A diterima");

    // 6. Komputasi (Perkalian Matriks)
    unsigned long t0 = millis();
    if (!row_block_a.matmul(matrix_b, result_block)) {
        Serial.println("[main] matmul gagal!");
        StatusLed::blink_error();
        return;
    }
    unsigned long t1 = millis();
    Serial.printf("[main] Matmul selesai dalam %lu ms\n", t1 - t0);
    heap_checkpoint("setelah matmul selesai");

    // 7. Kirim Hasil
    while (!Comm::send_result(PI_SERVER_RESULT, NODE_ID, result_block)) { 
        Serial.println("[main] Retry kirim hasil...");
        delay(3000); 
    }
    Serial.println("[main] Hasil terkirim, masuk ke mode siaga (polling).");
    
    StatusLed::task_done(); 
    StatusLed::idle();
}

void loop() {
    // Mode Siaga: Cek server Pi setiap 5 detik
    delay(5000);
    
    size_t new_N = 0;
    int new_NUM_NODES = 0;
    
    // Cek apakah server Pi mengubah ukuran matriks
    if (Comm::fetch_config(PI_SERVER_CONFIG, new_N, new_NUM_NODES)) {
        if (new_N != current_N || new_NUM_NODES != current_NUM_NODES) {
            Serial.printf("\n[main] Server mengubah ukuran (N lama: %u, N baru: %u)!\n", current_N, new_N);
            Serial.println("[main] Merestart ESP32 untuk membersihkan memori RAM...");
            delay(1000);
            ESP.restart(); // Kembalikan RAM 100% bersih tanpa fragmentasi
        }
    }
}
