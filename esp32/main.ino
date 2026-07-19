// main.ino — sketch utama, siklus federated learning per node
#include "matrix_ops.h"
#include "dense_layer.h"
#include "losses.h"
#include "trainer.h"
#include "comm.h"
#include "data_node1.h"  // GANTI: data_node2.h / data_node3.h / dst sesuai board ini

// ===== KONFIGURASI — SESUAIKAN PER NODE =====
const char* WIFI_SSID     = "GANTI_SSID";
const char* WIFI_PASSWORD = "GANTI_PASSWORD";
const char* PI_SERVER_UPLOAD  = "http://192.168.1.10:5000/upload_weights";
const char* PI_SERVER_GLOBAL  = "http://192.168.1.10:5000/get_global_weights";
const int   NODE_ID = 1; // GANTI: 1-5, beda tiap board, harus cocok dengan data_nodeN.h yang di-include

// ===== ARSITEKTUR MODEL =====
const size_t CONTEXT_LEN  = 8;    // jumlah karakter konteks
const size_t VOCAB_SIZE   = 128;  // ASCII dasar
const size_t HIDDEN_DIM   = 64;
const size_t INPUT_DIM    = CONTEXT_LEN; // id char dinormalisasi, bukan one-hot

// ===== FEDERATED CONFIG =====
const int    LOCAL_EPOCHS      = 3;
const size_t BATCH_SIZE        = 8;
const float  LEARNING_RATE     = 0.01f;
const uint32_t ROUND_INTERVAL_MS = 30000; // jeda antar ronde federated

SimpleMLP model;
Matrix train_inputs;
std::vector<uint16_t> train_targets;
size_t num_samples = 0;

// Muat data corpus asli dari data_nodeN.h (hasil data_prep.py), disimpan di flash
// lewat array const, lalu disalin ke Matrix (RAM) hanya untuk training.
void load_local_data() {
    num_samples = NODE_NUM_SAMPLES;
    train_inputs = Matrix(num_samples, INPUT_DIM, 0.0f);
    train_targets.resize(num_samples);

    for (size_t i = 0; i < num_samples; ++i) {
        for (size_t c = 0; c < CONTEXT_LEN; ++c) {
            train_inputs(i, c) = static_cast<float>(NODE_CONTEXT_IDS[i][c]) / VOCAB_SIZE;
        }
        train_targets[i] = NODE_TARGET_IDS[i];
    }

    Serial.print("[main] Data corpus dimuat, jumlah sample: ");
    Serial.println(num_samples);
}

void build_model() {
    model.layers.clear();
    model.layers.push_back(DenseLayer(INPUT_DIM, HIDDEN_DIM, ActivationType::RELU));
    model.layers.push_back(DenseLayer(HIDDEN_DIM, VOCAB_SIZE, ActivationType::SOFTMAX));
    Serial.print("[main] Model dibangun, total param: ");
    Serial.println(model.total_param_count());
}

bool run_federated_round() {
    size_t param_count = model.total_param_count();

    // 1. Ambil weight global terbaru dari Pi
    std::vector<float> global_weights;
    if (!Comm::receive_global_weights(PI_SERVER_GLOBAL, global_weights, param_count)) {
        Serial.println("[main] Gagal ambil global weight, skip ronde ini");
        return false;
    }
    if (!model.set_weights_flat(global_weights)) {
        Serial.println("[main] Gagal set weight ke model");
        return false;
    }

    // 2. Training lokal beberapa epoch
    float final_loss = Trainer::train_local_epochs(
        model, train_inputs, train_targets.data(), num_samples,
        LOCAL_EPOCHS, BATCH_SIZE, LEARNING_RATE
    );
    Serial.print("[main] Selesai training lokal, loss akhir: ");
    Serial.println(final_loss);

    // 3. Kirim weight hasil training balik ke Pi
    std::vector<float> local_weights;
    model.get_weights_flat(local_weights);
    if (!Comm::send_weights(PI_SERVER_UPLOAD, NODE_ID, local_weights)) {
        Serial.println("[main] Gagal kirim weight ke Pi");
        return false;
    }

    return true;
}

void setup() {
    Serial.begin(115200);
    delay(1000);
    Serial.println("[main] Boot node federated learning");

    randomSeed(analogRead(0)); // seed random biar tiap board beda

    build_model();
    load_local_data();

    if (!Comm::connect_wifi(WIFI_SSID, WIFI_PASSWORD)) {
        Serial.println("[main] WiFi gagal, restart ESP32...");
        delay(3000);
        ESP.restart();
    }
}

void loop() {
    Serial.println("[main] === Mulai ronde federated ===");
    run_federated_round();
    Serial.print("[main] Ronde selesai, tunggu ");
    Serial.print(ROUND_INTERVAL_MS / 1000);
    Serial.println(" detik...");
    delay(ROUND_INTERVAL_MS);
}