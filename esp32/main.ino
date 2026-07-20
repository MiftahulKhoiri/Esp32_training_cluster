// main.ino — sketch utama, siklus federated learning per node
#include "matrix_ops.h"
#include "dense_layer.h"
#include "losses.h"
#include "trainer.h"
#include "comm.h"
#include "status_led.h"

// ===== KONFIGURASI — SESUAIKAN PER NODE =====
const char* WIFI_SSID     = "GANTI_SSID";
const char* WIFI_PASSWORD = "GANTI_PASSWORD";
const char* PI_SERVER_UPLOAD    = "http://192.168.1.10:5000/upload_weights";
const char* PI_SERVER_GLOBAL    = "http://192.168.1.10:5000/get_global_weights";
const char* PI_SERVER_DATA      = "http://192.168.1.10:5000/get_training_data";
const char* PI_SERVER_STATUS    = "http://192.168.1.10:5000/training_status_flag";
const char* PI_SERVER_VOCAB     = "http://192.168.1.10:5000/vocab_size";
const char* PI_SERVER_HEARTBEAT = "http://192.168.1.10:5000/heartbeat";
const int   NODE_ID = 1; // GANTI: 1-5, beda tiap board

// ===== ARSITEKTUR MODEL =====
const size_t CONTEXT_LEN  = 8;
size_t VOCAB_SIZE = 0;
const size_t HIDDEN_DIM   = 64;
const size_t INPUT_DIM    = CONTEXT_LEN;
const size_t MAX_SAMPLES  = 300;

// ===== FEDERATED CONFIG =====
const int    LOCAL_EPOCHS      = 3;
const size_t BATCH_SIZE        = 8;
const float  LEARNING_RATE     = 0.01f;
const uint32_t ROUND_INTERVAL_MS = 30000;
const uint32_t STATUS_CHECK_INTERVAL_MS = 10000;

SimpleMLP model;
Matrix train_inputs;
std::vector<uint16_t> train_targets;
size_t num_samples = 0;
bool training_finished = false;

bool load_local_data() {
    std::vector<uint16_t> context_ids;
    std::vector<uint16_t> target_ids;

    bool ok = Comm::receive_training_data(
        PI_SERVER_DATA, NODE_ID, CONTEXT_LEN, MAX_SAMPLES,
        num_samples, context_ids, target_ids
    );
    if (!ok) {
        Serial.println("[main] Gagal ambil training data dari server");
        return false;
    }

    train_inputs = Matrix(num_samples, INPUT_DIM, 0.0f);
    train_targets.resize(num_samples);

    for (size_t i = 0; i < num_samples; ++i) {
        for (size_t c = 0; c < CONTEXT_LEN; ++c) {
            uint16_t token_id = context_ids[i * CONTEXT_LEN + c];
            train_inputs(i, c) = static_cast<float>(token_id) / VOCAB_SIZE;
        }
        train_targets[i] = target_ids[i];
    }

    Serial.print("[main] Data dari server dimuat, jumlah sample: ");
    Serial.println(num_samples);
    return true;
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

    StatusLed::working();
    std::vector<float> global_weights;
    if (!Comm::receive_global_weights(PI_SERVER_GLOBAL, global_weights, param_count)) {
        Serial.println("[main] Gagal ambil global weight, skip ronde ini");
        StatusLed::blink_error();
        StatusLed::idle();
        return false;
    }
    if (!model.set_weights_flat(global_weights)) {
        Serial.println("[main] Gagal set weight ke model");
        StatusLed::blink_error();
        StatusLed::idle();
        return false;
    }

    float final_loss = Trainer::train_local_epochs(
        model, train_inputs, train_targets.data(), num_samples,
        LOCAL_EPOCHS, BATCH_SIZE, LEARNING_RATE
    );
    Serial.print("[main] Selesai training lokal, loss akhir: ");
    Serial.println(final_loss);

    std::vector<float> local_weights;
    model.get_weights_flat(local_weights);
    if (!Comm::send_weights(PI_SERVER_UPLOAD, NODE_ID, local_weights)) {
        Serial.println("[main] Gagal kirim weight ke Pi");
        StatusLed::blink_error();
        StatusLed::idle();
        return false;
    }

    StatusLed::idle();
    return true;
}

void setup() {
    Serial.begin(115200);
    delay(1000);
    Serial.println("[main] Boot node federated learning");

    StatusLed::init();
    randomSeed(analogRead(0));

    StatusLed::working();
    if (!Comm::connect_wifi(WIFI_SSID, WIFI_PASSWORD)) {
        Serial.println("[main] WiFi gagal, restart ESP32...");
        StatusLed::blink_error();
        delay(3000);
        ESP.restart();
    }

    while (!Comm::fetch_vocab_size(PI_SERVER_VOCAB, VOCAB_SIZE) || VOCAB_SIZE == 0) {
        Serial.println("[main] Retry ambil vocab_size dalam 5 detik...");
        StatusLed::blink_error();
        delay(5000);
    }

    build_model();

    while (!load_local_data()) {
        Serial.println("[main] Retry ambil training data dalam 5 detik...");
        StatusLed::blink_error();
        delay(5000);
    }

    Comm::send_heartbeat(PI_SERVER_HEARTBEAT, NODE_ID); // kabari server: node ini hidup, dari awal

    StatusLed::idle();
}

void loop() {
    // Heartbeat SELALU dikirim tiap loop, terlepas status training — ini yang
    // bikin server tahu node ini masih hidup walau lagi idle nunggu ronde lain.
    Comm::send_heartbeat(PI_SERVER_HEARTBEAT, NODE_ID);

    bool is_complete = false;
    bool status_ok = Comm::check_training_complete(PI_SERVER_STATUS, is_complete);

    if (status_ok && is_complete) {
        if (!training_finished) {
            Serial.println("[main] *** TRAINING SELESAI menurut server — berhenti training, node idle ***");
            StatusLed::training_done();
            training_finished = true;
        }
        StatusLed::idle();
        delay(STATUS_CHECK_INTERVAL_MS);
        return;
    }

    if (status_ok && !is_complete && training_finished) {
        Serial.println("[main] Training di server aktif lagi, node lanjut ikut training");
        training_finished = false;
    }

    Serial.println("[main] === Mulai ronde federated ===");
    run_federated_round();
    Serial.print("[main] Ronde selesai, tunggu ");
    Serial.print(ROUND_INTERVAL_MS / 1000);
    Serial.println(" detik...");
    delay(ROUND_INTERVAL_MS);
}