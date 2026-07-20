// bpe_tokenizer.h — byte-level BPE tokenizer, C++ murni (jalan di Pi, bukan ESP32)
// Base vocab dinamis (dari byte yang muncul di corpus), bukan reserve 0-255 penuh —
// supaya vocab kecil (target ~200) tidak habis buat token special+base saja.
#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <cstdint>

class BPETokenizer {
public:
    // Token khusus — selalu id tetap 0-3, sama seperti desain ml_manual_cpp
    static constexpr uint16_t PAD_ID = 0;
    static constexpr uint16_t BOS_ID = 1;
    static constexpr uint16_t EOS_ID = 2;
    static constexpr uint16_t UNK_ID = 3;
    static constexpr uint16_t NUM_SPECIAL_TOKENS = 4;

    BPETokenizer() = default;

    // Latih tokenizer dari corpus mentah. Base vocab (byte unik di corpus) dibuat
    // dulu, lalu merge pair paling sering diulang sampai vocab_size() == target_vocab_size
    // atau tidak ada lagi pair yang layak digabung (frekuensi < 2).
    void train(const std::string& corpus, size_t target_vocab_size);

    // Encode teks jadi urutan token id, pakai merge rank-based greedy
    // (merge dengan rank/urutan belajar paling awal diprioritaskan lebih dulu).
    std::vector<uint16_t> encode(const std::string& text) const;

    // Decode urutan token id balik jadi teks (byte string).
    std::string decode(const std::vector<uint16_t>& ids) const;

    size_t vocab_size() const { return id_to_bytes_.size(); }

    // Simpan/muat tokenizer: cuma base vocab (byte mentah) + urutan merge yang disimpan,
    // bukan seluruh id_to_bytes_ (hemat ukuran file, direkonstruksi ulang saat load).
    bool save(const std::string& path) const;
    bool load(const std::string& path);

private:
    // id_to_bytes_[id] = urutan byte mentah yang direpresentasikan token itu.
    // id 0-3: token khusus (kosong/placeholder, tidak didecode jadi byte apa pun)
    // id 4..4+base_count-1: base vocab (1 byte per token)
    // id base_count+4...: hasil merge (konkatenasi byte dari 2 token yang digabung)
    std::vector<std::vector<uint8_t>> id_to_bytes_;

    std::unordered_map<uint8_t, uint16_t> byte_to_id_; // base vocab: byte mentah -> id

    // Urutan merge yang dipelajari (rank = index di vector ini).
    // merges_[rank] = {id_kiri, id_kanan} -> hasil gabungnya id = NUM_SPECIAL + base_count + rank
    std::vector<std::pair<uint16_t, uint16_t>> merges_;

    // Lookup cepat: pair (id_kiri, id_kanan) -> rank, dipakai saat encode
    std::unordered_map<uint64_t, size_t> merge_rank_;

    static uint64_t pair_key(uint16_t a, uint16_t b) {
        return (static_cast<uint64_t>(a) << 32) | static_cast<uint64_t>(b);
    }

    void build_base_vocab(const std::string& corpus);
};