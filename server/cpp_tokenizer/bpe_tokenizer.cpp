// bpe_tokenizer.cpp — implementasi BPETokenizer (C++ murni, jalan di Pi)
#include "bpe_tokenizer.h"
#include <fstream>
#include <algorithm>
#include <iostream>

void BPETokenizer::build_base_vocab(const std::string& corpus) {
    id_to_bytes_.clear();
    byte_to_id_.clear();

    // id 0-3: token khusus, placeholder kosong (tidak pernah didecode jadi byte)
    for (uint16_t i = 0; i < NUM_SPECIAL_TOKENS; ++i) {
        id_to_bytes_.push_back({});
    }

    // Base vocab: hanya byte yang benar-benar muncul di corpus, urut kemunculan pertama
    for (unsigned char ch : corpus) {
        uint8_t b = static_cast<uint8_t>(ch);
        if (byte_to_id_.find(b) == byte_to_id_.end()) {
            uint16_t new_id = static_cast<uint16_t>(id_to_bytes_.size());
            byte_to_id_[b] = new_id;
            id_to_bytes_.push_back({b});
        }
    }
}

void BPETokenizer::train(const std::string& corpus, size_t target_vocab_size) {
    merges_.clear();
    merge_rank_.clear();

    build_base_vocab(corpus);

    if (id_to_bytes_.size() >= target_vocab_size) {
        std::cout << "[BPETokenizer] Base vocab (" << id_to_bytes_.size()
                  << ") sudah >= target_vocab_size (" << target_vocab_size
                  << "), tidak ada merge dilakukan.\n";
        return;
    }

    // Sequence awal: tiap byte di corpus -> id base vocab-nya
    std::vector<uint16_t> seq;
    seq.reserve(corpus.size());
    for (unsigned char ch : corpus) {
        seq.push_back(byte_to_id_[static_cast<uint8_t>(ch)]);
    }

    while (id_to_bytes_.size() < target_vocab_size) {
        // Hitung frekuensi tiap pair yang bersebelahan
        std::unordered_map<uint64_t, size_t> pair_counts;
        std::unordered_map<uint64_t, std::pair<uint16_t, uint16_t>> pair_lookup;

        for (size_t i = 0; i + 1 < seq.size(); ++i) {
            uint64_t key = pair_key(seq[i], seq[i + 1]);
            pair_counts[key]++;
            pair_lookup[key] = {seq[i], seq[i + 1]};
        }

        if (pair_counts.empty()) {
            std::cout << "[BPETokenizer] Tidak ada pair tersisa, berhenti di vocab_size="
                      << id_to_bytes_.size() << "\n";
            break;
        }

        // Cari pair dengan frekuensi tertinggi
        uint64_t best_key = 0;
        size_t best_count = 0;
        for (const auto& kv : pair_counts) {
            if (kv.second > best_count) {
                best_count = kv.second;
                best_key = kv.first;
            }
        }

        if (best_count < 2) {
            std::cout << "[BPETokenizer] Pair terbanyak cuma muncul 1x, berhenti di vocab_size="
                      << id_to_bytes_.size() << "\n";
            break;
        }

        uint16_t a = pair_lookup[best_key].first;
        uint16_t b = pair_lookup[best_key].second;

        // Buat token baru dari gabungan byte a + byte b
        uint16_t new_id = static_cast<uint16_t>(id_to_bytes_.size());
        std::vector<uint8_t> merged_bytes = id_to_bytes_[a];
        merged_bytes.insert(merged_bytes.end(), id_to_bytes_[b].begin(), id_to_bytes_[b].end());
        id_to_bytes_.push_back(merged_bytes);

        size_t rank = merges_.size();
        merges_.push_back({a, b});
        merge_rank_[best_key] = rank;

        // Terapkan merge ke seluruh sequence, satu pass
        std::vector<uint16_t> new_seq;
        new_seq.reserve(seq.size());
        for (size_t i = 0; i < seq.size(); ++i) {
            if (i + 1 < seq.size() && seq[i] == a && seq[i + 1] == b) {
                new_seq.push_back(new_id);
                ++i; // skip elemen berikutnya, sudah tergabung
            } else {
                new_seq.push_back(seq[i]);
            }
        }
        seq = std::move(new_seq);
    }

    std::cout << "[BPETokenizer] Training selesai, vocab_size=" << id_to_bytes_.size()
              << " (" << merges_.size() << " merge dipelajari)\n";
}

std::vector<uint16_t> BPETokenizer::encode(const std::string& text) const {
    // Langkah 1: byte mentah -> id base vocab (UNK kalau tidak dikenal)
    std::vector<uint16_t> seq;
    seq.reserve(text.size());
    for (unsigned char ch : text) {
        auto it = byte_to_id_.find(static_cast<uint8_t>(ch));
        seq.push_back(it != byte_to_id_.end() ? it->second : UNK_ID);
    }

    // Langkah 2: greedy merge rank-based — tiap iterasi, cari pair dengan rank
    // terkecil (dipelajari paling awal = prioritas tertinggi), merge semua
    // kemunculannya, ulangi sampai tidak ada pair yang punya rank lagi.
    bool merged_something = true;
    while (merged_something && seq.size() > 1) {
        merged_something = false;

        size_t best_rank = merges_.size(); // nilai "tak terhingga" relatif
        size_t best_pos = 0;

        for (size_t i = 0; i + 1 < seq.size(); ++i) {
            uint64_t key = pair_key(seq[i], seq[i + 1]);
            auto it = merge_rank_.find(key);
            if (it != merge_rank_.end() && it->second < best_rank) {
                best_rank = it->second;
                best_pos = i;
            }
        }

        if (best_rank == merges_.size()) break; // tidak ada pair yang bisa di-merge lagi

        uint16_t a = seq[best_pos];
        uint16_t b = seq[best_pos + 1];
        uint16_t merged_id = static_cast<uint16_t>(NUM_SPECIAL_TOKENS + 0); // placeholder, dihitung di bawah
        // id hasil merge = base_count + NUM_SPECIAL + rank (karena disimpan berurutan saat training)
        merged_id = static_cast<uint16_t>(id_to_bytes_.size() - merges_.size() + best_rank);

        std::vector<uint16_t> new_seq;
        new_seq.reserve(seq.size());
        for (size_t i = 0; i < seq.size(); ++i) {
            if (i == best_pos) {
                new_seq.push_back(merged_id);
                ++i; // skip elemen pasangannya
            } else {
                new_seq.push_back(seq[i]);
            }
        }
        seq = std::move(new_seq);
        merged_something = true;
    }

    return seq;
}

std::string BPETokenizer::decode(const std::vector<uint16_t>& ids) const {
    std::string result;
    for (uint16_t id : ids) {
        if (id >= NUM_SPECIAL_TOKENS && id < id_to_bytes_.size()) {
            const auto& bytes = id_to_bytes_[id];
            result.append(reinterpret_cast<const char*>(bytes.data()), bytes.size());
        }
        // id < NUM_SPECIAL_TOKENS atau di luar jangkauan: dilewati, tidak kontribusi output
    }
    return result;
}

bool BPETokenizer::save(const std::string& path) const {
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        std::cerr << "[BPETokenizer] Gagal buka file untuk save: " << path << "\n";
        return false;
    }

    // Base vocab: jumlah byte + byte mentahnya (urut id, mulai dari id NUM_SPECIAL_TOKENS)
    uint32_t num_base = static_cast<uint32_t>(id_to_bytes_.size() - NUM_SPECIAL_TOKENS - merges_.size());
    out.write(reinterpret_cast<const char*>(&num_base), sizeof(num_base));
    for (uint16_t id = NUM_SPECIAL_TOKENS; id < NUM_SPECIAL_TOKENS + num_base; ++id) {
        uint8_t b = id_to_bytes_[id][0];
        out.write(reinterpret_cast<const char*>(&b), sizeof(b));
    }

    // Merge list: jumlah + pasangan (a,b) berurutan sesuai rank
    uint32_t num_merges = static_cast<uint32_t>(merges_.size());
    out.write(reinterpret_cast<const char*>(&num_merges), sizeof(num_merges));
    for (const auto& m : merges_) {
        out.write(reinterpret_cast<const char*>(&m.first), sizeof(m.first));
        out.write(reinterpret_cast<const char*>(&m.second), sizeof(m.second));
    }

    return true;
}

bool BPETokenizer::load(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        std::cerr << "[BPETokenizer] Gagal buka file untuk load: " << path << "\n";
        return false;
    }

    id_to_bytes_.clear();
    byte_to_id_.clear();
    merges_.clear();
    merge_rank_.clear();

    for (uint16_t i = 0; i < NUM_SPECIAL_TOKENS; ++i) {
        id_to_bytes_.push_back({});
    }

    uint32_t num_base = 0;
    in.read(reinterpret_cast<char*>(&num_base), sizeof(num_base));
    for (uint32_t i = 0; i < num_base; ++i) {
        uint8_t b = 0;
        in.read(reinterpret_cast<char*>(&b), sizeof(b));
        uint16_t new_id = static_cast<uint16_t>(id_to_bytes_.size());
        byte_to_id_[b] = new_id;
        id_to_bytes_.push_back({b});
    }

    uint32_t num_merges = 0;
    in.read(reinterpret_cast<char*>(&num_merges), sizeof(num_merges));
    for (uint32_t rank = 0; rank < num_merges; ++rank) {
        uint16_t a = 0, b = 0;
        in.read(reinterpret_cast<char*>(&a), sizeof(a));
        in.read(reinterpret_cast<char*>(&b), sizeof(b));

        uint16_t new_id = static_cast<uint16_t>(id_to_bytes_.size());
        std::vector<uint8_t> merged_bytes = id_to_bytes_[a];
        merged_bytes.insert(merged_bytes.end(), id_to_bytes_[b].begin(), id_to_bytes_[b].end());
        id_to_bytes_.push_back(merged_bytes);

        merges_.push_back({a, b});
        merge_rank_[pair_key(a, b)] = rank;
    }

    return true;
}