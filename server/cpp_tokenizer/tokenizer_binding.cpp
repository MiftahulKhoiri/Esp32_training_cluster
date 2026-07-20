// tokenizer_binding.cpp — pybind11 module, expose BPETokenizer ke Python
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "bpe_tokenizer.h"

namespace py = pybind11;

PYBIND11_MODULE(bpe_tokenizer_py, m) {
    m.doc() = "BPETokenizer C++ (byte-level BPE) untuk proyek esp32_federated_learning";

    py::class_<BPETokenizer>(m, "BPETokenizer")
        .def(py::init<>())
        .def("train", &BPETokenizer::train,
             py::arg("corpus"), py::arg("target_vocab_size"),
             "Latih tokenizer dari corpus mentah sampai vocab_size mencapai target")
        .def("encode", &BPETokenizer::encode,
             py::arg("text"),
             "Encode teks jadi list token id (greedy merge rank-based)")
        .def("decode", [](const BPETokenizer& self, const std::vector<uint16_t>& ids) {
                std::string raw = self.decode(ids);
                // py::bytes, BUKAN otomatis str/UTF-8 — byte-level BPE bisa hasilkan
                // urutan byte yang tidak valid UTF-8 berdiri sendiri (pelajaran yang
                // sama seperti waktu ml_manual_cpp kena UnicodeDecodeError)
                return py::bytes(raw);
             }, py::arg("ids"))
        .def("vocab_size", &BPETokenizer::vocab_size)
        .def("save", &BPETokenizer::save, py::arg("path"))
        .def("load", &BPETokenizer::load, py::arg("path"))
        .def_readonly_static("PAD_ID", &BPETokenizer::PAD_ID)
        .def_readonly_static("BOS_ID", &BPETokenizer::BOS_ID)
        .def_readonly_static("EOS_ID", &BPETokenizer::EOS_ID)
        .def_readonly_static("UNK_ID", &BPETokenizer::UNK_ID);
}