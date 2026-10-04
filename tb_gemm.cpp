#include "include/Int8Gemm.h"
#include "include/hw_mlp_params.h"

#include <cstdint>
#include <iostream>
#include <vector>

static int InputA(unsigned k, unsigned n) {
  return static_cast<int>((k * 17 + n * 29) % 201) - 100;
}

static int InputB(unsigned k, unsigned m) {
  return static_cast<int>((k * 31 + m * 13) % 201) - 100;
}

static void SetByte(ap_uint<128> &word, unsigned lane, int value) {
  word.range(lane * 8 + 7, lane * 8) = static_cast<uint8_t>(value);
}

static uint8_t GetByte(const ap_uint<128> &word, unsigned lane) {
  return static_cast<uint8_t>(word.range(lane * 8 + 7, lane * 8).to_uint());
}

static int32_t GetInt32(const ap_uint<128> &word, unsigned lane) {
  ap_int<32> value;
  value.range() = word.range(lane * 32 + 31, lane * 32);
  return value.to_int();
}

static int32_t GoldenAcc(unsigned n, unsigned m, unsigned k_size) {
  int64_t acc = 0;
  for (unsigned k = 0; k < k_size; ++k) {
    acc += static_cast<int64_t>(InputA(k, n)) * InputB(k, m);
  }
  return static_cast<int32_t>(acc);
}

static uint8_t Golden(unsigned n, unsigned m, unsigned k_size,
                      unsigned layer, bool gelu) {
  const int64_t acc = GoldenAcc(n, m, k_size);
  const int64_t product = acc * MLP_REQUANT_M[layer];
  const int64_t scaled = product >= 0
                             ? product >> 26
                             : -(((-product) + ((int64_t{1} << 26) - 1)) >> 26);
  int64_t value = scaled + MLP_MID_ZP[layer];
  if (value < 0) value = 0;
  if (value > 255) value = 255;
  const uint8_t quantized = static_cast<uint8_t>(value);
  return gelu ? GELU_LUT[layer][quantized] : quantized;
}

static bool RunCase(unsigned n_size, unsigned m_size, unsigned k_size,
                    unsigned layer, bool gelu, bool output_int32) {
  const unsigned n_words = (n_size + 15) / 16;
  const unsigned m_words = (m_size + 15) / 16;
  const unsigned c_words = output_int32 ? (m_size + 3) / 4 : m_words;
  // HLS cosim copies the full m_axi depth from each pointer.
  constexpr unsigned kInputDepth = 4096;
  constexpr unsigned kOutputDepth = 8192;
  std::vector<MemoryPackN_t> a(kInputDepth, 0);
  std::vector<MemoryPackM_t> b(kInputDepth, 0);
  std::vector<MemoryPackM_t> c(kOutputDepth, 0);

  for (unsigned k = 0; k < k_size; ++k) {
    for (unsigned n = 0; n < n_size; ++n) {
      SetByte(a[k * n_words + n / 16], n % 16, InputA(k, n));
    }
    for (unsigned m = 0; m < m_size; ++m) {
      SetByte(b[k * m_words + m / 16], m % 16, InputB(k, m));
    }
  }

  MatrixMultiplicationKernelInt8(a.data(), b.data(), c.data(), n_size, k_size,
                                 m_size, layer, gelu, output_int32);

  unsigned errors = 0;
  for (unsigned n = 0; n < n_size; ++n) {
    for (unsigned m = 0; m < m_size; ++m) {
      const int64_t actual = output_int32
                                 ? GetInt32(c[n * c_words + m / 4], m % 4)
                                 : GetByte(c[n * c_words + m / 16], m % 16);
      const int64_t expected = output_int32
                                   ? GoldenAcc(n, m, k_size)
                                   : Golden(n, m, k_size, layer, gelu);
      if (actual != expected) {
        if (errors < 8) {
          std::cerr << "Mismatch N=" << n << " M=" << m
                    << " got=" << actual
                    << " expected=" << expected << '\n';
        }
        ++errors;
      }
    }
  }
  std::cout << "N=" << n_size << " M=" << m_size << " K=" << k_size
            << " layer=" << layer << " gelu=" << gelu
            << " int32=" << output_int32
            << " errors=" << errors << '\n';
  return errors == 0;
}

int main() {
  bool ok = true;
  ok &= RunCase(16, 16, 7, 0, false, false);
  ok &= RunCase(16, 16, 7, 0, false, true);
  ok &= RunCase(17, 19, 5, 1, true, false);
  ok &= RunCase(17, 19, 5, 1, true, true);
  ok &= RunCase(129, 129, 5, 0, false, false);
  ok &= RunCase(129, 129, 7, 1, false, true);
  ok &= RunCase(129, 129, 7, 1, true, false);
  return ok ? 0 : 1;
}
