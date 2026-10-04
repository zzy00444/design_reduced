#pragma once

#include <cstddef>
#include "ap_int.h"
#include <hls_stream.h>
#include "hlslib/xilinx/DataPack.h"

using Data_t = ap_int<8>;
using Acc_t = ap_int<32>;

// Keep the same 128-bit AXI interfaces as the active fp32 design.
constexpr unsigned kMemoryWidthBytesN = 16;
constexpr unsigned kMemoryWidthBytesM = 16;
constexpr unsigned kMemoryWidthBytesC = 16;

// Defect 1 fix: vectorize the K dimension in groups of four int8 values.
constexpr unsigned kKVector = 4;

constexpr unsigned kMemoryWidthN = kMemoryWidthBytesN / sizeof(Data_t); // 16/1=16
constexpr unsigned kMemoryWidthM = kMemoryWidthBytesM / sizeof(Data_t);
constexpr unsigned kMemoryWidthC = kMemoryWidthBytesC / sizeof(Acc_t);

constexpr unsigned kOuterTileSizeN = 128;
constexpr unsigned kOuterTileSizeM = 128;
constexpr unsigned kInnerTileSizeN = 16; // 4 to 8 n维度并行数，即PE数
constexpr unsigned kComputeTileSizeN = 1;
constexpr unsigned kComputeTileSizeM = 8; // m维度并行数，即每个PE内的SIMD宽度
constexpr unsigned kPipeDepth = 16;       // 缓冲区深度,8 to 4

constexpr unsigned kComputeTilesN = kInnerTileSizeN / kComputeTileSizeN;
// zzy 修改 2D
/////////////////////////////////////////////////////////////////////////////////
// --- Int8Gemm.h ---

// 定义 2D 阵列维度
constexpr unsigned kGridRows = 2;
constexpr unsigned kGridCols = 2;

// 每个 PE 现在处理的“本地” Tile 大小
// 原本是 128 (OuterN) / 16 (PEs) = 8
// 现在是 128 / 4 = 32
constexpr unsigned kPeTileN = kOuterTileSizeN / kGridRows;
constexpr unsigned kPeTileM = kOuterTileSizeM / kGridCols;

// 内部计算参数
constexpr unsigned kPeInnerTilesN = kPeTileN / kComputeTileSizeN; // 32 / 1 = 32
constexpr unsigned kPeInnerTilesM = kPeTileM / kComputeTileSizeM; // 32 / 8 = 4

// 验证 PE 总数
static_assert(kGridRows * kGridCols == 4, "Reduced design uses four PEs.");
static_assert(kOuterTileSizeN % kGridRows == 0, "N tile must split evenly across rows.");
static_assert(kOuterTileSizeM % kGridCols == 0, "M tile must split evenly across columns.");
/////////////////////////////////////////////////////////////////////////////////////
constexpr unsigned kInnerTilesN = kOuterTileSizeN / kInnerTileSizeN;
constexpr unsigned kInnerTilesM = kOuterTileSizeM / kComputeTileSizeM;

constexpr unsigned kOuterTileSizeNMemory = kOuterTileSizeN / kMemoryWidthN;
constexpr unsigned kOuterTileSizeMMemory = kOuterTileSizeM / kMemoryWidthM;
constexpr unsigned kOuterTileSizeMMemoryC = kOuterTileSizeM / kMemoryWidthC;

static_assert(kMemoryWidthBytesN % sizeof(Data_t) == 0, "Invalid AXI width for A.");
static_assert(kMemoryWidthBytesM % sizeof(Data_t) == 0, "Invalid AXI width for B.");
static_assert(kMemoryWidthBytesC % sizeof(Acc_t) == 0, "Invalid AXI width for C.");
static_assert(kInnerTileSizeN % kComputeTileSizeN == 0, "Invalid N compute tiling.");
static_assert(kOuterTileSizeN % kInnerTileSizeN == 0, "Invalid N tile hierarchy.");
static_assert(kOuterTileSizeM % kComputeTileSizeM == 0, "Invalid M tile hierarchy.");
static_assert(kOuterTileSizeN % kMemoryWidthN == 0, "A tile must align to A AXI width.");
static_assert(kOuterTileSizeM % kMemoryWidthM == 0, "B tile must align to B AXI width.");
static_assert(kOuterTileSizeM % kMemoryWidthC == 0, "C tile must align to C AXI width.");
static_assert(kMemoryWidthM % kComputeTileSizeM == 0,
              "B memory width must split evenly into compute vectors.");
static_assert(kComputeTilesN == 16, "Top.cpp manually instantiates 4 processing elements."); // PE同步更改
static_assert(kInnerTilesM >= kInnerTilesN,
              "Current double-buffering scheme expects inner M tiles >= inner N tiles.");

using MemoryPackN_t = ap_uint<kMemoryWidthN * 8>;
using MemoryPackM_t = ap_uint<kMemoryWidthM * 8>;
using MemoryPackC_t = hlslib::DataPack<Acc_t, kMemoryWidthC>;

// Compute streams use raw packed bit-vectors to avoid DataPack proxy accesses
// that Vitis HLS 2023.2 rejects during synthesizability checking.

// using ComputePackN_t = ap_uint<128>;// <--- 致命的资源浪费！？
using ComputePackN_t = ap_uint<kKVector * 8>; // 即 ap_uint<32>
using ComputePackM_t = ap_uint<kComputeTileSizeM * kKVector * 8>;

using AccPack_t = hlslib::DataPack<Acc_t, kComputeTileSizeM>;

template <typename T, size_t depth = 0>
using Stream = hls::stream<T>;

inline unsigned DivCeil(const unsigned value, const unsigned factor)
{
#pragma HLS INLINE
  return (value + factor - 1) / factor;
}

inline unsigned KGroups(const unsigned k)
{
#pragma HLS INLINE
  return DivCeil(k, kKVector);
}

inline unsigned SizeNMemory(const unsigned n)
{
#pragma HLS INLINE
  return DivCeil(n, kMemoryWidthN);
}

inline unsigned SizeMMemory(const unsigned m)
{
#pragma HLS INLINE
  return DivCeil(m, kMemoryWidthM);
}

inline unsigned SizeCMemory(const unsigned m)
{
#pragma HLS INLINE
  return DivCeil(m, kMemoryWidthC);
}

inline unsigned OuterTilesN(const unsigned n)
{
#pragma HLS INLINE
  return DivCeil(n, kOuterTileSizeN);
}

inline unsigned OuterTilesM(const unsigned m)
{
#pragma HLS INLINE
  return DivCeil(m, kOuterTileSizeM);
}

template <unsigned Idx>
inline Data_t GetMemoryNElement(const MemoryPackN_t &pack)
{
#pragma HLS INLINE
  static_assert(Idx < kMemoryWidthN, "MemoryPackN index out of range.");
  Data_t out;
  out.range() = pack.range((Idx + 1) * 8 - 1, Idx * 8);
  return out;
}

template <unsigned Idx>
inline Data_t GetMemoryMElement(const MemoryPackM_t &pack)
{
#pragma HLS INLINE
  static_assert(Idx < kMemoryWidthM, "MemoryPackM index out of range.");
  Data_t out;
  out.range() = pack.range((Idx + 1) * 8 - 1, Idx * 8);
  return out;
}

template <unsigned Idx>
inline Data_t GetComputeNElement(const ComputePackN_t &pack)
{
#pragma HLS INLINE
  static_assert(Idx < kKVector, "ComputePackN index out of range.");
  Data_t out;
  out.range() = pack.range((Idx + 1) * 8 - 1, Idx * 8);
  return out;
}

inline Data_t GetComputeNElement(const ComputePackN_t &pack,
                                 const unsigned idx)
{
#pragma HLS INLINE
  switch (idx)
  {
  case 0:
    return GetComputeNElement<0>(pack);
  case 1:
    return GetComputeNElement<1>(pack);
  case 2:
    return GetComputeNElement<2>(pack);
  default:
    return GetComputeNElement<3>(pack);
  }
}

template <unsigned Idx>
inline void SetComputeNElement(ComputePackN_t &pack, const Data_t value)
{
#pragma HLS INLINE
  static_assert(Idx < kKVector, "ComputePackN index out of range.");
  pack.range((Idx + 1) * 8 - 1, Idx * 8) = value.range();
}

inline void SetComputeNElement(ComputePackN_t &pack, const unsigned idx,
                               const Data_t value)
{
#pragma HLS INLINE
  switch (idx)
  {
  case 0:
    SetComputeNElement<0>(pack, value);
    break;
  case 1:
    SetComputeNElement<1>(pack, value);
    break;
  case 2:
    SetComputeNElement<2>(pack, value);
    break;
  default:
    SetComputeNElement<3>(pack, value);
    break;
  }
}

// template <unsigned Idx>
// inline Data_t GetComputeMElement(const ComputePackM_t &pack)
// {
// #pragma HLS INLINE
//   static_assert(Idx < kComputeTileSizeM * kKVector,
//                 "ComputePackM index out of range.");
//   Data_t out;
//   out.range() = pack.range((Idx + 1) * 8 - 1, Idx * 8);
//   return out;
// }

// inline Data_t GetComputeMElement(const ComputePackM_t &pack,
//                                  const unsigned idx)
// {
// #pragma HLS INLINE
//   switch (idx)
//   {
//   case 0:
//     return GetComputeMElement<0>(pack);
//   case 1:
//     return GetComputeMElement<1>(pack);
//   case 2:
//     return GetComputeMElement<2>(pack);
//   case 3:
//     return GetComputeMElement<3>(pack);
//   case 4:
//     return GetComputeMElement<4>(pack);
//   case 5:
//     return GetComputeMElement<5>(pack);
//   case 6:
//     return GetComputeMElement<6>(pack);
//   case 7:
//     return GetComputeMElement<7>(pack);
//   case 8:
//     return GetComputeMElement<8>(pack);
//   case 9:
//     return GetComputeMElement<9>(pack);
//   case 10:
//     return GetComputeMElement<10>(pack);
//   case 11:
//     return GetComputeMElement<11>(pack);
//   case 12:
//     return GetComputeMElement<12>(pack);
//   case 13:
//     return GetComputeMElement<13>(pack);
//   case 14:
//     return GetComputeMElement<14>(pack);
//   default:
//     return GetComputeMElement<15>(pack);
//   }
// }

// template <unsigned Idx>
// inline void SetComputeMElement(ComputePackM_t &pack, const Data_t value)
// {
// #pragma HLS INLINE
//   static_assert(Idx < kComputeTileSizeM * kKVector,
//                 "ComputePackM index out of range.");
//   pack.range((Idx + 1) * 8 - 1, Idx * 8) = value.range();
// }

// inline void SetComputeMElement(ComputePackM_t &pack, const unsigned idx,
//                                const Data_t value)
// {
// #pragma HLS INLINE
//   switch (idx)
//   {
//   case 0:
//     SetComputeMElement<0>(pack, value);
//     break;
//   case 1:
//     SetComputeMElement<1>(pack, value);
//     break;
//   case 2:
//     SetComputeMElement<2>(pack, value);
//     break;
//   case 3:
//     SetComputeMElement<3>(pack, value);
//     break;
//   case 4:
//     SetComputeMElement<4>(pack, value);
//     break;
//   case 5:
//     SetComputeMElement<5>(pack, value);
//     break;
//   case 6:
//     SetComputeMElement<6>(pack, value);
//     break;
//   case 7:
//     SetComputeMElement<7>(pack, value);
//     break;
//   case 8:
//     SetComputeMElement<8>(pack, value);
//     break;
//   case 9:
//     SetComputeMElement<9>(pack, value);
//     break;
//   case 10:
//     SetComputeMElement<10>(pack, value);
//     break;
//   case 11:
//     SetComputeMElement<11>(pack, value);
//     break;
//   case 12:
//     SetComputeMElement<12>(pack, value);
//     break;
//   case 13:
//     SetComputeMElement<13>(pack, value);
//     break;
//   case 14:
//     SetComputeMElement<14>(pack, value);
//     break;
//   default:
//     SetComputeMElement<15>(pack, value);
//     break;
//   }
// }

template <unsigned Idx>
inline Data_t GetComputeMElement(const ComputePackM_t &pack)
{
#pragma HLS INLINE
  static_assert(Idx < kComputeTileSizeM * kKVector, "ComputePackM index out of range.");
  Data_t out;
  out.range() = pack.range((Idx + 1) * 8 - 1, Idx * 8);
  return out;
}

// 暴力展开 0-31 (Vitis HLS 2023.2 强要求)
inline Data_t GetComputeMElement(const ComputePackM_t &pack, const unsigned idx)
{
#pragma HLS INLINE
  switch (idx)
  {
  case 0:
    return GetComputeMElement<0>(pack);
  case 1:
    return GetComputeMElement<1>(pack);
  case 2:
    return GetComputeMElement<2>(pack);
  case 3:
    return GetComputeMElement<3>(pack);
  case 4:
    return GetComputeMElement<4>(pack);
  case 5:
    return GetComputeMElement<5>(pack);
  case 6:
    return GetComputeMElement<6>(pack);
  case 7:
    return GetComputeMElement<7>(pack);
  case 8:
    return GetComputeMElement<8>(pack);
  case 9:
    return GetComputeMElement<9>(pack);
  case 10:
    return GetComputeMElement<10>(pack);
  case 11:
    return GetComputeMElement<11>(pack);
  case 12:
    return GetComputeMElement<12>(pack);
  case 13:
    return GetComputeMElement<13>(pack);
  case 14:
    return GetComputeMElement<14>(pack);
  case 15:
    return GetComputeMElement<15>(pack);
  case 16:
    return GetComputeMElement<16>(pack);
  case 17:
    return GetComputeMElement<17>(pack);
  case 18:
    return GetComputeMElement<18>(pack);
  case 19:
    return GetComputeMElement<19>(pack);
  case 20:
    return GetComputeMElement<20>(pack);
  case 21:
    return GetComputeMElement<21>(pack);
  case 22:
    return GetComputeMElement<22>(pack);
  case 23:
    return GetComputeMElement<23>(pack);
  case 24:
    return GetComputeMElement<24>(pack);
  case 25:
    return GetComputeMElement<25>(pack);
  case 26:
    return GetComputeMElement<26>(pack);
  case 27:
    return GetComputeMElement<27>(pack);
  case 28:
    return GetComputeMElement<28>(pack);
  case 29:
    return GetComputeMElement<29>(pack);
  case 30:
    return GetComputeMElement<30>(pack);
  default:
    return GetComputeMElement<31>(pack);
  }
}

template <unsigned Idx>
inline void SetComputeMElement(ComputePackM_t &pack, const Data_t value)
{
#pragma HLS INLINE
  static_assert(Idx < kComputeTileSizeM * kKVector, "ComputePackM index out of range.");
  pack.range((Idx + 1) * 8 - 1, Idx * 8) = value.range();
}

// 暴力展开 0-31
inline void SetComputeMElement(ComputePackM_t &pack, const unsigned idx, const Data_t value)
{
#pragma HLS INLINE
  switch (idx)
  {
  case 0:
    SetComputeMElement<0>(pack, value);
    break;
  case 1:
    SetComputeMElement<1>(pack, value);
    break;
  case 2:
    SetComputeMElement<2>(pack, value);
    break;
  case 3:
    SetComputeMElement<3>(pack, value);
    break;
  case 4:
    SetComputeMElement<4>(pack, value);
    break;
  case 5:
    SetComputeMElement<5>(pack, value);
    break;
  case 6:
    SetComputeMElement<6>(pack, value);
    break;
  case 7:
    SetComputeMElement<7>(pack, value);
    break;
  case 8:
    SetComputeMElement<8>(pack, value);
    break;
  case 9:
    SetComputeMElement<9>(pack, value);
    break;
  case 10:
    SetComputeMElement<10>(pack, value);
    break;
  case 11:
    SetComputeMElement<11>(pack, value);
    break;
  case 12:
    SetComputeMElement<12>(pack, value);
    break;
  case 13:
    SetComputeMElement<13>(pack, value);
    break;
  case 14:
    SetComputeMElement<14>(pack, value);
    break;
  case 15:
    SetComputeMElement<15>(pack, value);
    break;
  case 16:
    SetComputeMElement<16>(pack, value);
    break;
  case 17:
    SetComputeMElement<17>(pack, value);
    break;
  case 18:
    SetComputeMElement<18>(pack, value);
    break;
  case 19:
    SetComputeMElement<19>(pack, value);
    break;
  case 20:
    SetComputeMElement<20>(pack, value);
    break;
  case 21:
    SetComputeMElement<21>(pack, value);
    break;
  case 22:
    SetComputeMElement<22>(pack, value);
    break;
  case 23:
    SetComputeMElement<23>(pack, value);
    break;
  case 24:
    SetComputeMElement<24>(pack, value);
    break;
  case 25:
    SetComputeMElement<25>(pack, value);
    break;
  case 26:
    SetComputeMElement<26>(pack, value);
    break;
  case 27:
    SetComputeMElement<27>(pack, value);
    break;
  case 28:
    SetComputeMElement<28>(pack, value);
    break;
  case 29:
    SetComputeMElement<29>(pack, value);
    break;
  case 30:
    SetComputeMElement<30>(pack, value);
    break;
  default:
    SetComputeMElement<31>(pack, value);
    break;
  }
}

// inline Data_t GetBElement(const ComputePackM_t &pack, const unsigned m_lane,
//                           const unsigned k_lane)
// {
// #pragma HLS INLINE
//   switch (m_lane)
//   {
//   case 0:
//     switch (k_lane)
//     {
//     case 0:
//       return GetComputeMElement<0>(pack);
//     case 1:
//       return GetComputeMElement<1>(pack);
//     case 2:
//       return GetComputeMElement<2>(pack);
//     default:
//       return GetComputeMElement<3>(pack);
//     }
//   case 1:
//     switch (k_lane)
//     {
//     case 0:
//       return GetComputeMElement<4>(pack);
//     case 1:
//       return GetComputeMElement<5>(pack);
//     case 2:
//       return GetComputeMElement<6>(pack);
//     default:
//       return GetComputeMElement<7>(pack);
//     }
//   case 2:
//     switch (k_lane)
//     {
//     case 0:
//       return GetComputeMElement<8>(pack);
//     case 1:
//       return GetComputeMElement<9>(pack);
//     case 2:
//       return GetComputeMElement<10>(pack);
//     default:
//       return GetComputeMElement<11>(pack);
//     }
//   default:
//     switch (k_lane)
//     {
//     case 0:
//       return GetComputeMElement<12>(pack);
//     case 1:
//       return GetComputeMElement<13>(pack);
//     case 2:
//       return GetComputeMElement<14>(pack);
//     default:
//       return GetComputeMElement<15>(pack);
//     }
//   }
// }

inline Data_t GetBElement(const ComputePackM_t &pack, const unsigned m_lane, const unsigned k_lane)
{
#pragma HLS INLINE
  switch (m_lane)
  {
  case 0:
    switch (k_lane)
    {
    case 0:
      return GetComputeMElement<0>(pack);
    case 1:
      return GetComputeMElement<1>(pack);
    case 2:
      return GetComputeMElement<2>(pack);
    default:
      return GetComputeMElement<3>(pack);
    }
  case 1:
    switch (k_lane)
    {
    case 0:
      return GetComputeMElement<4>(pack);
    case 1:
      return GetComputeMElement<5>(pack);
    case 2:
      return GetComputeMElement<6>(pack);
    default:
      return GetComputeMElement<7>(pack);
    }
  case 2:
    switch (k_lane)
    {
    case 0:
      return GetComputeMElement<8>(pack);
    case 1:
      return GetComputeMElement<9>(pack);
    case 2:
      return GetComputeMElement<10>(pack);
    default:
      return GetComputeMElement<11>(pack);
    }
  case 3:
    switch (k_lane)
    {
    case 0:
      return GetComputeMElement<12>(pack);
    case 1:
      return GetComputeMElement<13>(pack);
    case 2:
      return GetComputeMElement<14>(pack);
    default:
      return GetComputeMElement<15>(pack);
    }
  case 4:
    switch (k_lane)
    {
    case 0:
      return GetComputeMElement<16>(pack);
    case 1:
      return GetComputeMElement<17>(pack);
    case 2:
      return GetComputeMElement<18>(pack);
    default:
      return GetComputeMElement<19>(pack);
    }
  case 5:
    switch (k_lane)
    {
    case 0:
      return GetComputeMElement<20>(pack);
    case 1:
      return GetComputeMElement<21>(pack);
    case 2:
      return GetComputeMElement<22>(pack);
    default:
      return GetComputeMElement<23>(pack);
    }
  case 6:
    switch (k_lane)
    {
    case 0:
      return GetComputeMElement<24>(pack);
    case 1:
      return GetComputeMElement<25>(pack);
    case 2:
      return GetComputeMElement<26>(pack);
    default:
      return GetComputeMElement<27>(pack);
    }
  default:
    switch (k_lane)
    {
    case 0:
      return GetComputeMElement<28>(pack);
    case 1:
      return GetComputeMElement<29>(pack);
    case 2:
      return GetComputeMElement<30>(pack);
    default:
      return GetComputeMElement<31>(pack);
    }
  }
}

// extern "C"
// {
//   void MatrixMultiplicationKernelInt8(MemoryPackN_t const a[],
//                                       MemoryPackM_t const b[],
//                                       MemoryPackC_t c[],
//                                       const unsigned size_n,
//                                       const unsigned size_k,
//                                       const unsigned size_m);
// }
extern "C"
{
  // C is 128 bits per word: 16 UINT8 values or 4 signed INT32 values.
  // output_int32 bypasses requantization and GELU; row strides are
  // ceil(size_m / 16) and ceil(size_m / 4) words, respectively.
  void MatrixMultiplicationKernelInt8(MemoryPackN_t const a[],
                                      MemoryPackM_t const b[],
                                      MemoryPackM_t c[],
                                      const unsigned size_n,
                                      const unsigned size_k,
                                      const unsigned size_m,
                                      const unsigned layer_idx,
                                      const bool enable_gelu,
                                      const bool output_int32);
}
