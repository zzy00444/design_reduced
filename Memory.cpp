#include "include/Memory.h"
#include "include/hw_mlp_params.h"

// -------------------------------------------------------------------------
// 閲嶉噺鍖栦笌GELU婵?娲荤殑浣嶅杞崲妯″潡 (INT8 鍘嬬缉)
// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// 閲嶉噺鍖栦笌GELU婵?娲荤殑浣嶅杞崲妯″潡 (璧勬簮鏋佽嚧浼樺寲鐗?)
// -------------------------------------------------------------------------
void ConvertWidthC_Int8(Stream<AccPack_t> &narrow, Stream<MemoryPackM_t> &wide,
                        const unsigned size_n, const unsigned size_k,
                        const unsigned size_m, const unsigned layer_idx,
                        const bool enable_gelu)
{
  int32_t m_int = MLP_REQUANT_M[layer_idx];
  int32_t zp = MLP_MID_ZP[layer_idx];

  // =========================================================================
  // 鈿? 浼樺寲 1锛氬鍓湰鍒嗙瀛楀吀 (Multi-Copy LUT) 鈿?
  // 鍒涘缓 8 涓嫭绔嬬殑 256 瀛楄妭寰瀷 RAM銆?
  // dim=1 灞曞紑鍚庯紝姣忎釜閫氶亾鎷ユ湁鑷繁鐙珛鐨勮绔彛锛屽交搴曟秷鐏? 256-to-1 MUX 鏍戯紒
  // =========================================================================
  uint8_t local_lut[kComputeTileSizeM][256];
#pragma HLS ARRAY_PARTITION variable = local_lut complete dim = 1
#pragma HLS BIND_STORAGE variable = local_lut type = ram_1p impl = lutram

  // 鍒濆鍖栬繖 8 涓瓧鍏搞?傝繖 256 涓懆鏈熷湪鏁翠釜澶х煩闃佃绠楀墠鍙墽琛屼竴娆★紝鑰楁椂鍗犳瘮瓒嬭繎浜? 0%
  for (int i = 0; i < 256; ++i)
  {
#pragma HLS PIPELINE II = 1
    uint8_t val = GELU_LUT[layer_idx][i];
    for (int j = 0; j < kComputeTileSizeM; ++j)
    {
      local_lut[j][i] = val;
    }
  }

ConvertWidthC_OuterTile_N:
  for (unsigned n0 = 0; n0 < OuterTilesN(size_n); ++n0)
  {
  ConvertWidthC_OuterTile_M:
    for (unsigned m0 = 0; m0 < OuterTilesM(size_m); ++m0)
    {
    ConvertWidthC_N1:
      for (unsigned n1 = 0; n1 < kOuterTileSizeN; ++n1)
      {

        MemoryPackM_t w;

      ConvertWidthC_M1:
        for (unsigned m1c = 0; m1c < kInnerTilesM; ++m1c)
        {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_FLATTEN

          AccPack_t acc = narrow.read();

        ConvertWidthC_Unroll:
          for (int i = 0; i < kComputeTileSizeM; ++i)
          {
#pragma HLS UNROLL
            Acc_t raw_acc = acc[i];

            ap_int<27> hw_val = static_cast<ap_int<27>>(raw_acc);
            ap_int<18> hw_m_int = m_int;

            // =========================================================================
            // 鈿? 浼樺寲 2锛氬己鍒? DSP 缁戝畾 鈿?
            // 鍛婅瘔缁煎悎鍣細鍒姽璞紝杩欏繀椤绘槸涓?涓師鐢熺殑 DSP48E2 涔樻硶
            // =========================================================================
            ap_int<45> mul = hw_val * hw_m_int;
#pragma HLS BIND_OP variable = mul op = mul impl = dsp

            // =========================================================================
            // 鈿? 浼樺寲 3锛氬父閲忔姌鍙? (娑堥櫎妗跺舰绉讳綅鍣?) 鈿?
            // 鍥犱负浣犵殑鍙傛暟涓墍鏈夌殑 shift 閮芥槸 26锛岀洿鎺ュ啓姝伙紒杩欎細杞寲涓? 0 LUT 鐨勭函鐗╃悊瀵肩嚎杩炵嚎
            // =========================================================================
            int32_t scaled = static_cast<int32_t>(mul >> 26);
            int32_t with_zp = scaled + zp;

            int32_t clamped = with_zp;
            if (clamped < 0)
              clamped = 0;
            if (clamped > 255)
              clamped = 255;

            // =========================================================================
            // 鈿? 浼樺寲 4锛氶?氶亾鐙珛鏌ヨ〃 鈿?
            // i 鏄? UNROLL 鐨勯?氶亾鍙凤紝姣忎釜閫氶亾鍘昏嚜宸变笓灞炵殑 RAM (local_lut[i]) 閲岄潰璇绘暟鎹?
            // =========================================================================
            uint8_t out_val = enable_gelu ? local_lut[i][clamped] : (uint8_t)clamped;

            // 鎷艰鍏? 128-bit 瀹藉瓧
            if ((m1c % 2) == 0)
            {
              w.range(i * 8 + 7, i * 8) = out_val;
            }
            else
            {
              w.range((i + 8) * 8 + 7, (i + 8) * 8) = out_val;
            }
          } // End Unroll

          // 鍑戦綈涓ゆ璇诲彇鍚庡彂灏?
          if ((m1c % 2) == 1)
          {
            wide.write(w);
          }
        } // End M1
      } // End N1
    } // End Outer_M
  } // End Outer_N
}

// -------------------------------------------------------------------------
// 閫傞厤 INT8 鍘嬬缉灏哄鐨勫啓鍏ユā鍧?
// -------------------------------------------------------------------------
void WriteC_Int8(Stream<MemoryPackM_t> &pipe, MemoryPackM_t memory[],
                 const unsigned size_n, const unsigned size_k,
                 const unsigned size_m)
{
WriteC_OuterTile_N:
  for (unsigned n0 = 0; n0 < OuterTilesN(size_n); ++n0)
  {
  WriteC_OuterTile_M:
    for (unsigned m0 = 0; m0 < OuterTilesM(size_m); ++m0)
    {
    WriteC_N1:
      for (unsigned n1 = 0; n1 < kOuterTileSizeN; ++n1)
      {

        const unsigned row = n0 * kOuterTileSizeN + n1;
        const bool valid_row = row < size_n;
        const unsigned base_addr = row * SizeMMemory(size_m);

      WriteC_M1:
        for (unsigned m1m = 0; m1m < kOuterTileSizeMMemory; ++m1m)
        {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_FLATTEN
          const auto val = pipe.read();
          const unsigned packed_m = m0 * kOuterTileSizeMMemory + m1m;

          if (valid_row && packed_m < SizeMMemory(size_m))
          {
            memory[base_addr + packed_m] = val;
          }
        } // 缁撴潫 M1
      } // 缁撴潫 N1
    } // 缁撴潫 Outer_M
  } // 缁撴潫 Outer_N
} // 缁撴潫 WriteC_Int8 鍑芥暟

namespace
{

  static_assert(kMemoryWidthN == 16, "ConvertWidthATransposed assumes 16 A lanes.");
  static_assert(kMemoryWidthM / kComputeTileSizeM == 2,
                "ConvertWidthB assumes 2 output segments."); // 4->2 // kComputeTileSizeM 锟????4鏀逛负8

  unsigned IndexATransposed(const unsigned k, const unsigned packed_n,
                            const unsigned size_n)
  {
#pragma HLS INLINE
    return k * SizeNMemory(size_n) + packed_n;
  }

  unsigned IndexB(const unsigned k, const unsigned packed_m,
                  const unsigned size_m)
  {
#pragma HLS INLINE
    return k * SizeMMemory(size_m) + packed_m;
  }

  unsigned IndexC(const unsigned row, const unsigned packed_m,
                  const unsigned size_m)
  {
#pragma HLS INLINE
    return row * SizeCMemory(size_m) + packed_m;
  }

  unsigned MinUnsigned(const unsigned a, const unsigned b)
  {
#pragma HLS INLINE
    return (a < b) ? a : b;
  }

  unsigned ValidTileSpan(const unsigned base, const unsigned total,
                         const unsigned tile)
  {
#pragma HLS INLINE
    return (base < total) ? MinUnsigned(total - base, tile) : 0;
  }

  Data_t GetMemoryNElementRuntime(const MemoryPackN_t &pack,
                                  const unsigned lane)
  {
#pragma HLS INLINE
    switch (lane)
    {
    case 0:
      return GetMemoryNElement<0>(pack);
    case 1:
      return GetMemoryNElement<1>(pack);
    case 2:
      return GetMemoryNElement<2>(pack);
    case 3:
      return GetMemoryNElement<3>(pack);
    case 4:
      return GetMemoryNElement<4>(pack);
    case 5:
      return GetMemoryNElement<5>(pack);
    case 6:
      return GetMemoryNElement<6>(pack);
    case 7:
      return GetMemoryNElement<7>(pack);
    case 8:
      return GetMemoryNElement<8>(pack);
    case 9:
      return GetMemoryNElement<9>(pack);
    case 10:
      return GetMemoryNElement<10>(pack);
    case 11:
      return GetMemoryNElement<11>(pack);
    case 12:
      return GetMemoryNElement<12>(pack);
    case 13:
      return GetMemoryNElement<13>(pack);
    case 14:
      return GetMemoryNElement<14>(pack);
    default:
      return GetMemoryNElement<15>(pack);
    }
  }

  template <int Lane>
  ComputePackN_t MakeComputePackN(MemoryPackN_t const &p0,
                                  MemoryPackN_t const &p1,
                                  MemoryPackN_t const &p2,
                                  MemoryPackN_t const &p3)
  {
#pragma HLS INLINE
    ComputePackN_t out = 0;
    SetComputeNElement<0>(out, GetMemoryNElement<Lane>(p0));
    SetComputeNElement<1>(out, GetMemoryNElement<Lane>(p1));
    SetComputeNElement<2>(out, GetMemoryNElement<Lane>(p2));
    SetComputeNElement<3>(out, GetMemoryNElement<Lane>(p3));
    return out;
  }

  ComputePackN_t MakeComputePackNRuntime(MemoryPackN_t const &p0,
                                         MemoryPackN_t const &p1,
                                         MemoryPackN_t const &p2,
                                         MemoryPackN_t const &p3,
                                         const unsigned lane)
  {
#pragma HLS INLINE
    ComputePackN_t out = 0;
    SetComputeNElement(out, 0, GetMemoryNElementRuntime(p0, lane));
    SetComputeNElement(out, 1, GetMemoryNElementRuntime(p1, lane));
    SetComputeNElement(out, 2, GetMemoryNElementRuntime(p2, lane));
    SetComputeNElement(out, 3, GetMemoryNElementRuntime(p3, lane));
    return out;
  }

  //   template <int Segment>
  //   ComputePackM_t MakeComputePackM(MemoryPackM_t const &p0,
  //                                   MemoryPackM_t const &p1,
  //                                   MemoryPackM_t const &p2,
  //                                   MemoryPackM_t const &p3)
  //   {
  // #pragma HLS INLINE
  //     constexpr int base = Segment * kComputeTileSizeM;
  //     ComputePackM_t out = 0;
  //     SetComputeMElement<0>(out, GetMemoryMElement<base + 0>(p0));
  //     SetComputeMElement<1>(out, GetMemoryMElement<base + 0>(p1));
  //     SetComputeMElement<2>(out, GetMemoryMElement<base + 0>(p2));
  //     SetComputeMElement<3>(out, GetMemoryMElement<base + 0>(p3));
  //     SetComputeMElement<4>(out, GetMemoryMElement<base + 1>(p0));
  //     SetComputeMElement<5>(out, GetMemoryMElement<base + 1>(p1));
  //     SetComputeMElement<6>(out, GetMemoryMElement<base + 1>(p2));
  //     SetComputeMElement<7>(out, GetMemoryMElement<base + 1>(p3));
  //     SetComputeMElement<8>(out, GetMemoryMElement<base + 2>(p0));
  //     SetComputeMElement<9>(out, GetMemoryMElement<base + 2>(p1));
  //     SetComputeMElement<10>(out, GetMemoryMElement<base + 2>(p2));
  //     SetComputeMElement<11>(out, GetMemoryMElement<base + 2>(p3));
  //     SetComputeMElement<12>(out, GetMemoryMElement<base + 3>(p0));
  //     SetComputeMElement<13>(out, GetMemoryMElement<base + 3>(p1));
  //     SetComputeMElement<14>(out, GetMemoryMElement<base + 3>(p2));
  //     SetComputeMElement<15>(out, GetMemoryMElement<base + 3>(p3));
  //     return out;
  //   }

  template <int Segment>
  ComputePackM_t MakeComputePackM(MemoryPackM_t const &p0,
                                  MemoryPackM_t const &p1,
                                  MemoryPackM_t const &p2,
                                  MemoryPackM_t const &p3)
  {
#pragma HLS INLINE
    constexpr int base = Segment * kComputeTileSizeM;
    ComputePackM_t out = 0;
    SetComputeMElement<0>(out, GetMemoryMElement<base + 0>(p0));
    SetComputeMElement<1>(out, GetMemoryMElement<base + 0>(p1));
    SetComputeMElement<2>(out, GetMemoryMElement<base + 0>(p2));
    SetComputeMElement<3>(out, GetMemoryMElement<base + 0>(p3));
    SetComputeMElement<4>(out, GetMemoryMElement<base + 1>(p0));
    SetComputeMElement<5>(out, GetMemoryMElement<base + 1>(p1));
    SetComputeMElement<6>(out, GetMemoryMElement<base + 1>(p2));
    SetComputeMElement<7>(out, GetMemoryMElement<base + 1>(p3));
    SetComputeMElement<8>(out, GetMemoryMElement<base + 2>(p0));
    SetComputeMElement<9>(out, GetMemoryMElement<base + 2>(p1));
    SetComputeMElement<10>(out, GetMemoryMElement<base + 2>(p2));
    SetComputeMElement<11>(out, GetMemoryMElement<base + 2>(p3));
    SetComputeMElement<12>(out, GetMemoryMElement<base + 3>(p0));
    SetComputeMElement<13>(out, GetMemoryMElement<base + 3>(p1));
    SetComputeMElement<14>(out, GetMemoryMElement<base + 3>(p2));
    SetComputeMElement<15>(out, GetMemoryMElement<base + 3>(p3));
    SetComputeMElement<16>(out, GetMemoryMElement<base + 4>(p0));
    SetComputeMElement<17>(out, GetMemoryMElement<base + 4>(p1));
    SetComputeMElement<18>(out, GetMemoryMElement<base + 4>(p2));
    SetComputeMElement<19>(out, GetMemoryMElement<base + 4>(p3));
    SetComputeMElement<20>(out, GetMemoryMElement<base + 5>(p0));
    SetComputeMElement<21>(out, GetMemoryMElement<base + 5>(p1));
    SetComputeMElement<22>(out, GetMemoryMElement<base + 5>(p2));
    SetComputeMElement<23>(out, GetMemoryMElement<base + 5>(p3));
    SetComputeMElement<24>(out, GetMemoryMElement<base + 6>(p0));
    SetComputeMElement<25>(out, GetMemoryMElement<base + 6>(p1));
    SetComputeMElement<26>(out, GetMemoryMElement<base + 6>(p2));
    SetComputeMElement<27>(out, GetMemoryMElement<base + 6>(p3));
    SetComputeMElement<28>(out, GetMemoryMElement<base + 7>(p0));
    SetComputeMElement<29>(out, GetMemoryMElement<base + 7>(p1));
    SetComputeMElement<30>(out, GetMemoryMElement<base + 7>(p2));
    SetComputeMElement<31>(out, GetMemoryMElement<base + 7>(p3));
    return out;
  }

  // =========================================================================
  // ReadA Helpers for Ping-Pong DATAFLOW
  // =========================================================================

  //   void ReadA_Load_Block(MemoryPackN_t const memory[],
  //                         MemoryPackN_t localA[kKVector][kOuterTileSizeNMemory],
  //                         const unsigned size_n, const unsigned size_k,
  //                         const unsigned packed_n_base, const unsigned k_base)
  //   {
  //     for (unsigned kk = 0; kk < kKVector; ++kk)
  //     {
  //       const unsigned k = k_base + kk;
  //       const bool valid_k = k < size_k;
  //       const unsigned valid_packed_n = ValidTileSpan(packed_n_base, SizeNMemory(size_n), kOuterTileSizeNMemory);
  //       const unsigned load_count = valid_k ? valid_packed_n : 0;

  //       if (valid_k)
  //       {
  //       ReadA_BurstLoad:
  //         for (unsigned n1m = 0; n1m < load_count; ++n1m)
  //         {
  // #pragma HLS PIPELINE II = 1
  //           localA[kk][n1m] = memory[IndexATransposed(k, packed_n_base + n1m, size_n)];
  //         }
  //       }

  //     ReadA_ZeroFill:
  //       for (unsigned n1m = load_count; n1m < kOuterTileSizeNMemory; ++n1m)
  //       {
  // #pragma HLS PIPELINE II = 1
  //         localA[kk][n1m] = 0;
  //       }
  //     }
  //   }
  // zzy 淇敼
  void ReadA_Load_Block(MemoryPackN_t const memory[],
                        MemoryPackN_t localA[kKVector][kOuterTileSizeNMemory],
                        const unsigned size_n, const unsigned size_k,
                        const unsigned packed_n_base, const unsigned k_base)
  {
    const unsigned size_n_mem = SizeNMemory(size_n); // 鎻愬墠鎻愬彇

    for (unsigned kk = 0; kk < kKVector; ++kk)
    {
      const unsigned k = k_base + kk;
      const bool valid_k = k < size_k;
      const unsigned valid_packed_n = ValidTileSpan(packed_n_base, size_n_mem, kOuterTileSizeNMemory);
      const unsigned load_count = valid_k ? valid_packed_n : 0;

      if (valid_k)
      {
        // 鎵嬪姩灏嗕箻娉曞拰鍩哄湴锟???鍔犳硶鎻愬嚭娴佹按绾匡紒
        const unsigned base_addr = k * size_n_mem + packed_n_base;

      ReadA_BurstLoad:
        for (unsigned n1m = 0; n1m < load_count; ++n1m)
        {
#pragma HLS PIPELINE II = 1
          // 娴佹按绾垮唴鍙墿涓嬩竴涓渶锟???鍗曠殑 + n1m 绱姞锟???
          localA[kk][n1m] = memory[base_addr + n1m];
        }
      }

    ReadA_ZeroFill:
      for (unsigned n1m = load_count; n1m < kOuterTileSizeNMemory; ++n1m)
      {
#pragma HLS PIPELINE II = 1
        localA[kk][n1m] = 0;
      }
    }
  }

  //   void ReadA_Feed_Block(MemoryPackN_t localA[kKVector][kOuterTileSizeNMemory],
  //                         Stream<ComputePackN_t> &pipe)
  //   {
  //   ReadA_StreamFeed_N1:
  //     for (unsigned n1m = 0; n1m < kOuterTileSizeNMemory; ++n1m)
  //     {
  //       const auto p0 = localA[0][n1m];
  //       const auto p1 = localA[1][n1m];
  //       const auto p2 = localA[2][n1m];
  //       const auto p3 = localA[3][n1m];

  //     ReadA_StreamFeed_Lane:
  //       for (unsigned lane = 0; lane < kMemoryWidthN; ++lane)
  //       {
  // #pragma HLS PIPELINE II = 1
  //         pipe.write(MakeComputePackNRuntime(p0, p1, p2, p3, lane));
  //       }
  //     }
  //   }
  // zzy 淇敼
  void ReadA_Feed_Block(MemoryPackN_t localA[kKVector][kOuterTileSizeNMemory],
                        Stream<ComputePackN_t> &pipe)
  {
  ReadA_StreamFeed_N1:
    for (unsigned n1m = 0; n1m < kOuterTileSizeNMemory; ++n1m)
    {
      // 锟??? 128-bit 瀹芥暟鎹浇鍏ュ眬閮ㄥ瘎瀛樺櫒
      MemoryPackN_t p0 = localA[0][n1m];
      MemoryPackN_t p1 = localA[1][n1m];
      MemoryPackN_t p2 = localA[2][n1m];
      MemoryPackN_t p3 = localA[3][n1m];

    ReadA_StreamFeed_Lane:
      for (unsigned lane = 0; lane < kMemoryWidthN; ++lane)
      {
#pragma HLS PIPELINE II = 1
        ComputePackN_t out = 0;

        // 姘歌繙鍙彇锟???浣庣殑 8 bits (鍗冲綋锟??? lane 鐨勬暟锟???)
        out.range(7, 0) = p0.range(7, 0);
        out.range(15, 8) = p1.range(7, 0);
        out.range(23, 16) = p2.range(7, 0);
        out.range(31, 24) = p3.range(7, 0);

        pipe.write(out);

        // 鏍稿績榄旀硶锛氭墍鏈夊瘎瀛樺櫒鍙崇Щ 8 浣嶏紝灏嗕笅锟???涓瓧鑺傛帹鍒版渶浣庝綅
        // 杩欏湪 FPGA 涓細缁煎悎涓烘瀬楂橀鐨勭Щ浣嶈繛绾匡紝涓嶄粎锟??? LUT锛屾椂搴忚繕鏋佷匠
        p0 >>= 8;
        p1 >>= 8;
        p2 >>= 8;
        p3 >>= 8;
      }
    }
  }

  // =========================================================================
  // ReadB Helpers for Ping-Pong DATAFLOW
  // =========================================================================

  //   void ReadB_Load_Block(MemoryPackM_t const memory[],
  //                         MemoryPackM_t localB[kKVector][kOuterTileSizeMMemory],
  //                         const unsigned size_m, const unsigned size_k,
  //                         const unsigned packed_m_base, const unsigned k_base)
  //   {
  //     for (unsigned kk = 0; kk < kKVector; ++kk)
  //     {
  //       const unsigned k = k_base + kk;
  //       const bool valid_k = k < size_k;
  //       const unsigned valid_packed_m = ValidTileSpan(packed_m_base, SizeMMemory(size_m), kOuterTileSizeMMemory);
  //       const unsigned load_count = valid_k ? valid_packed_m : 0;

  //       if (valid_k)
  //       {
  //       ReadB_BurstLoad:
  //         for (unsigned m1m = 0; m1m < load_count; ++m1m)
  //         {
  // #pragma HLS PIPELINE II = 1
  //           localB[kk][m1m] = memory[IndexB(k, packed_m_base + m1m, size_m)];
  //         }
  //       }

  //     ReadB_ZeroFill:
  //       for (unsigned m1m = load_count; m1m < kOuterTileSizeMMemory; ++m1m)
  //       {
  // #pragma HLS PIPELINE II = 1
  //         localB[kk][m1m] = 0;
  //       }
  //     }
  //   }
  // zzy 淇敼
  void ReadB_Load_Block(MemoryPackM_t const memory[],
                        MemoryPackM_t localB[kKVector][kOuterTileSizeMMemory],
                        const unsigned size_m, const unsigned size_k,
                        const unsigned packed_m_base, const unsigned k_base)
  {
    const unsigned size_m_mem = SizeMMemory(size_m); // 鎻愬墠鎻愬彇

    for (unsigned kk = 0; kk < kKVector; ++kk)
    {
      const unsigned k = k_base + kk;
      const bool valid_k = k < size_k;
      const unsigned valid_packed_m = ValidTileSpan(packed_m_base, size_m_mem, kOuterTileSizeMMemory);
      const unsigned load_count = valid_k ? valid_packed_m : 0;

      if (valid_k)
      {
        // 鏍稿績锛氭妸涔樻硶鎻愬嚭娴佹按绾垮
        const unsigned base_addr = k * size_m_mem + packed_m_base;

      ReadB_BurstLoad:
        for (unsigned m1m = 0; m1m < load_count; ++m1m)
        {
#pragma HLS PIPELINE II = 1
          // 鍐呭眰鍙墿鍔犳硶
          localB[kk][m1m] = memory[base_addr + m1m];
        }
      }

    ReadB_ZeroFill:
      for (unsigned m1m = load_count; m1m < kOuterTileSizeMMemory; ++m1m)
      {
#pragma HLS PIPELINE II = 1
        localB[kk][m1m] = 0;
      }
    }
  }

  void ReadB_Feed_Block(MemoryPackM_t localB[kKVector][kOuterTileSizeMMemory],
                        Stream<MemoryPackM_t> &pipe)
  {
  ReadB_StreamFeed_M1:
    for (unsigned m1m = 0; m1m < kOuterTileSizeMMemory; ++m1m)
    {
    ReadB_StreamFeed_KVector:
      for (unsigned kk = 0; kk < kKVector; ++kk)
      {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_FLATTEN
        pipe.write(localB[kk][m1m]);
      }
    }
  }

} // namespace

// =========================================================================
// Main Functions
// =========================================================================

void ReadATransposed(MemoryPackN_t const memory[], Stream<ComputePackN_t> &pipe,
                     const unsigned size_n, const unsigned size_k,
                     const unsigned size_m)
{
ReadA_OuterTile_N:
  for (unsigned n0 = 0; n0 < OuterTilesN(size_n); ++n0)
  {
    const unsigned packed_n_base = n0 * kOuterTileSizeNMemory;

  ReadA_OuterTile_M:
    for (unsigned m0 = 0; m0 < OuterTilesM(size_m); ++m0)
    {
    ReadA_KGroup:
      for (unsigned k0 = 0; k0 < KGroups(size_k); ++k0)
      {
// Enforce Ping-Pong Buffer (PIPO) creation and concurrent execution
#pragma HLS DATAFLOW

        MemoryPackN_t localA[kKVector][kOuterTileSizeNMemory];
#pragma HLS ARRAY_PARTITION variable = localA complete dim = 1
// zzy new寮哄埗浣跨敤鍒嗗竷锟????? RAM
#pragma HLS BIND_STORAGE variable = localA type = ram_2p impl = bram

        const unsigned k_base = k0 * kKVector;

        ReadA_Load_Block(memory, localA, size_n, size_k, packed_n_base, k_base);
        ReadA_Feed_Block(localA, pipe);
      }
    }
  }
}

void ReadB(MemoryPackM_t const memory[], Stream<MemoryPackM_t> &pipe,
           const unsigned size_n, const unsigned size_k,
           const unsigned size_m)
{
ReadB_OuterTile_N:
  for (unsigned n0 = 0; n0 < OuterTilesN(size_n); ++n0)
  {
  ReadB_OuterTile_M:
    for (unsigned m0 = 0; m0 < OuterTilesM(size_m); ++m0)
    {
      const unsigned packed_m_base = m0 * kOuterTileSizeMMemory;

    ReadB_KGroup:
      for (unsigned k0 = 0; k0 < KGroups(size_k); ++k0)
      {
// Enforce Ping-Pong Buffer (PIPO) creation and concurrent execution
#pragma HLS DATAFLOW

        MemoryPackM_t localB[kKVector][kOuterTileSizeMMemory];
#pragma HLS ARRAY_PARTITION variable = localB complete dim = 1
// zzy new寮哄埗浣跨敤鍒嗗竷锟????? RAM
#pragma HLS BIND_STORAGE variable = localB type = ram_2p impl = bram

        const unsigned k_base = k0 * kKVector;

        ReadB_Load_Block(memory, localB, size_m, size_k, packed_m_base, k_base);
        ReadB_Feed_Block(localB, pipe);
      }
    }
  }
}

// void ConvertWidthB(Stream<MemoryPackM_t> &wide, Stream<ComputePackM_t> &narrow,
//                    const unsigned size_n, const unsigned size_k,
//                    const unsigned size_m)
// {
// ConvertWidthB_OuterTile_N:
//   for (unsigned n0 = 0; n0 < OuterTilesN(size_n); ++n0)
//   {
//   ConvertWidthB_OuterTile_M:
//     for (unsigned m0 = 0; m0 < OuterTilesM(size_m); ++m0)
//     {
//     ConvertWidthB_KGroup:
//       for (unsigned k0 = 0; k0 < KGroups(size_k); ++k0)
//       {
//       ConvertWidthB_Buffer_M1:
//         for (unsigned m1m = 0; m1m < kOuterTileSizeMMemory; ++m1m)
//         {
//           const auto p0 = wide.read();
//           const auto p1 = wide.read();
//           const auto p2 = wide.read();
//           const auto p3 = wide.read();

//           narrow.write(MakeComputePackM<0>(p0, p1, p2, p3));
//           narrow.write(MakeComputePackM<1>(p0, p1, p2, p3));
//           narrow.write(MakeComputePackM<2>(p0, p1, p2, p3));
//           narrow.write(MakeComputePackM<3>(p0, p1, p2, p3));
//         }
//       }
//     }
//   }
// }

void ConvertWidthB(Stream<MemoryPackM_t> &wide, Stream<ComputePackM_t> &narrow,
                   const unsigned size_n, const unsigned size_k,
                   const unsigned size_m)
{
ConvertWidthB_OuterTile_N:
  for (unsigned n0 = 0; n0 < OuterTilesN(size_n); ++n0)
  {
  ConvertWidthB_OuterTile_M:
    for (unsigned m0 = 0; m0 < OuterTilesM(size_m); ++m0)
    {
    ConvertWidthB_KGroup:
      for (unsigned k0 = 0; k0 < KGroups(size_k); ++k0)
      {
      ConvertWidthB_Buffer_M1:
        for (unsigned m1m = 0; m1m < kOuterTileSizeMMemory; ++m1m)
        {
          // 鏍稿績淇敼 1锛氶『搴旂墿鐞嗚寰嬶紝璁剧疆涓? II=4锛堟垨鑰呭共鑴嗗彧鍐? #pragma HLS PIPELINE锛岃缁煎悎鍣ㄨ嚜鍔ㄦ帹鏂负 4锛?
#pragma HLS PIPELINE II = 4
// 鏍稿績淇敼 2锛氬姞涓婃媿骞虫寚浠わ紒鏋佸害鍏抽敭锛?
#pragma HLS LOOP_FLATTEN
          const auto p0 = wide.read();
          const auto p1 = wide.read();
          const auto p2 = wide.read();
          const auto p3 = wide.read();

          // 鏍稿績淇敼锛氱幇鍦ㄥ彧鍐欏叆 0 锟???? 1
          narrow.write(MakeComputePackM<0>(p0, p1, p2, p3));
          narrow.write(MakeComputePackM<1>(p0, p1, p2, p3));
        }
      }
    }
  }
}

void FeedB(Stream<ComputePackM_t> &fromMemory, Stream<ComputePackM_t> &toKernel,
           const unsigned size_n, const unsigned size_k,
           const unsigned size_m)
{
FeedB_OuterTile_N:
  for (unsigned n0 = 0; n0 < OuterTilesN(size_n); ++n0)
  {
  FeedB_OuterTile_M:
    for (unsigned m0 = 0; m0 < OuterTilesM(size_m); ++m0)
    {
    FeedB_KGroup:
      for (unsigned k0 = 0; k0 < KGroups(size_k); ++k0)
      {
        ComputePackM_t buffer[kInnerTilesM];
#pragma HLS ARRAY_PARTITION variable = buffer complete

      FeedB_Pipeline_N:
        for (unsigned n1 = 0; n1 < kInnerTilesN; ++n1)
        {
        FeedB_Pipeline_M:
          for (unsigned m1 = 0; m1 < kInnerTilesM; ++m1)
          {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_FLATTEN
            ComputePackM_t val;
            if (n1 == 0)
            {
              val = fromMemory.read();
              buffer[m1] = val;
            }
            else
            {
              val = buffer[m1];
            }
            toKernel.write(val);
          }
        }
      }
    }
  }
}

// void ConvertWidthC(Stream<AccPack_t> &narrow, Stream<MemoryPackC_t> &wide,
//                    const unsigned size_n, const unsigned size_k,
//                    const unsigned size_m)
// {
// ConvertWidthC_OuterTile_N:
//   for (unsigned n0 = 0; n0 < OuterTilesN(size_n); ++n0)
//   {
//   ConvertWidthC_OuterTile_M:
//     for (unsigned m0 = 0; m0 < OuterTilesM(size_m); ++m0)
//     {
//     ConvertWidthC_N1:
//       for (unsigned n1 = 0; n1 < kOuterTileSizeN; ++n1)
//       {
//       ConvertWidthC_M1:
//         for (unsigned m1 = 0; m1 < kInnerTilesM; ++m1)
//         {
// #pragma HLS PIPELINE II = 1
// #pragma HLS LOOP_FLATTEN
//           wide.write(narrow.read());
//         }
//       }
//     }
//   }
// }

void ConvertWidthC(Stream<AccPack_t> &narrow, Stream<MemoryPackC_t> &wide,
                   const unsigned size_n, const unsigned size_k,
                   const unsigned size_m)
{
ConvertWidthC_OuterTile_N:
  for (unsigned n0 = 0; n0 < OuterTilesN(size_n); ++n0)
  {
  ConvertWidthC_OuterTile_M:
    for (unsigned m0 = 0; m0 < OuterTilesM(size_m); ++m0)
    {
    ConvertWidthC_N1:
      for (unsigned n1 = 0; n1 < kOuterTileSizeN; ++n1)
      {
        AccPack_t acc;
      ConvertWidthC_M1C:
        // 寰幆娆℃暟瀵归綈锟???? 128-bit AXI 杈撳嚭娆℃暟 (256/4 = 64)
        for (unsigned m1c = 0; m1c < kOuterTileSizeMMemoryC; ++m1c)
        {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_FLATTEN
          // 锟???? 2 娆″惊鐜粠 PE 璇诲彇锟????娆″鏁版嵁 (8涓狪NT32)
          if ((m1c % 2) == 0)
          {
            acc = narrow.read();
          }

          MemoryPackC_t w;
          const unsigned offset = ((m1c % 2) == 0) ? 0 : 4;
          // 鍙栧嚭瀵瑰簲锟???? 4 锟???? INT32 鍖呰锟???? 128-bit 鍐欏叆 AXI 婕忔枟
          w[0] = acc[offset + 0];
          w[1] = acc[offset + 1];
          w[2] = acc[offset + 2];
          w[3] = acc[offset + 3];

          wide.write(w);
        }
      }
    }
  }
}

// void WriteC(Stream<MemoryPackC_t> &pipe, MemoryPackC_t memory[],
//             const unsigned size_n, const unsigned size_k,
//             const unsigned size_m)
// {
// WriteC_OuterTile_N:
//   for (unsigned n0 = 0; n0 < OuterTilesN(size_n); ++n0)
//   {
//   WriteC_OuterTile_M:
//     for (unsigned m0 = 0; m0 < OuterTilesM(size_m); ++m0)
//     {
//     WriteC_N1:
//       for (unsigned n1 = 0; n1 < kOuterTileSizeN; ++n1)
//       {
//       WriteC_M1:
//         for (unsigned m1c = 0; m1c < kOuterTileSizeMMemoryC; ++m1c)
//         {
// #pragma HLS PIPELINE II = 1
// #pragma HLS LOOP_FLATTEN
//           const auto val = pipe.read();
//           const unsigned row = n0 * kOuterTileSizeN + n1;
//           const unsigned packed_m = m0 * kOuterTileSizeMMemoryC + m1c;
//           if (row < size_n && packed_m < SizeCMemory(size_m))
//           {
//             memory[IndexC(row, packed_m, size_m)] = val;
//           }
//         }
//       }
//     }
//   }
// }
// zzy 淇敼
void WriteC(Stream<MemoryPackC_t> &pipe, MemoryPackC_t memory[],
            const unsigned size_n, const unsigned size_k,
            const unsigned size_m)
{
WriteC_OuterTile_N:
  for (unsigned n0 = 0; n0 < OuterTilesN(size_n); ++n0)
  {
  WriteC_OuterTile_M:
    for (unsigned m0 = 0; m0 < OuterTilesM(size_m); ++m0)
    {
    WriteC_N1:
      for (unsigned n1 = 0; n1 < kOuterTileSizeN; ++n1)
      {
        // 鏍稿績锛氬湪 M 缁村害灞曞紑涔嬪墠锛屾彁鍓嶇畻濂藉綋鍓嶈鐨勫熀鍦板潃
        const unsigned row = n0 * kOuterTileSizeN + n1;
        const bool valid_row = row < size_n;
        const unsigned base_addr = row * SizeCMemory(size_m);

      WriteC_M1:
        for (unsigned m1c = 0; m1c < kOuterTileSizeMMemoryC; ++m1c)
        {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_FLATTEN
          const auto val = pipe.read();
          const unsigned packed_m = m0 * kOuterTileSizeMMemoryC + m1c;

          if (valid_row && packed_m < SizeCMemory(size_m))
          {
            // 鍐呭眰娴佹按绾块伩锟???锟???鏈変箻锟???
            memory[base_addr + packed_m] = val;
          }
        }
      }
    }
  }
}