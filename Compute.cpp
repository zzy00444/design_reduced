#include "include/Compute.h"
#include "hlslib/xilinx/Utility.h"

// -------------------------------------------------------------------------
// 鏍稿績纭欢鎺ㄦ柇锛欴SP48E2 鏋侀檺鍙屾嫾 (1 DSP 锟??? 2 锟??? INT8 涔樻硶)
// -------------------------------------------------------------------------
inline void PackedMAC(Data_t weight, Data_t b0, Data_t b1, Acc_t &res0, Acc_t &res1)
{
#pragma HLS INLINE

  // weight 鐙崰 18-bit 绔彛
  ap_int<18> w = weight;

  // b0, b1 鍏变韩 27-bit 绔彛銆俠0 灞忚斀楂樹綅闃茬鍙锋薄鏌擄紝b1 宸︾Щ 18 锟???
  ap_int<27> pack_b = (ap_int<27>(b1) << 18) | (ap_int<27>(b0) & 0x3FFFF);

  // 銆愭瀬闄愬帇姒拷?锟斤細杩欓噷鍙細娑堬拷?? 1 锟??? DSP48E2
  ap_int<45> p = w * pack_b;
#pragma HLS BIND_OP variable = p op = mul impl = dsp

  // 鎻愬彇浣庝綅涔樼Н (W * B0)锛屽己杞负 16 浣嶄互鑷姩澶勭悊锟???楂樹綅绗﹀彿鎵╁睍
  res0 = (Acc_t)(ap_int<16>(p.range(15, 0)));

  // 鎻愬彇楂樹綅涔樼Н (W * B1)
  ap_int<18> upper = p.range(44, 18);

  // -----------------------------------------------------------------------
  // 鏈夌鍙锋暟纭欢淇閫昏緫锛堢敱鏋佸皯閲忕殑 LUT 鍔犳硶鍣ㄥ疄鐜帮紝鐪佸幓锟???鏁村潡 DSP锟???
  // -----------------------------------------------------------------------
  ap_int<18> corr = upper;

  // corr = corr + p[15] - (b0[7] ? w : 0);
  if (b0[7])
  {
    // 濡傛灉 B0 鏄礋鏁帮紝瀹冨湪搴曞眰鐨勬棤绗﹀彿鎷兼帴涓浉褰撲簬鍔犱簡 2^18锛岄渶瑕佸噺锟??? W 琛ュ伩
    corr -= w;
  }
  if (p[15])
  {
    // 濡傛灉浣庝綅涔樼Н鏈韩鏄礋鏁帮紝瀹冨悜楂樹綅鍊熶簡 1锛岄渶瑕佸姞鍥炴潵
    corr += 1;
  }

  // #pragma HLS PIPELINE II = 1
  // 灏? p 鐨勭粨鏋滅敤灞?閮ㄥ彉閲忔殏瀛橈紝寮哄埗 HLS 鎻掑叆瀵勫瓨鍣?

  res1 = (Acc_t)corr;
}

void ProcessingElement(Stream<ComputePackN_t> &aIn,
                       Stream<ComputePackN_t> &aOut,
                       Stream<ComputePackM_t> &bIn,
                       Stream<ComputePackM_t> &bOut,
                       Stream<AccPack_t> &cOut,
                       Stream<AccPack_t> &cIn,
                       const unsigned locationN,
                       const unsigned size_n,
                       const unsigned size_k,
                       const unsigned size_m)
{
  ComputePackN_t aBuffer[2 * kInnerTilesN];
#pragma HLS ARRAY_PARTITION variable = aBuffer complete

  // Tile-local accumulation buffer (int32).
  AccPack_t cBuffer[kInnerTilesN * kInnerTilesM][kComputeTileSizeN];
#pragma HLS ARRAY_PARTITION variable = cBuffer complete dim = 2
  // 銆愭柊澧炶繖锟????琛岋拷?锟斤細寮哄埗浣跨敤 0% 鍒╃敤鐜囩殑 URAM锛屾嫰锟???? BRAM锟????
#pragma HLS BIND_STORAGE variable = cBuffer type = ram_t2p impl = uram

InitializeABuffer_Inner:
  for (unsigned n2 = 0; n2 < kInnerTilesN; ++n2)
  {
    if (locationN < kComputeTilesN - 1)
    {
    InitializeABuffer_Outer:
      for (unsigned n1 = 0; n1 < kComputeTilesN - locationN; ++n1)
      {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_FLATTEN
        const auto read = aIn.read();
        if (n1 == 0)
        {
          aBuffer[n2] = read;
        }
        else
        {
          aOut.write(read);
        }
      }
    }
    else
    {
      aBuffer[n2] = aIn.read();
    }
  }

OuterTile_N:
  for (unsigned n0 = 0; n0 < OuterTilesN(size_n); ++n0)
  {
  OuterTile_M:
    for (unsigned m0 = 0; m0 < OuterTilesM(size_m); ++m0)
    {
    Collapse_K:
      for (unsigned k = 0; k < KGroups(size_k); ++k)
      {
      Pipeline_N:
        for (unsigned n1 = 0; n1 < kInnerTilesN; ++n1)
        {
        Pipeline_M:
          for (unsigned m1 = 0; m1 < kInnerTilesM; ++m1)
          {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_FLATTEN

            if ((n0 < OuterTilesN(size_n) - 1 || m0 < OuterTilesM(size_m) - 1 ||
                 k < KGroups(size_k) - 1) &&
                m1 >= locationN && m1 < kComputeTilesN)
            {
              const auto read = aIn.read();
              if (m1 == locationN)
              {
                aBuffer[n1 + (k % 2 == 0 ? kInnerTilesN : 0)] = read;
#pragma HLS DEPENDENCE variable = aBuffer false
              }
              else if (locationN < kComputeTilesN - 1)
              {
                aOut.write(read);
              }
            }

            const auto aVal = aBuffer[n1 + (k % 2 == 0 ? 0 : kInnerTilesN)];
#pragma HLS DEPENDENCE variable = aBuffer false
            const auto bVal = bIn.read();
            if (locationN < kComputeTilesN - 1)
            {
              bOut.write(bVal);
            }

          Unroll_N:
            for (unsigned n2 = 0; n2 < kComputeTileSizeN; ++n2)
            {
#pragma HLS UNROLL
              const bool inBoundsN =
                  (n0 * kInnerTilesN * kComputeTileSizeN +
                   n1 * kComputeTileSizeN + n2) < size_n;

              AccPack_t cStore;
              const auto cPrev = (k > 0)
                                     ? cBuffer[n1 * kInnerTilesM + m1][n2]
                                     : AccPack_t(Acc_t(0));

              // Unroll_M:
            //               for (unsigned m2 = 0; m2 < kComputeTileSizeM; ++m2)
            //               {
            // #pragma HLS UNROLL
            //                 const bool inBoundsM =
            //                     (m0 * kInnerTilesM * kComputeTileSizeM +
            //                      m1 * kComputeTileSizeM + m2) < size_m;
            //                 const bool inBounds = inBoundsN && inBoundsM;

            //                 // Four packed INT8 products along K, followed by a local
            //                 // reduction to INT32. This keeps the pack-4 throughput gain
            //                 // while avoiding the synthesis issue triggered by the previous
            //                 // helper-based dot-product form in Vitis HLS 2023.2.
            //                 // const Acc_t mul0 =
            //                 //     static_cast<Acc_t>(GetComputeNElement<0>(aVal)) *
            //                 //     static_cast<Acc_t>(GetBElement(bVal, m2, 0));
            //                 // const Acc_t mul1 =
            //                 //     static_cast<Acc_t>(GetComputeNElement<1>(aVal)) *
            //                 //     static_cast<Acc_t>(GetBElement(bVal, m2, 1));
            //                 // const Acc_t mul2 =
            //                 //     static_cast<Acc_t>(GetComputeNElement<2>(aVal)) *
            //                 //     static_cast<Acc_t>(GetBElement(bVal, m2, 2));
            //                 // const Acc_t mul3 =
            //                 //     static_cast<Acc_t>(GetComputeNElement<3>(aVal)) *
            //                 //     static_cast<Acc_t>(GetBElement(bVal, m2, 3));
            //                 // 鏍稿績淇锛氱Щ闄ゆ彁鍓嶇殑 static_cast锛岃 ap_int<8> 鐩存帴鐩镐箻锛屽崟 DSP 鍗冲彲瀹屾垚锟????
            //                 const Acc_t mul0 = static_cast<Acc_t>(GetComputeNElement<0>(aVal) * GetBElement(bVal, m2, 0));
            //                 const Acc_t mul1 = static_cast<Acc_t>(GetComputeNElement<1>(aVal) * GetBElement(bVal, m2, 1));
            //                 const Acc_t mul2 = static_cast<Acc_t>(GetComputeNElement<2>(aVal) * GetBElement(bVal, m2, 2));
            //                 const Acc_t mul3 = static_cast<Acc_t>(GetComputeNElement<3>(aVal) * GetBElement(bVal, m2, 3));

            //                 const Acc_t sum01 = mul0 + mul1;
            //                 const Acc_t sum23 = mul2 + mul3;
            //                 const Acc_t dot = sum01 + sum23;
            //                 const Acc_t sum = cPrev[m2] + dot;

            //                 cStore[m2] = inBounds ? sum : cPrev[m2];
            // #pragma HLS DEPENDENCE variable = cBuffer false
            //               }

            // Compute MACs (浣跨敤 DSP 鍙屾嫾锛屾瘡娆″锟??? m2 锟??? m2+1)
            Unroll_M:
              for (unsigned m2 = 0; m2 < kComputeTileSizeM; m2 += 2)
              {
#pragma HLS UNROLL

                Acc_t mul0_0, mul0_1;
                PackedMAC(GetComputeNElement<0>(aVal), GetBElement(bVal, m2, 0), GetBElement(bVal, m2 + 1, 0), mul0_0, mul0_1);

                Acc_t mul1_0, mul1_1;
                PackedMAC(GetComputeNElement<1>(aVal), GetBElement(bVal, m2, 1), GetBElement(bVal, m2 + 1, 1), mul1_0, mul1_1);

                Acc_t mul2_0, mul2_1;
                PackedMAC(GetComputeNElement<2>(aVal), GetBElement(bVal, m2, 2), GetBElement(bVal, m2 + 1, 2), mul2_0, mul2_1);

                Acc_t mul3_0, mul3_1;
                PackedMAC(GetComputeNElement<3>(aVal), GetBElement(bVal, m2, 3), GetBElement(bVal, m2 + 1, 3), mul3_0, mul3_1);

                // 鍒嗗埆锟??? m2 锟??? m2+1 鐨勶拷?锟介亾杩涜 K 缁村害鐨勭疮锟???
                Acc_t dot_0 = mul0_0 + mul1_0 + mul2_0 + mul3_0;
                Acc_t dot_1 = mul0_1 + mul1_1 + mul2_1 + mul3_1;

                // --- 鍐欏叆 m2 閫氶亾 ---
                const bool inBounds_0 = (locationN * kInnerTilesN + n1) * kComputeTileSizeN + n2 < size_n && m1 * kComputeTileSizeM + m2 < size_m;
                Acc_t sum_0 = cPrev[m2] + dot_0;
                cStore[m2] = inBounds_0 ? sum_0 : cPrev[m2];

                // --- 鍐欏叆 m2+1 閫氶亾 ---
                const bool inBounds_1 = (locationN * kInnerTilesN + n1) * kComputeTileSizeN + n2 < size_n && m1 * kComputeTileSizeM + m2 + 1 < size_m;
                Acc_t sum_1 = cPrev[m2 + 1] + dot_1;
                cStore[m2 + 1] = inBounds_1 ? sum_1 : cPrev[m2 + 1];

#pragma HLS DEPENDENCE variable = cBuffer false
              }

              cBuffer[n1 * kInnerTilesM + m1][n2] = cStore;
            }
          }
        }
      }

      const unsigned writeFlattenedInner =
          (kComputeTileSizeN * kInnerTilesM +
           (kComputeTilesN - locationN - 1) * kComputeTileSizeN * kInnerTilesM);
      const unsigned writeFlattened = kInnerTilesN * writeFlattenedInner;
      // ap_uint<hlslib::ConstLog2(kInnerTilesN)> n1 = 0;
      // ap_uint<((kComputeTileSizeN > 1) ? hlslib::ConstLog2(kComputeTileSizeN)
      //                                  : 1)>
      //     n2 = 0;
      // ap_uint<hlslib::ConstLog2(kInnerTilesM)> m1 = 0;
      // zzy 淇敼 鍏ㄩ儴鏇挎崲涓烘渶绋冲仴鐨勫熀纭?绫诲瀷锛孒LS 鐨勭患鍚堝櫒浼氳嚜鍔ㄤ紭鍖栦负瀵勫瓨鍣ㄥ搴?
      unsigned n1 = 0;
      unsigned n2 = 0;
      unsigned m1 = 0;

      unsigned inner = 0;

    WriteC_Flattened:
      for (unsigned i = 0; i < writeFlattened; ++i)
      {
#pragma HLS PIPELINE II = 1

        if (inner < kComputeTileSizeN * kInnerTilesM)
        {
          cOut.write(cBuffer[n1 * kInnerTilesM + m1][n2]);
          if (m1 == kInnerTilesM - 1)
          {
            m1 = 0;
            if (n2 == kComputeTileSizeN - 1)
            {
              n2 = 0;
            }
            else
            {
              ++n2;
            }
          }
          else
          {
            ++m1;
          }
        }
        else if (locationN < kComputeTilesN - 1)
        {
          cOut.write(cIn.read());
        }

        if (inner == writeFlattenedInner - 1)
        {
          inner = 0;
          ++n1;
        }
        else
        {
          ++inner;
        }
      }
    }
  }
}

// --- Compute.cpp ---
// zzy 淇敼 2D
/////////////////////////////////////////////////////////////////////////////////////////////
// inline void PackedMAC(Data_t weight, Data_t b0, Data_t b1, Acc_t &res0, Acc_t &res1)
// {
// #pragma HLS INLINE
//   ap_int<18> w = weight;
//   ap_int<27> pack_b = (ap_int<27>(b1) << 18) | (ap_int<27>(b0) & 0x3FFFF);
//   ap_int<45> p = w * pack_b;
// #pragma HLS BIND_OP variable = p op = mul impl = dsp
//   res0 = (Acc_t)(ap_int<16>(p.range(15, 0)));
//   ap_int<18> upper = p.range(44, 18);
//   ap_int<18> corr = upper;
//   if (b0[7])
//     corr -= w;
//   if (p[15])
//     corr += 1;
//   res1 = (Acc_t)corr;
// }

// --- 淇敼 Compute.cpp 涓殑涓変釜 Adapter ---

// -------------------------------------------------------------------------
// 閫傞厤鍣? 1锛氬畬缇庡苟鍙戝垎鍙? A 娴? (瑙ｅ喅楗ラタ闂)
// -------------------------------------------------------------------------
void FeedA_Adapter(Stream<ComputePackN_t> &aIn, Stream<ComputePackN_t> aPipes[kGridRows][kGridCols + 1],
                   const unsigned size_n, const unsigned size_k, const unsigned size_m)
{
FeedA_Outer_N:
  for (unsigned n0 = 0; n0 < OuterTilesN(size_n); ++n0)
  {
  FeedA_Outer_M:
    for (unsigned m0 = 0; m0 < OuterTilesM(size_m); ++m0)
    {
    FeedA_Collapse_K:
      for (unsigned k = 0; k < KGroups(size_k); ++k)
      {

        // 灞?閮ㄥ井缂撳瓨锛氭秷鑰楁瀬灏戠殑 LUTRAM 鍚告敹涓茶鏁版嵁
        ComputePackN_t localA[kGridRows][kPeInnerTilesN];
#pragma HLS ARRAY_PARTITION variable = localA complete dim = 1

      // 1. 鍏ㄩ?熷惛鍏ヤ覆琛屾暟鎹? (128涓懆鏈?)
      FeedA_Read_R:
        for (unsigned r = 0; r < kGridRows; ++r)
        {
        FeedA_Read_N1:
          for (unsigned n1 = 0; n1 < kPeInnerTilesN; ++n1)
          {
#pragma HLS PIPELINE II = 1
            localA[r][n1] = aIn.read();
          }
        }

      // 2. 榻愭璧帮紒瀹屽叏骞惰鍒嗗彂缁? 4 涓
      FeedA_Distribute_N1:
        for (unsigned n1 = 0; n1 < kPeInnerTilesN; ++n1)
        {
#pragma HLS PIPELINE II = 1
        FeedA_Distribute_R:
          for (unsigned r = 0; r < kGridRows; ++r)
          {
#pragma HLS UNROLL
            aPipes[r][0].write(localA[r][n1]);
          }
        }
      }
    }
  }
}

// -------------------------------------------------------------------------
// 閫傞厤鍣? 2锛氬畬缇庡苟鍙戝垎鍙? B 娴? (瑙ｅ喅楗ラタ闂)
// -------------------------------------------------------------------------
void FeedB_Adapter(Stream<ComputePackM_t> &bIn, Stream<ComputePackM_t> bPipes[kGridRows + 1][kGridCols],
                   const unsigned size_n, const unsigned size_k, const unsigned size_m)
{
FeedB_Outer_N:
  for (unsigned n0 = 0; n0 < OuterTilesN(size_n); ++n0)
  {
  FeedB_Outer_M:
    for (unsigned m0 = 0; m0 < OuterTilesM(size_m); ++m0)
    {
    FeedB_Collapse_K:
      for (unsigned k = 0; k < KGroups(size_k); ++k)
      {

        // 灞?閮ㄥ井缂撳瓨
        ComputePackM_t localB[kGridCols][kPeInnerTilesM];
#pragma HLS ARRAY_PARTITION variable = localB complete dim = 1

      // 1. 鍏ㄩ?熷惛鍏ヤ覆琛屾暟鎹? (16涓懆鏈?)
      FeedB_Read_C:
        for (unsigned c = 0; c < kGridCols; ++c)
        {
        FeedB_Read_M1:
          for (unsigned m1 = 0; m1 < kPeInnerTilesM; ++m1)
          {
#pragma HLS PIPELINE II = 1
            localB[c][m1] = bIn.read();
          }
        }

      // 2. 榻愭璧帮紒瀹屽叏骞惰鍒嗗彂缁? 4 涓垪
      FeedB_Distribute_M1:
        for (unsigned m1 = 0; m1 < kPeInnerTilesM; ++m1)
        {
#pragma HLS PIPELINE II = 1
        FeedB_Distribute_C:
          for (unsigned c = 0; c < kGridCols; ++c)
          {
#pragma HLS UNROLL
            bPipes[0][c].write(localB[c][m1]);
          }
        }
      }
    }
  }
}

// -------------------------------------------------------------------------
// 閫傞厤鍣? 3锛氬苟鍙戝惛鏀? 4 琛岀粨鏋滐紝涓茶瀵规帴鍏夋爡鎵弿 (娑堥櫎鎺掔┖鑳屽帇)
// -------------------------------------------------------------------------
void CollectC_Adapter(Stream<AccPack_t> cPipes[kGridRows][kGridCols + 1], Stream<AccPack_t> &cOut,
                      const unsigned size_n, const unsigned size_k, const unsigned size_m)
{
Collect_Outer_N:
  for (unsigned n0 = 0; n0 < OuterTilesN(size_n); ++n0)
  {
  Collect_Outer_M:
    for (unsigned m0 = 0; m0 < OuterTilesM(size_m); ++m0)
    {

      // 姣忎釜 PE 浼氬悙鍑?: 32 (n1) * 16 (鎺掔┖鑺傛媿) = 512 涓? AccPack_t
      constexpr unsigned kDrainLength = kPeInnerTilesN * kGridCols * kPeInnerTilesM;

      // 灞?閮ㄥ井缂撳瓨锛氱敤浜庡苟鍙戝惛鏀堕樀鍒楀悙鍑虹殑缁撴灉 (鎬昏娑堣?楃害 64KB URAM/BRAM)
      AccPack_t localC[kGridRows][kDrainLength];
#pragma HLS ARRAY_PARTITION variable = localC complete dim = 1
#pragma HLS BIND_STORAGE variable = localC type = ram_t2p impl = uram

    // 1. 榻愭璧帮紒骞跺彂鍚告敹 4 涓鐨勬帓绌烘暟鎹? (浠呴渶 512 涓懆鏈燂紝闃靛垪绔嬪埢瑙ｆ斁)
    Collect_Concurrent_Drain:
      for (unsigned i = 0; i < kDrainLength; ++i)
      {
#pragma HLS PIPELINE II = 1
      Collect_Drain_R:
        for (unsigned r = 0; r < kGridRows; ++r)
        {
#pragma HLS UNROLL
          localC[r][i] = cPipes[r][kGridCols].read();
        }
      }

    // 2. 涓茶鍖栧啓鍑猴紝閫傞厤 WriteC 鏈熷緟鐨勫厜鏍呮壂鎻忛『搴? (闇?瑕? 2048 涓懆鏈?)
    Collect_Serial_Write_R:
      for (unsigned r = 0; r < kGridRows; ++r)
      {
      Collect_Serial_Write_I:
        for (unsigned i = 0; i < kDrainLength; ++i)
        {
#pragma HLS PIPELINE II = 1
          cOut.write(localC[r][i]);
        }
      }
    }
  }
}

// -------------------------------------------------------------------------
// 鏍稿績 2D 鑴夊姩闃靛垪璁＄畻鍗曞厓
// -------------------------------------------------------------------------
void ProcessingElement2D(
    Stream<ComputePackN_t> &aIn, Stream<ComputePackN_t> &aOut,
    Stream<ComputePackM_t> &bIn, Stream<ComputePackM_t> &bOut,
    Stream<AccPack_t> &cIn, Stream<AccPack_t> &cOut,
    const unsigned r, const unsigned c,
    const unsigned size_n, const unsigned size_k, const unsigned size_m)
{
  // C 鐨勫眬閮ㄧ紦瀛橈細浣跨敤 0% 鍒╃敤鐜囩殑 URAM 鎷晳 BRAM
  AccPack_t cBuffer[kPeInnerTilesN * kPeInnerTilesM][1];
#pragma HLS ARRAY_PARTITION variable = cBuffer complete dim = 2
#pragma HLS BIND_STORAGE variable = cBuffer type = ram_t2p impl = uram

OuterTile_N:
  for (unsigned n0 = 0; n0 < OuterTilesN(size_n); ++n0)
  {
  OuterTile_M:
    for (unsigned m0 = 0; m0 < OuterTilesM(size_m); ++m0)
    {
    Collapse_K:
      for (unsigned k = 0; k < KGroups(size_k); ++k)
      {

        // PE 鏈湴鐨? A, B 琛屽垪寰紦鍐?
        ComputePackN_t aBuffer[kPeInnerTilesN];
        ComputePackM_t bBuffer[kPeInnerTilesM];
#pragma HLS ARRAY_PARTITION variable = aBuffer complete dim = 1
#pragma HLS ARRAY_PARTITION variable = bBuffer complete dim = 1

      Pipeline_N:
        for (unsigned n1 = 0; n1 < kPeInnerTilesN; ++n1)
        {
        Pipeline_M:
          for (unsigned m1 = 0; m1 < kPeInnerTilesM; ++m1)
          {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_FLATTEN

            // --- 1. A 鏁版嵁鑾峰彇涓庢í鍚戝箍鎾? ---
            ComputePackN_t aVal;
            if (m1 == 0)
            {
              aVal = aIn.read();
              if (c < kGridCols - 1)
                aOut.write(aVal);
              aBuffer[n1] = aVal;
            }
            else
            {
              aVal = aBuffer[n1];
            }

            // --- 2. B 鏁版嵁鑾峰彇涓庣旱鍚戝箍鎾? ---
            ComputePackM_t bVal;
            if (n1 == 0)
            {
              bVal = bIn.read();
              if (r < kGridRows - 1)
                bOut.write(bVal);
              bBuffer[m1] = bVal;
            }
            else
            {
              bVal = bBuffer[m1];
            }

            // --- 3. 鏍稿績鍙屾嫾 MAC 璁＄畻 ---
            AccPack_t cStore;
            AccPack_t cPrev = (k > 0) ? cBuffer[n1 * kPeInnerTilesM + m1][0] : AccPack_t(Acc_t(0));

          Unroll_M:
            for (unsigned m2 = 0; m2 < kComputeTileSizeM; m2 += 2)
            {
#pragma HLS UNROLL
              Acc_t mul0_0, mul0_1;
              PackedMAC(GetComputeNElement<0>(aVal), GetBElement(bVal, m2, 0), GetBElement(bVal, m2 + 1, 0), mul0_0, mul0_1);
              Acc_t mul1_0, mul1_1;
              PackedMAC(GetComputeNElement<1>(aVal), GetBElement(bVal, m2, 1), GetBElement(bVal, m2 + 1, 1), mul1_0, mul1_1);
              Acc_t mul2_0, mul2_1;
              PackedMAC(GetComputeNElement<2>(aVal), GetBElement(bVal, m2, 2), GetBElement(bVal, m2 + 1, 2), mul2_0, mul2_1);
              Acc_t mul3_0, mul3_1;
              PackedMAC(GetComputeNElement<3>(aVal), GetBElement(bVal, m2, 3), GetBElement(bVal, m2 + 1, 3), mul3_0, mul3_1);

              Acc_t dot_0 = mul0_0 + mul1_0 + mul2_0 + mul3_0;
              Acc_t dot_1 = mul0_1 + mul1_1 + mul2_1 + mul3_1;

              // const bool inBounds_0 = ((n0 * kOuterTileSizeN + r * kPeInnerTilesN + n1) * kComputeTileSizeN) < size_n &&
              //                         (m0 * kOuterTileSizeM + c * (kPeInnerTilesM * kComputeTileSizeM) + m1 * kComputeTileSizeM + m2) < size_m;
              // cStore[m2] = inBounds_0 ? (Acc_t)(cPrev[m2] + dot_0) : (Acc_t)cPrev[m2];

              // const bool inBounds_1 = ((n0 * kOuterTileSizeN + r * kPeInnerTilesN + n1) * kComputeTileSizeN) < size_n &&
              //                         (m0 * kOuterTileSizeM + c * (kPeInnerTilesM * kComputeTileSizeM) + m1 * kComputeTileSizeM + m2 + 1) < size_m;
              // cStore[m2 + 1] = inBounds_1 ? (Acc_t)(cPrev[m2 + 1] + dot_1) : (Acc_t)cPrev[m2 + 1];
              const bool inBounds_0 = ((n0 * kOuterTileSizeN + r * kPeInnerTilesN + n1) * kComputeTileSizeN) < size_n &&
                                      (m0 * kOuterTileSizeM + c * (kPeInnerTilesM * kComputeTileSizeM) + m1 * kComputeTileSizeM + m2) < size_m;
              const bool inBounds_1 = ((n0 * kOuterTileSizeN + r * kPeInnerTilesN + n1) * kComputeTileSizeN) < size_n &&
                                      (m0 * kOuterTileSizeM + c * (kPeInnerTilesM * kComputeTileSizeM) + m1 * kComputeTileSizeM + m2 + 1) < size_m;

              // 1. 寮哄埗鎻愬彇 Proxy 涓殑鏁版嵁锛岄殣寮?/鏄惧紡杞崲涓哄師鐢? Acc_t
              Acc_t prev_val_0 = static_cast<Acc_t>(cPrev[m2]);
              Acc_t prev_val_1 = static_cast<Acc_t>(cPrev[m2 + 1]);

              // 2. 浣跨敤鍘熺敓绫诲瀷杩涜鍔犳硶
              Acc_t sum_0 = prev_val_0 + dot_0;
              Acc_t sum_1 = prev_val_1 + dot_1;

              // 3. 缁撴灉鍐欏洖 Proxy
              cStore[m2] = inBounds_0 ? sum_0 : prev_val_0;
              cStore[m2 + 1] = inBounds_1 ? sum_1 : prev_val_1;
            }
            cBuffer[n1 * kPeInnerTilesM + m1][0] = cStore;
          }
        }
      } // End Collapse_K

    // --- 4. 缁撴灉鎺掔┖闃舵 (姘村钩鏂瑰悜鍚戝彸鎺掔┖) ---
    Drain_N:
      for (unsigned n = 0; n < kPeInnerTilesN; ++n)
      {
      Drain_M:
        for (unsigned i = 0; i < kGridCols * kPeInnerTilesM; ++i)
        {
#pragma HLS PIPELINE II = 1
          if (i < c * kPeInnerTilesM)
          {
            if (c > 0)
              cOut.write(cIn.read()); // 杞彂宸︿晶浼犳潵鐨勭粨鏋?
          }
          else if (i < (c + 1) * kPeInnerTilesM)
          {
            // 鍐欏嚭鏈湴缁撴灉
            cOut.write(cBuffer[n * kPeInnerTilesM + (i - c * kPeInnerTilesM)][0]);
          }
        }
      }
    } // End OuterTile_M
  } // End OuterTile_N
}

//////////////////////////////////////////////////////////////////////////////////////////