// #include "include/Int8Gemm.h"
// #include "include/Compute.h"
// #include "include/Memory.h"
// #include "hlslib/xilinx/Simulation.h"

// extern "C" void MatrixMultiplicationKernelInt8(MemoryPackN_t const a[],
//                                                MemoryPackM_t const b[],
//                                                MemoryPackC_t c[],
//                                                const unsigned size_n,
//                                                const unsigned size_k,
//                                                const unsigned size_m)
// {
// #pragma HLS INTERFACE m_axi port=a offset=slave bundle=gmem0 max_read_burst_length=256 num_read_outstanding=64
// #pragma HLS INTERFACE m_axi port=b offset=slave bundle=gmem1 max_read_burst_length=256 num_read_outstanding=64
// #pragma HLS INTERFACE m_axi port=c offset=slave bundle=gmem2 max_write_burst_length=256 num_write_outstanding=64

// #pragma HLS INTERFACE s_axilite port = return
// #pragma HLS INTERFACE s_axilite port = size_n
// #pragma HLS INTERFACE s_axilite port = size_k
// #pragma HLS INTERFACE s_axilite port = size_m

// // PE 锟斤拷展锟斤拷 16
// #pragma HLS DATAFLOW

//   // 1. 锟斤拷展 A 锟侥管碉拷 (锟斤拷要 17 锟斤拷锟斤拷锟接达拷锟斤拷 16 锟斤拷 PE)
//   Stream<ComputePackN_t, kPipeDepth> aPipes_0("aPipes_0");
//   Stream<ComputePackN_t, kPipeDepth> aPipes_1("aPipes_1");
//   Stream<ComputePackN_t, kPipeDepth> aPipes_2("aPipes_2");
//   Stream<ComputePackN_t, kPipeDepth> aPipes_3("aPipes_3");
//   Stream<ComputePackN_t, kPipeDepth> aPipes_4("aPipes_4");
//   Stream<ComputePackN_t, kPipeDepth> aPipes_5("aPipes_5");
//   Stream<ComputePackN_t, kPipeDepth> aPipes_6("aPipes_6");
//   Stream<ComputePackN_t, kPipeDepth> aPipes_7("aPipes_7");
//   Stream<ComputePackN_t, kPipeDepth> aPipes_8("aPipes_8");
//   Stream<ComputePackN_t, kPipeDepth> aPipes_9("aPipes_9");
//   Stream<ComputePackN_t, kPipeDepth> aPipes_10("aPipes_10");
//   Stream<ComputePackN_t, kPipeDepth> aPipes_11("aPipes_11");
//   Stream<ComputePackN_t, kPipeDepth> aPipes_12("aPipes_12");
//   Stream<ComputePackN_t, kPipeDepth> aPipes_13("aPipes_13");
//   Stream<ComputePackN_t, kPipeDepth> aPipes_14("aPipes_14");
//   Stream<ComputePackN_t, kPipeDepth> aPipes_15("aPipes_15");
//   Stream<ComputePackN_t, kPipeDepth> aPipes_16("aPipes_16");

//   Stream<MemoryPackM_t, 2 * kOuterTileSizeMMemory * kKVector> bMemory("bMemory");
// #pragma HLS STREAM variable = bMemory depth = 8

//   // 2. 锟斤拷展 B 锟侥管碉拷
//   Stream<ComputePackM_t, kPipeDepth> bPipes_0("bPipes_0");
//   Stream<ComputePackM_t, kPipeDepth> bPipes_1("bPipes_1");
//   Stream<ComputePackM_t, kPipeDepth> bPipes_2("bPipes_2");
//   Stream<ComputePackM_t, kPipeDepth> bPipes_3("bPipes_3");
//   Stream<ComputePackM_t, kPipeDepth> bPipes_4("bPipes_4");
//   Stream<ComputePackM_t, kPipeDepth> bPipes_5("bPipes_5");
//   Stream<ComputePackM_t, kPipeDepth> bPipes_6("bPipes_6");
//   Stream<ComputePackM_t, kPipeDepth> bPipes_7("bPipes_7");
//   Stream<ComputePackM_t, kPipeDepth> bPipes_8("bPipes_8");
//   Stream<ComputePackM_t, kPipeDepth> bPipes_9("bPipes_9");
//   Stream<ComputePackM_t, kPipeDepth> bPipes_10("bPipes_10");
//   Stream<ComputePackM_t, kPipeDepth> bPipes_11("bPipes_11");
//   Stream<ComputePackM_t, kPipeDepth> bPipes_12("bPipes_12");
//   Stream<ComputePackM_t, kPipeDepth> bPipes_13("bPipes_13");
//   Stream<ComputePackM_t, kPipeDepth> bPipes_14("bPipes_14");
//   Stream<ComputePackM_t, kPipeDepth> bPipes_15("bPipes_15");
//   Stream<ComputePackM_t, kPipeDepth> bPipes_16("bPipes_16");
//   Stream<ComputePackM_t, kPipeDepth> bFeed("bFeed");

//   // 3. 锟斤拷展 C 锟侥管碉拷
//   Stream<AccPack_t> cPipes_0("cPipes_0");
//   Stream<AccPack_t> cPipes_1("cPipes_1");
//   Stream<AccPack_t> cPipes_2("cPipes_2");
//   Stream<AccPack_t> cPipes_3("cPipes_3");
//   Stream<AccPack_t> cPipes_4("cPipes_4");
//   Stream<AccPack_t> cPipes_5("cPipes_5");
//   Stream<AccPack_t> cPipes_6("cPipes_6");
//   Stream<AccPack_t> cPipes_7("cPipes_7");
//   Stream<AccPack_t> cPipes_8("cPipes_8");
//   Stream<AccPack_t> cPipes_9("cPipes_9");
//   Stream<AccPack_t> cPipes_10("cPipes_10");
//   Stream<AccPack_t> cPipes_11("cPipes_11");
//   Stream<AccPack_t> cPipes_12("cPipes_12");
//   Stream<AccPack_t> cPipes_13("cPipes_13");
//   Stream<AccPack_t> cPipes_14("cPipes_14");
//   Stream<AccPack_t> cPipes_15("cPipes_15");
//   Stream<AccPack_t> cPipes_16("cPipes_16");

//   Stream<MemoryPackC_t, 2 * kOuterTileSizeMMemoryC> cMemory("cMemory");

//   HLSLIB_DATAFLOW_INIT();

//   HLSLIB_DATAFLOW_FUNCTION(ReadATransposed, a, aPipes_0, size_n, size_k, size_m);

//   HLSLIB_DATAFLOW_FUNCTION(ReadB, b, bMemory, size_n, size_k, size_m);
//   HLSLIB_DATAFLOW_FUNCTION(ConvertWidthB, bMemory, bFeed, size_n, size_k, size_m);
//   HLSLIB_DATAFLOW_FUNCTION(FeedB, bFeed, bPipes_0, size_n, size_k, size_m);

//   // 4. 锟斤拷锟斤拷 16 锟斤拷 PE
//   HLSLIB_DATAFLOW_FUNCTION(ProcessingElement, aPipes_0, aPipes_1, bPipes_0, bPipes_1, cPipes_0, cPipes_1, 0, size_n, size_k, size_m);
//   HLSLIB_DATAFLOW_FUNCTION(ProcessingElement, aPipes_1, aPipes_2, bPipes_1, bPipes_2, cPipes_1, cPipes_2, 1, size_n, size_k, size_m);
//   HLSLIB_DATAFLOW_FUNCTION(ProcessingElement, aPipes_2, aPipes_3, bPipes_2, bPipes_3, cPipes_2, cPipes_3, 2, size_n, size_k, size_m);
//   HLSLIB_DATAFLOW_FUNCTION(ProcessingElement, aPipes_3, aPipes_4, bPipes_3, bPipes_4, cPipes_3, cPipes_4, 3, size_n, size_k, size_m);
//   HLSLIB_DATAFLOW_FUNCTION(ProcessingElement, aPipes_4, aPipes_5, bPipes_4, bPipes_5, cPipes_4, cPipes_5, 4, size_n, size_k, size_m);
//   HLSLIB_DATAFLOW_FUNCTION(ProcessingElement, aPipes_5, aPipes_6, bPipes_5, bPipes_6, cPipes_5, cPipes_6, 5, size_n, size_k, size_m);
//   HLSLIB_DATAFLOW_FUNCTION(ProcessingElement, aPipes_6, aPipes_7, bPipes_6, bPipes_7, cPipes_6, cPipes_7, 6, size_n, size_k, size_m);
//   HLSLIB_DATAFLOW_FUNCTION(ProcessingElement, aPipes_7, aPipes_8, bPipes_7, bPipes_8, cPipes_7, cPipes_8, 7, size_n, size_k, size_m);
//   HLSLIB_DATAFLOW_FUNCTION(ProcessingElement, aPipes_8, aPipes_9, bPipes_8, bPipes_9, cPipes_8, cPipes_9, 8, size_n, size_k, size_m);
//   HLSLIB_DATAFLOW_FUNCTION(ProcessingElement, aPipes_9, aPipes_10, bPipes_9, bPipes_10, cPipes_9, cPipes_10, 9, size_n, size_k, size_m);
//   HLSLIB_DATAFLOW_FUNCTION(ProcessingElement, aPipes_10, aPipes_11, bPipes_10, bPipes_11, cPipes_10, cPipes_11, 10, size_n, size_k, size_m);
//   HLSLIB_DATAFLOW_FUNCTION(ProcessingElement, aPipes_11, aPipes_12, bPipes_11, bPipes_12, cPipes_11, cPipes_12, 11, size_n, size_k, size_m);
//   HLSLIB_DATAFLOW_FUNCTION(ProcessingElement, aPipes_12, aPipes_13, bPipes_12, bPipes_13, cPipes_12, cPipes_13, 12, size_n, size_k, size_m);
//   HLSLIB_DATAFLOW_FUNCTION(ProcessingElement, aPipes_13, aPipes_14, bPipes_13, bPipes_14, cPipes_13, cPipes_14, 13, size_n, size_k, size_m);
//   HLSLIB_DATAFLOW_FUNCTION(ProcessingElement, aPipes_14, aPipes_15, bPipes_14, bPipes_15, cPipes_14, cPipes_15, 14, size_n, size_k, size_m);
//   HLSLIB_DATAFLOW_FUNCTION(ProcessingElement, aPipes_15, aPipes_16, bPipes_15, bPipes_16, cPipes_15, cPipes_16, 15, size_n, size_k, size_m);

//   HLSLIB_DATAFLOW_FUNCTION(ConvertWidthC, cPipes_0, cMemory, size_n, size_k, size_m);
//   HLSLIB_DATAFLOW_FUNCTION(WriteC, cMemory, c, size_n, size_k, size_m);

//   HLSLIB_DATAFLOW_FINALIZE();
// }

// zzy 淇敼 2D
//////////////////////////////////////////////////////////////////////////////////////
#include "include/Int8Gemm.h"
#include "include/Compute.h"
#include "include/Memory.h"
#include "hlslib/xilinx/Simulation.h"

// 澹版槑澶栭儴 Adapter (锟??瑕佸湪 Compute.h 涓０锟??)
extern "C" void MatrixMultiplicationKernelInt8(MemoryPackN_t const a[],
                                               MemoryPackM_t const b[],
                                               MemoryPackM_t c[], // <--- 娉ㄦ剰锛氳緭鍑虹被鍨嬪彉锟?? MemoryPackM_t
                                               const unsigned size_n,
                                               const unsigned size_k,
                                               const unsigned size_m,
                                               const unsigned layer_idx, // <--- 鍛婅瘔 FPGA 褰撳墠鏄摢锟??锟??
                                               const bool enable_gelu)   // <--- GELU 鏃佽矾锟??锟??
{
#pragma HLS INTERFACE m_axi port = a offset = slave bundle = gmem0 depth = 4096 max_read_burst_length = 64 num_read_outstanding = 8
#pragma HLS INTERFACE m_axi port = b offset = slave bundle = gmem1 depth = 4096 max_read_burst_length = 64 num_read_outstanding = 8
#pragma HLS INTERFACE m_axi port = c offset = slave bundle = gmem2 depth = 4096 max_write_burst_length = 64 num_write_outstanding = 8
#pragma HLS INTERFACE s_axilite port = return
#pragma HLS INTERFACE s_axilite port = size_n
#pragma HLS INTERFACE s_axilite port = size_k
#pragma HLS INTERFACE s_axilite port = size_m
#pragma HLS INTERFACE s_axilite port = layer_idx
#pragma HLS INTERFACE s_axilite port = enable_gelu

#pragma HLS DATAFLOW

  Stream<ComputePackN_t, kPipeDepth> aFeed("aFeed");
  Stream<MemoryPackM_t, 2 * kOuterTileSizeMMemory * kKVector> bMemory("bMemory");
  Stream<ComputePackM_t, kPipeDepth> bFeed("bFeed");
  Stream<AccPack_t, kPipeDepth> cMerged("cMerged");

  // 淇敼 cMemory 鐨勪綅瀹斤紝娣卞害鍑忓崐 (鐢变簬 INT8 瀵嗗害鎻愰珮锛岀幇鍦ㄤ竴锟?? tile 浠呴渶瑕佸啓锟?? 16 锟??)
  Stream<MemoryPackM_t, 2 * kOuterTileSizeMMemory> cMemory("cMemory");

#pragma HLS STREAM variable = aFeed depth = kPipeDepth
#pragma HLS STREAM variable = bMemory depth = kPipeDepth
#pragma HLS STREAM variable = bFeed depth = kPipeDepth
#pragma HLS STREAM variable = cMerged depth = kPipeDepth
#pragma HLS STREAM variable = cMemory depth = kPipeDepth

  // 鏍稿績淇锛氬幓闄ょ被鍨嬩腑锟?? kPipeDepth 妯℃澘鍙傛暟锛屼娇锟?? pragma 鎸囧畾娣卞害
  // 杩欐牱瀹冧滑锟?? C++ 灞傞潰灏卞彉鎴愪簡鏍囧噯绫诲瀷锛孉dapter 鍙互瀹岀編鎺ユ敹
  Stream<ComputePackN_t> aPipes[kGridRows][kGridCols + 1]; // 4 * 5
#pragma HLS STREAM variable = aPipes depth = kPipeDepth

  Stream<ComputePackM_t> bPipes[kGridRows + 1][kGridCols];
#pragma HLS STREAM variable = bPipes depth = kPipeDepth

  Stream<AccPack_t> cPipes[kGridRows][kGridCols + 1];
#pragma HLS STREAM variable = cPipes depth = kPipeDepth

  HLSLIB_DATAFLOW_INIT();

  // 1. 鏁版嵁鍒嗗彂
  HLSLIB_DATAFLOW_FUNCTION(ReadATransposed, a, aFeed, size_n, size_k, size_m);
  HLSLIB_DATAFLOW_FUNCTION(FeedA_Adapter, aFeed, aPipes, size_n, size_k, size_m);

  HLSLIB_DATAFLOW_FUNCTION(ReadB, b, bMemory, size_n, size_k, size_m);
  HLSLIB_DATAFLOW_FUNCTION(ConvertWidthB, bMemory, bFeed, size_n, size_k, size_m);
  HLSLIB_DATAFLOW_FUNCTION(FeedB_Adapter, bFeed, bPipes, size_n, size_k, size_m);

  // 2. 闃靛垪渚嬪寲
  for (unsigned r = 0; r < kGridRows; ++r)
  {
#pragma HLS UNROLL
    for (unsigned c = 0; c < kGridCols; ++c)
    {
#pragma HLS UNROLL
      HLSLIB_DATAFLOW_FUNCTION(
          ProcessingElement2D,
          aPipes[r][c], aPipes[r][c + 1],
          bPipes[r][c], bPipes[r + 1][c],
          cPipes[r][c], cPipes[r][c + 1],
          r, c, size_n, size_k, size_m);
    }
  }

  // 3. 鏀堕泦缁撴灉 -> INT8 閲忓寲锟?? GELU -> 鍐欏叆鍐呭瓨
  HLSLIB_DATAFLOW_FUNCTION(CollectC_Adapter, cPipes, cMerged, size_n, size_k, size_m);

  // 鏇挎崲涓轰綘鏂板啓鐨勬ā锟??
  HLSLIB_DATAFLOW_FUNCTION(ConvertWidthC_Int8, cMerged, cMemory, size_n, size_k, size_m, layer_idx, enable_gelu);
  HLSLIB_DATAFLOW_FUNCTION(WriteC_Int8, cMemory, c, size_n, size_k, size_m);

  HLSLIB_DATAFLOW_FINALIZE();
}
/////////////////////////////////////////////////////////////////////////
