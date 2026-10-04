#pragma once

#include "Int8Gemm.h"

void ProcessingElement(Stream<ComputePackN_t> &aIn,
                       Stream<ComputePackN_t> &aOut,
                       Stream<ComputePackM_t> &bIn,
                       Stream<ComputePackM_t> &bOut,
                       Stream<AccPack_t> &cOut,
                       Stream<AccPack_t> &cIn,
                       const unsigned locationN,
                       const unsigned size_n,
                       const unsigned size_k,
                       const unsigned size_m);

// zzy 修改 2D
// --- 添加到 Compute.h 中 ---

// 声明 2D PE 阵列
void ProcessingElement2D(
    Stream<ComputePackN_t> &aIn, Stream<ComputePackN_t> &aOut,
    Stream<ComputePackM_t> &bIn, Stream<ComputePackM_t> &bOut,
    Stream<AccPack_t> &cIn, Stream<AccPack_t> &cOut,
    const unsigned r, const unsigned c,
    const unsigned size_n, const unsigned size_k, const unsigned size_m);

// 声明 3 个适配器（注意这里直接接收 2D 数组）
void FeedA_Adapter(Stream<ComputePackN_t> &aIn, Stream<ComputePackN_t> aPipes[kGridRows][kGridCols + 1],
                   const unsigned size_n, const unsigned size_k, const unsigned size_m);

void FeedB_Adapter(Stream<ComputePackM_t> &bIn, Stream<ComputePackM_t> bPipes[kGridRows + 1][kGridCols],
                   const unsigned size_n, const unsigned size_k, const unsigned size_m);

void CollectC_Adapter(Stream<AccPack_t> cPipes[kGridRows][kGridCols + 1], Stream<AccPack_t> &cOut,
                      const unsigned size_n, const unsigned size_k, const unsigned size_m);