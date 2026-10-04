#pragma once

#include "Int8Gemm.h"

void ReadATransposed(MemoryPackN_t const memory[], Stream<ComputePackN_t> &pipe,
                     const unsigned size_n, const unsigned size_k,
                     const unsigned size_m);

void ReadB(MemoryPackM_t const memory[], Stream<MemoryPackM_t> &pipe,
           const unsigned size_n, const unsigned size_k,
           const unsigned size_m);

void ConvertWidthB(Stream<MemoryPackM_t> &wide, Stream<ComputePackM_t> &narrow,
                   const unsigned size_n, const unsigned size_k,
                   const unsigned size_m);

void FeedB(Stream<ComputePackM_t> &fromMemory, Stream<ComputePackM_t> &toKernel,
           const unsigned size_n, const unsigned size_k,
           const unsigned size_m);

void ConvertWidthC(Stream<AccPack_t> &narrow, Stream<MemoryPackC_t> &wide,
                   const unsigned size_n, const unsigned size_k,
                   const unsigned size_m);

void WriteC(Stream<MemoryPackC_t> &pipe, MemoryPackC_t memory[],
            const unsigned size_n, const unsigned size_k,
            const unsigned size_m);

void ConvertWidthC_Int8(Stream<AccPack_t> &narrow, Stream<MemoryPackM_t> &wide,
                        const unsigned size_n, const unsigned size_k,
                        const unsigned size_m, const unsigned layer_idx,
                        const bool enable_gelu);

void WriteC_Int8(Stream<MemoryPackM_t> &pipe, MemoryPackM_t memory[],
                 const unsigned size_n, const unsigned size_k,
                 const unsigned size_m);
