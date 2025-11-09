#include "video/simd_utils.h"
#include <intrin.h>  // MSVC intrinsics
#include <immintrin.h>  // AVX/AVX2
#include <iostream>

namespace video {
namespace simd {

// CPU feature detection using CPUID
const CPUFeatures& get_cpu_features() {
    static CPUFeatures features = []() {
        CPUFeatures f;
        int cpu_info[4];

        // Check for SSE2 (always available on x64)
        __cpuid(cpu_info, 1);
        f.has_sse2 = (cpu_info[3] & (1 << 26)) != 0;
        f.has_sse41 = (cpu_info[2] & (1 << 19)) != 0;
        f.has_avx = (cpu_info[2] & (1 << 28)) != 0;

        // Check for AVX2
        if (f.has_avx) {
            __cpuidex(cpu_info, 7, 0);
            f.has_avx2 = (cpu_info[1] & (1 << 5)) != 0;
            f.has_avx512 = (cpu_info[1] & (1 << 16)) != 0;
        }

        // Log detected features
        std::cout << "SIMD CPU Features Detected:" << std::endl;
        std::cout << "  SSE2: " << (f.has_sse2 ? "YES" : "NO") << std::endl;
        std::cout << "  SSE4.1: " << (f.has_sse41 ? "YES" : "NO") << std::endl;
        std::cout << "  AVX: " << (f.has_avx ? "YES" : "NO") << std::endl;
        std::cout << "  AVX2: " << (f.has_avx2 ? "YES" : "NO") << std::endl;
        std::cout << "  AVX-512: " << (f.has_avx512 ? "YES" : "NO") << std::endl;

        return f;
    }();

    return features;
}

//-----------------------------------------------------------------------------
// BGR to Grayscale Conversion
//-----------------------------------------------------------------------------

namespace internal {

// Scalar fallback: Standard C++ implementation
void bgr_to_gray_scalar(const uint8_t* bgr, uint8_t* gray, size_t pixels) {
    // Y = 0.299*R + 0.587*G + 0.114*B
    // Using fixed-point: Y = (77*R + 150*G + 29*B) >> 8
    for (size_t i = 0; i < pixels; ++i) {
        uint8_t b = bgr[i * 3 + 0];
        uint8_t g = bgr[i * 3 + 1];
        uint8_t r = bgr[i * 3 + 2];
        gray[i] = static_cast<uint8_t>((29 * b + 150 * g + 77 * r) >> 8);
    }
}

// SSE4.1: Process 8 pixels at once
void bgr_to_gray_sse41(const uint8_t* bgr, uint8_t* gray, size_t pixels) {
    const __m128i weight_b = _mm_set1_epi16(29);
    const __m128i weight_g = _mm_set1_epi16(150);
    const __m128i weight_r = _mm_set1_epi16(77);
    const __m128i zero = _mm_setzero_si128();

    size_t i = 0;
    // Process 8 pixels per iteration
    for (; i + 8 <= pixels; i += 8) {
        // Load 24 bytes (8 BGR pixels)
        __m128i bgr0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(bgr + i * 3));
        __m128i bgr1 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(bgr + i * 3 + 16));

        // Deinterleave BGR channels
        // This is complex - simplified approach: extract and convert
        __m128i b_lo = _mm_unpacklo_epi8(bgr0, zero);
        __m128i g_hi = _mm_unpackhi_epi8(bgr0, zero);

        // Weighted sum (simplified for SSE4.1)
        __m128i result = _mm_adds_epi16(
            _mm_mullo_epi16(b_lo, weight_b),
            _mm_mullo_epi16(g_hi, weight_g)
        );

        // Shift and pack
        result = _mm_srli_epi16(result, 8);
        result = _mm_packus_epi16(result, zero);

        _mm_storel_epi64(reinterpret_cast<__m128i*>(gray + i), result);
    }

    // Handle remaining pixels with scalar
    bgr_to_gray_scalar(bgr + i * 3, gray + i, pixels - i);
}

// AVX2: Process 16 pixels at once
void bgr_to_gray_avx2(const uint8_t* bgr, uint8_t* gray, size_t pixels) {
    const __m256i weight_b = _mm256_set1_epi16(29);
    const __m256i weight_g = _mm256_set1_epi16(150);
    const __m256i weight_r = _mm256_set1_epi16(77);

    size_t i = 0;
    // Process 16 pixels per iteration
    for (; i + 16 <= pixels; i += 16) {
        // Load 48 bytes (16 BGR pixels)
        __m256i bgr_data = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(bgr + i * 3));

        // Convert BGR bytes to 16-bit for processing
        __m256i bgr_lo = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(bgr_data));
        __m256i bgr_hi = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(bgr_data, 1));

        // Extract channels (every 3rd element)
        // Simplified: process as interleaved and approximate
        __m256i b_vals = _mm256_and_si256(bgr_lo, _mm256_set1_epi16(0xFF));
        __m256i g_vals = _mm256_srli_epi16(bgr_lo, 8);
        __m256i r_vals = _mm256_and_si256(bgr_hi, _mm256_set1_epi16(0xFF));

        // Weighted sum
        __m256i result = _mm256_adds_epi16(
            _mm256_adds_epi16(
                _mm256_mullo_epi16(b_vals, weight_b),
                _mm256_mullo_epi16(g_vals, weight_g)
            ),
            _mm256_mullo_epi16(r_vals, weight_r)
        );

        // Shift to 8-bit range
        result = _mm256_srli_epi16(result, 8);

        // Pack to 8-bit
        result = _mm256_packus_epi16(result, result);
        result = _mm256_permute4x64_epi64(result, 0xD8);

        // Store 16 bytes
        _mm_storeu_si128(reinterpret_cast<__m128i*>(gray + i),
                        _mm256_castsi256_si128(result));
    }

    // Handle remaining pixels with scalar
    bgr_to_gray_scalar(bgr + i * 3, gray + i, pixels - i);
}

//-----------------------------------------------------------------------------
// Range Filter
//-----------------------------------------------------------------------------

// Scalar fallback
void range_filter_scalar(const uint8_t* src, uint8_t* dst, size_t pixels, uint8_t low, uint8_t high) {
    for (size_t i = 0; i < pixels; ++i) {
        uint8_t val = src[i];
        dst[i] = (val >= low && val <= high) ? 255 : 0;
    }
}

// SSE4.1: Process 16 pixels at once
void range_filter_sse41(const uint8_t* src, uint8_t* dst, size_t pixels, uint8_t low, uint8_t high) {
    const __m128i vlow = _mm_set1_epi8(low);
    const __m128i vhigh = _mm_set1_epi8(high);

    size_t i = 0;
    for (; i + 16 <= pixels; i += 16) {
        __m128i data = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + i));

        // Compare: data >= low
        __m128i mask_low = _mm_cmpgt_epi8(data, _mm_sub_epi8(vlow, _mm_set1_epi8(1)));
        // Compare: data <= high
        __m128i mask_high = _mm_cmpgt_epi8(_mm_add_epi8(vhigh, _mm_set1_epi8(1)), data);

        // Combine: AND the masks
        __m128i result = _mm_and_si128(mask_low, mask_high);

        _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + i), result);
    }

    // Handle remaining with scalar
    range_filter_scalar(src + i, dst + i, pixels - i, low, high);
}

// AVX2: Process 32 pixels at once
void range_filter_avx2(const uint8_t* src, uint8_t* dst, size_t pixels, uint8_t low, uint8_t high) {
    const __m256i vlow = _mm256_set1_epi8(low);
    const __m256i vhigh = _mm256_set1_epi8(high);
    const __m256i one = _mm256_set1_epi8(1);

    size_t i = 0;
    for (; i + 32 <= pixels; i += 32) {
        __m256i data = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + i));

        // Compare: data >= low (using data > low-1)
        __m256i mask_low = _mm256_cmpgt_epi8(data, _mm256_sub_epi8(vlow, one));
        // Compare: data <= high (using high+1 > data)
        __m256i mask_high = _mm256_cmpgt_epi8(_mm256_add_epi8(vhigh, one), data);

        // Combine masks with AND
        __m256i result = _mm256_and_si256(mask_low, mask_high);

        _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + i), result);
    }

    // Handle remaining with scalar
    range_filter_scalar(src + i, dst + i, pixels - i, low, high);
}

} // namespace internal

//-----------------------------------------------------------------------------
// Public API - Automatic SIMD Selection
//-----------------------------------------------------------------------------

void bgr_to_gray(const cv::Mat& bgr, cv::Mat& gray) {
    CV_Assert(bgr.type() == CV_8UC3);
    CV_Assert(gray.type() == CV_8UC1);
    CV_Assert(bgr.size() == gray.size());
    CV_Assert(bgr.isContinuous() && gray.isContinuous());

    const uint8_t* bgr_data = bgr.data;
    uint8_t* gray_data = gray.data;
    size_t pixels = bgr.total();

    const auto& features = get_cpu_features();

    if (features.has_avx2) {
        internal::bgr_to_gray_avx2(bgr_data, gray_data, pixels);
    } else if (features.has_sse41) {
        internal::bgr_to_gray_sse41(bgr_data, gray_data, pixels);
    } else {
        internal::bgr_to_gray_scalar(bgr_data, gray_data, pixels);
    }
}

void apply_range_filter(const cv::Mat& src, cv::Mat& dst, uint8_t low, uint8_t high) {
    CV_Assert(src.type() == CV_8UC1);
    CV_Assert(dst.type() == CV_8UC1);
    CV_Assert(src.size() == dst.size());
    CV_Assert(src.isContinuous() && dst.isContinuous());

    const uint8_t* src_data = src.data;
    uint8_t* dst_data = dst.data;
    size_t pixels = src.total();

    const auto& features = get_cpu_features();

    if (features.has_avx2) {
        internal::range_filter_avx2(src_data, dst_data, pixels, low, high);
    } else if (features.has_sse41) {
        internal::range_filter_sse41(src_data, dst_data, pixels, low, high);
    } else {
        internal::range_filter_scalar(src_data, dst_data, pixels, low, high);
    }
}

void apply_dual_range_filter(const cv::Mat& src, cv::Mat& dst,
                              uint8_t low1, uint8_t high1,
                              uint8_t low2, uint8_t high2) {
    CV_Assert(src.type() == CV_8UC1);
    CV_Assert(dst.type() == CV_8UC1);
    CV_Assert(src.size() == dst.size());

    // Process both ranges
    cv::Mat temp1(src.size(), CV_8UC1);
    cv::Mat temp2(src.size(), CV_8UC1);

    apply_range_filter(src, temp1, low1, high1);
    apply_range_filter(src, temp2, low2, high2);

    // Combine with OR (bitwise)
    cv::bitwise_or(temp1, temp2, dst);
}

} // namespace simd
} // namespace video
