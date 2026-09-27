//go:build arm64 && cgo
// +build arm64,cgo

#include <stdint.h>
#include <string.h>
#include <arm_neon.h>

/*
 * ARM64/NEON implementation of Zahak's NNUE hot paths.
 *
 * The x86-64 build has hand-written AVX assembly for these operations.
 * Android ARM64 used to fall back to the scalar Go implementation because
 * there was no corresponding arm64 implementation.  AArch64 guarantees
 * Advanced SIMD (NEON), so these paths are safe on every arm64 Android CPU.
 */

void zahak_update_hidden(
    const float *previous_outputs,
    const int16_t *update_indices,
    const int8_t *update_coeffs,
    int update_size,
    const float *weights,
    float *outputs,
    int outputs_len)
{
    memcpy(outputs, previous_outputs, (size_t)outputs_len * sizeof(float));

    for (int i = 0; i < update_size; ++i) {
        const int index = (int)update_indices[i];
        const float coeff = (float)update_coeffs[i];
        const float *w = weights + (size_t)index * (size_t)outputs_len;
        const float32x4_t coeff_vec = vdupq_n_f32(coeff);

        int j = 0;
        for (; j + 16 <= outputs_len; j += 16) {
            float32x4_t o0 = vld1q_f32(outputs + j);
            float32x4_t o1 = vld1q_f32(outputs + j + 4);
            float32x4_t o2 = vld1q_f32(outputs + j + 8);
            float32x4_t o3 = vld1q_f32(outputs + j + 12);

            o0 = vaddq_f32(o0, vmulq_f32(coeff_vec, vld1q_f32(w + j)));
            o1 = vaddq_f32(o1, vmulq_f32(coeff_vec, vld1q_f32(w + j + 4)));
            o2 = vaddq_f32(o2, vmulq_f32(coeff_vec, vld1q_f32(w + j + 8)));
            o3 = vaddq_f32(o3, vmulq_f32(coeff_vec, vld1q_f32(w + j + 12)));

            vst1q_f32(outputs + j, o0);
            vst1q_f32(outputs + j + 4, o1);
            vst1q_f32(outputs + j + 8, o2);
            vst1q_f32(outputs + j + 12, o3);
        }

        for (; j + 4 <= outputs_len; j += 4) {
            float32x4_t o = vld1q_f32(outputs + j);
            o = vaddq_f32(o, vmulq_f32(coeff_vec, vld1q_f32(w + j)));
            vst1q_f32(outputs + j, o);
        }

        for (; j < outputs_len; ++j) {
            outputs[j] += coeff * w[j];
        }
    }
}

float zahak_quick_feed(
    const float *hidden_outputs,
    int hidden_outputs_len,
    const float *weights,
    int weights_len)
{
    const float32x4_t zero = vdupq_n_f32(0.0f);
    float32x4_t sum = vdupq_n_f32(0.0f);

    int i = 0;
    const int limit = weights_len < hidden_outputs_len
        ? weights_len
        : hidden_outputs_len;

    for (; i + 16 <= limit; i += 16) {
        float32x4_t h0 = vmaxq_f32(vld1q_f32(hidden_outputs + i), zero);
        float32x4_t h1 = vmaxq_f32(vld1q_f32(hidden_outputs + i + 4), zero);
        float32x4_t h2 = vmaxq_f32(vld1q_f32(hidden_outputs + i + 8), zero);
        float32x4_t h3 = vmaxq_f32(vld1q_f32(hidden_outputs + i + 12), zero);

        sum = vaddq_f32(sum, vmulq_f32(h0, vld1q_f32(weights + i)));
        sum = vaddq_f32(sum, vmulq_f32(h1, vld1q_f32(weights + i + 4)));
        sum = vaddq_f32(sum, vmulq_f32(h2, vld1q_f32(weights + i + 8)));
        sum = vaddq_f32(sum, vmulq_f32(h3, vld1q_f32(weights + i + 12)));
    }

    for (; i + 4 <= limit; i += 4) {
        float32x4_t h = vmaxq_f32(vld1q_f32(hidden_outputs + i), zero);
        sum = vaddq_f32(sum, vmulq_f32(h, vld1q_f32(weights + i)));
    }

    float result = vaddvq_f32(sum);

    for (; i < limit; ++i) {
        const float h = hidden_outputs[i] < 0.0f ? 0.0f : hidden_outputs[i];
        result += h * weights[i];
    }

    return result;
}
