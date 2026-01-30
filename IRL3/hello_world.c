// File to prepare at home. (Making sure SIMD instructions are available, and
// you can use them on your machine before the session.

#include <stdio.h>
#include <stdlib.h>

#ifdef __arm__
#include <arm_neon.h>
#elif defined(__x86_64__)
#include <immintrin.h>
#include <emmintrin.h>
#else
#error "Unsupported Architecture"
#endif

int main () {
    const int N = 64;
    float *array;
    if (posix_memalign((void **)&array, 64, sizeof(float)*N) != 0) {
        perror("Memory allocaiton failed:");
        exit(1);
    }

    printf("Filling an array of %d floating point elements\n", N);

    for (int i = 0; i < N; i++) {
        array[i] = i;
    }

    printf("Array content before: [");
    for (int i = 0; i < N; i++) {
        printf("%3.1f%s", array[i], i==(N-1) ? "]\n":", ");
    }

    printf("Now incrementing each value of the element by one!\n");

#ifdef __arm__
    float32x4_t all_ones = vdupq_n_f32(1.0);
    for (int i = 0; i < N; i+= 4) {
        float32x4_t values = vld1q_f32(&array[i]);
        values = vaddq_f32(all_ones, values);
        vst1q_f32(&array[i], values);
    }
#elif defined(__x86_64__)
    __m128 all_ones = _mm_set1_ps(1.0);
    for (int i = 0; i < N; i += 4) {
        __m128 values = _mm_load_ps(&array[i]);
        values = _mm_add_ps(values, all_ones);
        _mm_store_ps(&array[i], values);
    }
#endif
    printf("Array content after: [");
    for (int i = 0; i < N; i++) {
        printf("%3.1f%s", array[i], i==(N-1) ? "]\n":", ");
    }

    return 0;
}
