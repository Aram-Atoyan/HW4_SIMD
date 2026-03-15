#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <time.h>
#include <arm_neon.h>

typedef struct {
    int width;
    int height;
    int max;
    unsigned char *data;
} Image;

double timer() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}

Image read_ppm(char *name) {
    Image img;
    img.data = NULL;

    FILE *f = fopen(name, "rb");
    if (f == NULL) {
        printf("Cannot open file\n");
        img.width = img.height = img.max = 0;
        return img;
    }

    char type[3];
    fscanf(f, "%2s", type);

    if (type[0] != 'P' || type[1] != '6') {
        printf("Only P6 format is supported\n");
        fclose(f);
        img.width = img.height = img.max = 0;
        return img;
    }

    fscanf(f, "%d %d %d", &img.width, &img.height, &img.max);
    fgetc(f);

    int size = img.width * img.height * 3;
    img.data = malloc(size);

    if (img.data == NULL) {
        printf("Couldn't Allocate\n");
        fclose(f);
        img.width = img.height = img.max = 0;
        return img;
    }

    fread(img.data, 1, size, f);
    fclose(f);

    return img;
}

void write_ppm(char *name, Image img) {
    FILE *f = fopen(name, "wb");
    if (f == NULL) {
        printf("Cannot write file\n");
        return;
    }

    fprintf(f, "P6\n%d %d\n%d\n", img.width, img.height, img.max);
    fwrite(img.data, 1, img.width * img.height * 3, f);
    fclose(f);
}

unsigned char make_gray(unsigned char r, unsigned char g, unsigned char b) {
    return (unsigned char)((77 * r + 150 * g + 29 * b) >> 8);
}

void scalar(unsigned char *in, unsigned char *out, int pixels) {
    for (int i = 0; i < pixels; i++) {
        int j = 3 * i;

        unsigned char r = in[j];
        unsigned char g = in[j + 1];
        unsigned char b = in[j + 2];

        unsigned char gray = make_gray(r, g, b);

        out[j] = gray;
        out[j + 1] = gray;
        out[j + 2] = gray;
    }
}


void simd(unsigned char *in, unsigned char *out, int pixels) {
    int limit = pixels - (pixels % 8);

    for (int i = 0; i < limit; i += 8) {
        uint8x8x3_t rgb = vld3_u8(in + 3 * i);

        unsigned char r_arr[8];
        unsigned char g_arr[8];
        unsigned char b_arr[8];
        unsigned char gray_arr[8];

        vst1_u8(r_arr, rgb.val[0]);
        vst1_u8(g_arr, rgb.val[1]);
        vst1_u8(b_arr, rgb.val[2]);

        for (int k = 0; k < 8; k++) {
            gray_arr[k] = make_gray(r_arr[k], g_arr[k], b_arr[k]);
        }

        uint8x8_t gray = vld1_u8(gray_arr);

        uint8x8x3_t out_rgb;
        out_rgb.val[0] = gray;
        out_rgb.val[1] = gray;
        out_rgb.val[2] = gray;

        vst3_u8(out + 3 * i, out_rgb);
    }

    for (int i = limit; i < pixels; i++) {
        int j = 3 * i;

        unsigned char r = in[j];
        unsigned char g = in[j + 1];
        unsigned char b = in[j + 2];

        unsigned char gray = make_gray(r, g, b);

        out[j] = gray;
        out[j + 1] = gray;
        out[j + 2] = gray;
    }
}

typedef struct {
    unsigned char *in;
    unsigned char *out;
    int start;
    int end;
} DataThread;

void *thread_function(void *arg) {
    DataThread *p = arg;
    scalar(p->in + 3 * p->start, p->out + 3 * p->start, p->end - p->start);
    return NULL;
}

void threads_only(unsigned char *in, unsigned char *out, int pixels, int th) {
    pthread_t tid[th];
    DataThread chunk[th];

    int size = pixels / th;

    for (int i = 0; i < th; i++) {
        chunk[i].in = in;
        chunk[i].out = out;
        chunk[i].start = i * size;

        if (i == th - 1) {
            chunk[i].end = pixels;
        }
        else {
            chunk[i].end = (i + 1) * size;
        }

        pthread_create(&tid[i], NULL, thread_function, &chunk[i]);
    }

    for (int i = 0; i < th; i++) {
        pthread_join(tid[i], NULL);
    }
}

typedef struct {
    unsigned char *in;
    unsigned char *out;
    int start;
    int end;
} DataThreadSIMD;

void *thread_simd(void *arg) {
    DataThreadSIMD *p = arg;
    simd(p->in + 3 * p->start, p->out + 3 * p->start, p->end - p->start);
    return NULL;
}

void simd_plus_threads(unsigned char *in, unsigned char *out, int pixels, int th) {
    pthread_t tid[th];
    DataThreadSIMD chunk[th];

    int size = pixels / th;

    for (int i = 0; i < th; i++) {
        chunk[i].in = in;
        chunk[i].out = out;
        chunk[i].start = i * size;

        if (i == th - 1) {
            chunk[i].end = pixels;
        }
        else {
            chunk[i].end = (i + 1) * size;
        }

        pthread_create(&tid[i], NULL, thread_simd, &chunk[i]);
    }

    for (int i = 0; i < th; i++) {
        pthread_join(tid[i], NULL);
    }
}

int same(unsigned char *a, unsigned char *b, int n) {
    for (int i = 0; i < n; i++) {
        if (a[i] != b[i]) return 0;
    }
    return 1;
}

int main() {
    char input_file[] = "input.ppm";
    int threads = 8;

    Image img = read_ppm(input_file);
    if (img.data == NULL) {
        return 1;
    }

    int pixels = img.width * img.height;
    int bytes = pixels * 3;

    unsigned char *out1 = malloc(bytes);
    unsigned char *out2 = malloc(bytes);
    unsigned char *out3 = malloc(bytes);
    unsigned char *out4 = malloc(bytes);

    if (out1 == NULL || out2 == NULL || out3 == NULL || out4 == NULL) {
        printf("Couldn't Allocate\n");
        free(img.data);
        return 1;
    }

    double start_timer, end_timer;

    start_timer = timer();
    scalar(img.data, out1, pixels);
    end_timer = timer();
    double time1 = end_timer - start_timer;

    start_timer = timer();
    simd(img.data, out2, pixels);
    end_timer = timer();
    double time2 = end_timer - start_timer;

    start_timer = timer();
    threads_only(img.data, out3, pixels, threads);
    end_timer = timer();
    double time3 = end_timer - start_timer;

    start_timer = timer();
    simd_plus_threads(img.data, out4, pixels, threads);
    end_timer = timer();
    double time4 = end_timer - start_timer;

    int ok1 = same(out1, out2, bytes);
    int ok2 = same(out1, out3, bytes);
    int ok3 = same(out1, out4, bytes);

    Image result;
    result.width = img.width;
    result.height = img.height;
    result.max = img.max;
    result.data = out1;

    write_ppm("gray_output.ppm", result);

    printf("Image size: %d x %d\n", img.width, img.height);
    printf("Threads used: %d\n\n", threads);

    printf("Scalar time: %.6f sec\n", time1);
    printf("SIMD time: %.6f sec\n", time2);
    printf("Multithreading time: %.6f sec\n", time3);
    printf("Multithreading + SIMD time: %.6f sec\n\n", time4);

    if (ok1) {
        printf("Scalar vs SIMD: OK\n");
    }
    else {
        printf("Scalar vs SIMD: FAIL\n");
    }

    if (ok2) {
        printf("Scalar vs Threads: OK\n");
    }
    else {
        printf("Scalar vs Threads: FAIL\n");
    }

    if (ok3) {
        printf("Scalar vs SIMD+Threads: OK\n");
    }
    else {
        printf("Scalar vs SIMD+Threads: FAIL\n");
    }

    if (ok1 && ok2 && ok3) {
        printf("Verification: PASSED\n");
    }
    else {
        printf("Verification: FAILED\n");
    }

    printf("Output image: gray_output.ppm\n");

    free(img.data);
    free(out1);
    free(out2);
    free(out3);
    free(out4);

    return 0;
}