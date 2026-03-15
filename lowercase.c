#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <time.h>
#include <arm_neon.h>

double timer() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}

char random_symbol() {
    int x = rand() % 5;

    if (x == 0) return 'a' + rand() % 26;     
    if (x == 1) return 'A' + rand() % 26;     
    if (x == 2) return '0' + rand() % 10;     
    if (x == 3) return '!' + rand() % 15;     
    return ' ';                               
}

void fill_buffer(char *buffer, int n) {
    for (int i = 0; i < n; i++) {
        buffer[i] = random_symbol();
    }
}

void copy_buffer(char *a, char *b, int n) {
    for (int i = 0; i < n; i++) {
        b[i] = a[i];
    }
}


typedef struct {
    char *buffer;
    int start;
    int end;
} DataThread;

void *thread_function(void *arg) {
    DataThread *p = arg;

    for (int i = p->start; i < p->end; i++) {
        if (p->buffer[i] >= 'a' && p->buffer[i] <= 'z') {
            p->buffer[i] = p->buffer[i] - 32;
        }
    }

    return NULL;
}

void threads_only(char *buffer, int n, int th) {
    pthread_t tid[th];
    DataThread chunk[th];

    int size = n / th;

    for (int i = 0; i < th; i++) {
        chunk[i].buffer = buffer;
        chunk[i].start = i * size;

        if (i == th - 1) {
            chunk[i].end = n;
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


void simd(char *buffer, int n) {
    uint8x16_t a = vdupq_n_u8('a');
    uint8x16_t z = vdupq_n_u8('z');
    uint8x16_t diff = vdupq_n_u8(32);

    int limit = n - (n % 16);

    for (int i = 0; i < limit; i += 16) {
        uint8x16_t v = vld1q_u8((uint8_t *)(buffer + i));

        uint8x16_t ge_a = vcgeq_u8(v, a);
        uint8x16_t le_z = vcleq_u8(v, z);
        uint8x16_t mask = vandq_u8(ge_a, le_z);

        uint8x16_t sub = vandq_u8(mask, diff);
        v = vsubq_u8(v, sub);

        vst1q_u8((uint8_t *)(buffer + i), v);
    }

    for (int i = limit; i < n; i++) {
        if (buffer[i] >= 'a' && buffer[i] <= 'z') {
            buffer[i] = buffer[i] - 32;
        }
    }
}


typedef struct {
    char *buffer;
    int start;
    int end;
} DataThreadSIMD;

void *thread_simd(void *arg) {
    DataThreadSIMD *p = arg;
    simd(p->buffer + p->start, p->end - p->start);
    return NULL;
}

void simd_plus_threads(char *buffer, int n, int th) {
    pthread_t tid[th];
    DataThreadSIMD chunk[th];

    int size = n / th;

    for (int i = 0; i < th; i++) {
        chunk[i].buffer = buffer;
        chunk[i].start = i * size;

        if (i == th - 1) {
            chunk[i].end = n;
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

int main() {
    int mb = 256;
    int threads = 8;

    int n = mb * 1024 * 1024;

    char *buffer1 = malloc(n);
    char *buffer2 = malloc(n);
    char *buffer3 = malloc(n);

    if (buffer1 == NULL || buffer2 == NULL || buffer3 == NULL) {
        printf("Couldn't Allocate\n");
        return 1;
    }

    fill_buffer(buffer1, n);
    copy_buffer(buffer1, buffer2, n);
    copy_buffer(buffer1, buffer3, n);

    double start_timer, end_timer;

    start_timer = timer();
    threads_only(buffer1, n, threads);
    end_timer = timer();
    double time1 = end_timer - start_timer;

    start_timer = timer();
    simd(buffer2, n);
    end_timer = timer();
    double time2 = end_timer - start_timer;

    start_timer = timer();
    simd_plus_threads(buffer3, n, threads);
    end_timer = timer();
    double time3 = end_timer - start_timer;

    printf("Buffer size: %d MB\n", mb);
    printf("Threads used: %d\n\n", threads);

    printf("Multithreading time: %.6f sec\n", time1);
    printf("SIMD time: %.6f sec\n", time2);
    printf("SIMD + Multithreading: %.6f sec\n", time3);

    free(buffer1);
    free(buffer2);
    free(buffer3);
    return 0;
}