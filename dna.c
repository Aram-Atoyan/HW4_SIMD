#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <time.h>
#include <arm_neon.h>

typedef struct {
    long long a, c, g, t;
} Count;

Count global = {0, 0, 0, 0};
pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;

double timer() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}

void generate_dna(char *buffer, int n) {
    char arr[4] = {'A', 'C', 'G', 'T'};
    for (int i = 0; i < n; i++) {
        buffer[i] = arr[rand() % 4];
    }
}

Count scalar_method(char *buffer, int n) {
    Count x = {0, 0, 0, 0};

    for (int i = 0; i < n; i++) {
        if (buffer[i] == 'A') x.a++;
        else if (buffer[i] == 'C') x.c++;
        else if (buffer[i] == 'G') x.g++;
        else if (buffer[i] == 'T') x.t++;
    }

    return x;
}


typedef struct {
    char *dna;
    int start;
    int end;
} DataThread;

void *thread_function(void *arg) {
    DataThread *p = arg;
    Count local = {0, 0, 0, 0};

    for (int i = p->start; i < p->end; i++) {
        if (p->dna[i] == 'A') local.a++;
        else if (p->dna[i] == 'C') local.c++;
        else if (p->dna[i] == 'G') local.g++;
        else if (p->dna[i] == 'T') local.t++;
    }

    pthread_mutex_lock(&lock);
    global.a += local.a;
    global.c += local.c;
    global.g += local.g;
    global.t += local.t;
    pthread_mutex_unlock(&lock);

    return NULL;
}

Count threads_global_count(char *dna, int n, int th) {
    pthread_t tid[th];
    DataThread chunk[th];

    global.a = global.c = global.g = global.t = 0;

    int size = n / th;

    for (int i = 0; i < th; i++) {
        chunk[i].dna = dna;
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

    return global;
}


Count simd(char *dna, int n) {
    Count x = {0, 0, 0, 0};

    uint8x16_t A = vdupq_n_u8('A');
    uint8x16_t C = vdupq_n_u8('C');
    uint8x16_t G = vdupq_n_u8('G');
    uint8x16_t T = vdupq_n_u8('T');
    uint8x16_t one = vdupq_n_u8(1);

    int limit = n - (n % 16);

    for (int i = 0; i < limit; i += 16) {
        uint8x16_t v = vld1q_u8((uint8_t *)(dna + i));

        uint8x16_t ma = vandq_u8(vceqq_u8(v, A), one);
        uint8x16_t mc = vandq_u8(vceqq_u8(v, C), one);
        uint8x16_t mg = vandq_u8(vceqq_u8(v, G), one);
        uint8x16_t mt = vandq_u8(vceqq_u8(v, T), one);

        x.a += vaddvq_u8(ma);
        x.c += vaddvq_u8(mc);
        x.g += vaddvq_u8(mg);
        x.t += vaddvq_u8(mt);
    }

    for (int i = limit; i < n; i++) {
        if (dna[i] == 'A') x.a++;
        else if (dna[i] == 'C') x.c++;
        else if (dna[i] == 'G') x.g++;
        else if (dna[i] == 'T') x.t++;
    }

    return x;
}



typedef struct {
    char *dna;
    int start;
    int end;
    Count ans;
} DataThreadPlusSIMD;

void *thread_simd(void *arg) {
    DataThreadPlusSIMD *p = arg;
    p->ans = simd(p->dna + p->start, p->end - p->start);
    return NULL;
}

Count simd_plus_threads(char *dna, int n, int th) {
    pthread_t tid[th];
    DataThreadPlusSIMD chunk[th];
    Count total = {0, 0, 0, 0};

    int size = n / th;

    for (int i = 0; i < th; i++) {
        chunk[i].dna = dna;
        chunk[i].start = i * size;

        if (i == th - 1) chunk[i].end = n;
        else chunk[i].end = (i + 1) * size;

        chunk[i].ans.a = 0;
        chunk[i].ans.c = 0;
        chunk[i].ans.g = 0;
        chunk[i].ans.t = 0;

        pthread_create(&tid[i], NULL, thread_simd, &chunk[i]);
    }

    for (int i = 0; i < th; i++) {
        pthread_join(tid[i], NULL);
        total.a += chunk[i].ans.a;
        total.c += chunk[i].ans.c;
        total.g += chunk[i].ans.g;
        total.t += chunk[i].ans.t;
    }

    return total;
}

int same(Count x, Count y) {
    return x.a == y.a && x.c == y.c && x.g == y.g && x.t == y.t;
}


void print_count(Count x) {
    printf("%lld %lld %lld %lld\n", x.a, x.c, x.g, x.t);
}

int main() {
    int mb = 512;
    int threads = 8;

    int n = mb * 1024 * 1024;

    char *dna = malloc(n);
    if (dna == NULL) {
        printf("couldn't allocate\n");
        return 1;
    }

    generate_dna(dna, n);

    double start_timer, end_timer;

    Count scalar, onlyThread, onlySIMD, threadsAndSIMD;

    start_timer = timer();
    scalar = scalar_method(dna, n);
    end_timer = timer();
    double t1 = end_timer - start_timer;

    start_timer = timer();
    onlyThread = threads_global_count(dna, n, threads);
    end_timer = timer();
    double t2 = end_timer - start_timer;

    start_timer = timer();
    onlySIMD = simd(dna, n);
    end_timer = timer();
    double t3 = end_timer - start_timer;

    start_timer = timer();
    threadsAndSIMD = simd_plus_threads(dna, n, threads);
    end_timer = timer();
    double t4 = end_timer - start_timer;

    printf("DNA size: %d MB\n", mb);
    printf("Threads used: %d\n\n", threads);

    printf("Counts (A C G T):\n");
    print_count(scalar);

    printf("\nCheck:\n");
    printf("Scalar vs Threads: %s\n", same(scalar, onlyThread) ? "Same" : "Not Same");
    printf("Scalar vs SIMD: %s\n", same(scalar, onlySIMD) ? "Same" : "Not Same");
    printf("Scalar vs SIMD+Threads: %s\n", same(scalar, threadsAndSIMD) ? "Same" : "Not Same");

    printf("\nScalar time: %.6f sec\n", t1);
    printf("Threads time: %.6f sec\n", t2);
    printf("SIMD time: %.6f sec\n", t3);
    printf("SIMD + Threads time: %.6f sec\n", t4);

    free(dna);
    return 0;
}