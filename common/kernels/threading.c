/*
 * threading.c - Thread pool for smol-genius inference kernels
 * Extracted from qwen_asr_kernels.c, prefixes renamed qwen_->smol_, QWEN_->SMOL_.
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#ifdef __APPLE__
#include <sys/sysctl.h>
#else
#include <unistd.h>
#endif

/* ========================================================================
 * Thread Pool
 * ======================================================================== */

int smol_verbose = 0;

#define SMOL_MAX_THREADS 16

typedef void (*parallel_fn_t)(int tid, int n_threads, void *arg);

static struct {
    pthread_t threads[SMOL_MAX_THREADS - 1];
    int tids[SMOL_MAX_THREADS - 1];
    int n_threads;
    int shutdown;

    parallel_fn_t fn;
    void *arg;
    int generation;

    pthread_mutex_t mutex;
    pthread_cond_t cond_work;
    pthread_cond_t cond_done;
    int n_done;
} tp = {
    .n_threads = 1,
    .shutdown = 0,
    .generation = 0,
    .mutex = PTHREAD_MUTEX_INITIALIZER,
    .cond_work = PTHREAD_COND_INITIALIZER,
    .cond_done = PTHREAD_COND_INITIALIZER,
};

static void *worker_loop(void *arg) {
    int tid = *(int *)arg;
    int my_gen = 0;

    for (;;) {
        pthread_mutex_lock(&tp.mutex);
        while (tp.generation == my_gen && !tp.shutdown)
            pthread_cond_wait(&tp.cond_work, &tp.mutex);
        if (tp.shutdown) {
            pthread_mutex_unlock(&tp.mutex);
            return NULL;
        }
        my_gen = tp.generation;
        parallel_fn_t fn = tp.fn;
        void *a = tp.arg;
        int nt = tp.n_threads;
        pthread_mutex_unlock(&tp.mutex);

        fn(tid, nt, a);

        pthread_mutex_lock(&tp.mutex);
        if (++tp.n_done >= tp.n_threads - 1)
            pthread_cond_signal(&tp.cond_done);
        pthread_mutex_unlock(&tp.mutex);
    }
}

void smol_set_threads(int n) {
    if (n < 1) n = 1;
    if (n > SMOL_MAX_THREADS) n = SMOL_MAX_THREADS;

    /* Shutdown existing workers */
    if (tp.n_threads > 1) {
        pthread_mutex_lock(&tp.mutex);
        tp.shutdown = 1;
        pthread_cond_broadcast(&tp.cond_work);
        pthread_mutex_unlock(&tp.mutex);
        for (int i = 0; i < tp.n_threads - 1; i++)
            pthread_join(tp.threads[i], NULL);
        tp.shutdown = 0;
        tp.generation = 0;
    }

    tp.n_threads = n;
    if (n <= 1) return;

    for (int i = 0; i < n - 1; i++) {
        tp.tids[i] = i + 1;
        pthread_create(&tp.threads[i], NULL, worker_loop, &tp.tids[i]);
    }

    if (smol_verbose >= 2)
        fprintf(stderr, "Thread pool: %d threads\n", n);
}

int smol_get_num_cpus(void) {
#ifdef __APPLE__
    int n = 0;
    size_t len = sizeof(n);
    sysctlbyname("hw.ncpu", &n, &len, NULL, 0);
    return n > 0 ? n : 1;
#else
    int n = (int)sysconf(_SC_NPROCESSORS_ONLN);
    return n > 0 ? n : 1;
#endif
}

int smol_get_thread_count(void) {
    return tp.n_threads;
}

/* Dispatch work to all threads; main thread is tid=0 */
void smol_parallel_for(void (*fn)(int tid, int n_threads, void *arg), void *arg) {
    if (tp.n_threads <= 1) {
        fn(0, 1, arg);
        return;
    }

    pthread_mutex_lock(&tp.mutex);
    tp.fn = fn;
    tp.arg = arg;
    tp.n_done = 0;
    tp.generation++;
    pthread_cond_broadcast(&tp.cond_work);
    pthread_mutex_unlock(&tp.mutex);

    fn(0, tp.n_threads, arg);

    pthread_mutex_lock(&tp.mutex);
    while (tp.n_done < tp.n_threads - 1)
        pthread_cond_wait(&tp.cond_done, &tp.mutex);
    pthread_mutex_unlock(&tp.mutex);
}
