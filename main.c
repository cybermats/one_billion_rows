#define _GNU_SOURCE
#include <sched.h>

#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <unistd.h>
#include <threads.h>
#include <stdatomic.h>
#include <emmintrin.h>

#define WINDOW_SIZE (1 * 1024 * 1024)

#define handle_error(msg) \
    do { perror(msg); exit(EXIT_FAILURE); } while (0)

int get_core_count() {
#ifdef NDEBUG
    
    cpu_set_t cpuset;
    sched_getaffinity(0, sizeof(cpuset), &cpuset);

    return CPU_COUNT(&cpuset);
#else
    return 1;
#endif
}
struct thread_data_t {
    char* addr;
    size_t length;
    atomic_int * window;
    atomic_int * line_count;
};

int process_file(void* thread_data_arg) {
    size_t counts = 0;
    size_t window_count = 0;

    struct thread_data_t* thread_data = thread_data_arg;
    const __m128i new_line = _mm_set1_epi8('\n');
    while(1) {
        ++window_count;
        size_t window = ++*thread_data->window;
        size_t start_idx = WINDOW_SIZE * window;

        if (start_idx >= thread_data->length)
            break;
        size_t end_idx = start_idx + WINDOW_SIZE;
        if (end_idx >= thread_data->length)
            end_idx = thread_data->length;
        
        char* curr = thread_data->addr + start_idx;
        struct foobar {
            long a;
            long b;
        } line_count;
        __m128i lc = _mm_setzero_si128();

        const int line_size = 16;
        const int max_lines = 85; // 85
        while (curr + (line_size * max_lines) < thread_data->addr + end_idx) {
            __m128i acc = _mm_setzero_si128();
            for (char l = 0; l < max_lines; ++l) {
                const __m128i vec = _mm_load_si128((__m128i const*)curr);
                const __m128i val = _mm_cmpeq_epi8(vec, new_line);
                acc = _mm_sub_epi8(acc, val);
                curr += line_size;
                ++counts;
            }
            __m128i agg = _mm_sad_epu8(acc, _mm_setzero_si128());
            lc = _mm_add_epi64(lc, agg);
        }
        _mm_storeu_si128( ( __m128i* )&line_count, lc);

        for (;curr < thread_data->addr + end_idx; ++curr) {
            if (*curr == '\n')
                ++line_count.a;
        }
        
        *thread_data->line_count += line_count.a + line_count.b;
    }
    return 0;
}

int main(int argc, char *argv[]) {
    
    if (argc < 2) {
        fprintf(stderr, "%s file\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    int fd = open(argv[1], O_RDONLY);
    if (fd == -1)
        handle_error("open");

    struct stat sb;
    if (fstat(fd, &sb) == -1)
        handle_error("fstat");

    char *addr = mmap(NULL, sb.st_size,
        PROT_READ, MAP_PRIVATE, fd, 0);
    if (addr == MAP_FAILED)
        handle_error("mmap");

    int thread_count = get_core_count();
    printf("Threads: %d\n", thread_count);

    thrd_t* thrds = malloc(sizeof(thrd_t) * thread_count);
    if (thrds == NULL)
        handle_error("malloc");


    atomic_int window = -1;
    atomic_int line_count = 0;

    struct thread_data_t thrd_arg = {addr, sb.st_size, &window, &line_count};
    
    for (int i = 0; i < thread_count; ++i) {
        thrd_create(&thrds[i], process_file, &thrd_arg);
    }

    for (int i = 0; i < thread_count; ++i)
        thrd_join(thrds[i], NULL);

    int lc = line_count;
    printf("Line count: %d\n", lc);

    free(thrds);
    
    exit(EXIT_SUCCESS);
    
}
