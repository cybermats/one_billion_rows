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
#include <time.h>

#define WINDOW_SIZE (1 * 1024 * 1024)

#define handle_error(msg) \
    do { perror(msg); exit(EXIT_FAILURE); } while (0)

int get_core_count()
{
#ifdef NDEBUG
    
    cpu_set_t cpuset;
    sched_getaffinity(0, sizeof(cpuset), &cpuset);

    return CPU_COUNT(&cpuset);
#else
    return 1;
#endif
}

struct timespec diff(struct timespec start, struct timespec end)
{
    struct timespec temp;
    if ((end.tv_nsec - start.tv_nsec) < 0)
    {
        temp.tv_sec = end.tv_sec - start.tv_sec - 1;
        temp.tv_nsec = 1000000000 + end.tv_nsec - start.tv_nsec;
    }
    else
    {
        temp.tv_sec = end.tv_sec - start.tv_sec;
        temp.tv_nsec = end.tv_nsec - start.tv_nsec;
    }
    return temp;
}


struct thread_data_t
{
    char* addr;
    size_t length;
    atomic_int* window;
    atomic_int* line_count;
};

int process_file_sse(void* thread_data_arg)
{
    size_t counts = 0;
    size_t window_count = 0;

    struct thread_data_t* thread_data = thread_data_arg;
    const __m128i new_line = _mm_set1_epi8('\n');
    while (1)
    {
        ++window_count;
        size_t window = ++*thread_data->window;
        size_t start_idx = WINDOW_SIZE * window;

        if (start_idx >= thread_data->length)
            break;
        size_t end_idx = start_idx + WINDOW_SIZE;
        if (end_idx >= thread_data->length)
            end_idx = thread_data->length;

        char* curr = thread_data->addr + start_idx;
        struct foobar
        {
            long a;
            long b;
        } line_count;
        __m128i lc = _mm_setzero_si128();

        const int line_size = 16;
        const int max_lines = 85; // 85
        while (curr + (line_size * max_lines) < thread_data->addr + end_idx)
        {
            __m128i acc = _mm_setzero_si128();
            for (char l = 0; l < max_lines; ++l)
            {
                const __m128i vec = _mm_load_si128((__m128i const*)curr);
                const __m128i val = _mm_cmpeq_epi8(vec, new_line);
                acc = _mm_sub_epi8(acc, val);
                curr += line_size;
                ++counts;
            }
            __m128i agg = _mm_sad_epu8(acc, _mm_setzero_si128());
            lc = _mm_add_epi64(lc, agg);
        }
        _mm_storeu_si128((__m128i*)&line_count, lc);

        for (; curr < thread_data->addr + end_idx; ++curr)
        {
            if (*curr == '\n')
                ++line_count.a;
        }

        *thread_data->line_count += line_count.a + line_count.b;
    }
    return 0;
}

const unsigned long BROADCAST_SEMICOLON = 0x3B3B3B3B3B3B3B3BL;
const unsigned long BROADCAST_0x01 = 0x0101010101010101L;
const unsigned long BROADCAST_0x80 = 0x8080808080808080L;
const unsigned long DOT_BITS = 0x10101000;

unsigned long semicolon_match_bits(unsigned long word)
{
    unsigned long diff = word ^ BROADCAST_SEMICOLON;
    return (diff - BROADCAST_0x01) & (~diff & BROADCAST_0x80);
}

unsigned int get_name_len(unsigned long separator)
{
    return (__builtin_ctzl(separator) >> 3) + 1;
}

int get_dot_pos(unsigned long word)
{
    return __builtin_ctzl(~word & DOT_BITS);
}


const unsigned long MAGIC_MULTIPLIER = (100 * 0x1000000 + 10 * 0x10000 + 1);
int parse_temperature(unsigned long number_bytes, int dot_pos)
{
    // number_bytes contains the number: X.X, -X.X, XX.X or -XX.X
    const unsigned long inv_number_bytes = ~number_bytes;

    // Calculate the sign
    const unsigned long sign = (inv_number_bytes << 59) >> 63;
    const int _28_minus_dot_pos = dot_pos ^ 0b11100;
    const unsigned long minus_filter = ~(sign & 0xff);

    // Use the pre-calculated decimal position to adjust the values
    const unsigned long digits =
        ((number_bytes & minus_filter) << _28_minus_dot_pos) & 0x0f000f0f00l;

    // Multiply by a magic, to get the result
    const unsigned long abs_value = ((digits * MAGIC_MULTIPLIER) >> 32) & 0x3ff;
    // And apply the sign
    return (abs_value + sign) ^ sign;
}

int process_file_normal(void* thread_data_arg)
{
    size_t window_count = 0;

    struct thread_data_t* thread_data = thread_data_arg;
    while (1)
    {
        ++window_count;
        size_t window = ++*thread_data->window;
        size_t start_idx = WINDOW_SIZE * window;

        if (start_idx >= thread_data->length)
            break;
        size_t end_idx = start_idx + WINDOW_SIZE;
        if (end_idx >= thread_data->length)
            end_idx = thread_data->length;

        char* curr = thread_data->addr + start_idx;
        size_t line_count = 0;

        while (curr < thread_data->addr + end_idx)
        {
            int name_len = 0;
            while (1)
            {
                unsigned long name_word = *(long*)curr;
                unsigned long matchBits = semicolon_match_bits(name_word);

                if (matchBits != 0)
                {
                    int nl = get_name_len(matchBits);
                    name_len += nl;
                    curr += nl;
                    unsigned long temp_word = *(long*)curr;
                    int dot_pos = get_dot_pos(temp_word);
                    int temperature = parse_temperature(temp_word, dot_pos);
                    curr += (dot_pos >> 3) + 3;
                    break;
                }
                curr += sizeof(long);
                name_len += sizeof(long);
            }

            ++line_count;
        }
        *thread_data->line_count += line_count;
    }
    return 0;
}


int main(int argc, char* argv[])
{
    struct timespec time1, time2, time3, time4;
    clock_gettime(CLOCK_MONOTONIC, &time1);


    if (argc < 2)
    {
        fprintf(stderr, "%s file\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    int fd = open(argv[1], O_RDONLY);
    if (fd == -1)
        handle_error("open");

    struct stat sb;
    if (fstat(fd, &sb) == -1)
        handle_error("fstat");

    char* addr = mmap(NULL, sb.st_size,
                      PROT_READ, MAP_PRIVATE, fd, 0);
    if (addr == MAP_FAILED)
        handle_error("mmap");

    close(fd);

    int thread_count = get_core_count();
    printf("Threads: %d\n", thread_count);

    thrd_t* thrds = malloc(sizeof(thrd_t) * thread_count);
    if (thrds == NULL)
        handle_error("malloc");


    atomic_int window = -1;
    atomic_int line_count = 0;

    struct thread_data_t thrd_arg = {addr, sb.st_size, &window, &line_count};

    clock_gettime(CLOCK_MONOTONIC, &time2);

    for (int i = 0; i < thread_count; ++i)
    {
        thrd_create(&thrds[i], process_file_normal, &thrd_arg);
    }

    for (int i = 0; i < thread_count; ++i)
        thrd_join(thrds[i], NULL);

    clock_gettime(CLOCK_MONOTONIC, &time3);


    int lc = line_count;
    printf("Line count: %d\n", lc);

    free(thrds);

    munmap(addr, sb.st_size);

    clock_gettime(CLOCK_MONOTONIC, &time4);
    printf("Time1-2: %ld:%ld\n", diff(time1, time2).tv_sec, diff(time1, time2).tv_nsec);
    printf("Time2-3: %ld:%ld\n", diff(time2, time3).tv_sec, diff(time2, time3).tv_nsec);
    printf("Time3-4: %ld:%ld\n", diff(time3, time4).tv_sec, diff(time3, time4).tv_nsec);
    printf("Time1-4: %ld:%ld\n", diff(time1, time4).tv_sec, diff(time1, time4).tv_nsec);


    exit(EXIT_SUCCESS);
}
