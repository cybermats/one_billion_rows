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
#include <string.h>
#include <time.h>
#include <limits.h>

#define WINDOW_SIZE (1 * 1024 * 1024)
#define HASHTABLE_SIZE 2048

#define BROADCAST_SEMICOLON 0x3B3B3B3B3B3B3B3BL
#define BROADCAST_0x01 0x0101010101010101L
#define BROADCAST_0x80 0x8080808080808080L
#define DOT_BITS 0x10101000


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

struct stats_acc
{
    char name[100];
    unsigned int length;
    unsigned long hash;
    long sum;
    unsigned int count;
    short max;
    short min;
};


int name_equals(const struct stats_acc* acc, const char* cursor, const unsigned int name_len)
{
    return acc->length == name_len &&
        memcmp(acc->name, cursor, name_len) == 0;
}

struct stats_acc* find_acc(struct stats_acc* hash_table, const unsigned long hash, const char* cursor, const int name_len)
{
    int slot_pos = hash & (HASHTABLE_SIZE - 1);
    while (1)
    {
        struct stats_acc* acc = &hash_table[slot_pos];
        if (acc->name[0] == 0)
        {
            memcpy(acc->name, cursor, name_len);
            acc->length = name_len;
            acc->hash = hash;
            acc->max = SHRT_MIN;
            acc->min = SHRT_MAX;
            return acc;
        }
        if (acc->hash == hash && name_equals(acc, cursor, name_len))
            return acc;
        slot_pos = (slot_pos + 1) & (HASHTABLE_SIZE - 1);
    }
}

void update(struct stats_acc* acc, int temperature)
{
    if (acc->max < temperature)
        acc->max = temperature;
    if (acc->min > temperature)
        acc->min = temperature;
    ++(acc->count);
    acc->sum += temperature;
}

void print_single_item(struct stats_acc* hashmap)
{
    printf("%s=%0.1f/%0.1f/%0.1f", hashmap->name, hashmap->min/10.f, hashmap->sum / (10.f * hashmap->count), hashmap->max/10.f);
}

int hash_sort_comp(const void * p1, const void * p2)
{
    const struct stats_acc* i1 = p1;
    const struct stats_acc* i2 = p2;
    return strcmp(i1->name, i2->name);
}

void print_full_hashmap(struct stats_acc* hashmap)
{
    qsort(hashmap, HASHTABLE_SIZE, sizeof(struct stats_acc), hash_sort_comp);
    printf("{");

    int i = 0;
    for (; i < HASHTABLE_SIZE; ++i)
    {
        if (hashmap[i].name[0] == 0)
            continue;
        print_single_item(&hashmap[i]);
        break;
    }
    ++i;
    for (; i < HASHTABLE_SIZE; ++i)
    {
        if (hashmap[i].name[0] == 0)
            continue;
        printf(", ");
        print_single_item(&hashmap[i]);
    }
    printf("}\n");
    
}

void merge_hashmap(struct stats_acc* dst, struct stats_acc* src, mtx_t* mtx)
{
    
    for (int idx = 0; idx < HASHTABLE_SIZE; ++idx)
    {
        struct stats_acc* acc = &src[idx];
        if (acc->name[0] == 0)
            continue;
        mtx_lock(mtx);
        struct stats_acc* dst_acc = find_acc(dst, acc->hash, acc->name, acc->length);
        dst_acc->count += acc->count;
        dst_acc->sum += acc->sum;
        if (dst_acc->max < acc->max)
            dst_acc->max = acc->max;
        if (dst_acc->min > acc->min)
            dst_acc->min = acc->min;
        mtx_unlock(mtx);
    }
}


struct thread_data_t
{
    char* addr;
    size_t length;
    atomic_int* window;
    atomic_int* line_count;
    mtx_t* mtx;
    struct stats_acc* hash_map;
};


unsigned long semicolon_match_bits(const unsigned long word)
{
    unsigned long diff = word ^ BROADCAST_SEMICOLON;
    return (diff - BROADCAST_0x01) & (~diff & BROADCAST_0x80);
}

unsigned int get_name_len(const unsigned long separator)
{
    return (__builtin_ctzl(separator) >> 3) + 1;
}

unsigned long mask_word(unsigned long word, unsigned long match_bits)
{
    return word & (match_bits ^ (match_bits - 1));
}

int get_dot_pos(const unsigned long word)
{
    return __builtin_ctzl(~word & DOT_BITS);
}

unsigned long get_hash(const unsigned long prev_hash, const unsigned long word)
{
    const unsigned long temp = (prev_hash ^ word) * 0x517cc1b727220a95L;
    return (temp << 13) | (temp >> (64-13));
}

const unsigned long MAGIC_MULTIPLIER = (100 * 0x1000000 + 10 * 0x10000 + 1);
int parse_temperature(const unsigned long number_bytes, const int dot_pos)
{
    // number_bytes contains the number: X.X, -X.X, XX.X or -XX.X
    const long inv_number_bytes = ~number_bytes;

    // Calculate the sign
    const long sign = (inv_number_bytes << 59) >> 63;
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

/**
 * \brief Thread main 
 * \param thread_data_arg 
 * \return 
 */
int process_file_normal(void* thread_data_arg)
{
    struct stats_acc hash_table[HASHTABLE_SIZE];

    struct thread_data_t* thread_data = thread_data_arg;
    while (1)
    {
        size_t window = ++*thread_data->window;
        size_t start_idx = WINDOW_SIZE * window;

        if (start_idx >= thread_data->length)
            break;
        size_t end_idx = start_idx + WINDOW_SIZE;
        if (end_idx >= thread_data->length)
            end_idx = thread_data->length;
        else
        {
            while (*(thread_data->addr + end_idx++) != '\n');
        }

        char* curr = thread_data->addr + start_idx;
        size_t line_count = 0;

        if (window > 0)
        {
            while(*curr++ != '\n');
        }

        

        while (curr < thread_data->addr + end_idx)
        {
            int name_len = 0;
            unsigned long hash = 0;
            const char* name_start = curr;
            while (1)
            {
                unsigned long name_word = *(long*)curr;
                unsigned long match_bits = semicolon_match_bits(name_word);

                if (match_bits != 0)
                {
                    int nl = get_name_len(match_bits);
                    name_word = mask_word(name_word, match_bits);
                    hash = get_hash(hash, name_word);
                    name_len += nl - 1;
                    curr += nl;
                    unsigned long temp_word = *(long*)curr;
                    int dot_pos = get_dot_pos(temp_word);
                    int temperature = parse_temperature(temp_word, dot_pos);
                    curr += (dot_pos >> 3) + 3;
                    struct stats_acc* acc = find_acc(hash_table, hash, name_start, name_len);
                    update(acc, temperature);
                    break;
                }
                hash = get_hash(hash, name_word);
                curr += sizeof(long);
                name_len += sizeof(long);
            }

            ++line_count;
        }
        *thread_data->line_count += line_count;
    }
    merge_hashmap(thread_data->hash_map, hash_table, thread_data->mtx);
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


    int thread_count = get_core_count();
    fprintf(stderr, "Threads: %d\n", thread_count);

    thrd_t* thrds = malloc(sizeof(thrd_t) * thread_count);
    if (thrds == NULL)
        handle_error("malloc");

    atomic_int window = -1;
    atomic_int line_count = 0;
    mtx_t mtx;
    mtx_init(&mtx, mtx_plain);
    struct stats_acc hash_table[HASHTABLE_SIZE];
    bzero(hash_table, sizeof(struct stats_acc) * HASHTABLE_SIZE);


    struct thread_data_t thrd_arg = {addr, sb.st_size, &window, &line_count, &mtx, hash_table};

    clock_gettime(CLOCK_MONOTONIC, &time2);

    for (int i = 0; i < thread_count; ++i)
    {
        thrd_create(&thrds[i], process_file_normal, &thrd_arg);
    }

    close(fd);


    for (int i = 0; i < thread_count; ++i)
        thrd_join(thrds[i], NULL);

    print_full_hashmap(hash_table);

    clock_gettime(CLOCK_MONOTONIC, &time3);


    int lc = line_count;
    fprintf(stderr, "Line count: %d\n", lc);

//    free(thrds);

//    munmap(addr, sb.st_size);

    clock_gettime(CLOCK_MONOTONIC, &time4);
    fprintf(stderr, "Time1-2: %ld:%ld\n", diff(time1, time2).tv_sec, diff(time1, time2).tv_nsec);
    fprintf(stderr, "Time2-3: %ld:%ld\n", diff(time2, time3).tv_sec, diff(time2, time3).tv_nsec);
    fprintf(stderr, "Time3-4: %ld:%ld\n", diff(time3, time4).tv_sec, diff(time3, time4).tv_nsec);
    fprintf(stderr, "Time1-4: %ld:%ld\n", diff(time1, time4).tv_sec, diff(time1, time4).tv_nsec);


    exit(EXIT_SUCCESS);
}
