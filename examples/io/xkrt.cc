//  to generate a file:
//      dd if=/dev/urandom of=/tmp/file-10GB.bin bs=1M count=10240
//
//  to measure disk bw:
//      dd if=/tmp/file-10GB.bin of=/dev/null bs=1G iflag=direct

#include <iostream>
#include <fstream>
#include <chrono>
#include <fcntl.h>
#include <unistd.h>
#include <vector>
#include <cmath>
#include <sys/stat.h>
#include <sys/mman.h>

#include <xkrt/runtime.h>

XKRT_NAMESPACE_USE;

// Helper to compute mean and standard deviation
void stats(const std::vector<double> &v, double &mean, double &stdev) {
    mean = 0.0;
    int n = v.size();
    for (double x : v)
    {
        printf("%lf\n", x);
        mean += x;
    }
    mean /= n;
    stdev = 0.0;
    for (double x : v)
        stdev += (x - mean) * (x - mean);
    stdev = std::sqrt(stdev / n);
}

double median(std::vector<double> &nums) {
    if (nums.empty()) {
        throw std::runtime_error("List is empty");
    }

    std::sort(nums.begin(), nums.end()); // sort the list

    size_t n = nums.size();
    if (n % 2 == 1) {
        // odd number of elements, return the middle one
        return nums[n / 2];
    } else {
        // even number of elements, return the average of the two middle ones
        return (nums[n / 2 - 1] + nums[n / 2]) / 2.0;
    }
}

off_t get_file_size(int fd) {
    struct stat st;

    if (fstat(fd, &st) == -1) {
        perror("fstat");
        return -1;
    }

    return st.st_size;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <filename> [ntasks] [nruns]\n";
        return 1;
    }

    const char *filename = argv[1];
    const int NTASKS = atoi(argv[2]);
    const int runs = atoi(argv[3]);

    // Open file for reading
    int fd = open(filename, O_RDONLY | O_DIRECT);
    if (fd < 0) {
        perror("open");
        return 1;
    }
    off_t size = get_file_size(fd);
    printf("Iterating %d times on a file of size %ldGB\n", runs, size/1024/1024/1024);

    // Allocate page-aligned host buffer
    void * buffer = mmap(nullptr, size,
            PROT_READ | PROT_WRITE,
            MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    assert(buffer);
    memset(buffer, 0, size);

    runtime_t runtime;
    runtime.init();

    const device_global_id_t device_global_id = 1;

    // Vectors to store results
    std::vector<double> t_alloc, t_read, t_register, t_memcpy, t_total;

    for (int iter = 0; iter < runs ; ++iter) {

        // invalidate caches
        # if 1
        runtime.reset();
        # endif
        # if 1

        // reset file for reading
        lseek(fd, 0, SEEK_SET);

        // --- Start total timer ---
        auto t_start_total = std::chrono::high_resolution_clock::now();

        // --- 2. cuMemHostRegister() ---
        runtime.memory_register_async(buffer, size, NTASKS);
        # endif

        // --- 3. read() ---
        # if 1
        runtime.file_read_async(fd, buffer, size, NTASKS);
        // read(fd, buffer, size);
        # endif

        // --- 4. cuMemcpyHtoD() ---
        # if 1
        runtime.memory_coherent_async(device_global_id, buffer, size, NTASKS);
        # endif

        // sync
        runtime.task_wait();

        auto t_end_total = std::chrono::high_resolution_clock::now();

        // Compute durations
        double total_ms = std::chrono::duration<double, std::milli>(t_end_total - t_start_total).count();
        runtime.memory_unregister(buffer, size);

        // Store times
        t_total.push_back(total_ms);
    }

    // Compute stats
    double mean_total, stdev_total;

    stats(t_total, mean_total, stdev_total);

    // Print results
    std::cout << "Benchmark results (size = " << size / (1024.0 * 1024.0 * 1024.0) << " GB, "
              << runs << " runs):\n";
    std::cout << "  Total time          : avg = " << mean_total << " ms, stdev = " << stdev_total << " ms\n";
    std::cout << "  Total time          : med = " << median(t_total) <<  " ms\n";

    double gbs = size / (1024.0 * 1024.0 * 1024.0) / (median(t_total) * 1.0e-3);
    double gbsd = size / (1024.0 * 1024.0 * 1024.0) / ((median(t_total) + stdev_total) * 1.0e-3);
    double dgbs = gbs - gbsd;
    std::cout << "  Total GB/s          : avg = " << gbs <<  " GB/s , stdev = " << dgbs << " GB/s\n";

    // Cleanup
    munmap(buffer, size);
    close(fd);

    runtime.deinit();
    return 0;
}


