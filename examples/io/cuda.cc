#include <sys/stat.h>
#include <iostream>
#include <fstream>
#include <chrono>
#include <fcntl.h>
#include <unistd.h>
#include <vector>
#include <cmath>
#include <cuda.h>

#define CHECK_CUDA(call)                                                      \
    do {                                                                      \
        CUresult err = call;                                                  \
        if (err != CUDA_SUCCESS) {                                            \
            const char *errStr;                                               \
            cuGetErrorString(err, &errStr);                                   \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__      \
                      << " code=" << err << " (" << errStr << ")" << std::endl; \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

// Helper to compute mean and standard deviation
void stats(const std::vector<double> &v, double &mean, double &stdev) {
    mean = 0.0;
    for (double x : v) mean += x;
    mean /= v.size();
    stdev = 0.0;
    for (double x : v) stdev += (x - mean) * (x - mean);
    stdev = std::sqrt(stdev / v.size());
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
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <filename>\n";
        return 1;
    }

    const char *filename = argv[1];

    // Open file for reading
    int fd = open(filename, O_RDONLY | O_DIRECT);
    if (fd < 0) {
        perror("open");
        return 1;
    }

    size_t size = get_file_size(fd);
    const int runs = 10;



    // Allocate page-aligned host buffer
    void *buffer = nullptr;
    posix_memalign(&buffer, 4096, size);
    if (!buffer) {
        std::cerr << "Failed to allocate host buffer\n";
        return 1;
    }

    // Initialize CUDA
    CHECK_CUDA(cuInit(0));
    CUdevice device;
    CHECK_CUDA(cuDeviceGet(&device, 0));
    CUcontext context;
    CHECK_CUDA(cuCtxCreate(&context, 0, device));

    // Vectors to store results
    std::vector<double> t_alloc, t_read, t_register, t_memcpy, t_total;

    for (int iter = 0; iter < runs + 1; ++iter) {
        // --- Start total timer ---
        auto t_start_total = std::chrono::high_resolution_clock::now();

        // --- 1. cuMemAlloc() ---
        auto t0 = std::chrono::high_resolution_clock::now();
        CUdeviceptr dst;
        CHECK_CUDA(cuMemAlloc(&dst, size));
        auto t1 = std::chrono::high_resolution_clock::now();

        // --- 2. read() ---
        lseek(fd, 0, SEEK_SET);
        auto t2 = std::chrono::high_resolution_clock::now();
        ssize_t bytes_read = read(fd, buffer, size);
        if (bytes_read != size)
        {
            printf("Did not read all bytes\n");
            return 1;
        }
        auto t3 = std::chrono::high_resolution_clock::now();
        if (bytes_read <= 0) {
            perror("read");
            return 1;
        }

        // --- 3. cuMemHostRegister() ---
        auto t4 = std::chrono::high_resolution_clock::now();
        CHECK_CUDA(cuMemHostRegister(buffer, size, 0));
        auto t5 = std::chrono::high_resolution_clock::now();

        // --- 4. cuMemcpyHtoD() ---
        auto t6 = std::chrono::high_resolution_clock::now();
        CHECK_CUDA(cuMemcpyHtoD(dst, buffer, size));
        auto t7 = std::chrono::high_resolution_clock::now();

        auto t_end_total = std::chrono::high_resolution_clock::now();

        // Compute durations
        double alloc_ms    = std::chrono::duration<double, std::milli>(t1 - t0).count();
        double read_ms     = std::chrono::duration<double, std::milli>(t3 - t2).count();
        double register_ms = std::chrono::duration<double, std::milli>(t5 - t4).count();
        double memcpy_ms   = std::chrono::duration<double, std::milli>(t7 - t6).count();
        double total_ms    = std::chrono::duration<double, std::milli>(t_end_total - t_start_total).count();

        // Cleanup before next iteration
        CHECK_CUDA(cuMemFree(dst));
        CHECK_CUDA(cuMemHostUnregister(buffer));

        // Ignore warmup iteration
        if (iter == 0) continue;

        // Store times
        t_alloc.push_back(alloc_ms);
        t_read.push_back(read_ms);
        t_register.push_back(register_ms);
        t_memcpy.push_back(memcpy_ms);
        t_total.push_back(total_ms);
    }

    // Compute stats
    double mean_alloc, stdev_alloc;
    double mean_read, stdev_read;
    double mean_register, stdev_register;
    double mean_memcpy, stdev_memcpy;
    double mean_total, stdev_total;

    stats(t_alloc, mean_alloc, stdev_alloc);
    stats(t_read, mean_read, stdev_read);
    stats(t_register, mean_register, stdev_register);
    stats(t_memcpy, mean_memcpy, stdev_memcpy);
    stats(t_total, mean_total, stdev_total);

    // Print results
    std::cout << "Benchmark results (size = " << size / (1024.0 * 1024.0) << " MB, "
              << runs << " runs):\n";
    std::cout << "  cuMemAlloc()        : avg = " << mean_alloc << " ms, stdev = " << stdev_alloc << " ms\n";
    std::cout << "  read()              : avg = " << mean_read << " ms, stdev = " << stdev_read << " ms\n";
    std::cout << "  cuMemHostRegister() : avg = " << mean_register << " ms, stdev = " << stdev_register << " ms\n";
    std::cout << "  cuMemcpyHtoD()      : avg = " << mean_memcpy << " ms, stdev = " << stdev_memcpy << " ms\n";
    std::cout << "  Total time          : avg = " << mean_total << " ms, stdev = " << stdev_total << " ms\n";

    // Cleanup
    free(buffer);
    close(fd);
    CHECK_CUDA(cuCtxDestroy(context));

    return 0;
}

