/*
** Copyright 2024 INRIA
**
** Contributors :
**
** Romain PEREIRA, romain.pereira@inria.fr
** Romain PEREIRA, rpereira@anl.gov
**
** This software is a computer program whose purpose is to execute
** blas subroutines on multi-GPUs system.
**
** This software is governed by the CeCILL-C license under French law and
** abiding by the rules of distribution of free software.  You can  use,
** modify and/ or redistribute the software under the terms of the CeCILL-C
** license as circulated by CEA, CNRS and INRIA at the following URL
** "http://www.cecill.info".

** As a counterpart to the access to the source code and  rights to copy,
** modify and redistribute granted by the license, users are provided only
** with a limited warranty  and the software's author,  the holder of the
** economic rights,  and the successive licensors  have only  limited
** liability.

** In this respect, the user's attention is drawn to the risks associated
** with loading,  using,  modifying and/or developing or reproducing the
** software by the user in light of its specific status of free software,
** that may mean  that it is complicated to manipulate,  and  that  also
** therefore means  that it is reserved for developers  and  experienced
** professionals having in-depth computer knowledge. Users are therefore
** encouraged to load and test the software's suitability as regards their
** requirements in conditions enabling the security of their systems and/or
** data to be ensured and,  more generally, to use and operate it in the
** same conditions as regards security.

** The fact that you are presently reading this means that you have had
** knowledge of the CeCILL-C license and that you accept its terms.
**/

# include <xkrt/runtime.h>
# include <stdio.h>
# include <stdlib.h>

# include <heat/consts.h>

# define USE_COMMAND_GRAPH 0

XKRT_NAMESPACE_USE;

/* Making a global xkrt context of simplicity purposes */
static runtime_t runtime;
static int TSX = 0;
static int TSY = 0;

////////////////
// VTK EXPORT //
////////////////

/* Export the grid to a vtk file */
void
export_to_vtk(TYPE * g, const char * filename, int step)
{
    FILE * file = fopen(filename, "w");
    if (file == NULL)
    {
        printf("Error opening file!\n");
        exit(1);
    }

    fprintf(file, "# vtk DataFile Version 3.0\n");
    fprintf(file, "2D Heat Diffusion Time Step %d\n", step);
    fprintf(file, "ASCII\n");
    fprintf(file, "DATASET STRUCTURED_GRID\n");
    fprintf(file, "DIMENSIONS %d %d 1\n", NX, NY);
    fprintf(file, "POINTS %d float\n", NX * NY);

    for (int j = 0; j < NY; j++)
        for (int i = 0; i < NX; i++)
            fprintf(file, "%.2f %.2f 0.0\n", i*DX, j*DY);

    fprintf(file, "POINT_DATA %d\n", NX * NY);
    fprintf(file, "SCALARS temperature float\n");
    fprintf(file, "LOOKUP_TABLE default\n");

    for (int j = 0; j < NY; j++)
        for (int i = 0; i < NX; i++)
            fprintf(file, "%.2f\n", GRID(g, i, j, LD));

    fclose(file);
}

static void
export_to_vtk(int frame, TYPE * grid)
{
    // Export frame
    char filename[50];
    snprintf(filename, sizeof(filename), "temperature_grid_%03d.vtk", frame);
    export_to_vtk(grid, filename, frame);
}

typedef struct  args_export_t
{
    TYPE * grid;
    int frame;
    args_export_t(TYPE * grid, int frame) : grid(grid), frame(frame) {}
}               args_export_t;

static void
export_vtk(runtime_t * runtime, device_t * device, task_t * task)
{
    args_export_t * args = (args_export_t *) TASK_ARGS(task);
    LOGGER_INFO("Exporting frame %d/%d", args->frame+1, N_VTK);
    export_to_vtk(args->frame, args->grid);
}

static task_format_id_t export_vtk_format_id;

static void
maybe_export(int step, TYPE * grid)
{
    if (N_VTK)
    {
        if ((step + 1) % (N_STEP / N_VTK) == 0)
        {
            int frame = step / (N_STEP / N_VTK);

            # if 1
            thread_t * thread = thread_t::get_tls();
            assert(thread);

            # define AC 1
            constexpr task_flag_bitfield_t flags = TASK_FLAG_ACCESSES;
            constexpr size_t task_size = task_compute_size(flags, AC);
            constexpr size_t args_size = sizeof(args_export_t);
            constexpr size_t size      = task_size + args_size;

            task_t * task = runtime.task_new(export_vtk_format_id, flags, size);

            task_acs_info_t * dep = TASK_DEP_INFO(task);
            new (dep) task_acs_info_t(AC);

            args_export_t * args = (args_export_t *) TASK_ARGS(task, task_size);
            new (args) args_export_t(grid, frame);

            # if XKRT_SUPPORT_DEBUG
            snprintf(task->label, sizeof(task->label), "export-vtk-%d", frame);
            # endif

            static_assert(AC <= TASK_MAX_ACCESSES);
            access_t * accesses = TASK_ACCESSES(task, flags);
            new (accesses + 0) access_t(task, MATRIX_COLMAJOR, grid, LD, 0, 0, NX, NY, sizeof(TYPE), ACCESS_MODE_R);
            thread->resolve(accesses, AC);
            # undef AC

            runtime.task_commit(task);

            # else

            runtime.coherent_async(MATRIX_COLMAJOR, grid, LD, NX, NY, sizeof(TYPE));
            // runtime.task_wait();
            // export_to_vtk(frame, grid);

            # endif
        }
    }
}

////////////////////
// DIFFUSION TASK //
////////////////////

static task_format_id_t diffusion_format_id;

typedef struct  args_t
{
    TYPE * src;
    TYPE * dst;
    const int tile_x;
    const int tile_y;
    args_t(TYPE * src, TYPE * dst, int tx, int ty) : src(src), dst(dst), tile_x(tx), tile_y(ty) {}
}               args_t;

# if XKRT_SUPPORT_CUDA

# include <xkrt/driver/driver-cu.h>
# include <xkrt/logger/logger-cu.h>

extern "C"
void diffusion_cuda(
    cudaStream_t stream,
    TYPE * src, int ld_src,
    TYPE * dst, int ld_dst,
    int tile_x, int tile_y,
    int tsx, int tsy
);

static void
cuda(
    runtime_t * runtime,
    device_t * device,
    task_t * task,
    queue_cu_t * queue,
    command_t * cmd,
    command_queue_list_counter_t idx
) {
    args_t * args = (args_t *) TASK_ARGS(task);

    const access_t * accesses = TASK_ACCESSES(task);
    const access_t * a_src = accesses + 0;
    const access_t * a_dst = accesses + 1;

    TYPE * src = (TYPE *) a_src->device_view.addr;
    TYPE * dst = (TYPE *) a_dst->device_view.addr;
    const size_t ld_src = a_src->device_view.ld;
    const size_t ld_dst = a_dst->device_view.ld;

    // offset boundary access so the kernel receive the correct pointer
    if (args->tile_x == 0)  dst = dst - 1;
    else                    src = src + 1;

    if (args->tile_y == 0)  dst = dst - ld_dst;
    else                    src = src + ld_src;

    // submit kernel
    cudaStream_t custream = queue->cu.handle.high;
    diffusion_cuda(
        custream,
        src, ld_src,
        dst, ld_dst,
        args->tile_x, args->tile_y,
        TSX, TSY
    );
    CU_SAFE_CALL(cuEventRecord(queue->cu.events.buffer[idx], custream));
}
# endif /* XKRT_SUPPORT_CUDA */

# if XKRT_SUPPORT_ZE

int
read_from_binary(unsigned char **output, size_t * size, const char *name)
{
    FILE *fp = fopen(name, "rb");
    if (!fp)
        return -1;

    fseek(fp, 0, SEEK_END);
    *size = ftell(fp);
    fseek(fp, 0, SEEK_SET);

    *output = (unsigned char *)malloc(*size * sizeof(unsigned char));
    if (!*output) {
        fclose(fp);
        return -1;
    }

    fread(*output, *size, 1, fp);
    fclose(fp);
    return 0;
}

# include <xkrt/driver/driver-ze.h>
# include <xkrt/logger/logger-ze.h>
# include <ze_api.h>

static xkrt_driver_module_t ZE_MODULES[XKRT_DEVICES_MAX];
static xkrt_driver_module_t ZE_KERNELS[XKRT_DEVICES_MAX];

static void
ze(
    runtime_t * runtime,
    device_t * device,
    task_t * task,
    queue_ze_t * queue,
    command_t * cmd,
    command_queue_list_counter_t idx
) {
    args_t * args = (args_t *) TASK_ARGS(task);

    const access_t * accesses = TASK_ACCESSES(task);
    const access_t * a_src = accesses + 0;
    const access_t * a_dst = accesses + 1;

    TYPE * src = (TYPE *) a_src->device_view.addr;
    TYPE * dst = (TYPE *) a_dst->device_view.addr;
    const size_t ld_src = a_src->device_view.ld;
    const size_t ld_dst = a_dst->device_view.ld;

    // offset boundary access so the kernel receive the correct pointer
    if (args->tile_x == 0)  dst = dst - 1;
    else                    src = src + 1;

    if (args->tile_y == 0)  dst = dst - ld_dst;
    else                    src = src + ld_src;

    driver_t * driver = runtime->driver_get(XKRT_DRIVER_TYPE_ZE);
    assert(driver);

    // retrieve module, or compile it
    const int device_id = queue->ze.device->inherited.driver_id;
    xkrt_driver_module_t module = ZE_MODULES[device_id];
    if (module == NULL)
    {
        driver_t * driver = runtime->driver_get(XKRT_DRIVER_TYPE_ZE);
        assert(driver);

        unsigned char * file;
        size_t size;
        const char * name = "../src/kernels/diffusion.ar";
        if (read_from_binary(&file, &size, name))
            LOGGER_FATAL("Could not find the kernel binary file, have you precompiled the kernel ?");
        ZE_MODULES[device_id] = driver->f_module_load(device_id, file, size, XKRT_DRIVER_MODULE_FORMAT_NATIVE);
        assert(ZE_MODULES[device_id]);
        free(file);

        module = ZE_MODULES[device_id];
    }
    assert(module);

    ze_kernel_handle_t fn = (ze_kernel_handle_t) ZE_KERNELS[device_id];
    if (fn == NULL)
    {
        fn = (ze_kernel_handle_t) driver->f_module_get_fn(module, "diffusion_cl_kernel");
        ZE_KERNELS[device_id] = fn;
        if (fn == NULL)
            LOGGER_FATAL("Couldnt find the `diffusion_cl_kernel` within the module");
    }
    assert(fn);

    // not necessary, thats a hint for having memory resident during kernel execution
    // ZE_SAFE_CALL(zeKernelSetIndirectAccess(fn, ZE_KERNEL_INDIRECT_ACCESS_FLAG_DEVICE));

    // kernel(__global TYPE * src, int ld_src, __global TYPE * dst, int ld_dst, int tile_x, int tile_y, int tsx, int tsy)
    ZE_SAFE_CALL(zeKernelSetArgumentValue(fn, 0, sizeof(TYPE *),    &src));
    ZE_SAFE_CALL(zeKernelSetArgumentValue(fn, 1, sizeof(int),       &ld_src));
    ZE_SAFE_CALL(zeKernelSetArgumentValue(fn, 2, sizeof(TYPE *),    &dst));
    ZE_SAFE_CALL(zeKernelSetArgumentValue(fn, 3, sizeof(int),       &ld_dst));
    ZE_SAFE_CALL(zeKernelSetArgumentValue(fn, 4, sizeof(int),       &args->tile_x));
    ZE_SAFE_CALL(zeKernelSetArgumentValue(fn, 5, sizeof(int),       &args->tile_y));
    ZE_SAFE_CALL(zeKernelSetArgumentValue(fn, 6, sizeof(int),       &TSX));
    ZE_SAFE_CALL(zeKernelSetArgumentValue(fn, 7, sizeof(int),       &TSY));

    uint32_t gsX = 16;
    uint32_t gsY = 16;
    uint32_t gsZ = 1;
    ZE_SAFE_CALL(zeKernelSuggestGroupSize(fn, TSX, TSY, 1, &gsX, &gsY, &gsZ));
    ZE_SAFE_CALL(zeKernelSetGroupSize(fn, gsX, gsY, gsZ));

    ze_group_count_t dispatch;
    dispatch.groupCountX = TSX / gsX;
    dispatch.groupCountY = TSY / gsY;
    dispatch.groupCountZ = 1;

    ZE_SAFE_CALL(zeCommandListAppendLaunchKernel(queue->ze.command.list, fn, &dispatch, queue->ze.events.list[idx], 0, NULL));
}
# endif /* XKRT_SUPPORT_ZE */

# if XKRT_SUPPORT_HIP

# include <xkrt/driver/driver-hip.h>
# include <xkrt/logger/logger-hip.h>

extern "C"
void diffusion_hip(
    hipStream_t stream,
    TYPE * src, int ld_src,
    TYPE * dst, int ld_dst,
    int tile_x, int tile_y,
    int tsx, int tsy
);

static void
hip(
    runtime_t * runtime,
    device_t * device,
    task_t * task,
    queue_cu_t * queue,
    command_t * cmd,
    command_queue_list_counter_t idx
) {
    args_t * args = (args_t *) TASK_ARGS(task);

    const access_t * accesses = TASK_ACCESSES(task);
    const access_t * a_src = accesses + 0;
    const access_t * a_dst = accesses + 1;

    TYPE * src = (TYPE *) a_src->device_view.addr;
    TYPE * dst = (TYPE *) a_dst->device_view.addr;
    const size_t ld_src = a_src->device_view.ld;
    const size_t ld_dst = a_dst->device_view.ld;

    // offset boundary access so the kernel receive the correct pointer
    if (args->tile_x == 0)  dst = dst - 1;
    else                    src = src + 1;

    if (args->tile_y == 0)  dst = dst - ld_dst;
    else                    src = src + ld_src;

    // submit kernel
    hipStream_t hipstream = queue->hip.handle.high;
    diffusion_hip(
        hipstream,
        src, ld_src,
        dst, ld_dst,
        args->tile_x, args->tile_y,
        TSX, TSY
    );
    HIP_SAFE_CALL(hipEventRecord(queue->hip.events.buffer[idx], hipstream));
}
# endif /* XKRT_SUPPORT_HIP */

static void
update_cpu(TYPE * src, TYPE * dst)
{
    for (int i = 1; i < NX - 1; i++)
    {
        for (int j = 1; j < NY - 1; j++)
        {
            GRID(dst, i, j, LD) = GRID(src, i, j, LD) + ALPHA * DT / (DX * DY) * (
                (GRID(src, i+1,   j, LD) - 2 * GRID(src, i, j, LD) + GRID(src, i-1,   j, LD)) / (DX * DX) +
                (GRID(src,   i, j+1, LD) - 2 * GRID(src, i, j, LD) + GRID(src,   i, j-1, LD)) / (DY * DY)
            );
        }
    }
}

////////////////
// XKRT TASKS //
////////////////

template<auto F>
static inline void
routine_task(
    runtime_t * runtime,
    device_t * device,
    task_t * task
) {
    constexpr bool synchronous = false;
    return runtime->task_kernel_launch<synchronous>(device, task, (kernel_launcher_t) F);
}

static void
setup_tasks(void)
{
    // diffusion
    {
        task_format_t format;
        memset(&format, 0, sizeof(task_format_t));

        # if XKRT_SUPPORT_CUDA
        format.f[XKRT_DRIVER_TYPE_CUDA] = (task_format_func_t) routine_task<cuda>;
        # endif /* XKRT_SUPPORT_CUDA */

        # if XKRT_SUPPORT_ZE
        format.f[XKRT_DRIVER_TYPE_ZE] = (task_format_func_t) routine_task<ze>;
        # endif /* XKRT_SUPPORT_ZE */

        # if XKRT_SUPPORT_HIP
        format.f[XKRT_DRIVER_TYPE_HIP] = (task_format_func_t) routine_task<hip>;
        # endif /* XKRT_SUPPORT_HIP */

        snprintf(format.label, sizeof(format.label), "heat-diffusion");
        diffusion_format_id = runtime.task_format_create(&format);
    }

    // export vtk
    {
        task_format_t format;
        memset(&format, 0, sizeof(task_format_t));

        format.f[XKRT_DRIVER_TYPE_HOST] = (task_format_func_t) export_vtk;
        snprintf(format.label, sizeof(format.label), "export-vtk");
        export_vtk_format_id = runtime.task_format_create(&format);
    }
}

//////////////////
// GRID UPDATES //
//////////////////

/* Initialize the grid */
static void
initialize(TYPE * grid1, TYPE * grid2)
{
    LOGGER_WARN("Initializing grid on the host");

    memset(grid1, 0, NX*NY*sizeof(TYPE));
    memset(grid2, 0, NX*NY*sizeof(TYPE));

    for (int i = 0; i < NX; ++i)
    {
        GRID(grid1, i,    0, LD) = GRID(grid2, i,    0, LD) = TEMPERATURE_BOUNDARY;
        GRID(grid1, i, NY-1, LD) = GRID(grid2, i, NY-1, LD) = TEMPERATURE_BOUNDARY;
    }

    for (int i = 0; i < NY; ++i)
    {
        GRID(grid1,    0, i, LD) = GRID(grid2,    0, i, LD) = TEMPERATURE_BOUNDARY;
        GRID(grid1, NX-1, i, LD) = GRID(grid2, NX-1, i, LD) = TEMPERATURE_BOUNDARY;
    }

    LOGGER_WARN("Initialized grid on the host");
}

// omp interfaces would look like
//
//  # pragma omp task access(read:  matrix(colmajor, src, ld, x0, y0, sx, sy))        \
//                    access(write: matrix(colmajor, dst, LD, x0, y0, sx, sy))
//
//  [...]

/* Submit a tile */
static void
update_tile(TYPE * src, TYPE * dst, int tile_x, int tile_y, int step, unsigned int ngpus)
{
    thread_t * thread = thread_t::get_tls();
    assert(thread);

    # define AC 2
    constexpr task_flag_bitfield_t flags = TASK_FLAG_DEVICE | TASK_FLAG_ACCESSES | TASK_FLAG_DETACHABLE;
    constexpr size_t task_size = task_compute_size(flags, AC);
    constexpr size_t args_size = sizeof(args_t);
    constexpr size_t size      = task_size + args_size;

    task_t * task = runtime.task_new(diffusion_format_id, flags, size);

    task_acs_info_t * dep = TASK_DEP_INFO(task);
    new (dep) task_acs_info_t(AC);

    task_dev_info_t * dev = TASK_DEV_INFO(task);
    constexpr uint8_t ocr_access = 0;
    new (dev) task_dev_info_t(UNSPECIFIED_DEVICE_GLOBAL_ID, ocr_access);

    args_t * args = (args_t *) TASK_ARGS(task, task_size);
    new (args) args_t(src, dst, tile_x, tile_y);

    # if XKRT_SUPPORT_DEBUG
    snprintf(task->label, sizeof(task->label), "diffusion(%d, %d, s=%d)", tile_x, tile_y, step);
    # endif

    const int ntx = NUM_OF_TILES(NX, TSX);
    const int nty = NUM_OF_TILES(NY, TSY);

    const int x = (tile_x * TSX);
    const int y = (tile_y * TSY);

    static_assert(AC <= TASK_MAX_ACCESSES);
    access_t * accesses = TASK_ACCESSES(task, flags);
    {
        const ssize_t x0 = MAX(x-1, 0);
        const ssize_t y0 = MAX(y-1, 0);
        const ssize_t x1 = MIN(x+TSX+1, NX);
        const ssize_t y1 = MIN(y+TSY+1, NY);
        const  size_t sx = x1 - x0;
        const  size_t sy = y1 - y0;
        new (accesses + 0) access_t(task, MATRIX_COLMAJOR, src, LD, x0, y0, sx, sy, sizeof(TYPE), ACCESS_MODE_R);
    }
    {
        const ssize_t x0 = MAX(x, 1);
        const ssize_t y0 = MAX(y, 1);
        const ssize_t x1 = MIN(x+TSX, NX-1);
        const ssize_t y1 = MIN(y+TSY, NY-1);
        const  size_t sx = x1 - x0;
        const  size_t sy = y1 - y0;
        new (accesses + 1) access_t(task, MATRIX_COLMAJOR, dst, LD, x0, y0, sx, sy, sizeof(TYPE), ACCESS_MODE_W);
    }
    thread->resolve(accesses, AC);
    # undef AC

    runtime.task_commit(task);
}

/* Simulate 1 time step */
static void
update(TYPE * src, TYPE * dst, int step, unsigned int ngpus)
{
    # if 1
    const int ntx = NUM_OF_TILES(NX, TSX);
    const int nty = NUM_OF_TILES(NY, TSY);
    for (int tile_x = 0; tile_x < ntx; ++tile_x)
        for (int tile_y = 0; tile_y < nty; ++tile_y)
            update_tile(src, dst, tile_x, tile_y, step, ngpus);
    # else
    update_cpu(src, dst);
    # endif
}

//////////
// MAIN //
//////////

int
main(void)
{
    // Initialize xkrt runtime
    runtime.init();

    const int ngpus = runtime.drivers.devices.n - 1;
    if (ngpus == 0)
        LOGGER_FATAL("No devices found");
    LOGGER_INFO("Running on %d gpus", ngpus);

    // compute data partitioning
    TSX = NX / 1;
    TSY = NY / ngpus;

    // setup task format
    setup_tasks();

    // Allocate memory for the temperature grids on the CPU
    const size_t s = sizeof(TYPE);
    # if 1
    const uintptr_t alignon = NX * s;
    const size_t size = 2 * (NX * NY * s + alignon);
    const uintptr_t mem = (uintptr_t) runtime.memory_host_allocate(0, size);
    TYPE * grid1 = (TYPE *) (mem + (alignon - (mem % alignon)) + 0 * s * (NX * NY));
    TYPE * grid2 = (TYPE *) (mem + (alignon - (mem % alignon)) + 1 * s * (NX * NY));
    assert((uintptr_t)grid1 % alignon == 0);
    assert((uintptr_t)grid1 + NX * NY * s < mem + size);
    assert((uintptr_t)grid2 % alignon == 0);
    assert((uintptr_t)grid2 + NX * NY * s < mem + size);
    # else
    TYPE * grid1 = (TYPE *) runtime.memory_host_allocate(0, NX * NY * s);
    TYPE * grid2 = (TYPE *) runtime.memory_host_allocate(0, NX * NY * s);
    # endif

    // Set initial conditions
    initialize(grid1, grid2);

    // Distribute memory
    runtime.distribute_async(XKRT_DISTRIBUTION_TYPE_CYCLIC2D, MATRIX_COLMAJOR, grid1, LD, NX, NY, TSX, TSY, sizeof(TYPE), 1, 1);
    runtime.distribute_async(XKRT_DISTRIBUTION_TYPE_CYCLIC2D, MATRIX_COLMAJOR, grid2, LD, NX, NY, TSX, TSY, sizeof(TYPE), 1, 1);

    runtime.task_wait();

    // run simulation
    uint64_t t0 = get_nanotime();

    // Time stepping
    for (int step = 0; step < N_STEP; ++step)
    {
        // Grid swap
        TYPE * src = (step % 2 == 0) ? grid1 : grid2;
        TYPE * dst = (step % 2 == 0) ? grid2 : grid1;

        // Update
        update(src, dst, step, ngpus);
        if (step % (N_STEP / 10 + 1) == 0)
            LOGGER_WARN("(graph creation) Progress: %.2lf%%", step / (double)N_STEP*100);

        // Export every other frames
        maybe_export(step, dst);
    }

    uint64_t tf = get_nanotime();
    LOGGER_WARN("Graph Creation Took %.2lf s", (tf-t0)/1e9);

    // Finish remaining tasks
    runtime.task_wait();

    tf = get_nanotime();
    LOGGER_WARN("Graph Execution Took %.2lf s", (tf-t0)/1e9);

    // Deinitialize xkrt runtime
    runtime.deinit();

    return 0;
}
