/*
** Copyright 2024,2025 INRIA
**
** Contributors :
** Thierry Gautier, thierry.gautier@inrialpes.fr
** Romain PEREIRA, romain.pereira@inria.fr + rpereira@anl.gov
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

/**
 *  Counterpart of file-read.cc: writes a buffer to a file using parallel
 *  file_write_async() tasks, then reads the file back (plain read) and
 *  verifies the content matches.
 */

# include <fcntl.h>
# include <unistd.h>
# include <stdlib.h>
# include <stdio.h>
# include <string.h>
# include <errno.h>

# include <xkrt/runtime.h>
# include <xkrt/logger/logger.h>
# include <xkrt/logger/metric.h>

XKRT_NAMESPACE_USE;

static const char * filename = "/tmp/xkrt-file-write-test.bin";
static const size_t size     = 8L * 1024 * 1024;   // 8 MiB
static const int    nchunks  = 16;

static unsigned char
pattern(size_t i)
{
    return (unsigned char) ((i * 7 + 3) % 256);
}

int
main(void)
{
    runtime_t runtime;
    assert(runtime.init() == 0);

    // source buffer with a known pattern
    unsigned char * buffer = (unsigned char *) malloc(size);
    assert(buffer);
    for (size_t i = 0 ; i < size ; ++i)
        buffer[i] = pattern(i);

    int fd = open(filename, O_RDWR | O_CREAT | O_TRUNC, 0644);
    if (fd < 0)
    {
        perror("open");
        exit(EXIT_FAILURE);
    }

    // write the buffer with parallel async tasks
    assert(runtime.file_write_async(fd, buffer, size, nchunks) == 0);
    runtime.task_wait();

    // read it back independently and verify
    unsigned char * readback = (unsigned char *) malloc(size);
    assert(readback);
    memset(readback, 0, size);

    off_t off = lseek(fd, 0, SEEK_SET);
    assert(off == 0);

    size_t total = 0;
    while (total < size)
    {
        ssize_t r = read(fd, readback + total, size - total);
        if (r < 0)
        {
            perror("read");
            exit(EXIT_FAILURE);
        }
        if (r == 0)
            break;
        total += (size_t) r;
    }
    assert(total == size);

    for (size_t i = 0 ; i < size ; ++i)
        assert(readback[i] == pattern(i));

    close(fd);
    if (remove(filename) != 0)
        perror("remove");

    free(buffer);
    free(readback);

    assert(runtime.deinit() == 0);

    return 0;
}
