/*
** Copyright 2024,2025 INRIA
**
** Contributors :
** Thierry Gautier, thierry.gautier@inrialpes.fr
** Joao Lima joao.lima@inf.ufsm.br
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

# include <atomic>
# include <cassert>
# include <cstring>

# include <xkrt/logger/logger.h>
# include <xkrt/namespace.h>
# include <xkrt/runtime.h>

XKRT_NAMESPACE_USE;

void
runtime_t::task_formats_init(void)
{
    new (&this->formats.all) task_formats_t();

    task_format_t format;
    memset(&format, 0, sizeof(format));
    snprintf(format.label, sizeof(format.label), "(null)");

    task_format_id_t id = this->task_format_create(&format);
    assert(id == XKRT_TASK_FORMAT_NULL);
}

task_format_t *
runtime_t::task_format_get(task_format_id_t fmtid)
{
    return this->formats.all.list.get(fmtid);
}

task_format_id_t
runtime_t::task_format_put(const char * label)
{
    const size_t index = this->formats.all.list.size();
    assert(index <= (size_t) (task_format_id_t) ~0);
    const task_format_id_t fmtid = (task_format_id_t) index;

    task_format_t * format = this->formats.all.list.put();
    memset(format, 0, sizeof(task_format_t));
    snprintf(format->label, sizeof(format->label), "%s", label);

    // grow the per-format stats pool in lockstep so stats[fmtid] exists
    # if XKRT_SUPPORT_STATS
    new (this->stats.tasks.put()) task_stats_t();
    # endif /* XKRT_SUPPORT_STATS */

    LOGGER_DEBUG("Created new task format `%d` named `%s`", fmtid, label);

    return fmtid;
}

task_format_id_t
runtime_t::task_format_create(const task_format_t * format)
{
    const task_format_id_t fmtid = this->task_format_put(format->label);
    memcpy(this->formats.all.list.get(fmtid), format, sizeof(task_format_t));
    return fmtid;
}

int
runtime_t::task_format_set(
    task_format_id_t fmtid,
    task_format_target_t target,
    task_format_func_t func
) {
    task_format_t * format = this->task_format_get(fmtid);
    format->f[target] = func;
    return 0;
}

int
runtime_t::task_format_set_source(
    task_format_id_t fmtid,
    task_format_target_t target,
    cgir_command_prog_source_t source
) {
    task_format_t * format = this->task_format_get(fmtid);
    format->source[target] = source;
    return 0;
}
