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

# include <xkrt/runtime.h>
# include <stack>

XKRT_NAMESPACE_USE;

////////////////////////////
// RUNTIME API //
////////////////////////////

void
runtime_t::task_dependency_graph_record_start(
    task_dependency_graph_t * tdg,
    bool execute_commands
) {
    /* wait for the completion of previously spawned tasks */
    this->task_wait();

    assert(tdg);

    thread_t * thread = thread_t::get_tls();
    assert(thread);

    task_t * task = thread->current_task;
    assert(task);

    assert(task->flags & TASK_FLAG_GRAPH);
    task->flags |= TASK_FLAG_GRAPH_RECORDING;

    if (execute_commands)
        task->flags |= TASK_FLAG_GRAPH_EXECUTE_COMMAND;

    task_gph_info_t * gph = TASK_GPH_INFO(task, task->flags);
    assert(gph);

    new (gph) task_gph_info_t(tdg);
}

void
runtime_t::task_dependency_graph_record_stop(void)
{
    /* wait for the completion of previously spawned tasks */
    this->task_wait();

    thread_t * thread = thread_t::get_tls();
    assert(thread);

    task_t * task = thread->current_task;
    assert(task);

    assert(task->flags & TASK_FLAG_GRAPH);
    assert(task->flags & TASK_FLAG_GRAPH_RECORDING);
    task->flags &= ~(TASK_FLAG_GRAPH_RECORDING | TASK_FLAG_GRAPH_EXECUTE_COMMAND);

    task_gph_info_t * gph = TASK_GPH_INFO(task, task->flags);
    assert(gph);

    assert(gph->tdg);
    gph->tdg->postprocess();
}

void
runtime_t::task_dependency_graph_replay(task_dependency_graph_t * tdg)
{
    (void) tdg;
    // nothing to do
}

void
runtime_t::task_dependency_graph_destroy(task_dependency_graph_t * tdg)
{
    (void) tdg;
    // nothing to do
}

////////////////////////////
// TDG API //
////////////////////////////

void
task_dependency_graph_t::foreach_task(std::function<void(task_t *)> f)
{
    for (task_t * task : this->tasks)
        f(task);
}

void
task_dependency_graph_t::dump_tasks(FILE * f)
{
    fprintf(f, "digraph G {\n");
    this->foreach_task([&] (task_t * task)
    {
        # if XKRT_SUPPORT_DEBUG
        fprintf(f, "    \"%p\" [label=\"%s\"] ;\n", (void *) task, task->label);
        # else
        fprintf(f, "    \"%p\" [label=\"%p\"] ;\n", (void *) task, task);
        # endif /* XKRT_SUPPORT_DEBUG */
        if (task->flags & TASK_FLAG_ACCESSES)
        {
            task_acs_info_t * acs = TASK_ACS_INFO(task);
            assert(acs);

            access_t * accesses = TASK_ACCESSES(task);
            for (task_access_counter_t ac = 0 ; ac < acs->ac ; ++ac)
            {
                access_t * pred = accesses + ac;
                for (access_t * succ : pred->successors)
                    fprintf(f, "    \"%p\" -> \"%p\" ;\n", (void *) pred->task, (void *) succ->task);
            }
        }
    });
    fprintf(f, "}\n");
}

void
task_dependency_graph_t::dump_accesses(FILE * f)
{
    fprintf(f, "digraph G {\n");
    this->foreach_task([&] (task_t * task)
    {
        if (task->flags & TASK_FLAG_ACCESSES)
        {
            task_acs_info_t * acs = TASK_ACS_INFO(task);
            assert(acs);

            access_t * accesses = TASK_ACCESSES(task);
            for (task_access_counter_t ac = 0 ; ac < acs->ac ; ++ac)
            {
                access_t * pred = accesses + ac;
                # if XKRT_SUPPORT_DEBUG
                fprintf(f, "    \"%p\" [label=\"%s - ac %d\"] ;\n", (void *) pred, task->label, ac);
                # else
                fprintf(f, "    \"%p\" [label=\"%p - ac %d\"] ;\n", (void *) pred, task, ac);
                # endif /* XKRT_SUPPORT_DEBUG */
                for (access_t * succ : pred->successors)
                    fprintf(f, "    \"%p\" -> \"%p\" ;\n", (void *) pred, (void *) succ);
            }
        }
    });
    fprintf(f, "}\n");
}

void
task_dependency_graph_t::walk(std::function<void(task_t *)> f)
{
    std::stack<task_t *> tasks;

    for (task_t * task : this->roots)
        tasks.push(task);

    while (!tasks.empty())
    {
        task_t * task = tasks.top();
        tasks.pop();

        assert(task->flags & TASK_FLAG_RECORD);
        task_rec_info_t * rec = TASK_REC_INFO(task);
        assert(rec);

        if (rec->visited_flag == this->visited_flag_cmp)
            continue ;
        rec->visited_flag = this->visited_flag_cmp;

        f(task);

        if (task->flags & TASK_FLAG_ACCESSES)
        {
            task_acs_info_t * acs = TASK_ACS_INFO(task);
            assert(acs);

            access_t * accesses = TASK_ACCESSES(task);
            assert(accesses);

            for (task_access_counter_t ac = 0 ; ac < acs->ac ; ++ac)
                for (access_t * succ : (accesses + ac)->successors)
                    tasks.push(succ->task);
        }
    }

    this->visited_flag_cmp = !this->visited_flag_cmp;
}

void
task_dependency_graph_t::remove_transitive_edges(void)
{
    // TODO - remove transitive edges
    LOGGER_NOT_IMPLEMENTED_WARN("remove_transitive_edges");
}

void
task_dependency_graph_t::compute_leaves(void)
{
    this->foreach_task([&] (task_t * task)
    {
        size_t n_successors = 0;

        if (task->flags & TASK_FLAG_ACCESSES)
        {
            task_acs_info_t * acs = TASK_ACS_INFO(task);
            assert(acs);

            access_t * accesses = TASK_ACCESSES(task);
            for (task_access_counter_t ac = 0 ; ac < acs->ac ; ++ac)
                n_successors += (accesses + ac)->successors.size();
        }

        if (n_successors == 0)
            this->leaves.push_back(task);
    });
}

void
task_dependency_graph_t::list_tasks(void)
{
    assert(this->tasks.size() == 0);
    this->walk([&] (task_t * task) {

        assert(task->flags & TASK_FLAG_RECORD);
        task_rec_info_t * rec = TASK_REC_INFO(task);
        assert(rec);

        rec->index = this->tasks.size();
        this->tasks.push_back(task);
    });
}

void
task_dependency_graph_t::postprocess(void)
{
    this->list_tasks();
    this->compute_leaves();
    this->remove_transitive_edges();
}
