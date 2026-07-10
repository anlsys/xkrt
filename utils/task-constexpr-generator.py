zero_flag = "TASK_FLAG_ZERO"
max_flag  = "TASK_FLAG_MAX"

# Layout-affecting flags, in memory-layout order (each contributes an info block
# right after the task_t, in this order). Appending a new flag here is now O(1):
# it adds one term to task_get_extra_size and one TASK_*_INFO accessor.
flags = [
    ("TASK_FLAG_ACCESSES",      "acs"),
    ("TASK_FLAG_DETACHABLE",    "det"),
    ("TASK_FLAG_DEVICE",        "dev"),
    ("TASK_FLAG_DOMAIN",        "dom"),
    ("TASK_FLAG_MOLDABLE",      "mol"),
    ("TASK_FLAG_GRAPH",         "gph"),
    ("TASK_FLAG_RECORD",        "rec"),
    ("TASK_FLAG_TASKGROUP",     "grp"),
]

n = len(flags)

# Print the task extra-size function.
#
# This used to be an exhaustive `switch` over all 2^n flag combinations, which
# grew exponentially (and most combinations are impractical anyway). It is now a
# linear sum over the present flags: O(n), and adding a flag costs a single line.
print('static constexpr size_t')
print('task_get_extra_size(const task_flag_bitfield_t flags)')
print('{')
print('    // XKRT invariant: a task cannot be both a dependency domain and a device task')
print('    assert(!((flags & TASK_FLAG_DOMAIN) && (flags & TASK_FLAG_DEVICE)));')
print('')
print('    size_t size = 0;')
for (flag, short) in flags:
    print(f'    if (flags & {flag}) size += sizeof(task_{short}_info_t);')
print('    return size;')
print('}')
print('')

# Print functions to retrieve each flag info
for i in range(0, n):

    # constexpr version with flags given
    print(f'static constexpr task_{flags[i][1]}_info_t *')
    print(f'TASK_{flags[i][1].upper()}_INFO(const task_t * task, const task_flag_bitfield_t flags)')
    print( '{')
    print(f'    assert(flags & {flags[i][0]});')
    for j in range(i-1, -1, -1):
        print(f'    if (flags & {flags[j][0]})')
        print(f'        return (task_{flags[i][1]}_info_t *) (TASK_{flags[j][1].upper()}_INFO(task, flags) + 1);')
    print(f'    return (task_{flags[i][1]}_info_t *) (task + 1);')
    print( '}')
    print( '')

    # non constexpr version, using task run-time flags
    print(f'static inline task_{flags[i][1]}_info_t *')
    print(f'TASK_{flags[i][1].upper()}_INFO(const task_t * task)')
    print( '{')
    print(f'    return TASK_{flags[i][1].upper()}_INFO(task, task->flags);')
    print( '}')
    print('')
