zero_flag = "TASK_FLAG_ZERO"
max_flag  = "TASK_FLAG_MAX"

flags = [
    ("TASK_FLAG_ACCESSES",      "acs"),
    ("TASK_FLAG_DETACHABLE",    "det"),
    ("TASK_FLAG_DEVICE",        "dev"),
    ("TASK_FLAG_DOMAIN",        "dom"),
    ("TASK_FLAG_MOLDABLE",      "mol"),
    ("TASK_FLAG_GRAPH",         "gph"),
    ("TASK_FLAG_RECORD",        "rec")
]

n = len(flags)
width = 26

# Print switch case for task size
print('static constexpr size_t')
print('task_get_extra_size(const task_flag_bitfield_t flags)')
print('{')
print('    switch (flags)')
print('    {')
for combination in range(0,1<<n):
    bit_string = '.'.join(f'{combination:0{n}b}')
    combination_flags_str  = list(reversed([flags[bit][0]                          if combination & (1 << bit) else zero_flag for bit in range(0, n)]))
    combination_flags_size = list(reversed([f'sizeof(task_{flags[bit][1]}_info_t)' if combination & (1 << bit) else "0"       for bit in range(0, n)]))
    print('        case (      ' + ' | '.join(f'{item:>{width}}' for item in combination_flags_str) + '):')
    print('            return (' + ' + '.join(f'{item:>{width}}' for item in combination_flags_size) + f'); // {bit_string}')
    print('')

print('        default:')
print('            return task_get_base_size_fallback(flags);')
print('    }')
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
