# C API tests (link against the C API exposed by libxkrt).

xkrt_add_test(NAME capi-task-with-access SOURCE task-with-access.c LABELS capi)
xkrt_add_test(NAME capi-task-spawn       SOURCE task-spawn.c       LABELS capi)
xkrt_add_test(NAME capi-team-task-spawn  SOURCE team-task-spawn.c  LABELS capi)
xkrt_add_test(NAME capi-memory-register  SOURCE memory-register.c  LABELS capi)
