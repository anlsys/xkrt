# Task system tests: formats, dependencies, moldability, detachable, graphs.

# --- task formats ---
xkrt_add_test(NAME task-format             SOURCE task-format.cc             LABELS task format)
xkrt_add_test(NAME task-format-host        SOURCE task-format-host.cc        LABELS task format)
xkrt_add_test(NAME task-format-incremental SOURCE task-format-incremental.cc LABELS task format)

# --- low-level dependency resolution (point / interval / matrix) ---
xkrt_add_test(NAME task-dependency                SOURCE task-dependency.cc                LABELS task dependency)
xkrt_add_test(NAME task-dependency-handle         SOURCE task-dependency-handle.cc         LABELS task dependency)
xkrt_add_test(NAME task-dependency-interval       SOURCE task-dependency-interval.cc       LABELS task dependency)
xkrt_add_test(NAME task-dependency-interval-matrix SOURCE task-dependency-interval-matrix.cc LABELS task dependency)

# --- high-level capture-API dependencies / concurrency / moldability ---
xkrt_add_test(NAME task-spawn-dependencies SOURCE task-spawn-dependencies.cc LABELS task dependency)
xkrt_add_test(NAME task-concurrency        SOURCE task-concurrency.cc        LABELS task dependency)
xkrt_add_test(NAME moldability             SOURCE moldability.cc             LABELS task moldable)
xkrt_add_test(NAME task-moldable-split     SOURCE task-moldable-split.cc     LABELS task moldable)

# --- detachable tasks (completion gated on a reference counter) ---
xkrt_add_test(NAME task-detachable SOURCE task-detachable.cc LABELS task detachable)

# --- recursive tasking (fibonacci) ---
# these are tasking stress benchmarks; run a small input so they stay well
# within the default timeout (correctness is checked for any N).
xkrt_add_test(NAME fib-task-capture SOURCE fib-task-capture.cc LABELS task ARGS 20)
xkrt_add_test(NAME fib-task-format  SOURCE fib-task-format.cc  LABELS task ARGS 20)

# --- task dependency graph -> command graph record/replay ---
xkrt_add_test(NAME task-dependency-graph-host SOURCE task-dependency-graph-host.cc LABELS task graph)

# --- GPU-only tests (skipped gracefully when no GPU is present) ---
xkrt_add_test(NAME task-gpu-empty       SOURCE task-gpu-empty.cc       LABELS task           GPU)
xkrt_add_test(NAME task-dependency-graph SOURCE task-dependency-graph.cc LABELS task graph     GPU)
