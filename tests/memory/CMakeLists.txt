# Memory tests: allocators, registration (sync/async/parallel), touch, copy.

# --- allocators ---
xkrt_add_test(NAME memory-allocator             SOURCE memory-allocator.cc             LABELS memory allocator)
xkrt_add_test(NAME memory-allocator-host-unified SOURCE memory-allocator-host-unified.cc LABELS memory allocator)

# --- registration / touch (host) ---
xkrt_add_test(NAME register-async                          SOURCE register-async.cc                          LABELS memory register)
xkrt_add_test(NAME memory-register-async                   SOURCE memory-register-async.cc                   LABELS memory register)
xkrt_add_test(NAME memory-register-async-legacy            SOURCE memory-register-async-legacy.cc            LABELS memory register)
xkrt_add_test(NAME memory-register-async-legacy-memcpy-split SOURCE memory-register-async-legacy-memcpy-split.cc LABELS memory register)
xkrt_add_test(NAME memory-unregister-async                 SOURCE memory-unregister-async.cc                 LABELS memory register)
xkrt_add_test(NAME memory-touch-async                      SOURCE memory-touch-async.cc                      LABELS memory register)
xkrt_add_test(NAME memory-touch-register-unregister-async  SOURCE memory-touch-register-unregister-async.cc  LABELS memory register)

# --- GPU-only memory tests (use distribute_async / device copies) ---
xkrt_add_test(NAME memory-register-protection            SOURCE memory-register-protection.cc            LABELS memory register GPU)
xkrt_add_test(NAME memory-register-assisted              SOURCE memory-register-assisted.cc              LABELS memory register GPU)
xkrt_add_test(NAME memory-register-assisted-async        SOURCE memory-register-assisted-async.cc        LABELS memory register GPU)
xkrt_add_test(NAME memory-register-assisted-async-depend SOURCE memory-register-assisted-async-depend.cc LABELS memory register GPU)
xkrt_add_test(NAME memory-register-assisted-unregister   SOURCE memory-register-assisted-unregister.cc   LABELS memory register GPU)
xkrt_add_test(NAME memory-copy-async                     SOURCE memory-copy-async.cc                     LABELS memory copy     GPU)

# note: memory-register-async.conf.cc is a snippet #included by
# memory-register-async.cc, not a standalone test.
