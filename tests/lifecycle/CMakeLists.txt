# Runtime lifecycle tests (init / deinit / synchronization)

xkrt_add_test(NAME init   SOURCE init.cc   LABELS lifecycle)
xkrt_add_test(NAME deinit SOURCE deinit.cc LABELS lifecycle)
xkrt_add_test(NAME sync   SOURCE sync.cc   LABELS lifecycle)
