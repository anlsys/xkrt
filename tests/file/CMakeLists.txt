# Filesystem I/O tests: parallel async read / write.

xkrt_add_test(NAME file-read  SOURCE file-read.cc  LABELS file)
xkrt_add_test(NAME file-write SOURCE file-write.cc LABELS file)
