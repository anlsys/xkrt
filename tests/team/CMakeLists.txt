# Thread-team tests: spawn/join, barriers, parallel-for, critical sections.

xkrt_add_test(NAME team-cpus               SOURCE team-cpus.cc               LABELS team)
xkrt_add_test(NAME team-cpus-master-member SOURCE team-cpus-master-member.cc LABELS team)
xkrt_add_test(NAME team-cpus-parallel-for  SOURCE team-cpus-parallel-for.cc  LABELS team)
xkrt_add_test(NAME team-barrier            SOURCE team-barrier.cc            LABELS team)
xkrt_add_test(NAME team-device             SOURCE team-device.cc             LABELS team)
xkrt_add_test(NAME team-critical           SOURCE team-critical.cc           LABELS team)
xkrt_add_test(NAME team-parallel-for-bounds SOURCE team-parallel-for-bounds.cc LABELS team)
