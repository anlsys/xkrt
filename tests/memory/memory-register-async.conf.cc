    const size_t size        = (size_t) 256 * 1024 * 1024;  // kept small so touch/register tests run fast
    const size_t nchunks     = 2;

    team_t * team = runtime.team_get_any(~(1 << XKRT_DRIVER_TYPE_HOST));
    if (team == NULL)
        team = runtime.team_get(XKRT_DRIVER_TYPE_HOST);
    assert(team);

    void * ptr = malloc(size);
