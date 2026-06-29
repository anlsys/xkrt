#!/usr/bin/env zsh
# ============================================================================
# update.sh — in-place updater for the xkrt ecosystem (companion to install.sh)
#
# Reuses the configuration cached by install.sh (.xkrt_install.cache) and the
# environment file it generates (env.sh).  For every component already checked
# out under $REPO_DIR it:
#   1. git pulls the latest changes (on the cached branch)
#   2. incrementally `make` + `make install` in its EXISTING build directory
#
# env.sh is sourced first, so all components' modules are loaded and the
# (incremental) builds find each other's headers/libraries/cmake config
# (e.g. rebuilding xkrt resolves <cgir/cgir.hpp>).
#
# It refreshes the SAME install/module locations, so loaded modules and env.sh
# keep working unchanged.  It does NOT re-run cmake from scratch nor create new
# (hash-based) install dirs — for that, re-run install.sh.
#
# Components are processed in dependency order (llvm, hwloc, cgir, xkrt, xkblas,
# xkomp).  A component is rebuilt when its own HEAD moved OR an upstream
# dependency was rebuilt during this run; otherwise it is skipped.  Use --force
# to rebuild & reinstall everything regardless of changes.
#
# Caveat: the custom LLVM offload runtime (libomptarget) links XKRT/XKOMP.  If
# you change XKRT/XKOMP in a way that affects libomptarget, re-run install.sh
# for a consistent rebuild — update.sh does not re-stage that dependency loop.
#
# Usage: update.sh [--force] [path/to/.xkrt_install.cache]
# ============================================================================

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ─── Colours / UI (same conventions as install.sh) ────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; CYAN='\033[0;36m'; BOLD='\033[1m'; DIM='\033[2m'; NC='\033[0m'
_tty() { printf "$@" >/dev/tty 2>/dev/null || printf "$@"; }
hr()      { _tty '%s\n' "────────────────────────────────────────────────────────────────────────"; }
step()    { _tty "\n${BOLD}${CYAN}▶${NC} ${BOLD}%s${NC}\n" "$*"; }
info()    { _tty "  ${BLUE}·${NC} %s\n" "$*"; }
success() { _tty "  ${GREEN}✓${NC} %s\n" "$*"; }
warn()    { _tty "  ${YELLOW}!${NC} %s\n" "$*"; }
fatal()   { _tty "  ${RED}✗ FATAL:${NC} %s\n" "$*"; exit 1; }

trap 'fatal "error on line $LINENO – aborting."' ERR

# ─── Arguments ────────────────────────────────────────────────────────────────
FORCE=false
CACHE_ARG=""
for _arg in "$@"; do
    case "$_arg" in
        -f|--force) FORCE=true ;;
        -h|--help)
            _tty "Usage: %s [--force] [path/to/.xkrt_install.cache]\n" "$(basename "$0")"
            _tty "  --force   rebuild & reinstall every component, even if unchanged\n"
            exit 0 ;;
        *) CACHE_ARG="$_arg" ;;
    esac
done

# ─── Locate & load the cached configuration ───────────────────────────────────
if [[ -n "$CACHE_ARG" ]]; then
    CACHE_FILE="$CACHE_ARG"
elif [[ -f "$(pwd)/.xkrt_install.cache" ]]; then
    CACHE_FILE="$(pwd)/.xkrt_install.cache"
elif [[ -f "$SCRIPT_DIR/.xkrt_install.cache" ]]; then
    CACHE_FILE="$SCRIPT_DIR/.xkrt_install.cache"
else
    fatal "no .xkrt_install.cache found in $(pwd) or $SCRIPT_DIR — run install.sh first (or pass the cache path)."
fi
[[ -f "$CACHE_FILE" ]] || fatal "cache file not found: $CACHE_FILE"

# shellcheck disable=SC1090
source "$CACHE_FILE"

: "${BASE_DIR:?cache is missing BASE_DIR — is this a valid install cache?}"
: "${REPO_DIR:?cache is missing REPO_DIR — is this a valid install cache?}"

hr
_tty "  ${BOLD}xkrt ecosystem — in-place update${NC}\n"
_tty "  cache : %s\n" "$CACHE_FILE"
_tty "  repos : %s\n" "$REPO_DIR"
[[ "$FORCE" == "true" ]] && _tty "  mode  : ${BOLD}--force${NC} (rebuild everything)\n"
hr

# ─── Load the generated environment ───────────────────────────────────────────
# install.sh writes env.sh (module use + module load for every component).  Just
# source it: that loads every component's module, so the incremental builds below
# find each other's headers, libraries and cmake config (e.g. <cgir/cgir.hpp>).
ENV_FILE="$BASE_DIR/env.sh"
if [[ -f "$ENV_FILE" ]]; then
    step "Loading modules (source $ENV_FILE)"
    set +u   # 'module' (Lmod) and its init scripts are not always set -u clean

    # Make the 'module' command available first.  This script runs in bash, but it
    # may have been launched from a shell that does not export shell functions to
    # child processes (zsh, notably) — so the 'module' function is NOT inherited,
    # and env.sh would bail.  The Lmod/Modules *environment* (e.g. $LMOD_PKG) IS
    # inherited, so re-source the module-system init to define 'module' here.
    if ! command -v module >/dev/null 2>&1; then
        for _f in "${LMOD_PKG:-/nonexistent}/init/bash" \
                  /usr/share/lmod/lmod/init/bash /usr/local/lmod/lmod/init/bash \
                  /opt/apps/lmod/lmod/init/bash /usr/share/modules/init/bash \
                  /etc/profile.d/modules.sh; do
            [ -r "$_f" ] || continue
            # shellcheck disable=SC1090
            . "$_f" || true
            command -v module >/dev/null 2>&1 && break
        done
    fi

    # shellcheck disable=SC1090
    source "$ENV_FILE" || fatal "failed to source $ENV_FILE (no usable module system found?)"
    set -u
else
    warn "no env.sh at $ENV_FILE — modules not loaded; builds may fail to find their"
    warn "dependencies.  Re-run install.sh to (re)generate it."
fi

# ─── Helpers ──────────────────────────────────────────────────────────────────

# _pull REPO [BRANCH] — fetch and fast-forward the repo (switch branch if needed).
_pull() {
    local repo="$1" branch="${2:-}" current
    info "fetching"
    git -C "$repo" fetch --all --tags --progress
    current="$(git -C "$repo" rev-parse --abbrev-ref HEAD 2>/dev/null || echo '')"
    if [[ -n "$branch" && "$current" != "$branch" ]]; then
        info "checking out '$branch'"
        git -C "$repo" checkout "$branch"
    fi
    if git -C "$repo" rev-parse --abbrev-ref '@{u}' >/dev/null 2>&1; then
        git -C "$repo" pull --ff-only --progress
    else
        info "no upstream tracking branch — fetched only (not pulled)"
    fi
}

# _newest_build_dir REPO [BUILD_TYPE] — newest build dir containing a CMakeCache.
_newest_build_dir() {
    local repo="$1" btype="${2:-}" d t best=0 newest="" glob
    if [[ -n "$btype" ]]; then glob="$repo/build/*/$btype"; else glob="$repo/build/*/*"; fi
    # shellcheck disable=SC2086
    for d in $glob; do
        [[ -f "$d/CMakeCache.txt" ]] || continue
        t="$(stat -c %Y "$d/CMakeCache.txt" 2>/dev/null || echo 0)"
        if (( t > best )); then best="$t"; newest="$d"; fi
    done
    printf '%s' "$newest"
}

DIRTY=false   # set true once any component is (re)built → forces downstream rebuilds

# _changed BEFORE AFTER — true (0) if this component must be (re)built.
_changed() {
    [[ "$FORCE" == "true" || "$DIRTY" == "true" || "$1" != "$2" ]]
}

# update_cmake NAME REPO_SUBDIR BUILD_TYPE BRANCH
update_cmake() {
    local name="$1" repo="$REPO_DIR/$2" btype="${3:-}" branch="${4:-}"
    step "Updating $name"
    if [[ ! -d "$repo/.git" ]]; then
        info "not checked out ($repo) — skipping; install it with install.sh first"
        return 0
    fi
    local before after
    before="$(git -C "$repo" rev-parse HEAD 2>/dev/null || echo none)"
    _pull "$repo" "$branch"
    after="$(git -C "$repo" rev-parse HEAD 2>/dev/null || echo none)"

    if ! _changed "$before" "$after"; then
        success "$name already up to date — skipped"
        return 0
    fi
    DIRTY=true   # downstream components must now rebuild against this one

    local bdir; bdir="$(_newest_build_dir "$repo" "$btype")"
    if [[ -z "$bdir" ]]; then
        warn "no configured build dir under $repo/build — run install.sh for $name first; skipping"
        return 0
    fi
    info "build dir : $bdir"
    make -C "$bdir" -j "$(nproc)"
    make -C "$bdir" install
    success "$name rebuilt & reinstalled"
}

# update_autotools BRANCH  (hwloc — configured & built in-tree by install.sh)
update_autotools() {
    local name="hwloc" repo="$REPO_DIR/hwloc" branch="${1:-}"
    step "Updating $name"
    if [[ ! -d "$repo/.git" ]]; then
        info "not checked out ($repo) — skipping; install it with install.sh first"
        return 0
    fi
    local before after
    before="$(git -C "$repo" rev-parse HEAD 2>/dev/null || echo none)"
    _pull "$repo" "$branch"
    after="$(git -C "$repo" rev-parse HEAD 2>/dev/null || echo none)"

    if ! _changed "$before" "$after"; then
        success "$name already up to date — skipped"
        return 0
    fi
    DIRTY=true   # downstream components must now rebuild against this one

    if [[ ! -f "$repo/Makefile" ]]; then
        warn "hwloc is not configured ($repo/Makefile missing) — run install.sh first; skipping"
        return 0
    fi
    info "build dir : $repo (autotools, in-tree)"
    make -C "$repo" -j "$(nproc)"
    make -C "$repo" install
    success "$name rebuilt & reinstalled"
}

# ─── Update all components in dependency order ─────────────────────────────────
update_cmake     "llvm"   "llvm-project" "${LLVM_BUILD_TYPE:-}"   "${LLVM_BRANCH:-}"
update_autotools                          "${HWLOC_BRANCH:-}"
update_cmake     "cgir"   "cgir"          "${CGIR_BUILD_TYPE:-}"   "${CGIR_BRANCH:-}"
update_cmake     "xkrt"   "xkrt"          "${XKRT_BUILD_TYPE:-}"   "${XKRT_BRANCH:-}"
update_cmake     "xkblas" "xkblas"        "${XKBLAS_BUILD_TYPE:-}" "${XKBLAS_BRANCH:-}"
update_cmake     "xkomp"  "xkomp"         "${XKOMP_BUILD_TYPE:-}"  "${XKOMP_BRANCH:-}"

# ─── Done ─────────────────────────────────────────────────────────────────────
_tty "\n"; hr
if [[ "$DIRTY" == "true" ]]; then
    _tty "  ${BOLD}${GREEN}Update complete.${NC}\n"
else
    _tty "  ${BOLD}${GREEN}Everything was already up to date.${NC}\n"
fi
hr
_tty "\n  Install/module locations are unchanged, so your loaded modules and\n"
_tty "  %s keep working as-is.\n" "$ENV_FILE"
if [[ "${LLVM_RUNTIMES:-}" == *offload* ]]; then
    _tty "\n  ${DIM}Note: if you changed XKRT/XKOMP in a way that affects the custom\n"
    _tty "  libomptarget (offload runtime), re-run install.sh for a consistent\n"
    _tty "  rebuild of that dependency loop.${NC}\n"
fi
_tty "\n"
