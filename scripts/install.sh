#!/usr/bin/env bash
# ============================================================================
# Interactive installer: llvm · hwloc · cgir · xkrt · xkblas · xkomp
#
# Dependency order:
#   llvm ─▶ cgir ─┐
#                    ├─▶  xkrt  ─┬─▶  xkblas
#   hwloc ───────────┘           └─▶  xkomp
#
# The custom LLVM (anlsys/llvm-project) is a dependency of cgir and may
# optionally be used to build cgir/xkrt/xkblas/xkomp.  It can be bootstrapped
# with any compiler, but building those libraries requires an LLVM >= 20 (the
# custom one, or a system clang >= 20).
#
# The custom libomptarget (the LLVM "offload" runtime) forwards OpenMP target
# calls to XKRT and XKOMP, creating a build loop (libomptarget → xkrt + xkomp →
# cgir → clang).  It is therefore built in two stages: LLVM is first built
# WITHOUT offload, then (after xkrt + xkomp are installed) the offload runtime is
# added in place:
#   llvm (no offload) ─▶ cgir ─▶ xkrt ─▶ xkomp ─▶ llvm offload/libomptarget
#
# Requires: cmake >= 3.17, a C/C++ compiler, git, autoconf/automake (for hwloc)
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ─── Colours ─────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; CYAN='\033[0;36m'; BOLD='\033[1m'; DIM='\033[2m'; NC='\033[0m'

# ─── UI helpers ───────────────────────────────────────────────────────────────
# All user-visible output goes to /dev/tty so command-substitution $() captures
# only the actual return values of functions.
_tty() { printf "$@" >/dev/tty; }

hr()      { _tty '%s\n' "────────────────────────────────────────────────────────────────────────"; }
step()    { _tty "\n${BOLD}${CYAN}▶${NC} ${BOLD}%s${NC}\n" "$*"; }
info()    { _tty "  ${BLUE}·${NC} %s\n" "$*"; }
success() { _tty "  ${GREEN}✓${NC} %s\n" "$*"; }
warn()    { _tty "  ${YELLOW}!${NC} %s\n" "$*"; }
fatal()   { _tty "  ${RED}✗ FATAL:${NC} %s\n" "$*"; exit 1; }

# prompt_yn QUESTION [DEFAULT=yes]
# Returns 0 for yes, 1 for no.
prompt_yn() {
    local q="$1" dflt="${2:-yes}" tag
    [[ "$dflt" == "yes" ]] && tag="${BOLD}[Y/n]${NC}" || tag="${BOLD}[y/N]${NC}"
    while true; do
        _tty "  ${BOLD}${BLUE}?${NC} %b %b: " "$q" "$tag"
        local ans; read -r ans </dev/tty
        ans="${ans:-$dflt}"
        case "${ans,,}" in
            y|yes) return 0 ;;
            n|no)  return 1 ;;
            *) _tty "    Please answer yes or no.\n" ;;
        esac
    done
}

# prompt_value QUESTION DEFAULT
# Prints the user's answer (or the default) on stdout.
prompt_value() {
    _tty "  ${BOLD}${BLUE}?${NC} %s [${DIM}%s${NC}]: " "$1" "$2"
    local ans; read -r ans </dev/tty
    printf '%s' "${ans:-$2}"
}

# run_spinner MESSAGE [--] COMMAND [ARGS...]
# Runs COMMAND in the background while animating a spinner + elapsed seconds on
# /dev/tty, so long, quiet operations (e.g. a huge 'git checkout' updating the
# working tree) don't look frozen.  COMMAND's output is captured and shown only
# if it fails (then the script aborts via fatal).  On success the line is
# replaced by a green check.  Use ONLY for non-interactive commands — output is
# captured, so anything that might prompt (clone/fetch/pull on a private repo)
# must NOT be wrapped or it would hang invisibly.
run_spinner() {
    local msg="$1"; shift
    [[ "${1:-}" == "--" ]] && shift

    local logf; logf="$(mktemp "${TMPDIR:-/tmp}/xkrt_spin.XXXXXX")"
    # Reset the ERR trap in the subshell so a failure is reported by us (via the
    # captured log below) rather than printing a FATAL from the background.
    ( trap - ERR; "$@" ) >"$logf" 2>&1 &
    local pid=$! rc=0
    local frames='|/-\' i=0 start=$SECONDS

    while kill -0 "$pid" 2>/dev/null; do
        _tty "\r  ${BLUE}%s${NC} %s ${DIM}(%ds)${NC}\033[K" \
             "${frames:i++%4:1}" "$msg" "$((SECONDS - start))"
        sleep 0.2
    done
    wait "$pid" || rc=$?

    if (( rc == 0 )); then
        _tty "\r  ${GREEN}✓${NC} %s ${DIM}(%ds)${NC}\033[K\n" "$msg" "$((SECONDS - start))"
        rm -f "$logf"
    else
        _tty "\r  ${RED}✗${NC} %s ${DIM}(failed after %ds)${NC}\033[K\n" "$msg" "$((SECONDS - start))"
        _tty "  ${DIM}----- command output -----${NC}\n"
        cat "$logf" >/dev/tty 2>/dev/null || true
        rm -f "$logf"
        fatal "command failed: $*"
    fi
}

# ─── CMake option parsing ─────────────────────────────────────────────────────

# parse_cmake_opts FILE
# Prints one "VARNAME|description|DEFAULT" line per xkoption/ocgoption entry.
parse_cmake_opts() {
    local f="$1"
    [[ -f "$f" ]] || return 0
    grep -E '^\s*(xkoption|ocgoption)\s*\(' "$f" \
      | sed -E 's/^\s*(xkoption|ocgoption)\s*\(\s*([A-Za-z0-9_]+)\s+"([^"]+)"\s+(ON|OFF)\s*\)/\2|\3|\4/' \
      | grep -E '^[A-Za-z0-9_]+\|' \
      || true
}

# ask_cmake_opts LIBNAME CMAKE_FILE [CACHED_OPTS]
# Asks whether to use cmake defaults or customise, then prompts accordingly.
# If CACHED_OPTS is non-empty, first offers to reuse it verbatim.
# Prints the accumulated "-DVAR=VAL ..." flag string on stdout.
ask_cmake_opts() {
    local lib="$1" f="$2" cached="${3:-}" flags=""

    # Fast path: reuse the cmake options recorded in a previous run.
    if [[ -n "$cached" ]]; then
        _tty "\n  ${BOLD}Cached cmake options for %s${NC}:\n    ${DIM}%s${NC}\n" "$lib" "$cached"
        if prompt_yn "  Reuse these cached cmake options for $lib?" "yes"; then
            printf '%s' "$cached"; return
        fi
    fi

    if [[ ! -f "$f" ]]; then
        _tty "  ${YELLOW}!${NC} CMakeLists.txt not found at %s – skipping option prompts.\n" "$f"
        printf ''; return
    fi

    local opts
    opts=$(parse_cmake_opts "$f")

    # No xkoption/ocgoption entries: just offer the free-form extra-flags field.
    if [[ -z "$opts" ]]; then
        _tty "  ${DIM}(no xkoption/ocgoption entries found in CMakeLists.txt)${NC}\n"
        _tty "  ${BOLD}${BLUE}?${NC} Extra cmake flags for %s (or Enter to skip): " "$lib"
        local extra; read -r extra </dev/tty
        printf '%s' "${extra:-}"; return
    fi

    # There are options: ask whether to use all defaults or customise.
    _tty "\n  ${BOLD}CMake options for %s${NC} (parsed from CMakeLists.txt):\n" "$lib"
    _tty "  ${DIM}Note: these reflect the local clone; may differ on other branches.${NC}\n\n"

    if prompt_yn "  Use all default cmake options?" "yes"; then
        # Fast path: return empty flags; cmake will apply its own defaults.
        printf ''; return
    fi

    # Customise: ask each option individually, then offer extra flags.
    while IFS='|' read -r var desc default; do
        [[ -z "$var" ]] && continue
        local def_yn
        [[ "$default" == "ON" ]] && def_yn="yes" || def_yn="no"
        _tty "    ${DIM}%-45s${NC} %s\n" "$var" "$desc"
        if prompt_yn "      Enable ${BOLD}${var}${NC}?" "$def_yn"; then
            flags="${flags} -D${var}=ON"
        else
            flags="${flags} -D${var}=OFF"
        fi
    done <<< "$opts"

    _tty "\n  ${BOLD}${BLUE}?${NC} Extra cmake flags for %s (e.g. -DFOO=bar), or Enter to skip: " "$lib"
    local extra; read -r extra </dev/tty
    flags="${flags}${extra:+ $extra}"

    printf '%s' "${flags# }"   # strip leading space
}

# ─── Configuration cache ─────────────────────────────────────────────────────

# _write_cache FILE
# Serialises every Phase-1 output variable into a sourceable bash file so the
# user can skip re-answering all questions on a re-run.
_write_cache() {
    local f="$1"
    {
        printf '# xkrt install configuration — %s\n' "$(date '+%Y-%m-%d %H:%M:%S')"
        declare -p BASE_DIR REPO_DIR INSTALL_DIR MODULES_DIR
        declare -p CC CXX
        declare -p INSTALL_LLVM LLVM_BRANCH LLVM_BUILD_TYPE \
                   LLVM_PROJECTS LLVM_RUNTIMES \
                   LLVM_CMAKE_TARGETS LLVM_CMAKE_RUNTIME_TARGETS \
                   LLVM_EXTRA_CMAKE_OPTS LLVM_GPU_SUMMARY \
                   USE_LLVM_FOR_BUILD LLVM_BOOTSTRAP_CC LLVM_BOOTSTRAP_CXX
        declare -p INSTALL_HWLOC HWLOC_BRANCH HWLOC_CONFIGURE_OPTS
        declare -p INSTALL_CGIR CGIR_BRANCH CGIR_BUILD_TYPE CGIR_CMAKE_OPTS
        declare -p INSTALL_XKRT  XKRT_BRANCH  XKRT_BUILD_TYPE  XKRT_CMAKE_OPTS
        declare -p INSTALL_XKBLAS XKBLAS_BRANCH XKBLAS_BUILD_TYPE XKBLAS_CMAKE_OPTS
        declare -p INSTALL_XKOMP  XKOMP_BRANCH  XKOMP_BUILD_TYPE  XKOMP_CMAKE_OPTS
    } > "$f"
}

# ─── Module-file generation ──────────────────────────────────────────────────

# generate_modulefile LIBNAME PREFIX ENVVAR OUTFILE [DEP ...]
# ENVVAR : environment variable set to the install prefix (e.g. XKRT_HOME)
# DEP    : zero or more dependency module paths ("name/hash/type") that will be
#          auto-loaded via 'module load' when this module is loaded.
generate_modulefile() {
    local lib="$1" prefix="$2" envvar="$3" out="$4"
    shift 4
    local -a deps=("$@")

    mkdir -p "$(dirname "$out")"

    {
        cat <<MODEOF
#%Module1.0

set whatis    "$lib"
set software  "$lib"
set description "$lib"

conflict "\$software"

MODEOF

        # Emit dependency loads before path manipulation so their libs are
        # visible as soon as this module is loaded.
        if (( ${#deps[@]} > 0 )); then
            echo "# ── Dependencies ────────────────────────────────────────────────────────────"
            echo "module use \"$MODULES_DIR\""
            for dep in "${deps[@]}"; do
                echo "module load \"$dep\""
            done
            echo ""
        fi

        cat <<MODEOF
set prefix "$prefix"

prepend-path PATH               "\$prefix/bin"
prepend-path MANPATH            "\$prefix/share/man"
prepend-path INFOPATH           "\$prefix/share/info"
prepend-path LIBRARY_PATH       "\$prefix/lib"
prepend-path LIBRARY_PATH       "\$prefix/lib64"
prepend-path LD_LIBRARY_PATH    "\$prefix/lib"
prepend-path LD_LIBRARY_PATH    "\$prefix/lib64"
prepend-path CMAKE_PREFIX_PATH  "\$prefix"
prepend-path CMAKE_LIBRARY_PATH "\$prefix/lib"
prepend-path CMAKE_INCLUDE_PATH "\$prefix/include"
prepend-path C_INCLUDE_PATH     "\$prefix/include"
prepend-path CPATH              "\$prefix/include"
prepend-path PKG_CONFIG_PATH    "\$prefix/lib/pkgconfig"

setenv $envvar "\$prefix"
MODEOF
    } > "$out"
}

# ─── Git helpers ─────────────────────────────────────────────────────────────

# clone_or_update URL DEST REF
# REF can be a branch name or a tag.
clone_or_update() {
    local url="$1" dest="$2" ref="$3"

    if [[ ! -d "$dest/.git" ]]; then
        info "Cloning $(basename "$dest") from $url"
        # --progress forces the transfer meter even when stderr is not a TTY,
        # so large clones (e.g. LLVM) don't look frozen.
        git clone --progress "$url" "$dest"
    else
        info "Fetching latest for $(basename "$dest")"
        git -C "$dest" fetch --all --tags --progress
    fi

    # 'git checkout' can spend many seconds updating the working tree/index of a
    # huge repo (LLVM) with no output of its own, so show a spinner.  It is a
    # purely local operation, so capturing its output (in run_spinner) is safe.
    run_spinner "Checking out $ref" \
        git -C "$dest" checkout "$ref"

    # Pull only if this ref is a remote branch (tags are immutable)
    if git -C "$dest" ls-remote --exit-code --heads origin "$ref" >/dev/null 2>&1; then
        git -C "$dest" pull --progress
    fi
}

# ─── Build environment helpers ────────────────────────────────────────────────

# _init_module_system
# Tries to make the 'module' shell function available by sourcing common Lmod /
# Environment-Modules init files.  Sets _MODULE_AVAILABLE=true on success.
# Non-fatal: if no module system is found we fall back to manual env exports.
_MODULE_AVAILABLE=false
_init_module_system() {
    if declare -f module > /dev/null 2>&1; then
        _MODULE_AVAILABLE=true; return 0
    fi
    local f
    for f in \
        "${LMOD_PKG:-NOFILE}/init/bash" \
        "/usr/share/lmod/lmod/init/bash" \
        "/usr/local/lmod/lmod/init/bash" \
        "/opt/apps/lmod/lmod/init/bash" \
        "/usr/share/modules/init/bash" \
        "/etc/profile.d/modules.sh"
    do
        [[ -f "$f" ]] || continue
        # shellcheck disable=SC1090
        source "$f" 2>/dev/null && { _MODULE_AVAILABLE=true; return 0; }
    done
    warn "Module system not found; environment will be configured manually."
}

# _activate_prefix PREFIX [MOD_NAME MOD_SUBPATH]
# Exports every variable that a 'module load' would set for PREFIX:
#   PATH, CPATH, C_INCLUDE_PATH, CPLUS_INCLUDE_PATH,
#   LIBRARY_PATH, LD_LIBRARY_PATH,
#   CMAKE_PREFIX_PATH, CMAKE_LIBRARY_PATH, CMAKE_INCLUDE_PATH,
#   PKG_CONFIG_PATH
# If the module system is available it also runs 'module load MOD_NAME/MOD_SUBPATH'
# (belt-and-suspenders: both the env vars and the module are applied).
_activate_prefix() {
    local prefix="$1" mod_name="${2:-}" mod_sub="${3:-}"

    # Export all paths that compilers, linkers, cmake and pkg-config consult.
    export PATH="${prefix}/bin${PATH:+:$PATH}"
    export C_INCLUDE_PATH="${prefix}/include${C_INCLUDE_PATH:+:$C_INCLUDE_PATH}"
    export CPLUS_INCLUDE_PATH="${prefix}/include${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"
    export CPATH="${prefix}/include${CPATH:+:$CPATH}"
    export LIBRARY_PATH="${prefix}/lib:${prefix}/lib64${LIBRARY_PATH:+:$LIBRARY_PATH}"
    export LD_LIBRARY_PATH="${prefix}/lib:${prefix}/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    export CMAKE_PREFIX_PATH="${prefix}${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"
    export CMAKE_LIBRARY_PATH="${prefix}/lib${CMAKE_LIBRARY_PATH:+:$CMAKE_LIBRARY_PATH}"
    export CMAKE_INCLUDE_PATH="${prefix}/include${CMAKE_INCLUDE_PATH:+:$CMAKE_INCLUDE_PATH}"
    export PKG_CONFIG_PATH="${prefix}/lib/pkgconfig${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"

    # Also load the module if the system is available (idempotent, non-fatal).
    if [[ "$_MODULE_AVAILABLE" == "true" && -n "$mod_name" && -n "$mod_sub" ]]; then
        module use "$MODULES_DIR" 2>/dev/null || true
        module load "${mod_name}/${mod_sub}" 2>/dev/null || true
    fi
}

# _llvm_host_target
# Maps uname -m to the LLVM backend name needed in LLVM_TARGETS_TO_BUILD.
_llvm_host_target() {
    case "$(uname -m)" in
        x86_64)  printf 'X86'      ;;
        aarch64) printf 'AArch64'  ;;
        arm*)    printf 'ARM'      ;;
        ppc64le) printf 'PowerPC'  ;;
        ppc64)   printf 'PowerPC'  ;;
        riscv64) printf 'RISCV'    ;;
        s390x)   printf 'SystemZ'  ;;
        *)       printf 'X86'      ;;   # safe fallback
    esac
}

# _list_contains LIST TOKEN
# Returns 0 if the ';'-separated LIST contains TOKEN, 1 otherwise.
_list_contains() {
    local IFS=';' x
    for x in $1; do [[ "$x" == "$2" ]] && return 0; done
    return 1
}

# _strip_token LIST TOKEN
# Prints the ';'-separated LIST with every occurrence of TOKEN (and empty
# fields) removed.
_strip_token() {
    local IFS=';' x result=""
    for x in $1; do
        [[ -z "$x" || "$x" == "$2" ]] && continue
        result="${result:+$result;}$x"
    done
    printf '%s' "$result"
}

# build_llvm RUNTIMES LABEL
# (Re)configure, build and install LLVM in $LLVM_BUILD_DIR with the given
# LLVM_ENABLE_RUNTIMES value (RUNTIMES may be empty to enable no runtimes).
# cmake is always re-run, so this can be invoked twice for the staged build:
# first without the offload runtime, then again to add the custom libomptarget
# once XKRT exists.  The top-level build always uses the bootstrap compiler so a
# second invocation is not seen as a compiler change (which forces a full rebuild).
# Globals: LLVM_BUILD_DIR LLVM_INSTALL_DIR LLVM_REPO_DIR LLVM_BUILD_TYPE
#   LLVM_PROJECTS LLVM_CMAKE_RUNTIME_TARGETS LLVM_CMAKE_TARGETS
#   LLVM_EXTRA_CMAKE_OPTS LLVM_BOOTSTRAP_CC LLVM_BOOTSTRAP_CXX
build_llvm() {
    local runtimes="$1" label="$2"
    local rt_flag=""
    [[ -n "$runtimes" ]] && rt_flag="-DLLVM_ENABLE_RUNTIMES=${runtimes}"

    info "Configuring LLVM (${label}) …"
    cd "$LLVM_BUILD_DIR"

    # shellcheck disable=SC2086
    # The top-level LLVM build always uses the bootstrap compiler.  LLVM_DIR /
    # MLIR_DIR are scrubbed from the environment here (they are exported AFTER
    # stage 1 to pin the XK* libraries to this LLVM) so building LLVM itself — and
    # its runtimes sub-build, e.g. the deferred offload/libomptarget — never tries
    # to resolve an external LLVM install.  XKRT_DIR / XKOMP_DIR / CMAKE_PREFIX_PATH
    # are kept, so find_package(XKRT)/find_package(XKOMP) in libomptarget still work.
    env -u LLVM_DIR -u MLIR_DIR \
        CC="$LLVM_BOOTSTRAP_CC" CXX="$LLVM_BOOTSTRAP_CXX" \
        cmake \
        -DCMAKE_BUILD_TYPE="$LLVM_BUILD_TYPE" \
        -DCMAKE_INSTALL_PREFIX="$LLVM_INSTALL_DIR" \
        -DLLVM_ENABLE_PROJECTS="$LLVM_PROJECTS" \
        ${rt_flag:+"$rt_flag"} \
        -DLLVM_RUNTIME_TARGETS="$LLVM_CMAKE_RUNTIME_TARGETS" \
        -DLLVM_TARGETS_TO_BUILD="$LLVM_CMAKE_TARGETS" \
        -DCMAKE_CXX_FLAGS="-Wno-c2y-extensions" \
        $LLVM_EXTRA_CMAKE_OPTS \
        "$LLVM_REPO_DIR/llvm"

    info "Building & installing LLVM (${label}) — this may take a while …"
    env -u LLVM_DIR -u MLIR_DIR make install -j "$(nproc)"
}

# ─── Error trap ───────────────────────────────────────────────────────────────
trap 'fatal "error on line $LINENO – aborting."' ERR

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 1 – GATHER CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

# _dflt_yn CACHED_BOOL FALLBACK
# Maps a cached "true"/"false" to a "yes"/"no" prompt default; uses FALLBACK when
# there is no cached value.
_dflt_yn() {
    case "${1:-}" in
        true)  printf 'yes' ;;
        false) printf 'no'  ;;
        *)     printf '%s' "$2" ;;
    esac
}

# _dflt_member HAVE_CACHE LIST TOKEN FALLBACK
# "yes"/"no" prompt default for an option encoded as TOKEN's presence in a cached
# ';'-separated LIST; uses FALLBACK when there is no cache.
_dflt_member() {
    [[ "$1" == "true" ]] || { printf '%s' "$4"; return; }
    if _list_contains "$2" "$3"; then printf 'yes'; else printf 'no'; fi
}

CACHE_FILE="$(pwd)/.xkrt_install.cache"
REUSE_CACHE=false
HAVE_CACHE=false

if [[ -f "$CACHE_FILE" ]]; then
    # Load the cache up front so its values can seed the default answer of every
    # question below, whether or not the user skips straight to the build.
    # shellcheck disable=SC1090
    source "$CACHE_FILE"
    HAVE_CACHE=true

    # Snapshot the cached values: each component block below resets its working
    # variables before prompting, so keep a copy to use as the prompt defaults.
    _C_BASE_DIR="${BASE_DIR:-}"
    _C_CC="${CC:-}"; _C_CXX="${CXX:-}"
    _C_INSTALL_LLVM="${INSTALL_LLVM:-}"
    _C_LLVM_BRANCH="${LLVM_BRANCH:-}"; _C_LLVM_BUILD_TYPE="${LLVM_BUILD_TYPE:-}"
    _C_LLVM_PROJECTS="${LLVM_PROJECTS:-}"; _C_LLVM_RUNTIMES="${LLVM_RUNTIMES:-}"
    _C_LLVM_CMAKE_TARGETS="${LLVM_CMAKE_TARGETS:-}"
    _C_LLVM_EXTRA_CMAKE_OPTS="${LLVM_EXTRA_CMAKE_OPTS:-}"
    _C_USE_LLVM_FOR_BUILD="${USE_LLVM_FOR_BUILD:-}"
    _C_LLVM_BOOTSTRAP_CC="${LLVM_BOOTSTRAP_CC:-}"; _C_LLVM_BOOTSTRAP_CXX="${LLVM_BOOTSTRAP_CXX:-}"
    _C_INSTALL_HWLOC="${INSTALL_HWLOC:-}"; _C_HWLOC_BRANCH="${HWLOC_BRANCH:-}"
    _C_HWLOC_CONFIGURE_OPTS="${HWLOC_CONFIGURE_OPTS:-}"
    _C_INSTALL_CGIR="${INSTALL_CGIR:-}"; _C_CGIR_BRANCH="${CGIR_BRANCH:-}"; _C_CGIR_BUILD_TYPE="${CGIR_BUILD_TYPE:-}"; _C_CGIR_CMAKE_OPTS="${CGIR_CMAKE_OPTS:-}"
    _C_INSTALL_XKRT="${INSTALL_XKRT:-}"; _C_XKRT_BRANCH="${XKRT_BRANCH:-}"; _C_XKRT_BUILD_TYPE="${XKRT_BUILD_TYPE:-}"; _C_XKRT_CMAKE_OPTS="${XKRT_CMAKE_OPTS:-}"
    _C_INSTALL_XKBLAS="${INSTALL_XKBLAS:-}"; _C_XKBLAS_BRANCH="${XKBLAS_BRANCH:-}"; _C_XKBLAS_BUILD_TYPE="${XKBLAS_BUILD_TYPE:-}"; _C_XKBLAS_CMAKE_OPTS="${XKBLAS_CMAKE_OPTS:-}"
    _C_INSTALL_XKOMP="${INSTALL_XKOMP:-}"; _C_XKOMP_BRANCH="${XKOMP_BRANCH:-}"; _C_XKOMP_BUILD_TYPE="${XKOMP_BUILD_TYPE:-}"; _C_XKOMP_CMAKE_OPTS="${XKOMP_CMAKE_OPTS:-}"

    _tty "\n"
    hr
    _tty "  ${BOLD}${YELLOW}Cached configuration found${NC}\n"
    _tty "  %s\n" "$CACHE_FILE"
    _tty "  Created: %s\n" "$(date -r "$CACHE_FILE" '+%Y-%m-%d %H:%M:%S' 2>/dev/null \
                               || stat -c '%y' "$CACHE_FILE" 2>/dev/null | cut -d. -f1)"
    hr
    _tty "\n"
    _tty "  ${DIM}Tip: even if you answer 'no' below, these cached values are used as the\n"
    _tty "  default answer to every question — just press Enter to accept each one.${NC}\n\n"
    if prompt_yn "Reuse this configuration and skip straight to the build?" "yes"; then
        REUSE_CACHE=true
        success "Configuration loaded — jumping to build phase."
    else
        info "Re-running configuration (cached values pre-filled as the defaults)."
    fi
fi

if [[ "$REUSE_CACHE" == "false" ]]; then

_tty "\n"
hr
_tty "  ${BOLD}xkrt ecosystem – interactive installer${NC}\n"
_tty "  llvm  ·  hwloc  ·  cgir  ·  xkrt  ·  xkblas  ·  xkomp\n"
hr
_tty "\n"

# ── Installation base ─────────────────────────────────────────────────────────
step "Installation directory"
BASE_DIR=$(prompt_value "Base directory (repos, installs and modules go here)" "${_C_BASE_DIR:-$(pwd)}")
BASE_DIR="$(realpath -m "$BASE_DIR")"
REPO_DIR="$BASE_DIR/repo"
INSTALL_DIR="$BASE_DIR/install"
MODULES_DIR="$BASE_DIR/modules"
info "Repos    → $REPO_DIR"
info "Install  → $INSTALL_DIR"
info "Modules  → $MODULES_DIR"

# Create the repo directory now so Phase 1 can clone into it for CMakeLists.txt
# parsing before the full build phase begins.
mkdir -p "$REPO_DIR"

# ── Compilers ─────────────────────────────────────────────────────────────────
step "Compilers"

# Minimum required clang major version.
readonly MIN_CLANG_VER=20

# _detect_clang
# Scans PATH for clang-<N>/clang++-<N> pairs and the unversioned clang/clang++.
# Prints one line: "STATUS CC CXX VERSION"
#   STATUS "ok"   – found a pair with version >= MIN_CLANG_VER (CC/CXX/VERSION set)
#   STATUS "old"  – found clang but all versions < MIN_CLANG_VER (VERSION=max found)
#   STATUS "none" – no clang found at all
_detect_clang() {
    local best_ver=0 best_cc="" best_cxx="" max_any=0
    local d base ver cc cxx

    # 1. Search versioned clang-<N> / clang++-<N> in every PATH directory.
    local -a path_dirs
    IFS=: read -ra path_dirs <<< "${PATH:-}"
    for d in "${path_dirs[@]}"; do
        [[ -d "$d" ]] || continue
        for cc in "$d"/clang-[0-9]*; do
            [[ -x "$cc" ]] || continue
            base="${cc##*/}"
            # Accept only exact "clang-<digits>" (exclude clang-cpp-19, clangd-19, …)
            [[ "$base" =~ ^clang-([0-9]+)$ ]] || continue
            ver="${BASH_REMATCH[1]}"
            cxx="${d}/clang++-${ver}"
            [[ -x "$cxx" ]] || continue          # need the matching C++ compiler
            (( ver > max_any )) && max_any=$ver
            if (( ver >= MIN_CLANG_VER && ver > best_ver )); then
                best_ver=$ver; best_cc="$cc"; best_cxx="$cxx"
            fi
        done
    done

    # 2. Also check unversioned 'clang' / 'clang++' (may be a symlink to latest).
    if command -v clang &>/dev/null && command -v clang++ &>/dev/null; then
        local vs; vs=$(clang --version 2>/dev/null | head -1)
        if [[ "$vs" =~ ([0-9]+)\.[0-9]+ ]]; then
            ver="${BASH_REMATCH[1]}"
            (( ver > max_any )) && max_any=$ver
            if (( ver >= MIN_CLANG_VER && ver > best_ver )); then
                best_ver=$ver
                best_cc=$(command -v clang)
                best_cxx=$(command -v clang++)
            fi
        fi
    fi

    if [[ -n "$best_cc" ]]; then
        printf 'ok %s %s %d' "$best_cc" "$best_cxx" "$best_ver"
    elif (( max_any > 0 )); then
        printf 'old - - %d' "$max_any"
    else
        printf 'none - - 0'
    fi
}

info "Scanning PATH for clang >= ${MIN_CLANG_VER} …"
read -r _clang_status _def_cc _def_cxx _clang_ver <<< "$(_detect_clang)"

# A system clang >= MIN_CLANG_VER is required to build the libraries *unless*
# the user installs the custom LLVM and chooses to build with it.  We therefore
# only record the result here and defer any hard error until the build compiler
# is selected (see "Build compiler" below).
case "$_clang_status" in
    ok)   success "Found ${_def_cc} / ${_def_cxx}  (version ${_clang_ver})" ;;
    old)  warn "Found clang ${_clang_ver}, but >= ${MIN_CLANG_VER} is needed to build the libraries." ;;
    none) warn "No clang >= ${MIN_CLANG_VER} found in PATH." ;;
esac
[[ "$_clang_status" == "ok" ]] || \
    info "You can still proceed by installing the custom LLVM and building with it."

# Pick a sensible default *bootstrap* compiler (any compiler is fine for that):
# prefer a detected clang >= MIN_CLANG_VER, else any clang, else gcc, else cc/c++.
if [[ "$_clang_status" == "ok" ]]; then
    _boot_cc_default="$_def_cc"; _boot_cxx_default="$_def_cxx"
else
    _boot_cc_default="$(command -v clang  || command -v gcc || command -v cc  || echo cc)"
    _boot_cxx_default="$(command -v clang++ || command -v g++ || command -v c++ || echo c++)"
fi

# ── LLVM (custom patched) ─────────────────────────────────────────────────────
step "LLVM  [cmake; custom patched toolchain — anlsys/llvm-project; dependency of cgir]"
_tty "\n"
_tty "  The anlsys fork adds new OpenMP pragma support (access clauses, etc.) and\n"
_tty "  provides the LLVM that cgir depends on.\n"
_tty "  It can be bootstrapped with any compiler.  You may optionally use it to\n"
_tty "  build cgir/xkrt/xkblas/xkomp — those require an LLVM >= ${MIN_CLANG_VER}\n"
_tty "  (the custom one, or a system clang >= ${MIN_CLANG_VER} detected above).\n\n"

INSTALL_LLVM=false
LLVM_BRANCH="" LLVM_BUILD_TYPE=""
LLVM_PROJECTS="" LLVM_RUNTIMES=""
LLVM_CMAKE_TARGETS="" LLVM_CMAKE_RUNTIME_TARGETS=""
LLVM_EXTRA_CMAKE_OPTS=""
LLVM_GPU_SUMMARY=""   # human-readable, for summary display
USE_LLVM_FOR_BUILD=false                      # use custom LLVM to build the libs?
LLVM_BOOTSTRAP_CC="" LLVM_BOOTSTRAP_CXX=""    # compiler used to build LLVM itself

if prompt_yn "Install custom patched LLVM?" "$(_dflt_yn "${_C_INSTALL_LLVM:-}" "no")"; then
    INSTALL_LLVM=true
    LLVM_BRANCH=$(prompt_value "Branch" "${_C_LLVM_BRANCH:-main}")
    LLVM_BUILD_TYPE=$(prompt_value "Build type (Release/Debug/RelWithDebInfo)" "${_C_LLVM_BUILD_TYPE:-Release}")

    # ── Host LLVM backend (auto-detected from uname -m) ──────────────────────
    _host_tgt=$(_llvm_host_target)
    info "Host LLVM backend: ${_host_tgt}  (uname -m = $(uname -m))"

    # ── GPU offload targets (multi-select) ────────────────────────────────────
    _tty "\n"
    _tty "  ${BOLD}GPU offload targets${NC} — select the GPU architectures to support:\n"
    _tty "\n"
    _tty "    ${BOLD}1)${NC} NVIDIA / CUDA\n"
    _tty "       LLVM backend : NVPTX\n"
    _tty "       Runtime target: nvptx64-nvidia-cuda\n"
    _tty "\n"
    _tty "    ${BOLD}2)${NC} AMD / ROCm\n"
    _tty "       LLVM backend : AMDGPU\n"
    _tty "       Runtime target: amdgcn-amd-amdhsa\n"
    _tty "\n"
    # Reconstruct the previously-selected GPU numbers from the cached backends.
    _gpu_default=""
    _list_contains "${_C_LLVM_CMAKE_TARGETS:-}" NVPTX  && _gpu_default="${_gpu_default}${_gpu_default:+ }1"
    _list_contains "${_C_LLVM_CMAKE_TARGETS:-}" AMDGPU && _gpu_default="${_gpu_default}${_gpu_default:+ }2"
    _tty "  Enter numbers separated by spaces (e.g. \"1 2\"), or Enter for [%s]: " "${_gpu_default:-host-only}"
    read -r _gpu_sel </dev/tty
    _gpu_sel="${_gpu_sel:-$_gpu_default}"

    _backends="$_host_tgt"
    _runtime_tgts="default"
    _gpu_names=()

    for _g in $_gpu_sel; do
        case "$_g" in
            1)
                _backends="${_backends};NVPTX"
                _runtime_tgts="${_runtime_tgts};nvptx64-nvidia-cuda"
                _gpu_names+=("NVIDIA/CUDA")
                ;;
            2)
                _backends="${_backends};AMDGPU"
                _runtime_tgts="${_runtime_tgts};amdgcn-amd-amdhsa"
                _gpu_names+=("AMD/ROCm")
                ;;
            *)
                warn "Unknown selection '$_g' – ignored."
                ;;
        esac
    done

    LLVM_CMAKE_TARGETS="$_backends"
    LLVM_CMAKE_RUNTIME_TARGETS="$_runtime_tgts"

    if (( ${#_gpu_names[@]} == 0 )); then
        LLVM_GPU_SUMMARY="host-only"
    else
        LLVM_GPU_SUMMARY=$(IFS='+'; printf '%s' "${_gpu_names[*]}")
    fi

    info "LLVM_TARGETS_TO_BUILD  = $LLVM_CMAKE_TARGETS"
    info "LLVM_RUNTIME_TARGETS   = $LLVM_CMAKE_RUNTIME_TARGETS"

    # ── Projects ─────────────────────────────────────────────────────────────
    _tty "\n  ${BOLD}LLVM projects${NC} (clang is always included):\n"
    _projects="clang"
    if prompt_yn "  Include MLIR  (recommended for CGIR)?" "$(_dflt_member "$HAVE_CACHE" "${_C_LLVM_PROJECTS:-}" mlir "yes")"; then
        _projects="${_projects};mlir"
    fi
    if prompt_yn "  Include lld  (LLVM linker — recommended for GPU offload)?" "$(_dflt_member "$HAVE_CACHE" "${_C_LLVM_PROJECTS:-}" lld "yes")"; then
        _projects="${_projects};lld"
    fi
    if prompt_yn "  Include bolt (binary optimizer)?" "$(_dflt_member "$HAVE_CACHE" "${_C_LLVM_PROJECTS:-}" bolt "no")"; then
        _projects="${_projects};bolt"
    fi
    LLVM_PROJECTS="$_projects"

    # ── Runtimes ─────────────────────────────────────────────────────────────
    _tty "\n  ${BOLD}LLVM runtimes${NC}:\n"
    _runtimes=""
    if prompt_yn "  Build openmp  (OpenMP host runtime)?" "$(_dflt_member "$HAVE_CACHE" "${_C_LLVM_RUNTIMES:-}" openmp "yes")"; then
        _runtimes="${_runtimes}${_runtimes:+;}openmp"
    fi
    _tty "  ${DIM}Note: the custom libomptarget depends on XKRT and XKOMP,\n"
    _tty "  so when enabled it is built last — after XKRT and XKOMP.${NC}\n"
    if prompt_yn "  Build offload (OpenMP GPU offload runtime / libomptarget)?" "$(_dflt_member "$HAVE_CACHE" "${_C_LLVM_RUNTIMES:-}" offload "yes")"; then
        _runtimes="${_runtimes}${_runtimes:+;}offload"
    fi
    LLVM_RUNTIMES="$_runtimes"

    # ── Extra flags ───────────────────────────────────────────────────────────
    LLVM_EXTRA_CMAKE_OPTS=$(prompt_value "Extra cmake flags for LLVM (or Enter to keep/skip)" "${_C_LLVM_EXTRA_CMAKE_OPTS:-}")

    # ── Bootstrap compiler (any compiler may build LLVM) ──────────────────────
    _tty "\n  ${BOLD}Bootstrap compiler${NC} — used only to compile LLVM itself.\n"
    _tty "  ${DIM}Any C/C++ compiler works here; no minimum version.${NC}\n"
    LLVM_BOOTSTRAP_CC=$(prompt_value  "Bootstrap C compiler"   "${_C_LLVM_BOOTSTRAP_CC:-$_boot_cc_default}")
    LLVM_BOOTSTRAP_CXX=$(prompt_value "Bootstrap C++ compiler" "${_C_LLVM_BOOTSTRAP_CXX:-$_boot_cxx_default}")

    # ── Use the custom LLVM to build the libraries? ───────────────────────────
    _tty "\n"
    if prompt_yn "Use this custom LLVM to build cgir/xkrt/xkblas/xkomp?" "$(_dflt_yn "${_C_USE_LLVM_FOR_BUILD:-}" "yes")"; then
        USE_LLVM_FOR_BUILD=true
    else
        USE_LLVM_FOR_BUILD=false
    fi

fi

# ── Build compiler ────────────────────────────────────────────────────────────
# cgir/xkrt/xkblas/xkomp must be built with an LLVM >= MIN_CLANG_VER: either
# the custom LLVM (built above) or a system clang >= MIN_CLANG_VER.
step "Build compiler  [for cgir · xkrt · xkblas · xkomp]"
if [[ "$INSTALL_LLVM" == "true" && "$USE_LLVM_FOR_BUILD" == "true" ]]; then
    # The custom LLVM provides clang >= 20.  Its exact path depends on the git
    # hash, so CC/CXX are resolved in Phase 2 right after LLVM is installed.
    CC="" CXX=""
    success "Libraries will be built with the custom LLVM (clang/clang++)."
else
    # Not using the custom LLVM → a system clang >= MIN_CLANG_VER is required.
    if [[ "$_clang_status" != "ok" ]]; then
        _tty "\n  ${RED}Error:${NC} building the libraries needs an LLVM >= ${MIN_CLANG_VER}.\n"
        if [[ "$_clang_status" == "old" ]]; then
            _tty "  Found clang ${_clang_ver}, which is too old.\n"
        else
            _tty "  No clang/clang++ was found in PATH.\n"
        fi
        _tty "  Either install a newer clang, e.g.:\n"
        _tty "    sudo apt install clang-${MIN_CLANG_VER} clang++-${MIN_CLANG_VER}\n"
        _tty "  or re-run, install the custom LLVM and choose to build with it.\n\n"
        exit 1
    fi
    _tty "\n"
    CC=$(prompt_value  "C compiler (LLVM >= ${MIN_CLANG_VER})"   "${_C_CC:-$_def_cc}")
    CXX=$(prompt_value "C++ compiler (LLVM >= ${MIN_CLANG_VER})" "${_C_CXX:-$_def_cxx}")
fi
export CC CXX

# ── hwloc ─────────────────────────────────────────────────────────────────────
step "hwloc  [autotools; no cmake dependencies]"
_tty "\n"
_tty "  hwloc is required by xkrt for hardware topology discovery.\n"
_tty "  If you skip this step, xkrt will attempt to use a system-installed hwloc\n"
_tty "  (searched in standard paths by Findhwloc.cmake).\n\n"
_tty "  hwloc's optional GPU backends (cuda/nvml/rsmi/opencl/levelzero) make\n"
_tty "  libhwloc hard-link GPU libraries that are often not on the linker/loader\n"
_tty "  path, which breaks downstream links.  They are disabled by default; pass\n"
_tty "  e.g. '--enable-rsmi --with-rocm=/opt/rocm' below to re-enable any you need.\n\n"

INSTALL_HWLOC=false; HWLOC_BRANCH=""; HWLOC_CONFIGURE_OPTS=""
if prompt_yn "Install hwloc?" "$(_dflt_yn "${_C_INSTALL_HWLOC:-}" "yes")"; then
    INSTALL_HWLOC=true
    HWLOC_BRANCH=$(prompt_value "Branch / tag" "${_C_HWLOC_BRANCH:-v2.14}")
    HWLOC_CONFIGURE_OPTS=$(prompt_value "Extra hwloc configure flags (or Enter to keep/skip)" "${_C_HWLOC_CONFIGURE_OPTS:-}")
fi

# ── cgir ────────────────────────────────────────────────────────────────────
step "cgir  [cmake; no runtime dependencies; parallel to hwloc]"
INSTALL_CGIR=false
CGIR_BRANCH=""; CGIR_BUILD_TYPE=""; CGIR_CMAKE_OPTS=""
if prompt_yn "Install cgir?" "$(_dflt_yn "${_C_INSTALL_CGIR:-}" "yes")"; then
    INSTALL_CGIR=true
    CGIR_BRANCH=$(prompt_value "Branch" "${_C_CGIR_BRANCH:-release/latest}")
    CGIR_BUILD_TYPE=$(prompt_value "Build type (Release/Debug)" "${_C_CGIR_BUILD_TYPE:-Release}")
    clone_or_update "https://github.com/JLESC-Tasking-Group/opencg" \
        "$REPO_DIR/cgir" "$CGIR_BRANCH"
    CGIR_CMAKE_OPTS=$(ask_cmake_opts "cgir" "$REPO_DIR/cgir/CMakeLists.txt" "${_C_CGIR_CMAKE_OPTS:-}")
fi

# ── xkrt ──────────────────────────────────────────────────────────────────────
step "xkrt  [cmake; depends on hwloc + cgir]"
INSTALL_XKRT=false
XKRT_BRANCH=""; XKRT_BUILD_TYPE=""; XKRT_CMAKE_OPTS=""
if prompt_yn "Install xkrt?" "$(_dflt_yn "${_C_INSTALL_XKRT:-}" "yes")"; then
    INSTALL_XKRT=true
    XKRT_BRANCH=$(prompt_value "Branch" "${_C_XKRT_BRANCH:-release/latest}")
    XKRT_BUILD_TYPE=$(prompt_value "Build type (Release/Debug)" "${_C_XKRT_BUILD_TYPE:-Release}")
    clone_or_update "https://gitlab.inria.fr/xkaapi/dev-v2" \
        "$REPO_DIR/xkrt" "$XKRT_BRANCH"
    XKRT_CMAKE_OPTS=$(ask_cmake_opts "xkrt" "$REPO_DIR/xkrt/CMakeLists.txt" "${_C_XKRT_CMAKE_OPTS:-}")
fi

# ── xkblas ────────────────────────────────────────────────────────────────────
step "xkblas  [cmake; depends on xkrt; parallel to xkomp]"
_tty "\n"
_tty "  Tip: if you enable USE_CBLAS, also set USE_OPENBLAS, USE_MKL, or\n"
_tty "  USE_CRAYBLAS via the 'extra cmake flags' prompt below.\n\n"
INSTALL_XKBLAS=false
XKBLAS_BRANCH=""; XKBLAS_BUILD_TYPE=""; XKBLAS_CMAKE_OPTS=""
if prompt_yn "Install xkblas?" "$(_dflt_yn "${_C_INSTALL_XKBLAS:-}" "yes")"; then
    INSTALL_XKBLAS=true
    XKBLAS_BRANCH=$(prompt_value "Branch" "${_C_XKBLAS_BRANCH:-release/v2.0-latest}")
    XKBLAS_BUILD_TYPE=$(prompt_value "Build type (Release/Debug)" "${_C_XKBLAS_BUILD_TYPE:-Release}")
    clone_or_update "https://gitlab.inria.fr/xkblas/dev" \
        "$REPO_DIR/xkblas" "$XKBLAS_BRANCH"
    XKBLAS_CMAKE_OPTS=$(ask_cmake_opts "xkblas" "$REPO_DIR/xkblas/CMakeLists.txt" "${_C_XKBLAS_CMAKE_OPTS:-}")
fi

# ── xkomp ─────────────────────────────────────────────────────────────────────
step "xkomp  [cmake; depends on xkrt; parallel to xkblas]"
INSTALL_XKOMP=false
XKOMP_BRANCH=""; XKOMP_BUILD_TYPE=""; XKOMP_CMAKE_OPTS=""
if prompt_yn "Install xkomp?" "$(_dflt_yn "${_C_INSTALL_XKOMP:-}" "yes")"; then
    INSTALL_XKOMP=true
    XKOMP_BRANCH=$(prompt_value "Branch" "${_C_XKOMP_BRANCH:-release/latest}")
    XKOMP_BUILD_TYPE=$(prompt_value "Build type (Release/Debug)" "${_C_XKOMP_BUILD_TYPE:-Release}")
    clone_or_update "https://github.com/anlsys/xkomp" \
        "$REPO_DIR/xkomp" "$XKOMP_BRANCH"
    XKOMP_CMAKE_OPTS=$(ask_cmake_opts "xkomp" "$REPO_DIR/xkomp/CMakeLists.txt" "${_C_XKOMP_CMAKE_OPTS:-}")
fi

# ── Summary & confirmation ────────────────────────────────────────────────────
_tty "\n"; hr
_tty "  ${BOLD}Summary${NC}\n"; hr
_tty "\n"

_lib_row() {
    local name="$1" install="$2" branch="${3:-}" btype="${4:-}"
    if [[ "$install" == "true" ]]; then
        _tty "  ${GREEN}✓${NC}  %-10s  branch: %-22s  build: %s\n" \
             "$name" "$branch" "$btype"
    else
        _tty "  ${DIM}✗  %-10s  (skipped)${NC}\n" "$name"
    fi
}

if [[ "$INSTALL_LLVM" == "true" ]]; then
    _tty "  ${GREEN}✓${NC}  %-10s  branch: %-22s  build: %s\n" \
         "llvm" "$LLVM_BRANCH" "$LLVM_BUILD_TYPE"
    _tty "          projects: %-20s  runtimes: %s\n" \
         "$LLVM_PROJECTS" "${LLVM_RUNTIMES:-none}"
    _tty "          GPU targets: %s\n" "$LLVM_GPU_SUMMARY"
    _tty "          bootstrap: %-20s  build libs with llvm: %s\n" \
         "$LLVM_BOOTSTRAP_CC" "$USE_LLVM_FOR_BUILD"
else
    _tty "  ${DIM}✗  %-10s  (skipped)${NC}\n" "llvm"
fi
if [[ "$INSTALL_LLVM" == "true" && "$USE_LLVM_FOR_BUILD" == "true" ]]; then
    _tty "  ${DIM}library compiler: custom LLVM clang (resolved after LLVM build)${NC}\n"
else
    _tty "  ${DIM}library compiler: %s${NC}\n" "$CC"
fi
_lib_row "hwloc"  "$INSTALL_HWLOC"  "$HWLOC_BRANCH"  "autotools"
_lib_row "cgir" "$INSTALL_CGIR" "$CGIR_BRANCH" "$CGIR_BUILD_TYPE"
_lib_row "xkrt"   "$INSTALL_XKRT"   "$XKRT_BRANCH"   "$XKRT_BUILD_TYPE"
_lib_row "xkblas" "$INSTALL_XKBLAS" "$XKBLAS_BRANCH" "$XKBLAS_BUILD_TYPE"
_lib_row "xkomp"  "$INSTALL_XKOMP"  "$XKOMP_BRANCH"  "$XKOMP_BUILD_TYPE"
_tty "\n"

if ! prompt_yn "Proceed with installation?" "yes"; then
    info "Aborted by user."; exit 0
fi

_write_cache "$CACHE_FILE"
success "Configuration saved → $CACHE_FILE"

fi # REUSE_CACHE — end of Phase 1

# Edge-safe defaults — also covers reusing a cache written by an older version
# of this script that predates the LLVM-as-build-compiler options.
: "${INSTALL_LLVM:=false}"
: "${USE_LLVM_FOR_BUILD:=false}"
: "${LLVM_BOOTSTRAP_CC:=${CC:-cc}}"
: "${LLVM_BOOTSTRAP_CXX:=${CXX:-c++}}"
: "${HWLOC_CONFIGURE_OPTS:=}"

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 2 – BUILD & INSTALL
# ─────────────────────────────────────────────────────────────────────────────

mkdir -p "$REPO_DIR" "$INSTALL_DIR" "$MODULES_DIR"

declare -a MOD_LOAD=()   # module load lines for final usage message

_init_module_system      # try to make 'module' available for _activate_prefix

# ── LLVM ──────────────────────────────────────────────────────────────────────
if [[ "$INSTALL_LLVM" == "true" ]]; then
    step "Building & installing LLVM"

    # Use the local clone if it lives inside the scripts dir; otherwise clone.
    LLVM_REPO_DIR="$REPO_DIR/llvm-project"
    if [[ -d "$SCRIPT_DIR/llvm-project/.git" ]]; then
        info "Found local llvm-project clone at $SCRIPT_DIR/llvm-project"
        if [[ "$SCRIPT_DIR/llvm-project" != "$LLVM_REPO_DIR" ]]; then
            # Symlink so we don't duplicate the (large) checkout
            mkdir -p "$(dirname "$LLVM_REPO_DIR")"
            [[ -e "$LLVM_REPO_DIR" ]] || ln -s "$SCRIPT_DIR/llvm-project" "$LLVM_REPO_DIR"
        fi
    else
        clone_or_update \
            "https://github.com/anlsys/llvm-project" \
            "$LLVM_REPO_DIR" \
            "$LLVM_BRANCH"
    fi

    # Always fetch + checkout the requested branch in the repo we'll use.
    # LLVM is a very large repository, so keep the progress meters on (no -q)
    # to make it obvious the fetch/checkout are working and not hung.
    info "Updating LLVM sources for branch '$LLVM_BRANCH' (large repo — this can take a while) …"
    git -C "$LLVM_REPO_DIR" fetch --all --tags --progress
    # Checking out a branch in the LLVM tree rewrites a huge working tree and can
    # sit silently for many seconds — show a spinner so it does not look hung.
    run_spinner "Checking out '$LLVM_BRANCH' (updating the working tree)" \
        git -C "$LLVM_REPO_DIR" checkout "$LLVM_BRANCH"
    if git -C "$LLVM_REPO_DIR" ls-remote --exit-code --heads origin "$LLVM_BRANCH" >/dev/null 2>&1; then
        git -C "$LLVM_REPO_DIR" pull --progress
    fi

    LLVM_HASH=$(git -C "$LLVM_REPO_DIR" rev-parse HEAD | cut -c1-12)
    LLVM_INSTALL_DIR="$INSTALL_DIR/llvm/$LLVM_HASH/$LLVM_BUILD_TYPE"
    LLVM_BUILD_DIR="$LLVM_REPO_DIR/build/$LLVM_HASH/$LLVM_BUILD_TYPE"

    mkdir -p "$LLVM_BUILD_DIR"

    # The custom libomptarget (the "offload" runtime) forwards OpenMP target
    # calls to XKRT, so it depends on XKRT → CGIR → clang.  To break that loop
    # we build LLVM now WITHOUT offload, and (re)build the offload runtime later,
    # after CGIR and XKRT are installed (see "custom libomptarget" below).
    if _list_contains "$LLVM_RUNTIMES" offload; then
        LLVM_BUILD_OFFLOAD=true
    else
        LLVM_BUILD_OFFLOAD=false
    fi
    LLVM_STAGE1_RUNTIMES="$(_strip_token "$LLVM_RUNTIMES" offload)"

    if [[ "$LLVM_BUILD_OFFLOAD" == "true" ]]; then
        info "offload/libomptarget requested → deferred until after XKRT (dependency loop)."
        build_llvm "$LLVM_STAGE1_RUNTIMES" "1/2 — without libomptarget"
    else
        build_llvm "$LLVM_STAGE1_RUNTIMES" "single stage"
    fi

    # Pin this EXACT LLVM/MLIR for every downstream build (cgir directly, and
    # the rest transitively), independently of whether it is also the compiler.
    # _activate_prefix puts the prefix on CMAKE_PREFIX_PATH; LLVM_DIR / MLIR_DIR
    # point find_package() straight at this build's cmake packages.  Both env
    # vars are read by find_package and inherited by every downstream cmake
    # (including transitive find_dependency(LLVM) pulled in via cgir).
    _activate_prefix "$LLVM_INSTALL_DIR" "llvm" "$LLVM_HASH/$LLVM_BUILD_TYPE"
    if   [[ -d "$LLVM_INSTALL_DIR/lib/cmake/llvm"   ]]; then export LLVM_DIR="$LLVM_INSTALL_DIR/lib/cmake/llvm"
    elif [[ -d "$LLVM_INSTALL_DIR/lib64/cmake/llvm" ]]; then export LLVM_DIR="$LLVM_INSTALL_DIR/lib64/cmake/llvm"
    fi
    if   [[ -d "$LLVM_INSTALL_DIR/lib/cmake/mlir"   ]]; then export MLIR_DIR="$LLVM_INSTALL_DIR/lib/cmake/mlir"
    elif [[ -d "$LLVM_INSTALL_DIR/lib64/cmake/mlir" ]]; then export MLIR_DIR="$LLVM_INSTALL_DIR/lib64/cmake/mlir"
    fi
    info "Pinned LLVM_DIR=${LLVM_DIR:-<not found>}"
    if [[ -n "${MLIR_DIR:-}" ]]; then
        info "Pinned MLIR_DIR=$MLIR_DIR"
    else
        warn "No MLIR cmake package under $LLVM_INSTALL_DIR — enable the 'mlir' LLVM project if cgir needs MLIR."
    fi

    # If the user opted to build the libraries with the custom LLVM, also switch
    # the build compiler to its clang/clang++.
    if [[ "$USE_LLVM_FOR_BUILD" == "true" ]]; then
        CC="$LLVM_INSTALL_DIR/bin/clang"
        CXX="$LLVM_INSTALL_DIR/bin/clang++"
        export CC CXX
        success "Libraries will be built with: $CC"
    fi

    # ── Module file ───────────────────────────────────────────────────────────
    LLVM_MOD_DIR="$MODULES_DIR/llvm/$LLVM_HASH"
    LLVM_MOD="$LLVM_MOD_DIR/$LLVM_BUILD_TYPE"
    mkdir -p "$LLVM_MOD_DIR"
    cat > "$LLVM_MOD" <<MODEOF
#%Module1.0

set whatis    "LLVM (anlsys patched)"
set software  "llvm"
set description "LLVM toolchain — anlsys/llvm-project @ $LLVM_HASH"

conflict "\$software"

set prefix "$LLVM_INSTALL_DIR"

prepend-path PATH               "\$prefix/bin"
prepend-path MANPATH            "\$prefix/share/man"
prepend-path LIBRARY_PATH       "\$prefix/lib"
prepend-path LIBRARY_PATH       "\$prefix/lib64"
prepend-path LD_LIBRARY_PATH    "\$prefix/lib"
prepend-path LD_LIBRARY_PATH    "\$prefix/lib64"
prepend-path CMAKE_PREFIX_PATH  "\$prefix"
prepend-path CMAKE_LIBRARY_PATH "\$prefix/lib"
prepend-path CMAKE_INCLUDE_PATH "\$prefix/include"
prepend-path C_INCLUDE_PATH     "\$prefix/include"
prepend-path CPATH              "\$prefix/include"
prepend-path PKG_CONFIG_PATH    "\$prefix/lib/pkgconfig"

# Set CC/CXX so downstream builds automatically use this clang.
setenv CC  "\$prefix/bin/clang"
setenv CXX "\$prefix/bin/clang++"
setenv LLVM_HOME "\$prefix"
MODEOF

    MOD_LOAD+=("module load llvm/$LLVM_HASH/$LLVM_BUILD_TYPE")
    if [[ "$LLVM_BUILD_OFFLOAD" == "true" ]]; then
        success "LLVM (without libomptarget) installed → $LLVM_INSTALL_DIR"
        info    "libomptarget (offload) will be built after XKRT."
    else
        success "LLVM  installed → $LLVM_INSTALL_DIR"
    fi
    success "module file     → $LLVM_MOD"
fi

# ── hwloc ─────────────────────────────────────────────────────────────────────
if [[ "$INSTALL_HWLOC" == "true" ]]; then
    step "Building & installing hwloc"
    clone_or_update \
        "https://github.com/open-mpi/hwloc" \
        "$REPO_DIR/hwloc" \
        "$HWLOC_BRANCH"

    HWLOC_HASH=$(git -C "$REPO_DIR/hwloc" rev-parse HEAD | cut -c1-12)
    HWLOC_INSTALL_DIR="$INSTALL_DIR/hwloc/$HWLOC_HASH"

    cd "$REPO_DIR/hwloc"
    if [[ ! -f configure ]]; then
        info "Running autogen.sh …"
        ./autogen.sh
    fi
    info "Configuring …"
    # Disable hwloc's optional GPU backends by default — they make libhwloc
    # hard-depend on GPU libraries (librocm_smi64, libcudart, …) that are often
    # not on the linker/loader search path and break downstream links (xkrt).
    # Any user HWLOC_CONFIGURE_OPTS come last so they can re-enable a specific
    # backend, e.g. "--enable-rsmi --with-rocm=/opt/rocm".
    # shellcheck disable=SC2086
    ./configure --prefix="$HWLOC_INSTALL_DIR" --quiet \
        --disable-cuda --disable-nvml --disable-rsmi \
        --disable-opencl --disable-levelzero \
        $HWLOC_CONFIGURE_OPTS
    info "Building …"
    make -j "$(nproc)"
    info "Installing …"
    make install

    # Activate the installed hwloc so downstream builds (xkrt) see its headers,
    # libraries and cmake config.  This sets PATH, CPATH, LD_LIBRARY_PATH,
    # CMAKE_PREFIX_PATH, etc. — equivalent to 'module load hwloc/…'.
    HWLOC_MOD="$MODULES_DIR/hwloc/$HWLOC_HASH/default"
    generate_modulefile "hwloc" "$HWLOC_INSTALL_DIR" "HWLOC_HOME" "$HWLOC_MOD"
    _activate_prefix "$HWLOC_INSTALL_DIR" "hwloc" "$HWLOC_HASH/default"
    MOD_LOAD+=("module load hwloc/$HWLOC_HASH/default")
    success "hwloc  installed → $HWLOC_INSTALL_DIR"
    success "module file      → $HWLOC_MOD"
fi

# ── cgir ────────────────────────────────────────────────────────────────────
if [[ "$INSTALL_CGIR" == "true" ]]; then
    step "Building & installing cgir"
    clone_or_update \
        "https://github.com/JLESC-Tasking-Group/opencg" \
        "$REPO_DIR/cgir" \
        "$CGIR_BRANCH"

    CGIR_HASH=$(git -C "$REPO_DIR/cgir" rev-parse HEAD | cut -c1-12)
    CGIR_INSTALL_DIR="$INSTALL_DIR/cgir/$CGIR_HASH/$CGIR_BUILD_TYPE"
    CGIR_BUILD_DIR="$REPO_DIR/cgir/build/$CGIR_HASH/$CGIR_BUILD_TYPE"

    rm -rf "$CGIR_BUILD_DIR" && mkdir -p "$CGIR_BUILD_DIR"
    cd "$CGIR_BUILD_DIR"

    # If a custom LLVM was built, force cgir's find_package(LLVM)/find_package(MLIR)
    # onto that exact build so it can never silently fall back to a system LLVM/MLIR.
    _cgir_llvm_flags=""
    if [[ "$INSTALL_LLVM" == "true" ]]; then
        [[ -n "${LLVM_DIR:-}" ]] && _cgir_llvm_flags="-DLLVM_DIR=$LLVM_DIR"
        [[ -n "${MLIR_DIR:-}" ]] && _cgir_llvm_flags="${_cgir_llvm_flags:+$_cgir_llvm_flags }-DMLIR_DIR=$MLIR_DIR"
    fi

    # shellcheck disable=SC2086
    cmake $CGIR_CMAKE_OPTS $_cgir_llvm_flags \
        -DCMAKE_BUILD_TYPE="$CGIR_BUILD_TYPE" \
        -DCMAKE_INSTALL_PREFIX="$CGIR_INSTALL_DIR" \
        "$REPO_DIR/cgir"
    make install -j "$(nproc)"

    # Activate cgir so xkrt finds its headers, libs and cmake config.
    CGIR_MOD="$MODULES_DIR/cgir/$CGIR_HASH/$CGIR_BUILD_TYPE"
    # cgir builds against (and links) the custom LLVM/MLIR whenever one was
    # built, so its module must load that llvm module at runtime — regardless of
    # whether the custom LLVM was also used as the compiler.  (A system LLVM, like
    # a system hwloc, ships no module to load.)
    declare -a _cgir_deps=()
    [[ "$INSTALL_LLVM" == "true" ]] && \
        _cgir_deps+=("llvm/$LLVM_HASH/$LLVM_BUILD_TYPE")
    generate_modulefile "cgir" "$CGIR_INSTALL_DIR" "CGIR_HOME" "$CGIR_MOD" \
        ${_cgir_deps[@]+"${_cgir_deps[@]}"}
    _activate_prefix "$CGIR_INSTALL_DIR" "cgir" "$CGIR_HASH/$CGIR_BUILD_TYPE"
    MOD_LOAD+=("module load cgir/$CGIR_HASH/$CGIR_BUILD_TYPE")
    success "cgir installed → $CGIR_INSTALL_DIR"
    success "module file      → $CGIR_MOD"
fi

# ── xkrt ──────────────────────────────────────────────────────────────────────
if [[ "$INSTALL_XKRT" == "true" ]]; then
    step "Building & installing xkrt"
    clone_or_update \
        "https://gitlab.inria.fr/xkaapi/dev-v2" \
        "$REPO_DIR/xkrt" \
        "$XKRT_BRANCH"

    XKRT_HASH=$(git -C "$REPO_DIR/xkrt" rev-parse HEAD | cut -c1-12)
    XKRT_INSTALL_DIR="$INSTALL_DIR/xkrt/$XKRT_HASH/$XKRT_BUILD_TYPE"
    XKRT_BUILD_DIR="$REPO_DIR/xkrt/build/$XKRT_HASH/$XKRT_BUILD_TYPE"

    rm -rf "$XKRT_BUILD_DIR" && mkdir -p "$XKRT_BUILD_DIR"
    cd "$XKRT_BUILD_DIR"

    # shellcheck disable=SC2086
    cmake $XKRT_CMAKE_OPTS \
        -DCMAKE_BUILD_TYPE="$XKRT_BUILD_TYPE" \
        -DCMAKE_INSTALL_PREFIX="$XKRT_INSTALL_DIR" \
        "$REPO_DIR/xkrt"
    make install -j "$(nproc)"

    # Activate xkrt so xkblas and xkomp find its headers, libs and cmake config.
    XKRT_MOD="$MODULES_DIR/xkrt/$XKRT_HASH/$XKRT_BUILD_TYPE"
    # Build the dependency list: cgir is always required; hwloc only when
    # it was installed from source by this script (otherwise it is system-provided).
    declare -a _xkrt_deps=()
    [[ "$INSTALL_CGIR" == "true" ]] && _xkrt_deps+=("cgir/$CGIR_HASH/$CGIR_BUILD_TYPE")
    [[ "$INSTALL_HWLOC"  == "true" ]] && _xkrt_deps+=("hwloc/$HWLOC_HASH/default")
    generate_modulefile "xkrt" "$XKRT_INSTALL_DIR" "XKRT_HOME" "$XKRT_MOD" \
        ${_xkrt_deps[@]+"${_xkrt_deps[@]}"}
    _activate_prefix "$XKRT_INSTALL_DIR" "xkrt" "$XKRT_HASH/$XKRT_BUILD_TYPE"
    MOD_LOAD+=("module load xkrt/$XKRT_HASH/$XKRT_BUILD_TYPE")
    success "xkrt  installed  → $XKRT_INSTALL_DIR"
    success "module file      → $XKRT_MOD"
fi

# ── xkblas ────────────────────────────────────────────────────────────────────
if [[ "$INSTALL_XKBLAS" == "true" ]]; then
    step "Building & installing xkblas"
    clone_or_update \
        "https://gitlab.inria.fr/xkblas/dev" \
        "$REPO_DIR/xkblas" \
        "$XKBLAS_BRANCH"

    XKBLAS_HASH=$(git -C "$REPO_DIR/xkblas" rev-parse HEAD | cut -c1-12)
    XKBLAS_INSTALL_DIR="$INSTALL_DIR/xkblas/$XKBLAS_HASH/$XKBLAS_BUILD_TYPE"
    XKBLAS_BUILD_DIR="$REPO_DIR/xkblas/build/$XKBLAS_HASH/$XKBLAS_BUILD_TYPE"

    rm -rf "$XKBLAS_BUILD_DIR" && mkdir -p "$XKBLAS_BUILD_DIR"
    cd "$XKBLAS_BUILD_DIR"

    # shellcheck disable=SC2086
    cmake $XKBLAS_CMAKE_OPTS \
        -DCMAKE_BUILD_TYPE="$XKBLAS_BUILD_TYPE" \
        -DCMAKE_INSTALL_PREFIX="$XKBLAS_INSTALL_DIR" \
        "$REPO_DIR/xkblas"
    make install -j "$(nproc)"

    XKBLAS_MOD="$MODULES_DIR/xkblas/$XKBLAS_HASH/$XKBLAS_BUILD_TYPE"
    declare -a _xkblas_deps=()
    [[ "$INSTALL_XKRT" == "true" ]] && _xkblas_deps+=("xkrt/$XKRT_HASH/$XKRT_BUILD_TYPE")
    generate_modulefile "xkblas" "$XKBLAS_INSTALL_DIR" "XKBLAS_HOME" "$XKBLAS_MOD" \
        ${_xkblas_deps[@]+"${_xkblas_deps[@]}"}
    MOD_LOAD+=("module load xkblas/$XKBLAS_HASH/$XKBLAS_BUILD_TYPE")
    success "xkblas installed → $XKBLAS_INSTALL_DIR"
    success "module file      → $XKBLAS_MOD"
fi

# ── xkomp ─────────────────────────────────────────────────────────────────────
if [[ "$INSTALL_XKOMP" == "true" ]]; then
    step "Building & installing xkomp"
    clone_or_update \
        "https://github.com/anlsys/xkomp" \
        "$REPO_DIR/xkomp" \
        "$XKOMP_BRANCH"

    XKOMP_HASH=$(git -C "$REPO_DIR/xkomp" rev-parse HEAD | cut -c1-12)
    XKOMP_INSTALL_DIR="$INSTALL_DIR/xkomp/$XKOMP_HASH/$XKOMP_BUILD_TYPE"
    XKOMP_BUILD_DIR="$REPO_DIR/xkomp/build/$XKOMP_HASH/$XKOMP_BUILD_TYPE"

    rm -rf "$XKOMP_BUILD_DIR" && mkdir -p "$XKOMP_BUILD_DIR"
    cd "$XKOMP_BUILD_DIR"

    # shellcheck disable=SC2086
    cmake $XKOMP_CMAKE_OPTS \
        -DCMAKE_BUILD_TYPE="$XKOMP_BUILD_TYPE" \
        -DCMAKE_INSTALL_PREFIX="$XKOMP_INSTALL_DIR" \
        "$REPO_DIR/xkomp"
    make install -j "$(nproc)"

    XKOMP_MOD="$MODULES_DIR/xkomp/$XKOMP_HASH/$XKOMP_BUILD_TYPE"
    declare -a _xkomp_deps=()
    [[ "$INSTALL_XKRT" == "true" ]] && _xkomp_deps+=("xkrt/$XKRT_HASH/$XKRT_BUILD_TYPE")
    generate_modulefile "xkomp" "$XKOMP_INSTALL_DIR" "XKOMP_HOME" "$XKOMP_MOD" \
        ${_xkomp_deps[@]+"${_xkomp_deps[@]}"}
    MOD_LOAD+=("module load xkomp/$XKOMP_HASH/$XKOMP_BUILD_TYPE")
    success "xkomp installed  → $XKOMP_INSTALL_DIR"
    success "module file      → $XKOMP_MOD"
fi

# ── LLVM offload runtime (custom libomptarget) ─────────────────────────────────
# Deferred from the LLVM step above: the custom libomptarget depends on XKRT and
# XKOMP (it forwards OpenMP target calls to them), so it can only be built now
# that both exist.  We reconfigure the existing LLVM build to add the offload
# runtime and install it into the same prefix (only the offload runtime is
# compiled — the rest of LLVM is already built).
if [[ "$INSTALL_LLVM" == "true" && "$LLVM_BUILD_OFFLOAD" == "true" ]]; then
    step "Building & installing custom libomptarget (LLVM offload runtime)"

    # libomptarget does find_package(XKRT) and find_package(XKOMP), so both must
    # be discoverable BEFORE we reconfigure LLVM.  Explicitly (re)load their
    # modules and export the variables find_package consults: it looks at
    # <pkg>_DIR and CMAKE_PREFIX_PATH (NOT <pkg>_HOME).  _activate_prefix puts
    # each prefix on CMAKE_PREFIX_PATH, and the LLVM runtimes sub-build inherits
    # these environment variables, which is what resolves the lookups.
    if [[ "$INSTALL_XKRT" == "true" ]]; then
        info "Loading the XKRT module so libomptarget (offload) can find it …"
        _activate_prefix "$XKRT_INSTALL_DIR" "xkrt" "$XKRT_HASH/$XKRT_BUILD_TYPE"
        export XKRT_HOME="$XKRT_INSTALL_DIR"
        export XKRT_DIR="$XKRT_INSTALL_DIR"          # find_package(XKRT) hint
        success "XKRT activated (CMAKE_PREFIX_PATH now includes $XKRT_INSTALL_DIR)"
    else
        warn "XKRT was not installed by this script, but the custom libomptarget"
        warn "depends on it — load your XKRT module (or put its prefix on"
        warn "CMAKE_PREFIX_PATH, or set XKRT_DIR) before this step."
    fi

    if [[ "$INSTALL_XKOMP" == "true" ]]; then
        info "Loading the XKOMP module so libomptarget (offload) can find it …"
        _activate_prefix "$XKOMP_INSTALL_DIR" "xkomp" "$XKOMP_HASH/$XKOMP_BUILD_TYPE"
        export XKOMP_HOME="$XKOMP_INSTALL_DIR"
        export XKOMP_DIR="$XKOMP_INSTALL_DIR"        # find_package(XKOMP) hint
        success "XKOMP activated (CMAKE_PREFIX_PATH now includes $XKOMP_INSTALL_DIR)"
    else
        warn "XKOMP was not installed by this script, but the custom libomptarget"
        warn "depends on it — load your XKOMP module (or put its prefix on"
        warn "CMAKE_PREFIX_PATH, or set XKOMP_DIR) before this step."
    fi

    # Force the LLVM runtimes to fully reconfigure & rebuild.  Stage 1 configured
    # the runtimes WITHOUT offload; just re-running cmake with offload added does
    # NOT reconfigure the already-stamped runtimes sub-build (see the cached
    # runtimes-stamps/ under $LLVM_BUILD_DIR/runtimes), so the host libomptarget.so
    # gets silently skipped while the device *.bc still build.  Removing the
    # runtimes sub-build forces a clean openmp+offload build for every
    # LLVM_RUNTIME_TARGETS entry, while keeping the already-built clang/llvm/mlir
    # (far cheaper than wiping the whole LLVM build).
    rm -rf "$LLVM_BUILD_DIR/runtimes"
    build_llvm "$LLVM_RUNTIMES" "2/2 — with libomptarget"

    # Sanity-check that the host offload runtime (libomptarget.so) was produced;
    # without it, OpenMP target links fail with "unable to find library -lomptarget".
    _llvm_omptarget=""
    for _d in "$LLVM_INSTALL_DIR/lib" "$LLVM_INSTALL_DIR/lib64"; do
        for _f in "$_d"/libomptarget.so*; do
            [[ -e "$_f" ]] && { _llvm_omptarget="$_f"; break 2; }
        done
    done
    if [[ -n "$_llvm_omptarget" ]]; then
        success "libomptarget (offload) installed → $_llvm_omptarget"
    else
        warn "offload build finished but no host libomptarget.so was found under"
        warn "$LLVM_INSTALL_DIR/lib{,64} — OpenMP target links (-lomptarget) will fail."
        warn "Check the LLVM offload build output above."
    fi
fi

# ─────────────────────────────────────────────────────────────────────────────
# Done – write env.sh and print usage
# ─────────────────────────────────────────────────────────────────────────────

# Write a sourceable env.sh that loads every module installed in this run, so
# users can `source env.sh` instead of re-typing the module commands by hand.
ENV_FILE="$BASE_DIR/env.sh"
{
cat <<EOF
#!/usr/bin/env bash
# Auto-generated by install.sh on $(date '+%Y-%m-%d %H:%M:%S')
#
# Source this file to load the installed xkrt-ecosystem modules:
#   source "$ENV_FILE"

if ! command -v module >/dev/null 2>&1; then
    echo "env.sh: the 'module' command is not available — initialise your module" >&2
    echo "        system (Lmod / Environment Modules) first, then re-source this file." >&2
    return 1 2>/dev/null || exit 1
fi

module use "$MODULES_DIR"
EOF
for line in ${MOD_LOAD[@]+"${MOD_LOAD[@]}"}; do
    printf '%s\n' "$line"
done
} > "$ENV_FILE"

_tty "\n"; hr
_tty "  ${BOLD}${GREEN}Installation complete!${NC}\n"; hr
_tty "\n  A source-able environment file was written — load everything with:\n\n"
_tty "    ${BOLD}source %s${NC}\n" "$ENV_FILE"
_tty "\n  ${DIM}(it runs the following; add them to your shell rc or a job script if you prefer)${NC}\n\n"
_tty "    module use %s\n" "$MODULES_DIR"
for line in ${MOD_LOAD[@]+"${MOD_LOAD[@]}"}; do
    _tty "    %s\n" "$line"
done
_tty "\n  Then compile against the installed libraries, for example:\n"
_tty "    clang++ main.cc -lxkblas\n"
_tty "\n  You may also test the installation by running 'xkrt_info'\n"
_tty "\n"
