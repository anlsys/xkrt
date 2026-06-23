#!/usr/bin/env bash
# ============================================================================
# Interactive installer: llvm · hwloc · opencg · xkrt · xkblas · xkomp
#
# Dependency order:
#   llvm ─▶ opencg ─┐
#                    ├─▶  xkrt  ─┬─▶  xkblas
#   hwloc ───────────┘           └─▶  xkomp
#
# The custom LLVM (anlsys/llvm-project) is a dependency of opencg and may
# optionally be used to build opencg/xkrt/xkblas/xkomp.  It can be bootstrapped
# with any compiler, but building those libraries requires an LLVM >= 20 (the
# custom one, or a system clang >= 20).
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

# ask_cmake_opts LIBNAME CMAKE_FILE
# Asks whether to use cmake defaults or customise, then prompts accordingly.
# Prints the accumulated "-DVAR=VAL ..." flag string on stdout.
ask_cmake_opts() {
    local lib="$1" f="$2" flags=""

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
        declare -p INSTALL_HWLOC HWLOC_BRANCH
        declare -p INSTALL_OPENCG OPENCG_BRANCH OPENCG_BUILD_TYPE OPENCG_CMAKE_OPTS
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
        git clone -q "$url" "$dest"
    else
        info "Fetching latest for $(basename "$dest")"
        git -C "$dest" fetch -q --all --tags
    fi

    info "Checking out $ref"
    git -C "$dest" checkout -q "$ref"

    # Pull only if this ref is a remote branch (tags are immutable)
    if git -C "$dest" ls-remote --exit-code --heads origin "$ref" >/dev/null 2>&1; then
        git -C "$dest" pull -q
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

# ─── Error trap ───────────────────────────────────────────────────────────────
trap 'fatal "error on line $LINENO – aborting."' ERR

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 1 – GATHER CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

CACHE_FILE="$(pwd)/.xkrt_install.cache"
REUSE_CACHE=false

if [[ -f "$CACHE_FILE" ]]; then
    _tty "\n"
    hr
    _tty "  ${BOLD}${YELLOW}Cached configuration found${NC}\n"
    _tty "  %s\n" "$CACHE_FILE"
    _tty "  Created: %s\n" "$(date -r "$CACHE_FILE" '+%Y-%m-%d %H:%M:%S' 2>/dev/null \
                               || stat -c '%y' "$CACHE_FILE" 2>/dev/null | cut -d. -f1)"
    hr
    _tty "\n"
    if prompt_yn "Reuse this configuration and skip to the build?" "yes"; then
        # shellcheck disable=SC1090
        source "$CACHE_FILE"
        REUSE_CACHE=true
        success "Configuration loaded — jumping to build phase."
    else
        info "Starting fresh configuration."
    fi
fi

if [[ "$REUSE_CACHE" == "false" ]]; then

_tty "\n"
hr
_tty "  ${BOLD}xkrt ecosystem – interactive installer${NC}\n"
_tty "  llvm  ·  hwloc  ·  opencg  ·  xkrt  ·  xkblas  ·  xkomp\n"
hr
_tty "\n"

# ── Installation base ─────────────────────────────────────────────────────────
step "Installation directory"
BASE_DIR=$(prompt_value "Base directory (repos, installs and modules go here)" "$(pwd)")
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
step "LLVM  [cmake; custom patched toolchain — anlsys/llvm-project; dependency of opencg]"
_tty "\n"
_tty "  The anlsys fork adds new OpenMP pragma support (access clauses, etc.) and\n"
_tty "  provides the LLVM that opencg depends on.\n"
_tty "  It can be bootstrapped with any compiler.  You may optionally use it to\n"
_tty "  build opencg/xkrt/xkblas/xkomp — those require an LLVM >= ${MIN_CLANG_VER}\n"
_tty "  (the custom one, or a system clang >= ${MIN_CLANG_VER} detected above).\n\n"

INSTALL_LLVM=false
LLVM_BRANCH="" LLVM_BUILD_TYPE=""
LLVM_PROJECTS="" LLVM_RUNTIMES=""
LLVM_CMAKE_TARGETS="" LLVM_CMAKE_RUNTIME_TARGETS=""
LLVM_EXTRA_CMAKE_OPTS=""
LLVM_GPU_SUMMARY=""   # human-readable, for summary display
USE_LLVM_FOR_BUILD=false                      # use custom LLVM to build the libs?
LLVM_BOOTSTRAP_CC="" LLVM_BOOTSTRAP_CXX=""    # compiler used to build LLVM itself

if prompt_yn "Install custom patched LLVM?" "no"; then
    INSTALL_LLVM=true
    LLVM_BRANCH=$(prompt_value "Branch" "main")
    LLVM_BUILD_TYPE=$(prompt_value "Build type (Release/Debug)" "Release")

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
    _tty "  Enter numbers separated by spaces (e.g. \"1 2\"), or Enter for host-only: "
    read -r _gpu_sel </dev/tty

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
    if prompt_yn "  Include lld  (LLVM linker — recommended for GPU offload)?" "yes"; then
        _projects="${_projects};lld"
    fi
    if prompt_yn "  Include bolt (binary optimizer)?" "no"; then
        _projects="${_projects};bolt"
    fi
    LLVM_PROJECTS="$_projects"

    # ── Runtimes ─────────────────────────────────────────────────────────────
    _tty "\n  ${BOLD}LLVM runtimes${NC}:\n"
    _runtimes=""
    if prompt_yn "  Build openmp  (OpenMP host runtime)?" "yes"; then
        _runtimes="${_runtimes}${_runtimes:+;}openmp"
    fi
    if prompt_yn "  Build offload (OpenMP GPU offload runtime)?" "yes"; then
        _runtimes="${_runtimes}${_runtimes:+;}offload"
    fi
    LLVM_RUNTIMES="$_runtimes"

    # ── Extra flags ───────────────────────────────────────────────────────────
    _tty "\n  ${BOLD}${BLUE}?${NC} Extra cmake flags for LLVM (or Enter to skip): "
    read -r _llvm_extra </dev/tty
    LLVM_EXTRA_CMAKE_OPTS="${_llvm_extra:-}"

    # ── Bootstrap compiler (any compiler may build LLVM) ──────────────────────
    _tty "\n  ${BOLD}Bootstrap compiler${NC} — used only to compile LLVM itself.\n"
    _tty "  ${DIM}Any C/C++ compiler works here; no minimum version.${NC}\n"
    LLVM_BOOTSTRAP_CC=$(prompt_value  "Bootstrap C compiler"   "$_boot_cc_default")
    LLVM_BOOTSTRAP_CXX=$(prompt_value "Bootstrap C++ compiler" "$_boot_cxx_default")

    # ── Use the custom LLVM to build the libraries? ───────────────────────────
    _tty "\n"
    if prompt_yn "Use this custom LLVM to build opencg/xkrt/xkblas/xkomp?" "yes"; then
        USE_LLVM_FOR_BUILD=true
    else
        USE_LLVM_FOR_BUILD=false
    fi

fi

# ── Build compiler ────────────────────────────────────────────────────────────
# opencg/xkrt/xkblas/xkomp must be built with an LLVM >= MIN_CLANG_VER: either
# the custom LLVM (built above) or a system clang >= MIN_CLANG_VER.
step "Build compiler  [for opencg · xkrt · xkblas · xkomp]"
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
    CC=$(prompt_value  "C compiler (LLVM >= ${MIN_CLANG_VER})"   "$_def_cc")
    CXX=$(prompt_value "C++ compiler (LLVM >= ${MIN_CLANG_VER})" "$_def_cxx")
fi
export CC CXX

# ── hwloc ─────────────────────────────────────────────────────────────────────
step "hwloc  [autotools; no cmake dependencies]"
_tty "\n"
_tty "  hwloc is required by xkrt for hardware topology discovery.\n"
_tty "  If you skip this step, xkrt will attempt to use a system-installed hwloc\n"
_tty "  (searched in standard paths by Findhwloc.cmake).\n\n"

INSTALL_HWLOC=false; HWLOC_BRANCH=""
if prompt_yn "Install hwloc?" "yes"; then
    INSTALL_HWLOC=true
    HWLOC_BRANCH=$(prompt_value "Branch / tag" "v2.14")
fi

# ── opencg ────────────────────────────────────────────────────────────────────
step "opencg  [cmake; no runtime dependencies; parallel to hwloc]"
INSTALL_OPENCG=false
OPENCG_BRANCH=""; OPENCG_BUILD_TYPE=""; OPENCG_CMAKE_OPTS=""
if prompt_yn "Install opencg?" "yes"; then
    INSTALL_OPENCG=true
    OPENCG_BRANCH=$(prompt_value "Branch" "release/latest")
    OPENCG_BUILD_TYPE=$(prompt_value "Build type (Release/Debug)" "Release")
    clone_or_update "https://github.com/JLESC-Tasking-Group/opencg" \
        "$REPO_DIR/opencg" "$OPENCG_BRANCH"
    OPENCG_CMAKE_OPTS=$(ask_cmake_opts "opencg" "$REPO_DIR/opencg/CMakeLists.txt")
fi

# ── xkrt ──────────────────────────────────────────────────────────────────────
step "xkrt  [cmake; depends on hwloc + opencg]"
INSTALL_XKRT=false
XKRT_BRANCH=""; XKRT_BUILD_TYPE=""; XKRT_CMAKE_OPTS=""
if prompt_yn "Install xkrt?" "yes"; then
    INSTALL_XKRT=true
    XKRT_BRANCH=$(prompt_value "Branch" "release/latest")
    XKRT_BUILD_TYPE=$(prompt_value "Build type (Release/Debug)" "Release")
    clone_or_update "https://gitlab.inria.fr/xkaapi/dev-v2" \
        "$REPO_DIR/xkrt" "$XKRT_BRANCH"
    XKRT_CMAKE_OPTS=$(ask_cmake_opts "xkrt" "$REPO_DIR/xkrt/CMakeLists.txt")
fi

# ── xkblas ────────────────────────────────────────────────────────────────────
step "xkblas  [cmake; depends on xkrt; parallel to xkomp]"
_tty "\n"
_tty "  Tip: if you enable USE_CBLAS, also set USE_OPENBLAS, USE_MKL, or\n"
_tty "  USE_CRAYBLAS via the 'extra cmake flags' prompt below.\n\n"
INSTALL_XKBLAS=false
XKBLAS_BRANCH=""; XKBLAS_BUILD_TYPE=""; XKBLAS_CMAKE_OPTS=""
if prompt_yn "Install xkblas?" "yes"; then
    INSTALL_XKBLAS=true
    XKBLAS_BRANCH=$(prompt_value "Branch" "release/v2.0-latest")
    XKBLAS_BUILD_TYPE=$(prompt_value "Build type (Release/Debug)" "Release")
    clone_or_update "https://gitlab.inria.fr/xkblas/dev" \
        "$REPO_DIR/xkblas" "$XKBLAS_BRANCH"
    XKBLAS_CMAKE_OPTS=$(ask_cmake_opts "xkblas" "$REPO_DIR/xkblas/CMakeLists.txt")
fi

# ── xkomp ─────────────────────────────────────────────────────────────────────
step "xkomp  [cmake; depends on xkrt; parallel to xkblas]"
INSTALL_XKOMP=false
XKOMP_BRANCH=""; XKOMP_BUILD_TYPE=""; XKOMP_CMAKE_OPTS=""
if prompt_yn "Install xkomp?" "yes"; then
    INSTALL_XKOMP=true
    XKOMP_BRANCH=$(prompt_value "Branch" "release/latest")
    XKOMP_BUILD_TYPE=$(prompt_value "Build type (Release/Debug)" "Release")
    clone_or_update "https://github.com/anlsys/xkomp" \
        "$REPO_DIR/xkomp" "$XKOMP_BRANCH"
    XKOMP_CMAKE_OPTS=$(ask_cmake_opts "xkomp" "$REPO_DIR/xkomp/CMakeLists.txt")
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
_lib_row "opencg" "$INSTALL_OPENCG" "$OPENCG_BRANCH" "$OPENCG_BUILD_TYPE"
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
    info "Fetching and checking out branch: $LLVM_BRANCH"
    git -C "$LLVM_REPO_DIR" fetch -q --all --tags
    git -C "$LLVM_REPO_DIR" checkout -q "$LLVM_BRANCH"
    if git -C "$LLVM_REPO_DIR" ls-remote --exit-code --heads origin "$LLVM_BRANCH" >/dev/null 2>&1; then
        git -C "$LLVM_REPO_DIR" pull -q
    fi

    LLVM_HASH=$(git -C "$LLVM_REPO_DIR" rev-parse HEAD | cut -c1-12)
    LLVM_INSTALL_DIR="$INSTALL_DIR/llvm/$LLVM_HASH/$LLVM_BUILD_TYPE"
    LLVM_BUILD_DIR="$LLVM_REPO_DIR/build/$LLVM_HASH/$LLVM_BUILD_TYPE"

    mkdir -p "$LLVM_BUILD_DIR"

    # Only run cmake if there is no existing cache (supports incremental rebuilds).
    if [[ ! -f "$LLVM_BUILD_DIR/CMakeCache.txt" ]]; then
        info "Configuring LLVM …"
        cd "$LLVM_BUILD_DIR"

        # Build a semicolon-separated runtimes string only if non-empty.
        _llvm_runtime_flag=""
        if [[ -n "$LLVM_RUNTIMES" ]]; then
            _llvm_runtime_flag="-DLLVM_ENABLE_RUNTIMES=${LLVM_RUNTIMES}"
        fi
        _llvm_rtgt_flag="-DLLVM_RUNTIME_TARGETS=${LLVM_CMAKE_RUNTIME_TARGETS}"

        # shellcheck disable=SC2086
        # LLVM is bootstrapped with the (any-version) bootstrap compiler.
        CC="$LLVM_BOOTSTRAP_CC" CXX="$LLVM_BOOTSTRAP_CXX" \
        cmake \
            -DCMAKE_BUILD_TYPE="$LLVM_BUILD_TYPE" \
            -DCMAKE_INSTALL_PREFIX="$LLVM_INSTALL_DIR" \
            -DLLVM_ENABLE_PROJECTS="$LLVM_PROJECTS" \
            ${_llvm_runtime_flag:+"$_llvm_runtime_flag"} \
            "$_llvm_rtgt_flag" \
            -DLLVM_TARGETS_TO_BUILD="$LLVM_CMAKE_TARGETS" \
            -DCMAKE_CXX_FLAGS="-Wno-c2y-extensions" \
            $LLVM_EXTRA_CMAKE_OPTS \
            "$LLVM_REPO_DIR/llvm"
    else
        info "Reusing existing cmake cache (incremental build)."
        cd "$LLVM_BUILD_DIR"
    fi

    info "Building LLVM (this may take a while) …"
    make install -j "$(nproc)"

    # If the user opted to build the libraries with the custom LLVM, switch the
    # build compiler to it now and activate its prefix so opencg (which depends
    # on LLVM) and the others find its clang, headers, libraries and cmake config.
    if [[ "$USE_LLVM_FOR_BUILD" == "true" ]]; then
        CC="$LLVM_INSTALL_DIR/bin/clang"
        CXX="$LLVM_INSTALL_DIR/bin/clang++"
        export CC CXX
        _activate_prefix "$LLVM_INSTALL_DIR" "llvm" "$LLVM_HASH/$LLVM_BUILD_TYPE"
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
    success "LLVM  installed → $LLVM_INSTALL_DIR"
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
    ./configure --prefix="$HWLOC_INSTALL_DIR" --quiet
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

# ── opencg ────────────────────────────────────────────────────────────────────
if [[ "$INSTALL_OPENCG" == "true" ]]; then
    step "Building & installing opencg"
    clone_or_update \
        "https://github.com/JLESC-Tasking-Group/opencg" \
        "$REPO_DIR/opencg" \
        "$OPENCG_BRANCH"

    OPENCG_HASH=$(git -C "$REPO_DIR/opencg" rev-parse HEAD | cut -c1-12)
    OPENCG_INSTALL_DIR="$INSTALL_DIR/opencg/$OPENCG_HASH/$OPENCG_BUILD_TYPE"
    OPENCG_BUILD_DIR="$REPO_DIR/opencg/build/$OPENCG_HASH/$OPENCG_BUILD_TYPE"

    rm -rf "$OPENCG_BUILD_DIR" && mkdir -p "$OPENCG_BUILD_DIR"
    cd "$OPENCG_BUILD_DIR"

    # shellcheck disable=SC2086
    cmake $OPENCG_CMAKE_OPTS \
        -DCMAKE_BUILD_TYPE="$OPENCG_BUILD_TYPE" \
        -DCMAKE_INSTALL_PREFIX="$OPENCG_INSTALL_DIR" \
        "$REPO_DIR/opencg"
    make install -j "$(nproc)"

    # Activate opencg so xkrt finds its headers, libs and cmake config.
    OPENCG_MOD="$MODULES_DIR/opencg/$OPENCG_HASH/$OPENCG_BUILD_TYPE"
    # opencg depends on LLVM; declare the dependency when it was built with the
    # custom LLVM (a system LLVM, like a system hwloc, ships no module to load).
    declare -a _opencg_deps=()
    [[ "$INSTALL_LLVM" == "true" && "$USE_LLVM_FOR_BUILD" == "true" ]] && \
        _opencg_deps+=("llvm/$LLVM_HASH/$LLVM_BUILD_TYPE")
    generate_modulefile "opencg" "$OPENCG_INSTALL_DIR" "OPENCG_HOME" "$OPENCG_MOD" \
        ${_opencg_deps[@]+"${_opencg_deps[@]}"}
    _activate_prefix "$OPENCG_INSTALL_DIR" "opencg" "$OPENCG_HASH/$OPENCG_BUILD_TYPE"
    MOD_LOAD+=("module load opencg/$OPENCG_HASH/$OPENCG_BUILD_TYPE")
    success "opencg installed → $OPENCG_INSTALL_DIR"
    success "module file      → $OPENCG_MOD"
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
    # Build the dependency list: opencg is always required; hwloc only when
    # it was installed from source by this script (otherwise it is system-provided).
    declare -a _xkrt_deps=()
    [[ "$INSTALL_OPENCG" == "true" ]] && _xkrt_deps+=("opencg/$OPENCG_HASH/$OPENCG_BUILD_TYPE")
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

# ─────────────────────────────────────────────────────────────────────────────
# Done – print usage
# ─────────────────────────────────────────────────────────────────────────────
_tty "\n"; hr
_tty "  ${BOLD}${GREEN}Installation complete!${NC}\n"; hr
_tty "\n  Add the following to your shell environment (or job script):\n\n"
_tty "    module use %s\n" "$MODULES_DIR"
for line in "${MOD_LOAD[@]}"; do
    _tty "    %s\n" "$line"
done
_tty "\n  Then compile against the installed libraries, for example:\n"
_tty "    clang++ main.cc -lxkblas\n"
_tty "\n"
_tty "\n  You may also test the installation by running 'xkrt_info'\n"
_tty "\n"
