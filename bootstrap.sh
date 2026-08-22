#!/usr/bin/env bash
# Download a pinned Nautical release and delegate installation to its installer.

set -euo pipefail

REPOSITORY="https://github.com/catanadj/taskwarrior-nautical.git"
DEFAULT_VERSION="v6.5.2"
VERSION="${NAUTICAL_VERSION:-$DEFAULT_VERSION}"
TASKDATA="${TASKDATA:-$HOME/.task}"
LAUNCHER_PATH=""
HOOKS_DIR=""
DRY_RUN=0
KEEP_CHECKOUT=0
CHECKOUT=""

usage() {
    cat <<'EOF'
Usage: bootstrap.sh [options]

Download and install a pinned Nautical release.

Options:
  --version REF         Release tag or branch (default: v6.5.2)
  --taskdata PATH       Taskwarrior data directory (default: TASKDATA or ~/.task)
  --launcher-path PATH  User-facing launcher path (use $PREFIX/bin/nautical on Termux)
  --hooks-dir PATH      Taskwarrior hooks directory override
  --dry-run             Validate the release without changing the installation
  --keep-checkout       Keep the temporary release checkout for inspection
  -h, --help            Show this help
EOF
}

die() {
    printf 'Nautical bootstrap: %s\n' "$1" >&2
    exit 2
}

while (($#)); do
    case "$1" in
        --version)
            (($# >= 2)) || die "--version requires a release tag or branch"
            VERSION="$2"
            shift 2
            ;;
        --taskdata)
            (($# >= 2)) || die "--taskdata requires a path"
            TASKDATA="$2"
            shift 2
            ;;
        --launcher-path)
            (($# >= 2)) || die "--launcher-path requires a path"
            LAUNCHER_PATH="$2"
            shift 2
            ;;
        --hooks-dir)
            (($# >= 2)) || die "--hooks-dir requires a path"
            HOOKS_DIR="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        --keep-checkout)
            KEEP_CHECKOUT=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            die "unknown option: $1 (use --help for usage)"
            ;;
    esac
done

command -v git >/dev/null 2>&1 || die "git is required"
command -v python3 >/dev/null 2>&1 || die "python3 is required"
command -v task >/dev/null 2>&1 || die "Taskwarrior is required; install task before Nautical"

printf '\nTaskwarrior Nautical\n'
printf 'Installing release: %s\n' "$VERSION"
printf 'Taskdata: %s\n\n' "$TASKDATA"

if [[ -n "$LAUNCHER_PATH" ]]; then
    case "$LAUNCHER_PATH" in
        */*) ;;
        *) die "--launcher-path must be an executable path" ;;
    esac
fi

CHECKOUT="$(mktemp -d "${TMPDIR:-/tmp}/nautical-bootstrap.XXXXXX")"
cleanup() {
    if ((KEEP_CHECKOUT)); then
        printf 'Release checkout kept at: %s\n' "$CHECKOUT"
    else
        rm -rf "$CHECKOUT"
    fi
}
trap cleanup EXIT INT TERM

printf 'Downloading release...\n'
git -c advice.detachedHead=false clone --quiet --depth 1 --branch "$VERSION" "$REPOSITORY" "$CHECKOUT"

install_args=(install --source "$CHECKOUT" --taskdata "$TASKDATA")
[[ -n "$LAUNCHER_PATH" ]] && install_args+=(--launcher-path "$LAUNCHER_PATH")
[[ -n "$HOOKS_DIR" ]] && install_args+=(--hooks-dir "$HOOKS_DIR")
((DRY_RUN)) && install_args+=(--dry-run)

printf 'Running the validated installer...\n'
python3 "$CHECKOUT/nautical" "${install_args[@]}"

if (( ! DRY_RUN )); then
    launcher="${LAUNCHER_PATH:-$HOME/.local/bin/nautical}"
    if [[ -x "$launcher" ]]; then
        printf '\nRunning post-install Doctor check...\n'
        "$launcher" doctor --taskdata "$TASKDATA"
    else
        printf '\nInstall completed; run nautical doctor to validate the runtime.\n'
    fi
fi
