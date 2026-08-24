#!/usr/bin/env bash
# Download a pinned Nautical release and delegate installation to its installer.

set -euo pipefail

REPOSITORY="https://github.com/catanadj/taskwarrior-nautical.git"
DEFAULT_VERSION="v7.0.0"
VERSION="${NAUTICAL_VERSION:-$DEFAULT_VERSION}"
TASKDATA="${TASKDATA:-$HOME/.task}"
LAUNCHER_PATH=""
HOOKS_DIR=""
DRY_RUN=0
INSTALL_DEPS=0
KEEP_CHECKOUT=0
CHECKOUT=""
PLATFORM="Linux"

if [[ -t 1 && -z "${NO_COLOR:-}" ]]; then
    STYLE_BOLD=$'\033[1m'
    STYLE_CYAN=$'\033[36m'
    STYLE_GREEN=$'\033[32m'
    STYLE_YELLOW=$'\033[33m'
    STYLE_RESET=$'\033[0m'
else
    STYLE_BOLD=""
    STYLE_CYAN=""
    STYLE_GREEN=""
    STYLE_YELLOW=""
    STYLE_RESET=""
fi

usage() {
    cat <<'EOF'
Usage: bootstrap.sh [options]

Download and install a pinned Nautical release.

Options:
  --version REF         Release tag or branch (default: v6.5.3)
  --taskdata PATH       Taskwarrior data directory (default: TASKDATA or ~/.task)
  --launcher-path PATH  User-facing launcher path (use $PREFIX/bin/nautical on Termux)
  --hooks-dir PATH      Taskwarrior hooks directory override
  --dry-run             Validate the release without changing the installation
  --install-deps        Install missing Python requirements without prompting
  --keep-checkout       Keep the temporary release checkout for inspection
  -h, --help            Show this help
EOF
}

die() {
    printf '%sNautical bootstrap:%s %s\n' "$STYLE_YELLOW" "$STYLE_RESET" "$1" >&2
    exit 2
}

render_banner() {
    local columns="${COLUMNS:-80}"
    local detected=""
    if command -v tput >/dev/null 2>&1 && detected="$(tput cols 2>/dev/null)" && [[ "$detected" =~ ^[0-9]+$ ]]; then
        columns="$detected"
    fi
    if [[ -t 1 && "$columns" -ge 78 ]]; then
        printf '%s%s' "$STYLE_BOLD" "$STYLE_CYAN"
        cat <<'EOF'
                                           88                       88
                                     ,d    ""                       88
                                     88                             88
8b,dPPYba,  ,adPPYYba, 88       88 MM88MMM 88  ,adPPYba, ,adPPYYba, 88
88P'   `"8a ""     `Y8 88       88   88    88 a8"     "" ""     `Y8 88
88       88 ,adPPPPP88 88       88   88    88 8b         ,adPPPPP88 88
88       88 88,    ,88 "8a,   ,a88   88,   88 "8a,   ,aa 88,    ,88 88
88       88 `"8bbdP"Y8  `"YbbdP'Y8   "Y888 88  `"Ybbd8"' `"8bbdP"Y8 88
EOF
        printf '%s\n' "$STYLE_RESET"
    else
        printf '\n%s%sTaskwarrior Nautical%s\n\n' "$STYLE_BOLD" "$STYLE_CYAN" "$STYLE_RESET"
    fi
}

step() {
    printf '\n%s%s[%s/4]%s %s\n' "$STYLE_BOLD" "$STYLE_CYAN" "$1" "$STYLE_RESET" "$2"
}

missing_python_requirements() {
    python3 - "$1" <<'PY'
from importlib import metadata
from pathlib import Path
import re
import sys

missing = []
for raw in Path(sys.argv[1]).read_text(encoding="utf-8").splitlines():
    line = raw.strip()
    if not line or line.startswith("#"):
        continue
    match = re.match(r"([A-Za-z0-9_.-]+)", line)
    if match is None:
        continue
    name = match.group(1)
    try:
        metadata.version(name)
    except metadata.PackageNotFoundError:
        missing.append(name)
print(", ".join(missing))
PY
}

render_legacy_verification() {
    python3 - "$1" "$2" "$3" <<'PY'
import json
import os
import shutil
import sys
from pathlib import Path

report_path, platform, launcher_text = sys.argv[1:]
launcher = Path(launcher_text).expanduser()
try:
    payload = json.loads(Path(report_path).read_text(encoding="utf-8"))
except Exception as exc:
    print(f"Post-install verification could not be read: {exc}", file=sys.stderr)
    raise SystemExit(2)
findings = [item for item in payload.get("findings") or [] if isinstance(item, dict)]

def group(prefix, empty_status="failed"):
    items = [item for item in findings if str(item.get("id") or "").startswith(prefix)]
    if not items:
        return empty_status
    if any(item.get("severity") == "error" for item in items):
        return "failed"
    if any(item.get("severity") == "warn" for item in items):
        return "attention"
    return "passed"

checks = [
    ("Platform", "passed", platform),
    ("Taskwarrior", group("taskwarrior."), "command available"),
    ("Taskdata", group("taskdata."), str(payload.get("taskdata") or "")),
    ("Runtime", group("install."), "managed release active"),
    ("Hooks", group("hook."), "add, modify, and exit"),
    ("Launcher", "passed" if launcher.is_file() and os.access(launcher, os.X_OK) else "failed", str(launcher)),
    ("UDAs", group("uda.", empty_status="passed"), "Taskwarrior fields registered"),
    ("Timezone", group("config.timezone"), "explicit scheduling timezone"),
]
required_prefixes = ("integration.", "taskwarrior.", "taskdata.", "hook.", "uda.", "install.", "config.")
optional_prefixes = ("navigator.", "astronomy.")
manual = []
optional = []
seen = set()
for item in findings:
    if item.get("severity") == "ok":
        continue
    check_id = str(item.get("id") or "")
    destination = optional if check_id.startswith(optional_prefixes) else manual if check_id.startswith(required_prefixes) else None
    if destination is None:
        continue
    action = str(item.get("fix") or item.get("message") or "Inspect this finding.").strip()
    if (check_id, action) not in seen:
        seen.add((check_id, action))
        destination.append(action)
if shutil.which("nautical") is None and launcher.name == "nautical":
    optional.append(f"Add {launcher.parent} to PATH or invoke {launcher}.")

symbols = {"passed": "+", "attention": "!", "failed": "x"}
color = sys.stdout.isatty() and not os.environ.get("NO_COLOR")
styles = {
    "passed": "\033[32m" if color else "",
    "attention": "\033[33m" if color else "",
    "failed": "\033[31m" if color else "",
    "heading": "\033[1;36m" if color else "",
    "reset": "\033[0m" if color else "",
}
print(f"\n{styles['heading']}Post-install verification{styles['reset']}")
for name, status, detail in checks:
    print(f"  {styles[status]}{symbols[status]}{styles['reset']} {name}: {detail}")
if manual:
    print("\nManual action")
    for action in manual:
        print(f"  ! {action}")
if optional:
    print("\nOptional")
    for action in optional:
        print(f"  ! {action}")
failed = any(status == "failed" for _, status, _ in checks) or any(
    item.get("severity") == "error"
    and str(item.get("id") or "").startswith(required_prefixes + ("astronomy.",))
    for item in findings
)
if failed:
    print("\nInstallation verification failed. Resolve the required actions before using Nautical.")
    raise SystemExit(2)
if manual:
    print(f"\nInstallation completed; {len(manual)} manual action(s) remain.")
    raise SystemExit(1)
if optional:
    print("\nCore installation verified. Optional enhancements are listed above.")
    raise SystemExit(1)
print("\nInstallation verified. Nautical is ready.")
PY
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
        --install-deps)
            INSTALL_DEPS=1
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

if [[ -n "${TERMUX_VERSION:-}" ]] || [[ "${PREFIX:-}" == */com.termux/files/usr ]]; then
    PLATFORM="Termux"
    if [[ -z "$LAUNCHER_PATH" ]]; then
        [[ -n "${PREFIX:-}" ]] || die "Termux was detected but PREFIX is unavailable"
        LAUNCHER_PATH="$PREFIX/bin/nautical"
    fi
else
    PLATFORM="$(uname -s 2>/dev/null || printf 'Linux')"
fi

if [[ "$PLATFORM" == "Termux" ]]; then
    command -v git >/dev/null 2>&1 || die "git is required; run: pkg install git"
    command -v python3 >/dev/null 2>&1 || die "python is required; run: pkg install python"
    command -v task >/dev/null 2>&1 || die "Taskwarrior is required; run: pkg install taskwarrior"
else
    command -v git >/dev/null 2>&1 || die "git is required"
    command -v python3 >/dev/null 2>&1 || die "python3 is required"
    command -v task >/dev/null 2>&1 || die "Taskwarrior is required; install task before Nautical"
fi

render_banner
printf '%sRelease%s   %s\n' "$STYLE_BOLD" "$STYLE_RESET" "$VERSION"
printf '%sPlatform%s  %s\n' "$STYLE_BOLD" "$STYLE_RESET" "$PLATFORM"
printf '%sTaskdata%s  %s\n' "$STYLE_BOLD" "$STYLE_RESET" "$TASKDATA"

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

step 1 "Downloading release"
git -c advice.detachedHead=false clone --quiet --depth 1 --branch "$VERSION" "$REPOSITORY" "$CHECKOUT"

step 2 "Checking Python requirements"
requirements_file="$CHECKOUT/requirements.txt"
[[ -f "$requirements_file" ]] || die "release is missing requirements.txt"
missing_requirements="$(missing_python_requirements "$requirements_file")"
if [[ -n "$missing_requirements" ]]; then
    printf '\nMissing Python requirements: %s\n' "$missing_requirements"
    if ((DRY_RUN)); then
        die "dry-run cannot install dependencies; rerun normally and approve installation, or use --install-deps"
    fi
    if (( ! INSTALL_DEPS )); then
        if [[ -r /dev/tty && -w /dev/tty ]]; then
            printf 'Install them now with %s? [Y/n] ' "$(command -v python3)" >/dev/tty
            reply=""
            IFS= read -r reply </dev/tty || true
            case "${reply,,}" in
                ""|y|yes) INSTALL_DEPS=1 ;;
                *) die "Python requirements are required; rerun the bootstrap with --install-deps" ;;
            esac
        else
            die "Python requirements are required; rerun the bootstrap with --install-deps"
        fi
    fi
    python3 -m pip --version >/dev/null 2>&1 || die "pip is unavailable for python3; install python3-pip (or Termux's python package)"
    printf '%sInstalling Python requirements...%s\n' "$STYLE_GREEN" "$STYLE_RESET"
    python3 -m pip install -r "$requirements_file" || die "Python requirement installation failed"
    remaining_requirements="$(missing_python_requirements "$requirements_file")"
    [[ -z "$remaining_requirements" ]] || die "Python requirements remain unavailable: $remaining_requirements"
fi

install_args=(install --source "$CHECKOUT" --taskdata "$TASKDATA")
[[ -n "$LAUNCHER_PATH" ]] && install_args+=(--launcher-path "$LAUNCHER_PATH")
[[ -n "$HOOKS_DIR" ]] && install_args+=(--hooks-dir "$HOOKS_DIR")
((DRY_RUN)) && install_args+=(--dry-run)

step 3 "Installing Nautical"
python3 "$CHECKOUT/nautical" "${install_args[@]}"

if (( ! DRY_RUN )); then
    launcher="${LAUNCHER_PATH:-$HOME/.local/bin/nautical}"
    if [[ -x "$launcher" ]]; then
        step 4 "Verifying installation"
        doctor_report="$CHECKOUT/doctor-installation.json"
        doctor_args=(doctor --taskdata "$TASKDATA" --json)
        if "$launcher" doctor --help 2>&1 | grep -q -- '--installation-only'; then
            doctor_args+=(--installation-only)
        fi
        doctor_status=0
        "$launcher" "${doctor_args[@]}" >"$doctor_report" || doctor_status=$?
        verification_status=0
        verifier="$CHECKOUT/nautical_core/tools/nautical_install_verify.py"
        if [[ -f "$verifier" ]]; then
            python3 "$verifier" \
                --input "$doctor_report" \
                --platform "$PLATFORM" \
                --launcher "$launcher" || verification_status=$?
        else
            render_legacy_verification "$doctor_report" "$PLATFORM" "$launcher" || verification_status=$?
        fi
        if ((verification_status == 2)); then
            exit 2
        fi
        if ((doctor_status > 2)); then
            exit 2
        fi
    else
        printf '\nManual action required: installed launcher is not executable at %s\n' "$launcher"
        exit 2
    fi
else
    step 4 "Dry-run complete"
fi
