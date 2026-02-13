#!/usr/bin/env bash
set -u

TASKDATA="${TASKDATA:-$HOME/.task}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHECK_SCRIPT="${SCRIPT_DIR}/../nautical_health_check.py"
LOG_FILE="${TASKDATA}/.nautical_health_monitor.log"
LOG_DIR="$(dirname "${LOG_FILE}")"

mkdir -p "${TASKDATA}"

payload="$(/usr/bin/env python3 "${CHECK_SCRIPT}" --taskdata "${TASKDATA}" --json 2>&1)"
rc=$?
ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

can_write_log=0
if [ -e "${LOG_FILE}" ]; then
  if [ -w "${LOG_FILE}" ]; then
    can_write_log=1
  fi
elif [ -d "${LOG_DIR}" ] && [ -w "${LOG_DIR}" ]; then
  can_write_log=1
fi

if [ "${can_write_log}" -eq 1 ]; then
  if ! { printf '%s %s\n' "${ts}" "${payload}" >> "${LOG_FILE}"; } 2>/dev/null; then
    /usr/bin/logger -t nautical-health "log_write_failed path=${LOG_FILE}" >/dev/null 2>&1 || true
  fi
else
  /usr/bin/logger -t nautical-health "log_write_failed path=${LOG_FILE}" >/dev/null 2>&1 || true
fi

if [ "${rc}" -ge 1 ]; then
  /usr/bin/logger -t nautical-health "rc=${rc} taskdata=${TASKDATA} payload=${payload}" >/dev/null 2>&1 || true
fi

exit "${rc}"
