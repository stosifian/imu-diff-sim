#!/usr/bin/env bash
# SessionStart hook: ensure today's lab-notebook entry exists, and remind
# Claude to keep it current.
#
# The notes methodology (notes/README.md) only pays off if entries get written
# on the day, not reconstructed in week 4. Scaffolding the file is deterministic
# work a hook can do; filling it in is not, so this also injects a reminder
# into the session context.
#
# Outputs SessionStart JSON on stdout. Never fails the session: any error
# degrades to a silent no-op, because a broken notes hook must not block work.
set -uo pipefail

root="${CLAUDE_PROJECT_DIR:-$PWD}"
notes="$root/notes"
today="$(date +%F)"
file="$notes/$today.md"
template="$notes/_template.md"

created=false
if [ -d "$notes" ] && [ ! -f "$file" ]; then
  if [ -f "$template" ]; then
    sed "s/^# YYYY-MM-DD/# $today/" "$template" > "$file" 2>/dev/null && created=true
  else
    printf '# %s\n\n**Focus:**\n**Commit at end of session:**\n\n## Did\n\n## Found\n\n## Notes\n\n## Next session\n' \
      "$today" > "$file" 2>/dev/null && created=true
  fi
fi

[ -f "$file" ] || exit 0   # notes/ missing or unwritable: stay silent

if [ "$created" = true ]; then
  state="Created from the template; it is empty."
else
  state="Already exists; append rather than overwrite."
fi

read -r -d '' ctx <<CTX || true
Lab notebook: today's entry is notes/$today.md. $state

Keep it current as the session goes, without being asked. Append to it when
something happens that week 4 would otherwise have to reconstruct from memory:

  - Found   : a measurement, a verification result, something that turned out
              untrue. Numbers, not adjectives.
  - Notes   : tag inline with [DECISION] (a choice and why, including what was
              rejected), [LIMITATION] (known-wrong, approximate or unvalidated),
              [OPEN] (a question moved past), [SURPRISE] (a result that
              contradicted expectation -- the most valuable and most forgotten).

Reference artifacts, not memory: commit SHA, config path, figure path under
reports/figures/. Record negative results and wrong turns; per the brief those
build credibility and are most of what gets learned. See notes/README.md.
CTX

jq -nc --arg c "$ctx" \
  '{hookSpecificOutput:{hookEventName:"SessionStart",additionalContext:$c}}' \
  2>/dev/null || true
