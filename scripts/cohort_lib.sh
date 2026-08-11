# Shared cohort selection for the registry-driven drivers.
#
# Every registry entry declares a `cohorts` list - mandatory, possibly
# overlapping, and nothing is inferred from its absence. Selection is any-of.
#
# This mirrors select_cohorts() in the parent repo's steering/registry_utils.py,
# which owns the schema. That coupling is the reason this file exists: the
# predicate was previously inlined in four drivers, and when the registry moved
# from a `cohort` string to a `cohorts` list all four silently kept matching
# nothing. One copy is one place to keep in step.
#
# Lives in scripts/ rather than scripts/cumprobs/ because scripts/run_adl_kd.sh
# uses it too, and a driver should not have to source across sibling dirs.

MO_DEFAULT_COHORT=core

# Every cohort name carried by some entry, space-separated.
# Errors if any entry omits the field or declares it as a bare string - both
# would otherwise match nothing and read as "this family has no variants".
mo_known_cohorts() {
    jq -r '
        [ .models | to_entries[]
          | (.value.cohorts // []) as $c
          | if ($c | type) != "array" or ($c | length) == 0
            then error("registry entry \"\(.key)\" has no non-empty \"cohorts\" list")
            else $c[] end
        ] | unique | join(" ")' "$1"
}

# Reject a cohort name no entry carries. A typo enumerates nothing, which is
# indistinguishable from a family legitimately having none of that cohort, so
# it has to be caught here rather than surfacing as an empty sweep.
mo_validate_cohorts() {
    local registry="$1" cohorts="$2" known cohort
    known="$(mo_known_cohorts "$registry")" || exit 1
    for cohort in ${cohorts//,/ }; do
        if [[ "$cohort" != all && " $known " != *" $cohort "* ]]; then
            echo "unknown cohort '$cohort'; known: $known (or 'all')" >&2
            exit 1
        fi
    done
}

# Registry keys for $family in any of $cohorts, ordered by plot_order.
# Callers wanting bare variant suffixes strip the "<family>_" prefix themselves.
mo_registry_variants() {
    jq -r --arg fam "$2" --arg cohorts "$3" '
        ($cohorts | split(",")) as $want
        | .models
        | to_entries
        | map(select(
            .value.quirk_family_id == $fam
            and (($want | index("all")) != null
                 or (.value.cohorts | any(IN($want[]))))
          ))
        | sort_by(.value.plot_order)
        | .[].key
    ' "$1"
}

# Non-core runs write to a suffixed output tree. The per-combination output path
# is built from family and judge alone, so without the suffix a `--cohort kd`
# run would overwrite the core run's results in place. Core keeps the bare name.
mo_cohort_tree_suffix() {
    [[ "$1" == "$MO_DEFAULT_COHORT" ]] || printf '_%s' "${1//,/+}"
}

# Usage-text line describing the flag, so the four drivers agree on wording.
mo_usage_cohort_line() {
    echo "  --cohort <list>  registry cohorts, comma-separated or 'all'" >&2
    echo "                   (default: ${MO_DEFAULT_COHORT}; non-core writes to a suffixed tree)" >&2
}
