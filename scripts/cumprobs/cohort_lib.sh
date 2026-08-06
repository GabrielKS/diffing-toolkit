# Shared cohort handling for the cumprobs drivers. Source, do not execute.
#
# The registry holds two kinds of model: the original MO families (cohort
# "core") and the behavioural-distillation students (cohort "kd"). Entries
# written before cohorts existed carry no `cohort` field and count as "core",
# so the default below makes an existing invocation enumerate exactly the
# models it always did.
#
# Keep this in step with steering/registry_utils.py in the parent repo, which
# implements the same defaulting rule for the Python consumers.

MO_DEFAULT_COHORT="core"

# registry_variants <registry> <family> <cohorts>
#
# Print the registry keys for one quirk family, ordered by plot_order and
# restricted to the requested comma-separated cohorts ("all" keeps every one).
registry_variants() {
    jq -r --arg fam "$2" --arg cohorts "$3" '
        ($cohorts | split(",")) as $want
        | .models
        | to_entries
        | map(select(
            .value.quirk_family_id == $fam
            and (($want | index("all")) != null
                 or ((.value.cohort // "core") | IN($want[])))
          ))
        | sort_by(.value.plot_order)
        | .[].key
    ' "$1"
}

# cohort_tree_suffix <cohorts>
#
# Suffix for the output tree name. A sweep's output path is built from the
# family and judge only, so a `--cohort kd` run would otherwise overwrite the
# core run's relevance.csv in place — every other path component is identical.
# Core keeps the bare tree name, so existing output paths are unchanged.
cohort_tree_suffix() {
    if [[ "$1" == "$MO_DEFAULT_COHORT" ]]; then
        echo ""
    else
        echo "_${1//,/+}"
    fi
}

# parse_cohort_flag <arg> <next>
#
# Helper for the drivers' argument loops; echoes the value and the number of
# arguments consumed.
mo_usage_cohort_line() {
    echo "  --cohort <list>  registry cohorts to sweep, comma-separated or 'all'" >&2
    echo "                   (default: ${MO_DEFAULT_COHORT}; non-core writes to a" >&2
    echo "                   suffixed output tree so core results are never clobbered)" >&2
}
