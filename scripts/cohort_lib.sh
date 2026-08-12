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

# ── Lens axis ───────────────────────────────────────────────────────────────
#
# <mode> packs both axes of a lens-derived run into one CLI word: which lens
# (logit vs Jacobian) and which cached vector it is applied to (diff, ft,
# base). It is the lens's tag followed by the variant, with the tag omitted
# when empty - the logit lens's tag is empty, so `ft` is (logit_lens, ft)
# while `jlens_ft` is (jlens, ft).
#
# The variant is always spelled out, including `diff`: the logit lens has no
# tag, so omitting `diff` too would leave the empty string. LL_SUFFIX follows
# the other convention and does omit a `diff` variant, which is why mode
# `jlens_diff` produces suffix `_jlens` rather than `_jlens_diff`.
#
# src/diffing/analysis/lens_axis.py owns this grammar and derives the CSV
# `method` labels and artifact filenames from it. The two implementations have
# to agree exactly or a sweep writes files the plotters do not look for, so
# tests/analysis/test_lens_axis.py sources this file and asserts they do.
# Duplicated in bash rather than shelled out to because `uv run python` costs
# ~1.6s of interpreter startup on every driver invocation.

MO_LENS_MODES="diff ft base jlens_diff jlens_ft jlens_base"
MO_LENS_MODES_USAGE="${MO_LENS_MODES// /|}"

# Decode <mode> into LENS, LL_VARIANT and LL_SUFFIX. Returns 1 (setting
# nothing) on an unknown mode, so callers can fold it into arg parsing.
mo_lens_mode() {
    local mode="$1"
    case "$mode" in
        diff)       LENS="logit_lens"; LL_VARIANT="diff";           LL_SUFFIX="" ;;
        ft|base)    LENS="logit_lens"; LL_VARIANT="$mode";          LL_SUFFIX="_${mode}" ;;
        jlens_diff) LENS="jlens";      LL_VARIANT="diff";           LL_SUFFIX="_jlens" ;;
        jlens_ft|jlens_base)
                    LENS="jlens";      LL_VARIANT="${mode#jlens_}"; LL_SUFFIX="_${mode}" ;;
        *) return 1 ;;
    esac
}
