#!/bin/bash
set -euo pipefail

source "$(dirname "$0")/../sbatches/train_launcher_lib.sh"

fail() {
    echo "[FAIL] $*" >&2
    exit 1
}

assert_eq() {
    local name="$1"
    local got="$2"
    local expected="$3"
    if [[ "$got" != "$expected" ]]; then
        fail "$name: expected '$expected', got '$got'"
    fi
}

assert_parse() {
    local jobname="$1"
    local gpu_total="$2"
    local expected_model_size="$3"
    local expected_job_core="$4"
    local expected_model_name="$5"
    local expected_model_override="$6"
    local expected_batch_size="$7"
    local expected_is_v1="$8"
    local expected_attack_domain="$9"
    local expected_attack_eps="${10}"
    local expected_crit="${11}"
    local expected_advtrain="${12}"
    local expected_gradnorm="${13}"
    local expected_dvd_enabled="${14:-false}"
    local expected_dvd_variant="${15:-}"

    parse_train_job "$jobname" "$gpu_total"

    assert_eq "$jobname model_size" "$model_size" "$expected_model_size"
    assert_eq "$jobname job_core" "$job_core" "$expected_job_core"
    assert_eq "$jobname model_name" "$model_name" "$expected_model_name"
    assert_eq "$jobname model_override" "$model_override" "$expected_model_override"
    assert_eq "$jobname batch_size" "$BATCH_SIZE" "$expected_batch_size"
    assert_eq "$jobname is_v1" "$is_v1" "$expected_is_v1"
    assert_eq "$jobname attack_domain" "$attack_domain" "$expected_attack_domain"
    assert_eq "$jobname attack_eps" "$attack_eps" "$expected_attack_eps"
    assert_eq "$jobname crit" "$crit" "$expected_crit"
    assert_eq "$jobname advtrain" "$advtrain" "$expected_advtrain"
    assert_eq "$jobname gradnorm" "$gradnorm" "$expected_gradnorm"
    assert_eq "$jobname dvd_enabled" "$dvd_enabled" "$expected_dvd_enabled"
    assert_eq "$jobname dvd_variant" "$dvd_variant" "$expected_dvd_variant"
}

assert_eq "48GB env" "$(select_train_env 49140)" "tomer_advtrain"
assert_eq "90GB env" "$(select_train_env 90000)" "tomer_advtrain_pro"

assert_parse \
    "l2_16_init1" 49140 \
    "small" "l2_16_init1" "convnext_small_l2_16_init1" "" "256" \
    "false" "pixel" "16" "madry" "true" "false"

assert_parse \
    "convnext_base_l2_16_init1" 49140 \
    "base" "l2_16_init1" "convnext_base_l2_16_init1" "model=convnext_base" "256" \
    "false" "pixel" "16" "madry" "true" "false"

assert_parse \
    "convnext_large_l2_16_init1" 49140 \
    "large" "l2_16_init1" "convnext_large_l2_16_init1" "model=convnext_large" "192" \
    "false" "pixel" "16" "madry" "true" "false"

assert_parse \
    "convnext_large_l2trades_4_init2" 49140 \
    "large" "l2trades_4_init2" "convnext_large_l2trades_4_init2" "model=convnext_large" "144" \
    "false" "pixel" "4" "trades" "true" "false"

assert_parse \
    "v1clean_linf_8_init1" 49140 \
    "small" "v1clean_linf_8_init1" "convnext_small_v1_clean_linf_8_init1" "model=convnext_small_v1" "256" \
    "true" "v1_feature" "8" "madry" "true" "false"

assert_parse \
    "convnext_base_v1clean_linf_8_init1" 49140 \
    "base" "v1clean_linf_8_init1" "convnext_base_v1_clean_linf_8_init1" "model=convnext_base_v1" "256" \
    "true" "v1_feature" "8" "madry" "true" "false"

assert_parse \
    "convnext_large_v1clean_linf_8_init1" 49140 \
    "large" "v1clean_linf_8_init1" "convnext_large_v1_clean_linf_8_init1" "model=convnext_large_v1" "192" \
    "true" "v1_feature" "8" "madry" "true" "false"

assert_parse \
    "convnext_large_v1clean_linftrades_8_init1" 90000 \
    "large" "v1clean_linftrades_8_init1" "convnext_large_v1_clean_linftrades_8_init1" "model=convnext_large_v1" "288" \
    "true" "v1_feature" "8" "trades" "true" "false"

assert_parse \
    "gradnorm_l2_8_init3" 22000 \
    "small" "gradnorm_l2_8_init3" "convnext_small_gradnorm_l2_8_init3" "" "96" \
    "false" "pixel" "8" "madry" "false" "true"

assert_parse \
    "dvd_b_init1" 49140 \
    "small" "dvd_b_init1" "convnext_small_dvd_b_init1" "" "256" \
    "false" "pixel" "" "madry" "false" "false" "true" "dvd-b"

assert_parse \
    "dvd_b_l2_16_init1" 49140 \
    "small" "dvd_b_l2_16_init1" "convnext_small_dvd_b_l2_16_init1" "" "256" \
    "false" "pixel" "16" "madry" "true" "false" "true" "dvd-b"

assert_parse \
    "convnext_base_dvd_s_linftrades_8_init2" 90000 \
    "base" "dvd_s_linftrades_8_init2" "convnext_base_dvd_s_linftrades_8_init2" "model=convnext_base" "384" \
    "false" "pixel" "8" "trades" "true" "false" "true" "dvd-s"

assert_parse \
    "dvd_b_trades_l2_8_init2" 49140 \
    "small" "dvd_b_trades_l2_8_init2" "convnext_small_dvd_b_trades_l2_8_init2" "" "192" \
    "false" "pixel" "8" "trades" "true" "false" "true" "dvd-b"

if parse_train_job "v1noise_linf_8_init1" 49140 >/dev/null 2>&1; then
    fail "v1noise adversarial job should be rejected"
fi

if parse_train_job "convnext_large_v1clean_gradnorm_l1_8_init1" 49140 >/dev/null 2>&1; then
    fail "V1 gradnorm job should be rejected"
fi

if parse_train_job "l2_8_init1" 21000 >/dev/null 2>&1; then
    fail "GPU memory below 22GB should be rejected"
fi

if parse_train_job "dvd_x_l2_8_init1" 49140 >/dev/null 2>&1; then
    fail "invalid DVD variant should be rejected"
fi

echo "[OK] train launcher mapping tests passed"
