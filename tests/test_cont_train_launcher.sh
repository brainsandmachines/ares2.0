#!/bin/bash
set -euo pipefail

source "$(dirname "$0")/../sbatches/cont_train_launcher.sh"

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
    local expected_model_name="$2"
    local expected_source_model_name="$3"
    local expected_attack_norm="$4"
    local expected_crit="$5"
    local expected_attack_domain="$6"
    local expected_schedule_type="$7"
    local expected_epochs="$8"

    local tmp_root
    tmp_root="$(mktemp -d)"
    mkdir -p "$tmp_root/$expected_source_model_name"
    touch "$tmp_root/$expected_source_model_name/model_best_adv.pth.tar"
    unset CONTINUATION_CKPT || true

    parse_cont_train_job "$jobname" 49140 "$tmp_root"

    assert_eq "$jobname model_name" "$model_name" "$expected_model_name"
    assert_eq "$jobname source_model_name" "$source_model_name" "$expected_source_model_name"
    assert_eq "$jobname attack_norm" "$attack_norm" "$expected_attack_norm"
    assert_eq "$jobname crit" "$crit" "$expected_crit"
    assert_eq "$jobname attack_domain" "$attack_domain" "$expected_attack_domain"
    assert_eq "$jobname schedule_type" "$schedule_type" "$expected_schedule_type"
    assert_eq "$jobname epochs" "$epochs" "$expected_epochs"
    assert_eq "$jobname checkpoint" "$CONTINUATION_CKPT" "$tmp_root/$expected_source_model_name/model_best_adv.pth.tar"

    rm -rf "$tmp_root"
}

assert_eq "48GB env" "$(select_cont_train_env 49140)" "tomer_advtrain"
assert_eq "90GB env" "$(select_cont_train_env 90000)" "tomer_advtrain_pro"

assert_parse \
    "linf_cont4to8_direct_init1" \
    "convnext_small_linf_cont4to8_direct_init1" \
    "convnext_small_linf_4_init1" \
    "linf" "madry" "pixel" "fixed" "30"

assert_parse \
    "l1trades_cont4to8_ramp_init1" \
    "convnext_small_l1trades_cont4to8_ramp_init1" \
    "convnext_small_l1trades_4_init1" \
    "l1" "trades" "pixel" "warmup_ramp_fixed" "40"

assert_parse \
    "convnext_base_l2_cont8to16_direct_init2" \
    "convnext_base_l2_cont8to16_direct_init2" \
    "convnext_base_l2_8_init2" \
    "l2" "madry" "pixel" "fixed" "30"

assert_parse \
    "v1clean_linftrades_cont4to8_ramp_init1" \
    "convnext_small_v1_clean_linftrades_cont4to8_ramp_init1" \
    "convnext_small_v1_clean_linftrades_4_init1" \
    "linf" "trades" "v1_feature" "warmup_ramp_fixed" "40"

tmp_root="$(mktemp -d)"
mkdir -p "$tmp_root/convnext_small_l2_4_init1"
touch "$tmp_root/convnext_small_l2_4_init1/model_best.pth.tar"
unset CONTINUATION_CKPT || true
parse_cont_train_job "l2_cont4to8_direct_init1" 49140 "$tmp_root"
assert_eq "fallback checkpoint" "$CONTINUATION_CKPT" "$tmp_root/convnext_small_l2_4_init1/model_best.pth.tar"
rm -rf "$tmp_root"

tmp_ckpt="$(mktemp)"
CONTINUATION_CKPT="$tmp_ckpt"
parse_cont_train_job "l2_cont4to8_direct_init1" 49140 "/does/not/matter"
assert_eq "explicit checkpoint" "$CONTINUATION_CKPT" "$tmp_ckpt"
rm -f "$tmp_ckpt"
unset CONTINUATION_CKPT || true

if parse_cont_train_job "v1noise_linf_cont4to8_direct_init1" 49140 "/tmp" >/dev/null 2>&1; then
    fail "v1noise continuation should be rejected"
fi

echo "[OK] continuation launcher mapping tests passed"
