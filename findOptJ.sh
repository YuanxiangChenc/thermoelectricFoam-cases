#!/bin/bash
# =============================================================================
# findOptJ.sh  -  Sweep J and find optimum Seebeck efficiency
# =============================================================================
# Adjustable:
J_START=500
J_INTERVAL=100
T_MAX_LIMIT=520
# =============================================================================

MAT_PROPS="constant/materialProperties"
LOG="findOptJ.log"
RESULTS="findOptJ_results.txt"

# --- colours -----------------------------------------------------------------
R='\033[0;31m'; Y='\033[1;33m'; G='\033[0;32m'; C='\033[0;36m'; N='\033[0m'
B='\033[1m'

# --- logging helpers (all write to terminal AND log file) --------------------
_tee() { echo -e "$*" | tee -a "$LOG"; }
step() { _tee "\n${C}${B}[STEP]${N} $*"; }
info() { _tee "       $*"; }
ok()   { _tee "${G}[  OK]${N} $*"; }
warn() { _tee "${Y}[WARN]${N} $*"; }
err()  { _tee "${R}[ ERR]${N} $*"; }
dbg()  { _tee "  ${Y}(dbg)${N} $*"; }
sep()  { _tee "       $(printf '%.0s-' {1..60})"; }

# =============================================================================
# 1. SANITY CHECKS
# =============================================================================
echo "# findOptJ.sh  $(date)" > "$LOG"
echo "# J_START=$J_START  J_INTERVAL=$J_INTERVAL  T_MAX_LIMIT=$T_MAX_LIMIT" >> "$LOG"

step "Sanity checks"
info "Working dir        : $(pwd)"
info "materialProperties : $MAT_PROPS"

[ ! -f "$MAT_PROPS" ] && { err "Cannot find $MAT_PROPS -- run from the case folder"; exit 1; }
ok "materialProperties found"

for cmd in thermoelectricFoam python3 bc sed grep awk; do
    if command -v "$cmd" &>/dev/null; then
        ok "$cmd -> $(command -v $cmd)"
    else
        err "$cmd not found in PATH"; exit 1
    fi
done

[ ! -x "./Allsample" ] && { err "./Allsample not found or not executable"; exit 1; }
ok "./Allsample found"

[ ! -f "cal_all.py" ] && { err "cal_all.py not found"; exit 1; }
ok "cal_all.py found"

info "Current j line in materialProperties:"
dbg "$(grep 'j (0' "$MAT_PROPS")"

# =============================================================================
# 2. SET J
# =============================================================================
set_J() {
    local jval=$1
    step "set_J: writing J=$jval to $MAT_PROPS"
    dbg "BEFORE: $(grep 'j (0' "$MAT_PROPS")"
    sed -i "s/j (0 [0-9.-]* 0)/j (0 $jval 0)/" "$MAT_PROPS"
    local rc=$?
    dbg "AFTER : $(grep 'j (0' "$MAT_PROPS")"
    [ $rc -ne 0 ] && { err "sed failed (rc=$rc)"; return 1; }
    grep -q "j (0 $jval 0)" "$MAT_PROPS" || { err "Value not found after sed -- check format"; return 1; }
    ok "J=$jval written"
}

# =============================================================================
# 3. RUN SOLVER
# =============================================================================
run_solver() {
    step "run_solver: thermoelectricFoam"
    thermoelectricFoam > log 2>&1
    local rc=$?
    if [ $rc -ne 0 ]; then
        err "Solver exited with rc=$rc"
        err "Last 15 lines of solver log:"
        tail -15 log | while IFS= read -r line; do dbg "  $line"; done
        return 1
    fi
    local nsteps
    nsteps=$(grep -c "Time = " log 2>/dev/null || echo "?")
    ok "Solver done  (time steps: $nsteps)"
}

# =============================================================================
# 4. RUN ALLSAMPLE
# =============================================================================
run_sample() {
    step "run_sample: ./Allsample"
    ./Allsample >> "$LOG" 2>&1
    local rc=$?
    [ $rc -ne 0 ] && { err "Allsample failed (rc=$rc)"; return 1; }
    ok "Allsample done"
}

# =============================================================================
# 5. PARSE cal_all.py
#    Sets globals: G_QSH  G_QSC  G_QE  G_ETA  G_TMAX
# =============================================================================
parse_output() {
    local J=$1
    step "parse_output: python3 cal_all.py  (J=$J)"

    local raw
    raw=$(python3 cal_all.py 2>&1)
    local rc=$?

    if [ $rc -ne 0 ]; then
        err "cal_all.py exited rc=$rc"
        echo "$raw" | while IFS= read -r line; do dbg "  $line"; done
        return 1
    fi

    info "--- cal_all.py output ---"
    echo "$raw" | while IFS= read -r line; do info "  $line"; done
    info "--- end ---"

    # Extract: Tmax is the first decimal number on the "hot side" line
    G_TMAX=$(echo "$raw" | awk '/hot side/  { for(i=1;i<=NF;i++) if($i~/^[0-9]+\.[0-9]+$/) {print $i; exit} }')
    G_QSH=$( echo "$raw" | awk '/hot side/  { print $NF }')
    G_QSC=$( echo "$raw" | awk '/cold side/ { print $NF }')
    G_QE=$(  echo "$raw" | awk '/Q\.e/      { print $NF }')
    G_ETA=$( echo "$raw" | awk '/seebeck efficiency/ { print $NF }')

    # Strip leading + signs
    G_ETA=${G_ETA#+};   G_TMAX=${G_TMAX#+}
    G_QSH=${G_QSH#+};   G_QSC=${G_QSC#+};   G_QE=${G_QE#+}

    dbg "Extracted -> eta='$G_ETA'  Tmax='$G_TMAX'  qsh='$G_QSH'  qsc='$G_QSC'  qe='$G_QE'"

    if ! [[ "$G_ETA"  =~ ^-?[0-9]+(\.[0-9]+)?$ ]]; then
        err "eta='$G_ETA' is not a number"
        err "The awk pattern for 'seebeck efficiency' did not match -- check cal_all.py output above"
        return 1
    fi
    if ! [[ "$G_TMAX" =~ ^-?[0-9]+(\.[0-9]+)?$ ]]; then
        err "Tmax='$G_TMAX' is not a number"
        err "The awk pattern for 'hot side' did not match -- check cal_all.py output above"
        return 1
    fi

    ok "Parsed: eta=${G_ETA}%  Tmax=${G_TMAX}K"
}

# =============================================================================
# 6. RESULTS TABLE HELPERS
# =============================================================================
init_results() {
    printf "%-9s  %-13s  %-13s  %-11s  %-8s  %-8s\n" \
        "J(A/m2)" "q.s.hot" "q.s.cold" "Q.e" "eta[%]" "Tmax[K]" | tee "$RESULTS"
    printf '%.0s-' {1..68} | tee -a "$RESULTS"; echo | tee -a "$RESULTS"
}

append_results() {
    local J=$1
    printf "%-9s  %-13s  %-13s  %-11s  %-8s  %-8s\n" \
        "$J" "$G_QSH" "$G_QSC" "$G_QE" "$G_ETA" "$G_TMAX" | tee -a "$RESULTS"
}

# =============================================================================
# 7. RUN ONE J VALUE  (rc: 0=ok  1=error  2=Tmax exceeded)
# =============================================================================
run_one_J() {
    local J=$1
    sep
    _tee "${B}>>> J = $J A/m2 <<<${N}"
    sep

    set_J     "$J" || return 1
    run_solver      || return 1
    run_sample      || return 1
    parse_output "$J" || return 1
    append_results "$J"

    if (( $(echo "$G_TMAX > $T_MAX_LIMIT" | bc -l) )); then
        warn "Tmax=${G_TMAX}K exceeds limit ${T_MAX_LIMIT}K -- stopping"
        return 2
    fi
    return 0
}

# =============================================================================
# 8. MAIN SWEEP
# =============================================================================
sep
info "J_START=$J_START  J_INTERVAL=$J_INTERVAL  T_MAX_LIMIT=${T_MAX_LIMIT}K"
sep

init_results

best_J=0; best_eta=-999; consecutive_drops=0; STOP_REASON=""

# J = 0 baseline
run_one_J 0
rc=$?
[ $rc -eq 1 ] && { err "J=0 baseline failed -- aborting"; exit 1; }
[ $rc -eq 2 ] && { warn "J=0 already over temp limit -- check BCs"; exit 1; }
best_eta=$G_ETA; best_J=0
ok "Baseline J=0: eta=${best_eta}%"

# Sweep
J_cur=$J_START
while true; do
    run_one_J $J_cur
    rc=$?

    [ $rc -eq 1 ] && { err "J=$J_cur failed -- stopping"; break; }
    [ $rc -eq 2 ] && { STOP_REASON="Tmax exceeded at J=$J_cur"; break; }

    if (( $(echo "$G_ETA > $best_eta" | bc -l) )); then
        best_J=$J_cur; best_eta=$G_ETA; consecutive_drops=0
        ok "New best  J=${J_cur}  eta=${G_ETA}%"
    else
        consecutive_drops=$((consecutive_drops + 1))
        info "Drop #${consecutive_drops}: eta=${G_ETA}%  (best=${best_eta}%)"
        if [ $consecutive_drops -ge 2 ]; then
            STOP_REASON="2 consecutive drops after J=${best_J}"
            info "Optimum bracketed -- stopping"
            break
        fi
    fi

    J_cur=$((J_cur + J_INTERVAL))
done

# =============================================================================
# 9. FINAL SUMMARY
# =============================================================================
sep
_tee "\n${B}RESULTS:${N}"
cat "$RESULTS" | tee -a "$LOG"
_tee "\n${B}SUMMARY${N}"
ok  "Optimal J  = ${best_J} A/m2"
ok  "Best eta   = ${best_eta} %"
[ -n "$STOP_REASON" ] && info "Stop reason : $STOP_REASON"
info "Table  -> $RESULTS"
info "Log    -> $LOG"
