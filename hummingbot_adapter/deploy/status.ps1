# Forward-test status per bot (WO-HB2 deliverable 5). Usage, in hummingbot_adapter\deploy: .\status.ps1 [-Smoke]
param([switch]$Smoke)
$data = Join-Path $env:USERPROFILE "Documents\hb_data"; if ($Smoke) { $data = Join-Path $data "smoke" }
$nowUs = [DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds() * 1000
foreach ($bot in "hb_b0", "hb_b3") {
    $box = docker inspect -f "{{.State.Status}}|{{.State.StartedAt}}|{{.RestartCount}}" $bot 2>$null
    if ($box) {
        $p = $box -split '\|'; $up = [DateTimeOffset]::UtcNow - [DateTimeOffset]::Parse($p[1].Substring(0, 19) + "Z")
        $box = "{0}, up {1:N0} min, {2} restarts" -f $p[0], $up.TotalMinutes, $p[2]
    } else { $box = "no container" }
    $refused = Join-Path $data "$bot\start_refused.txt"; if (Test-Path $refused) { "{0}: START REFUSED: {1}" -f $bot, (Get-Content $refused -TotalCount 1) }
    $file = Join-Path $data "$bot\state.json"
    if (-not (Test-Path $file)) { "{0}: {1}; no state file" -f $bot, $box; continue }
    $s = Get-Content $file -Raw | ConvertFrom-Json; $t = $s.status; $c = $s.counters
    $ago = if ($s.last_step_us) { "{0:N1} s" -f (($nowUs - $s.last_step_us) / 1e6) } else { "never" }
    "{0}: {1}; day {2}; last step {3} ago; q {4}; gross P&L {5:N4} USDT; drawdown {6:N4} USDT; fills {7}/{8}" -f `
        $bot, $box, $s.day.date, $ago, $t.q, $t.wealth, $t.drawdown, $c.fills_buy, $c.fills_sell
    "    halted {0} {1}; paused stale/clock/drawdown {2}/{3}/{4}" -f $t.halted, $t.halt_reason, $t.stale_paused, `
        $t.clock_paused, $t.drawdown_paused
    $tr = ($c.triggers.PSObject.Properties | ForEach-Object { "$($_.Name)=$($_.Value)" }) -join " "
    "    triggers: {0}" -f $(if ($tr) { $tr } else { "none" })
    "    new orders {0} (max {1} per 10 s); rate-cap skips 10 s/day {2}/{3}; round-away steps {4}/{5}, orders {6}/{7}" -f `
        $t.new_orders, $t.max_new_orders_10s, $t.skipped_rate_10s, $t.skipped_rate_day, $c.round_bid_steps, `
        $c.round_ask_steps, $c.round_bid_orders, $c.round_ask_orders
    $inc = @(Get-Content (Join-Path $data "$bot\run_log_incidents.md") -ErrorAction SilentlyContinue); $re = @($inc -match 'clock re-anchor').Count; $back = @($inc -match 'clock backward').Count
    "    clock {0:N3} s (uncorrected raw {13:N3} s, wall {14:N3} s) at {1}, rate step {2:P3} raw {3:P3} over {4:N0} s; re-anchors run {5} (today {6}), backward run {11} (today {12}); snapshots {7} (ahead {8}, errors {9}); crossed/locked {10}" -f `
        $t.clock.offset_s, $t.clock.checked_utc, $t.clock.rate_step, $t.clock.rate_monotonic, $t.clock.rate_over_s, $re, $t.clock_jumps, $t.book_snapshots, $t.snapshots_ahead_of_diffs, $t.snapshot_errors, $t.crossed_books, $back, $t.clock_backward, $t.clock.raw_offset_s, $t.clock.wall_offset_s
}
"free disk on C: {0:N1} GB" -f ((Get-PSDrive C).Free / 1GB)
