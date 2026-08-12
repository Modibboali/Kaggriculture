# Runs chunked 100-game 5-day ablation for modes A-E vs starter.
# Idempotent: chunks already in the CSV are skipped.
#   powershell -File agent/ai/run_modes_chunks.ps1
$ErrorActionPreference = "Continue"
$py = ".\.venv\Scripts\python.exe"
$csv = "output\modes_5d_starter.csv"
$opponent = "starter"
$days = 5
$iters = 12
$chunk = 20
$total = 100

foreach ($mode in @("A", "B", "C", "D", "E")) {
    for ($start = 1; $start -lt $total; $start += $chunk) {
        $end = [Math]::Min($start + $chunk, $total + 1)
        $done = $false
        if (Test-Path $csv) {
            $done = (Select-String -Path $csv -Pattern "^$mode,new,$opponent,$days,$iters,$start,$end," -Quiet)
        }
        if ($done) {
            Write-Output "skip mode $mode seeds[$start,$end) (already done)"
            continue
        }
        Write-Output "RUN mode $mode vs $opponent seeds[$start,$end) at $(Get-Date -Format HH:mm:ss)"
        & $py -u -m agent.ai.run_chunked --kind new --mode $mode --opponent $opponent --days $days --iters $iters --seed-start $start --seed-end $end --out $csv
        Write-Output "DONE mode $mode seeds[$start,$end) at $(Get-Date -Format HH:mm:ss)"
    }
}
Write-Output "=== ALL MODES COMPLETE ==="
