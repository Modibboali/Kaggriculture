# Runs chunked 100-game matchups for the new evaluator (short, reliable subprocesses).
# Idempotent: chunks already present in the CSV are skipped, so an interrupted run
# can be resumed. Each chunk is a separate python invocation (~5-8 min, well inside
# the environment's reliable window).
#   powershell -File agent/ai/run_matchup_chunks.ps1
$ErrorActionPreference = "Continue"
$py = ".\.venv\Scripts\python.exe"
$csv = "output\matchups_new.csv"
$kind = "new"
$days = 5
$iters = 12
$chunk = 20   # games per chunk
$total = 100  # target games per matchup

foreach ($opponent in @("random", "starter", "heuristic")) {
    for ($start = 1; $start -lt $total; $start += $chunk) {
        $end = [Math]::Min($start + $chunk, $total + 1)
        $done = $false
        if (Test-Path $csv) {
            $done = (Select-String -Path $csv -Pattern "^$kind,$opponent,$days,$iters,$start,$end," -Quiet)
        }
        if ($done) {
            Write-Output "skip $kind vs $opponent seeds[$start,$end) (already done)"
            continue
        }
        Write-Output "RUN $kind vs $opponent seeds[$start,$end) at $(Get-Date -Format HH:mm:ss)"
        & $py -u -m agent.ai.run_chunked --kind $kind --opponent $opponent --days $days --iters $iters --seed-start $start --seed-end $end --out $csv
        Write-Output "DONE $kind vs $opponent seeds[$start,$end) at $(Get-Date -Format HH:mm:ss)"
    }
}
Write-Output "=== ALL CHUNKS COMPLETE ==="
