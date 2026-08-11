# Runs the remaining horizon-aware experiments sequentially, logging each to
# output/ with unbuffered output so progress is visible in the log files.
# Run after the regression completes:  powershell -File agent/ai/run_all_experiments.ps1

$ErrorActionPreference = "Continue"
$py = ".\.venv\Scripts\python.exe"

Write-Output "=== matchups (new + old vs random/starter/heuristic, 100 games, 5d) ==="
& $py -u -m agent.ai.run_horizon_experiments matchups --days 5 --games 100 --iterations 12 *> output\horizon_matchups.log

Write-Output "=== sweep (budgets 25..500, 3d, 5 games) ==="
& $py -u -m agent.ai.run_horizon_experiments sweep --days 3 --games 5 *> output\horizon_sweep.log

Write-Output "=== ablation (old/new/no-crop/no-animal-worker, 5d, 10 games) ==="
& $py -u -m agent.ai.run_horizon_experiments ablation --days 5 --games 10 --iterations 12 *> output\horizon_ablation.log

Write-Output "=== DONE ==="
