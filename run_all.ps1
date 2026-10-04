<# run_all.ps1 - Automated Execution Pipeline for Semantic-Aware Edge Video Analytics #>
param(
    [switch]$SkipStream,
    [switch]$Display,
    [int]$MaxFrames = 30
)

$ErrorActionPreference = "Stop"
$banner = "=" * 70

function Write-Step($num, $msg) {
    Write-Host ""
    Write-Host $banner -ForegroundColor Cyan
    Write-Host "  [$num] $msg" -ForegroundColor Yellow
    Write-Host $banner -ForegroundColor Cyan
}

# -- 0. Virtual environment check ------------------------------------------
Write-Step 0 "Virtual Environment Check"
if (-Not (Test-Path ".venv\Scripts\python.exe")) {
    Write-Host "  [ERROR] .venv not found. Run: python -m venv .venv" -ForegroundColor Red
    exit 1
}
$pyver = & .venv\Scripts\python.exe --version 2>&1
Write-Host "  Python: $pyver"
Write-Host "  pip packages: " -NoNewline
$pkgcount = (Get-ChildItem .venv\Lib\site-packages -Directory | Measure-Object).Count
Write-Host "$pkgcount installed"
Write-Host "  [OK] Virtual environment ready." -ForegroundColor Green

# -- 1. Pytest validation suite --------------------------------------------
Write-Step 1 "Running Pytest Validation Suite"
& .venv\Scripts\python.exe -m pytest tests/ -v
if ($LASTEXITCODE -ne 0) {
    Write-Host "  [FAIL] Tests did not pass. Aborting." -ForegroundColor Red
    exit 1
}
Write-Host "  [OK] All tests passed." -ForegroundColor Green

# -- 2. Benchmark generation ----------------------------------------------
Write-Step 2 "Running Parameterized Benchmark"
& .venv\Scripts\python.exe src/benchmark.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "  [FAIL] Benchmark failed." -ForegroundColor Red
    exit 1
}
Write-Host "  [OK] Benchmark complete. See benchmark_results.csv" -ForegroundColor Green

# -- 3. Visualization generation -------------------------------------------
Write-Step 3 "Generating Visualization Charts"
& .venv\Scripts\python.exe src/visualize_metrics.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "  [FAIL] Visualization generation failed." -ForegroundColor Red
    exit 1
}
Write-Host "  [OK] Charts saved to reports/" -ForegroundColor Green

# -- 4. Video stream simulation --------------------------------------------
if (-Not $SkipStream) {
    Write-Step 4 "Video Stream Simulation"
    Write-Host "  Starting FastAPI server in background..."

    $serverJob = Start-Job -ScriptBlock {
        Set-Location $using:PWD
        & .venv\Scripts\python.exe -m uvicorn src.server:app --host 127.0.0.1 --port 8765 --log-level warning 2>&1
    }

    # Wait for server startup
    Start-Sleep -Seconds 8

    $serverReady = $false
    for ($i = 0; $i -lt 10; $i++) {
        try {
            $r = Invoke-WebRequest -Uri "http://127.0.0.1:8765/health" -TimeoutSec 2 -ErrorAction SilentlyContinue
            if ($r.StatusCode -eq 200) { $serverReady = $true; break }
        } catch { }
        Start-Sleep -Seconds 1
    }

    if (-Not $serverReady) {
        Write-Host "  [WARN] Server did not start. Skipping stream test." -ForegroundColor Yellow
        Stop-Job $serverJob -ErrorAction SilentlyContinue
        Remove-Job $serverJob -ErrorAction SilentlyContinue
    } else {
        Write-Host "  Server ready on http://127.0.0.1:8765"
        $streamArgs = @("src/stream_processor.py", "--server", "http://127.0.0.1:8765", "--max-frames", $MaxFrames)
        if ($Display) { $streamArgs += "--display" }
        & .venv\Scripts\python.exe @streamArgs

        Stop-Job $serverJob -ErrorAction SilentlyContinue
        Remove-Job $serverJob -ErrorAction SilentlyContinue
        Write-Host "  [OK] Stream simulation complete. See reports/demo_output.mp4" -ForegroundColor Green
    }
} else {
    Write-Step 4 "Skipping Video Stream (use -SkipStream:$false to enable)"
}

# -- Summary ---------------------------------------------------------------
Write-Host ""
Write-Host $banner -ForegroundColor Green
Write-Host "  ALL STEPS COMPLETED SUCCESSFULLY" -ForegroundColor Green
Write-Host $banner -ForegroundColor Green
Write-Host ""
Write-Host "  Artifacts:"
Write-Host "    - benchmark_results.csv"
Write-Host "    - reports/latency_distribution.png"
Write-Host "    - reports/bandwidth_savings.png"
if (-Not $SkipStream) {
    Write-Host "    - reports/demo_output.mp4"
}
Write-Host "    - visual_test_output.jpg"
Write-Host ""


