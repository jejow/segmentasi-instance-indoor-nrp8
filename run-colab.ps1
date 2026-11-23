param(
    [Parameter(Mandatory)]
    [ValidateSet('Pilot','Full','Three')]
    [string]$Phase,
    [ValidatePattern('^0[1-9]$')]
    [string]$OnlyProfile,
    [ValidatePattern('^-[a-z0-9-]+$')]
    [string]$RunSuffix,
    [switch]$Check
)

$ErrorActionPreference = 'Stop'
$launcher = 'C:\Users\junxi\.agents\skills\colab\scripts\colab.ps1'
$stablePython = Join-Path $env:LOCALAPPDATA 'FP-K2-Training\segmentasi-nrp8-venv\Scripts\python.exe'
$backend = 'C:\Users\junxi\OneDrive\Documents\Semester 5\Kompt Biomeds\FP\fp_rsna\eksperimen_model\03_colab\colab_project.py'
$dataset = 'data/panoptic-indoor-v5-640.zip'
$selectedClasses = 'hard-classes.json'
$models = @(
    @{ Model='yolov8n'; Title='YOLOv8n-seg'; Requirements='requirements-yolo.txt' },
    @{ Model='yolo11n'; Title='YOLO11n-seg'; Requirements='requirements-yolo.txt' },
    @{ Model='maskrcnn'; Title='Mask R-CNN R50-FPN'; Requirements='requirements-maskrcnn.txt' }
)
$firstProfile = @{ Pilot=1; Full=4; Three=7 }[$Phase]
$methodLabel = @{ Pilot='Pilot semua kelas'; Full='Semua kelas'; Three='Tiga kelas tersulit' }[$Phase]

if ($Phase -eq 'Three' -and !(Test-Path -LiteralPath (Join-Path $PSScriptRoot $selectedClasses))) {
    throw 'Tiga kelas harus ditetapkan dari hasil pilot sebelum training kondisi ini.'
}

for ($index = 0; $index -lt $models.Count; $index++) {
    $model = $models[$index]
    $profileId = '{0:00}' -f ($firstProfile + $index)
    if ($OnlyProfile -and $OnlyProfile -ne $profileId) { continue }

    $epochLimit = if ($Phase -eq 'Pilot') { 3 } else { 16 }
    $runtimeLimit = if ($Phase -eq 'Pilot') { 10 } else { 45 }
    $runId = $Phase.ToLowerInvariant() + '-' + $model.Model + $RunSuffix
    $arguments = @(
        'start', 'train.py', '--root', $PSScriptRoot,
        '--project', 'segmentasi-indoor-nrp8', '--run', $runId,
        '--title', 'Segmentasi instance indoor (NRP 8)', '--dataset', 'Roboflow Panoptic Indoor Segmentation v5',
        '--profile', $profileId, '--model', $model.Title, '--method', $methodLabel,
        '--accelerator', 'T4', '--metric', 'Mask AP50-95 validasi', '--unit', '%', '--mode', 'max',
        '--epochs', $epochLimit, '--requirements', $model.Requirements,
        '--include', 'metrics.py', '--include', $dataset, '--output', 'output/result.zip',
        '--train-arg=--model', $('--train-arg=' + $model.Model),
        '--train-arg=--epochs', $('--train-arg=' + $epochLimit),
        '--train-arg=--minutes', $('--train-arg=' + $runtimeLimit)
    )
    if ($Phase -eq 'Pilot') {
        $arguments += @('--train-arg=--pilot','--train-arg=--eval-splits','--train-arg=valid')
    }
    if ($Phase -eq 'Three') {
        $arguments += @('--include', $selectedClasses, '--train-arg=--classes', ('--train-arg=' + $selectedClasses))
    }
    if ($Check) { $arguments += '--check' }

    if ((Test-Path -LiteralPath $stablePython) -and (Test-Path -LiteralPath $backend)) {
        $env:COLAB_CLI_REQUEST_TIMEOUT = '60'
        $env:COLAB_CLI_ASSIGN_RETRIES = '2'
        & $stablePython $backend @arguments
    } else {
        & $launcher @arguments
    }
    if ($LASTEXITCODE -ne 0) { throw "Peluncur gagal mendaftarkan profil $profileId, run $runId." }
}
