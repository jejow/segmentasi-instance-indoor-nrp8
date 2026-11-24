param(
    [switch]$Check,
    [string]$OnlyRun,
    [ValidatePattern('^0[1-9]$')][string]$ProfileOverride,
    [ValidatePattern('^-retry[0-9]+$')][string]$RetrySuffix
)
$ErrorActionPreference = 'Stop'
if (($ProfileOverride -or $RetrySuffix) -and !$OnlyRun) {
    throw 'Pilih satu run sebelum mengganti profil atau menambah akhiran.'
}
$dataset = 'data/panoptic-indoor-v5-train-valid-640.zip'
python (Join-Path $PSScriptRoot 'pack_training_splits.py') (Join-Path $PSScriptRoot 'data/panoptic-indoor-v5-640.zip') (Join-Path $PSScriptRoot $dataset)
if ($LASTEXITCODE -ne 0) { throw 'Paket train dan validasi gagal disiapkan.' }
$stablePython = Join-Path $env:LOCALAPPDATA 'FP-K2-Training/segmentasi-nrp8-venv/Scripts/python.exe'
$backend = 'C:/Users/junxi/OneDrive/Documents/Semester 5/Kompt Biomeds/FP/fp_rsna/eksperimen_model/03_colab/colab_project.py'
$launcher = 'C:/Users/junxi/.agents/skills/colab/scripts/colab.ps1'
$cases = @(
    @{Run='coco-640-42'; Init='coco-overlap'; Size=640; Seed=42; Profile='02'},
    @{Run='coco-640-97'; Init='coco-overlap'; Size=640; Seed=97; Profile='03'},
    @{Run='coco-800-42'; Init='coco-overlap'; Size=800; Seed=42; Profile='04'},
    @{Run='coco-800-97'; Init='coco-overlap'; Size=800; Seed=97; Profile='05'},
    @{Run='random-800-42'; Init='random'; Size=800; Seed=42; Profile='06'},
    @{Run='random-800-97'; Init='random'; Size=800; Seed=97; Profile='01'}
)
if ($OnlyRun -and $OnlyRun -notin $cases.Run) { throw "Run tidak dikenal: $OnlyRun" }
foreach ($case in $cases) {
    if ($OnlyRun -and $case.Run -ne $OnlyRun) { continue }
    $run = $case.Run + $RetrySuffix
    $profile = if ($ProfileOverride) { $ProfileOverride } else { $case.Profile }
    $launch = @(
        'start','train.py','--root',$PSScriptRoot,'--project','segmentasi-indoor-improvement',
        '--run',$run,'--profile',$profile,'--accelerator','T4',
        '--title','Segmentasi indoor - inisialisasi COCO dan resolusi','--dataset','Panoptic Indoor v5, train 397 dan valid 166',
        '--model','maskrcnn-v2','--method',"$($case.Init), $($case.Size) px, seed $($case.Seed)",
        '--epochs','20','--metric','Mask AP50-95 validasi','--unit','%','--mode','max',
        '--requirements','requirements-maskrcnn.txt','--include','metrics.py',
        '--include',$dataset,'--output','output/result.zip',
        '--train-arg=--model','--train-arg=maskrcnn-v2',
        '--train-arg=--seed',"--train-arg=$($case.Seed)",
        '--train-arg=--dataset',"--train-arg=$dataset",
        '--train-arg=--rcnn-head-init',"--train-arg=$($case.Init)",
        '--train-arg=--rcnn-size',"--train-arg=$($case.Size)",
        '--train-arg=--epochs','--train-arg=20','--train-arg=--minutes','--train-arg=55',
        '--train-arg=--eval-splits','--train-arg=valid'
    )
    if ($Check) { $launch += '--check' }
    if ((Test-Path -LiteralPath $stablePython) -and (Test-Path -LiteralPath $backend)) {
        $env:COLAB_CLI_REQUEST_TIMEOUT = '75'
        $env:COLAB_CLI_ASSIGN_RETRIES = '2'
        & $stablePython $backend @launch
    } else {
        & $launcher @launch
    }
    if ($LASTEXITCODE -ne 0) { throw "Gagal mendaftarkan $run." }
}
