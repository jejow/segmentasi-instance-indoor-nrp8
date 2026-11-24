param(
    [ValidatePattern('^0[1-9]$')][string]$Profile = '07',
    [ValidatePattern('^[a-z0-9-]+$')][string]$Run = 'selected-public-test',
    [string]$SelectionFile = 'output/research/research.json',
    [switch]$Check
)
$ErrorActionPreference = 'Stop'
$result = Get-Content -LiteralPath (Join-Path $PSScriptRoot $SelectionFile) -Raw | ConvertFrom-Json
$model = $result.selection.variant
$selected = $result.selection.checkpoint
$chosen = $result.runs | Where-Object run -EQ $selected | Select-Object -First 1
if (!$chosen) { throw 'Tidak ada checkpoint uji yang cocok dengan hasil pemilihan validasi.' }
$checkpoint = ((Join-Path (Split-Path $SelectionFile -Parent) "$selected/checkpoint.pt") -replace '\\','/')
$testDataset = 'output/tuning/test-v5.zip'
$stablePython = Join-Path $env:LOCALAPPDATA 'FP-K2-Training/segmentasi-nrp8-venv/Scripts/python.exe'
$backend = 'C:/Users/junxi/OneDrive/Documents/Semester 5/Kompt Biomeds/FP/fp_rsna/eksperimen_model/03_colab/colab_project.py'
$launcher = 'C:/Users/junxi/.agents/skills/colab/scripts/colab.ps1'
$launch = @(
    'start','train.py','--root',$PSScriptRoot,'--project','segmentasi-indoor-research-test',
    '--run',$Run,'--profile',$Profile,'--accelerator','T4','--title','Uji publik checkpoint Mask R-CNN',
    '--dataset','Test resmi Panoptic Indoor v5, 359 foto publik',
    '--model',$model,'--method',"Checkpoint $selected, dipilih dari validasi",
    '--metric','Mask AP50-95 test','--unit','%','--mode','max',
    '--requirements','requirements-maskrcnn.txt','--include','metrics.py',
    '--include',$checkpoint,'--include',$testDataset,'--output','output/result.zip',
    '--train-arg=--model',"--train-arg=$model",'--train-arg=--seed',"--train-arg=$($chosen.seed)",
    '--train-arg=--checkpoint',"--train-arg=$checkpoint",'--train-arg=--dataset',"--train-arg=$testDataset",
    '--train-arg=--eval-splits','--train-arg=test'
)
if ($chosen.rcnn_size) {
    $launch += @('--train-arg=--rcnn-size',"--train-arg=$($chosen.rcnn_size)")
}
if ($Check) { $launch += '--check' }
if ((Test-Path -LiteralPath $stablePython) -and (Test-Path -LiteralPath $backend)) {
    $env:COLAB_CLI_REQUEST_TIMEOUT = '60'
    $env:COLAB_CLI_ASSIGN_RETRIES = '2'
    & $stablePython $backend @launch
} else {
    & $launcher @launch
}
if ($LASTEXITCODE -ne 0) { throw "Gagal menjalankan evaluasi $Run." }
