param([switch]$Check)
$ErrorActionPreference = 'Stop'
$stablePython = Join-Path $env:LOCALAPPDATA 'FP-K2-Training\segmentasi-nrp8-venv\Scripts\python.exe'
$backend = 'C:\Users\junxi\OneDrive\Documents\Semester 5\Kompt Biomeds\FP\fp_rsna\eksperimen_model\03_colab\colab_project.py'
$launcher = 'C:\Users\junxi\.agents\skills\colab\scripts\colab.ps1'
$models = @('maskrcnn','maskrcnn-v2')
$seeds = @(42,97,123)
for ($modelIndex=0; $modelIndex -lt 2; $modelIndex++) {
    $model = $models[$modelIndex]
    for ($seedIndex=0; $seedIndex -lt 3; $seedIndex++) {
        $seed = $seeds[$seedIndex]
        $profileId = '{0:00}' -f (1 + 3*$modelIndex + $seedIndex)
        $launch = @(
            'start','train.py','--root',$PSScriptRoot,'--project','segmentasi-indoor-research',
            '--run',("$model-$seed"),'--profile',$profileId,'--accelerator','T4',
            '--title','Segmentasi indoor: replikasi Mask R-CNN','--dataset','Roboflow v5 train 397 dan valid 166',
            '--model',$model,'--method',("R50-FPN, seed $seed"),'--epochs','20',
            '--metric','Mask AP50-95 validasi','--unit','%','--mode','max',
            '--requirements','requirements-maskrcnn.txt','--include','metrics.py',
            '--include','data/panoptic-indoor-v5-640.zip','--output','output/result.zip',
            '--train-arg=--model',("--train-arg=$model"),'--train-arg=--seed',("--train-arg=$seed"),
            '--train-arg=--epochs','--train-arg=20','--train-arg=--minutes','--train-arg=38',
            '--train-arg=--eval-splits','--train-arg=valid'
        )
        if ($Check) { $launch += '--check' }
        if ((Test-Path -LiteralPath $stablePython) -and (Test-Path -LiteralPath $backend)) {
            $env:COLAB_CLI_REQUEST_TIMEOUT = '60'
            $env:COLAB_CLI_ASSIGN_RETRIES = '2'
            & $stablePython $backend @launch
        } else { & $launcher @launch }
        if ($LASTEXITCODE -ne 0) { throw "Gagal mendaftarkan $model-$seed." }
    }
}
