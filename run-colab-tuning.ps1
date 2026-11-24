param(
    [switch]$Check,
    [ValidateSet('full-uniform','full-repeat','full-repeat-mask2','full-repeat-mask2-light','hard-uniform','hard-foreground')]
    [string]$OnlyRun,
    [ValidatePattern('^0[1-9]$')]
    [string]$ProfileOverride,
    [ValidatePattern('^-retry[0-9]+$')]
    [string]$RetrySuffix
)
$ErrorActionPreference = 'Stop'
if (($ProfileOverride -or $RetrySuffix) -and !$OnlyRun) {
    throw 'Pilih satu run dengan -OnlyRun sebelum mengubah profil atau ID.'
}
$launcher = 'C:\Users\junxi\.agents\skills\colab\scripts\colab.ps1'
$stablePython = Join-Path $env:LOCALAPPDATA 'FP-K2-Training\segmentasi-nrp8-venv\Scripts\python.exe'
$backend = 'C:\Users\junxi\OneDrive\Documents\Semester 5\Kompt Biomeds\FP\fp_rsna\eksperimen_model\03_colab\colab_project.py'
$jobs = @(
    @{Id='full-uniform'; Profile='04'; Sampling='uniform'; Ratio=4; Label='Semua kelas, sampling seragam'},
    @{Id='full-repeat'; Profile='05'; Sampling='repeat-factor'; Ratio=4; Label='Semua kelas, repeat factor'},
    @{Id='full-repeat-mask2'; Profile='06'; Sampling='repeat-factor'; Ratio=2; Label='Repeat factor dan mask lebih rapat'},
    @{Id='full-repeat-mask2-light'; Profile='01'; Sampling='repeat-factor'; Ratio=2; Lighting=$true; Label='Repeat factor, mask rapat dan HSV'},
    @{Id='hard-uniform'; Profile='07'; Sampling='uniform'; Ratio=4; Hard=$true; Label='Tiga kelas, sampling seragam'},
    @{Id='hard-foreground'; Profile='08'; Sampling='foreground'; Ratio=4; Hard=$true; Label='Tiga kelas, oversampling gambar positif'}
)
foreach ($job in $jobs) {
    if ($OnlyRun -and $job.Id -ne $OnlyRun) { continue }
    $profile = if ($ProfileOverride) { $ProfileOverride } else { $job.Profile }
    $runId = $job.Id + $RetrySuffix
    $launch = @(
        'start','train.py','--root',$PSScriptRoot,'--project','segmentasi-indoor-tuning',
        '--run',$runId,'--profile',$profile,'--accelerator','T4',
        '--title','Segmentasi indoor: perbaikan preprocessing',
        '--dataset','Panoptic Indoor Segmentation v5, train 397 / valid 166',
        '--model','YOLOv8n-seg','--method',$job.Label,'--epochs','32',
        '--metric','Mask AP50-95 validasi','--unit','%','--mode','max',
        '--requirements','requirements-yolo.txt','--include','metrics.py',
        '--include','data/panoptic-indoor-v5-640.zip','--output','output/result.zip',
        '--train-arg=--model','--train-arg=yolov8n','--train-arg=--epochs','--train-arg=32',
        '--train-arg=--sampling',('--train-arg=' + $job.Sampling),
        '--train-arg=--mask-ratio',('--train-arg=' + $job.Ratio),
        '--train-arg=--eval-splits','--train-arg=valid'
    )
    if ($job.Lighting) { $launch += '--train-arg=--lighting-augmentation' }
    if ($job.Hard) { $launch += @('--include','hard-classes.json','--train-arg=--classes','--train-arg=hard-classes.json') }
    if ($Check) { $launch += '--check' }
    if ((Test-Path -LiteralPath $stablePython) -and (Test-Path -LiteralPath $backend)) {
        $env:COLAB_CLI_REQUEST_TIMEOUT = '60'
        $env:COLAB_CLI_ASSIGN_RETRIES = '2'
        & $stablePython $backend @launch
    } else {
        & $launcher @launch
    }
    if ($LASTEXITCODE -ne 0) { throw "Gagal mendaftarkan run $($job.Id)." }
}
