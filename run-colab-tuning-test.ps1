param(
    [switch]$Check,
    [ValidateSet('full','three','full-target')]
    [string]$OnlyScope,
    [ValidatePattern('^0[1-9]$')]
    [string]$ProfileOverride,
    [ValidatePattern('^-retry[0-9]+$')]
    [string]$RetrySuffix
)
$ErrorActionPreference = 'Stop'
if (($ProfileOverride -or $RetrySuffix) -and !$OnlyScope) {
    throw 'Pilih satu kasus evaluasi bila mengganti profil atau run ID.'
}
Set-Location -LiteralPath $PSScriptRoot
@"
import hashlib, json, shutil
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

output = Path("output/tuning")
ranking = json.loads((output / "ringkasan-validasi.json").read_text())
if ranking["incomplete"] or set(ranking["best_by_scope"]) != {"full", "three"}:
    raise ValueError("Select both winners after all validation runs are complete.")
dataset = Path("data/panoptic-indoor-v5-640.zip")
with dataset.open("rb") as stream:
    sha = hashlib.file_digest(stream, "sha256").hexdigest()
test = output / "test-v5.zip"
with ZipFile(dataset) as source, ZipFile(test.with_suffix(".part"), "w", ZIP_DEFLATED) as archive:
    manifest = json.loads(source.read("manifest.json"))
    manifest["splits"] = {"test": manifest["splits"]["test"]}
    manifest["parent_archive_sha256"] = sha
    archive.writestr("manifest.json", json.dumps(manifest))
    for name in source.namelist():
        if name.startswith("test/"):
            archive.writestr(name, source.read(name))
test.with_suffix(".part").replace(test)
choices = dict(ranking["best_by_scope"])
choices["full-target"] = max(
    (row for row in ranking["completed"] if row["scope"] == "full"),
    key=lambda row: row["target_ap"],
)
for scope, record in choices.items():
    model_path = output / "winners" / scope / "checkpoint.pt"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(output / "runs" / record["output_run"] / "result.zip") as archive:
        training = json.loads(archive.read("summary.json"))
        if training["dataset_sha256"] != sha:
            raise ValueError("Winner dataset does not match test archive provenance.")
        with archive.open("checkpoint.pt") as source, model_path.open("wb") as target:
            shutil.copyfileobj(source, target)
    print(scope, record["output_run"], model_path)
"@ | python -
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
$stablePython = Join-Path $env:LOCALAPPDATA 'FP-K2-Training\segmentasi-nrp8-venv\Scripts\python.exe'
$backend = 'C:\Users\junxi\OneDrive\Documents\Semester 5\Kompt Biomeds\FP\fp_rsna\eksperimen_model\03_colab\colab_project.py'
$launcher = 'C:\Users\junxi\.agents\skills\colab\scripts\colab.ps1'
$profiles = @{full='02';three='09';'full-target'='03'}
foreach ($scope in @('full','three','full-target')) {
    if ($OnlyScope -and $scope -ne $OnlyScope) { continue }
    $profile = if ($ProfileOverride) { $ProfileOverride } else { $profiles[$scope] }
    $checkpoint = "output/tuning/winners/$scope/checkpoint.pt"
    $testDataset = 'output/tuning/test-v5.zip'
    $launch = @(
        'start','train.py','--root',$PSScriptRoot,'--project','segmentasi-indoor-tuning',
        '--run',("test-best-" + $scope + $RetrySuffix),'--profile',$profile,'--accelerator','T4',
        '--title','Segmentasi indoor: perbaikan preprocessing',
        '--dataset','Panoptic Indoor Segmentation v5, train 397 / valid 166',
        '--model','YOLOv8n-seg','--method','Evaluasi checkpoint terpilih pada test resmi',
        '--metric','Mask AP50-95 validasi','--unit','%','--mode','max',
        '--requirements','requirements-yolo.txt','--include','metrics.py',
        '--include',$checkpoint,'--include',$testDataset,'--output','output/result.zip',
        '--train-arg=--model','--train-arg=yolov8n',
        '--train-arg=--checkpoint',('--train-arg=' + $checkpoint),
        '--train-arg=--dataset',('--train-arg=' + $testDataset),
        '--train-arg=--eval-splits','--train-arg=test'
    )
    if ($scope -eq 'three') {
        $launch += @('--include','hard-classes.json','--train-arg=--classes','--train-arg=hard-classes.json')
    }
    if ($Check) { $launch += '--check' }
    if ((Test-Path -LiteralPath $stablePython) -and (Test-Path -LiteralPath $backend)) {
        $env:COLAB_CLI_REQUEST_TIMEOUT = '60'
        $env:COLAB_CLI_ASSIGN_RETRIES = '2'
        & $stablePython $backend @launch
    } else {
        & $launcher @launch
    }
    if ($LASTEXITCODE -ne 0) { throw "Gagal mendaftarkan evaluasi $scope." }
}
