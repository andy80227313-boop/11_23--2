```powershell
$ErrorActionPreference = 'Stop'
$Tool = Join-Path $Bundle '날짜 작업.ps1'
$JobArgs = @{
    Repo = $Repo
    Origin = 'https://github.com/HighKcal/capture-masking.git'
    Branch = 'work/masking'
    Day = '11-23'
    Files = @(
        'experiments/ocr-coordinate-refinement-v0/coord_common.py'
        'experiments/ocr-coordinate-refinement-v0/detail_adapter.py'
    )
    Message = '글자 위치와 원본 좌표 대응 코드 이관'
}
& $Tool @JobArgs -Step Prepare
```

```powershell
& $Tool @JobArgs -Step Stage
```

```powershell
& $Tool @JobArgs -Step Publish
```
