# wipe_and_reupload_with_headers.ps1
# Deletes everything in the bucket, then uploads LOCAL_DIR -> s3://BUCKET/product/ in batches by extension,
# setting Content-Type + Cache-Control on upload (faster than post-upload metadata copy).
#
# Usage (PowerShell):
#   $AccountId = "YOUR_ACCOUNT_ID"
#   $Bucket    = "tptcoding"
#   $LocalDir  = "D:\path\to\your\files"
#   $Prefix    = "product/"        # optional
#   $Profile   = "r2"              # optional
#   $DryRun    = $true             # set to $false to actually delete+upload
#   .\wipe_and_reupload_with_headers.ps1
#
# Notes:
# - This uses aws s3 sync with --exclude/--include per batch.
# - It assumes your local directory is the ROOT of your content (not already containing a top-level "product\" folder).
# - If your local dir already contains a "pdf\" subfolder you want to preserve as product/pdf/... that's fine.
# - If you need to preserve existing metadata (ETag etc.) that's not applicable; object storage keys are rewritten.

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# --------- REQUIRED: set these in your session before running ----------
if (-not $AccountId) { throw "Set `$AccountId first." }
if (-not $Bucket)    { throw "Set `$Bucket first." }
if (-not $LocalDir)  { throw "Set `$LocalDir first." }

# --------- OPTIONAL defaults ----------
if (-not $Prefix)  { $Prefix  = "product/" }
if (-not $Profile) { $Profile = "r2" }
if ($null -eq $DryRun) { $DryRun = $true }

if (-not (Test-Path -LiteralPath $LocalDir)) {
  throw "LocalDir not found: $LocalDir"
}

$Endpoint = "https://$AccountId.r2.cloudflarestorage.com"
$Dest     = "s3://$Bucket/$Prefix"

Write-Host "Endpoint : $Endpoint"
Write-Host "Bucket   : $Bucket"
Write-Host "LocalDir : $LocalDir"
Write-Host "Dest     : $Dest"
Write-Host "DryRun   : $DryRun"
Write-Host ""

function Invoke-Aws {
  param([Parameter(Mandatory=$true)][string[]]$Args)
  $cmd = "aws --profile $Profile --endpoint-url $Endpoint " + ($Args -join " ")
  if ($DryRun) {
    Write-Host "DRYRUN: $cmd"
    return
  }
  & "C:\Program Files\Amazon\AWSCLIV2\aws.exe" --profile $Profile --endpoint-url $Endpoint @Args
}

# 1) WIPE BUCKET
Write-Host "=== STEP 1: Wiping bucket (all objects) ==="
Invoke-Aws -Args @("s3", "rm", "s3://$Bucket/", "--recursive")
Write-Host ""

# 2) UPLOAD BATCHES
# Cache policy:
$cacheLong  = "public, max-age=31536000, immutable"
$cacheShort = "public, max-age=3600"

# Extension batches: add/remove as needed.
# Each batch is: Name, Includes[], ContentType, CacheControl
$batches = @(
  @{
    Name        = "PDF"
    Includes    = @("*.pdf")
    ContentType = "application/pdf"
    Cache       = $cacheLong
  },
  @{
    Name        = "PNG"
    Includes    = @("*.png")
    ContentType = "image/png"
    Cache       = $cacheLong
  },
  @{
    Name        = "JPG"
    Includes    = @("*.jpg","*.jpeg")
    ContentType = "image/jpeg"
    Cache       = $cacheLong
  },
  @{
    Name        = "PPTX"
    Includes    = @("*.pptx")
    ContentType = "application/vnd.openxmlformats-officedocument.presentationml.presentation"
    Cache       = $cacheLong
  },
  @{
    Name        = "PPSX"
    Includes    = @("*.ppsx")
    ContentType = "application/vnd.openxmlformats-officedocument.presentationml.slideshow"
    Cache       = $cacheLong
  },
  @{
    Name        = "PPT"
    Includes    = @("*.ppt")
    ContentType = "application/vnd.ms-powerpoint"
    Cache       = $cacheLong
  },
  @{
    Name        = "DOCX"
    Includes    = @("*.docx")
    ContentType = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    Cache       = $cacheLong
  },
  @{
    Name        = "DOCM"
    Includes    = @("*.docm")
    ContentType = "application/vnd.ms-word.document.macroEnabled.12"
    Cache       = $cacheLong
  },
  @{
    Name        = "DOC"
    Includes    = @("*.doc")
    ContentType = "application/msword"
    Cache       = $cacheLong
  },
  @{
    Name        = "ZIP"
    Includes    = @("*.zip")
    ContentType = "application/zip"
    Cache       = $cacheLong
  },
  @{
    Name        = "MP4"
    Includes    = @("*.mp4")
    ContentType = "video/mp4"
    Cache       = $cacheLong
  },
  @{
    Name        = "MOV"
    Includes    = @("*.mov")
    ContentType = "video/quicktime"
    Cache       = $cacheLong
  },
  @{
    Name        = "MP3"
    Includes    = @("*.mp3")
    ContentType = "audio/mpeg"
    Cache       = $cacheLong
  },
  @{
    Name        = "TTF"
    Includes    = @("*.ttf")
    ContentType = "font/ttf"
    Cache       = $cacheLong
  },
  @{
    Name        = "PUB"
    Includes    = @("*.pub")
    ContentType = "application/x-mspublisher"
    Cache       = $cacheLong
  },
  @{
    Name        = "KEY (Apple Keynote - treat as octet-stream)"
    Includes    = @("*.key")
    ContentType = "application/octet-stream"
    Cache       = $cacheLong
  },
  @{
    Name        = "INI (short cache)"
    Includes    = @("*.ini")
    ContentType = "text/plain"
    Cache       = $cacheShort
  },
  @{
    Name        = "DB (short cache)"
    Includes    = @("*.db")
    ContentType = "application/octet-stream"
    Cache       = $cacheShort
  },
  @{
    Name        = "BRIDGESORT (short cache)"
    Includes    = @("*.bridgesort")
    ContentType = "application/octet-stream"
    Cache       = $cacheShort
  }
)

Write-Host "=== STEP 2: Uploading batches to $Dest ==="

# Helper: build --exclude/--include args
function Build-IncludeArgs {
  param([string[]]$Includes)
  $args = @("--exclude","*")
  foreach ($pat in $Includes) { $args += @("--include",$pat) }
  return $args
}

foreach ($b in $batches) {
  $name = $b.Name
  $inc  = $b.Includes
  $ct   = $b.ContentType
  $cc   = $b.Cache

  # Quick local count so you can see progress expectations
  $count = 0
  foreach ($pat in $inc) {
    $count += (Get-ChildItem -LiteralPath $LocalDir -Recurse -File -Filter $pat -ErrorAction SilentlyContinue | Measure-Object).Count
  }

  Write-Host ""
  Write-Host "--- Batch: $name | Files: $count | CT=$ct | Cache-Control=$cc ---"

  if ($count -eq 0) { continue }

  $includeArgs = Build-IncludeArgs -Includes $inc

  # aws s3 sync LOCAL -> DEST
  $syncArgs = @(
    "s3","sync",
    $LocalDir, $Dest,
    "--exact-timestamps",
    "--content-type", $ct,
    "--cache-control", $cc
  ) + $includeArgs

  Invoke-Aws -Args $syncArgs
}

Write-Host ""
Write-Host "=== STEP 3: Verify sample listing ==="
Invoke-Aws -Args @("s3","ls",$Dest,"--recursive")
Write-Host ""
Write-Host "DONE."
