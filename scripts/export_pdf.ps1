param([string]$Docx = (Join-Path $PSScriptRoot '..\research\SarcasTone-paper.docx'))
$ErrorActionPreference = 'Stop'
$inputPath = (Resolve-Path -LiteralPath $Docx).Path
$pdfPath = [System.IO.Path]::ChangeExtension($inputPath, '.pdf')
$word = New-Object -ComObject Word.Application
$word.Visible = $false
$word.DisplayAlerts = 0
try {
    $document = $word.Documents.Open($inputPath, $false, $true)
    try { $document.ExportAsFixedFormat($pdfPath,17) } finally { $document.Close(0) }
} finally { $word.Quit() }
Write-Output $pdfPath
