# Configura o GStreamer no Windows para o video do NVDR (POST /video).
#
# Acha o GStreamer instalado; se nao achar, roda o instalador .exe (ou .msi)
# que estiver em Downloads. Depois poe a pasta bin no PATH do usuario, grava
# NVDR_GST_LAUNCH e confere se os elementos que o server.js usa existem.
#
# Na pasta do NVDR:
#   powershell -ExecutionPolicy Bypass -File scripts\setup_gstreamer.ps1
# ou cole o arquivo inteiro numa janela do PowerShell.
#
# Mensagens sem acento de proposito: o PowerShell 5 le .ps1 sem BOM como ANSI.

& {
    $ErrorActionPreference = 'Stop'

    function Find-GstBin {
        $roots = @()
        foreach ($scope in 'Process', 'User', 'Machine') {
            foreach ($name in 'GSTREAMER_1_0_ROOT_MSVC_X86_64', 'GSTREAMER_1_0_ROOT_MINGW_X86_64',
                              'GSTREAMER_1_0_ROOT_X86_64') {
                $v = [Environment]::GetEnvironmentVariable($name, $scope)
                if ($v) { $roots += $v }
            }
        }
        $roots += "$env:ProgramFiles\gstreamer\1.0\msvc_x86_64",
                  'C:\gstreamer\1.0\msvc_x86_64', 'C:\gstreamer\1.0\mingw_x86_64'
        foreach ($root in $roots) {
            $exe = "$($root.TrimEnd('\'))\bin\gst-launch-1.0.exe"
            if (Test-Path $exe) { return Split-Path $exe }
        }
        return $null
    }

    $bin = Find-GstBin
    if (-not $bin) {
        # O pacote "devel" so tem cabecalhos e bibliotecas; o que roda e o runtime.
        $installer = Get-ChildItem "$env:USERPROFILE\Downloads" -File -ErrorAction SilentlyContinue |
            Where-Object { $_.Name -match '^gstreamer-1\.0-.*\.(exe|msi)$' -and $_.Name -notmatch 'devel' } |
            Sort-Object LastWriteTime -Descending | Select-Object -First 1
        if (-not $installer) {
            Write-Host 'GStreamer nao instalado e nenhum instalador gstreamer-1.0-*.exe em Downloads.' -ForegroundColor Red
            Write-Host 'Baixe o "runtime installer" MSVC 64-bit em https://gstreamer.freedesktop.org/download/'
            return
        }
        Write-Host "Instalando $($installer.Name)..." -ForegroundColor Cyan
        if ($installer.Extension -eq '.msi') {
            # ADDLOCAL=ALL e a instalacao completa, que traz o gst-libav (H.264, HEVC).
            Start-Process msiexec.exe -ArgumentList '/i', "`"$($installer.FullName)`"", 'ADDLOCAL=ALL', '/passive' -Wait
        } else {
            Write-Host 'No instalador, escolha a instalacao COMPLETA (Complete/Full): sem ela nao vem o H.264.' -ForegroundColor Yellow
            Start-Process $installer.FullName -Wait
        }
        $bin = Find-GstBin
        if (-not $bin) {
            Write-Host 'A instalacao terminou mas o gst-launch-1.0.exe nao apareceu. Ela foi concluida?' -ForegroundColor Red
            return
        }
    }
    $gst = Join-Path $bin 'gst-launch-1.0.exe'
    Write-Host "GStreamer em $bin" -ForegroundColor Green

    $userPath = [Environment]::GetEnvironmentVariable('Path', 'User')
    if (-not $userPath) { $userPath = '' }
    if (($userPath -split ';') -notcontains $bin) {
        [Environment]::SetEnvironmentVariable('Path', (($userPath.TrimEnd(';'), $bin) -join ';').TrimStart(';'), 'User')
        Write-Host 'Pasta bin adicionada ao PATH do usuario.'
    }
    if (($env:Path -split ';') -notcontains $bin) { $env:Path = "$bin;$env:Path" }
    [Environment]::SetEnvironmentVariable('NVDR_GST_LAUNCH', $gst, 'User')
    $env:NVDR_GST_LAUNCH = $gst
    Write-Host "NVDR_GST_LAUNCH = $gst"

    & $gst --version | Select-Object -First 1

    # Os elementos dos pipelines do server.js; avdec_h264 vem do gst-libav.
    Write-Host 'Conferindo plugins (a primeira vez demora um pouco)...'
    $inspect = Join-Path $bin 'gst-inspect-1.0.exe'
    $missing = @()
    foreach ($el in 'filesrc', 'decodebin', 'videoconvert', 'videoflip', 'videoscale', 'identity',
                    'pngenc', 'multifilesink', 'avdec_h264') {
        & $inspect --exists $el
        if ($LASTEXITCODE -ne 0) { $missing += $el }
    }
    if ($missing) {
        Write-Host "Faltando: $($missing -join ', ')" -ForegroundColor Red
        Write-Host 'Rode o instalador de novo e escolha a instalacao completa.'
        return
    }

    $out = Join-Path ([IO.Path]::GetTempPath()) 'nvdr_gst_teste'
    New-Item -ItemType Directory -Force $out | Out-Null
    Push-Location $out
    try {
        & $gst -q videotestsrc num-buffers=3 ! videoconvert ! 'video/x-raw,format=RGB' ! pngenc ! multifilesink 'location=f%04d.png'
        $ok = $LASTEXITCODE -eq 0 -and (Get-ChildItem -Filter 'f*.png').Count -eq 3
    } finally {
        Pop-Location
        Remove-Item -Recurse -Force $out
    }
    if (-not $ok) {
        Write-Host 'O pipeline de teste falhou; veja a mensagem acima.' -ForegroundColor Red
        return
    }

    Write-Host ''
    Write-Host 'Pronto. Feche este terminal, abra um novo e rode: node server.js' -ForegroundColor Green
    Write-Host 'Na linha "video:" ao iniciar deve aparecer o caminho do gst-launch-1.0.exe.'
}
