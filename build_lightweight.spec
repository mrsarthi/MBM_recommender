# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

# EXCLUDE heavy libraries that pandas/scikit-learn might pull in but we don't actually need for inference
# This shrinks the final .exe size significantly
excludes = [
    'PyQt5', 'PySide2', 'tk', 'tcl', 'seaborn',
    'scipy.spatial', 'scipy.integrate', 'scipy.optimize', 'scipy.stats',
    'notebook', 'jupyter', 'IPython', 'bokeh', 'plotly'
]

a = Analysis(
    ['app.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('.env', '.'), # Ensure TMDB key is packaged
        ('models/', 'models/'), # Generic fallback models if needed
    ],
    hiddenimports=['requests_cache', 'requests_cache.backends.sqlite'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='Personal_AI_Recommender',
    debug=False,
    bootloader_ignore_signals=False,
    strip=True,  # Strip debugging symbols to save space
    upx=True,    # UPX compression (if installed on host)
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False, # Set to False to hide the CMD window and only show the GUI
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='app_icon.ico' # Optional: Add an icon if you have one
)
