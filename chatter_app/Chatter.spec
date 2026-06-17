# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the Chatter Kivy desktop app.

Build (run from the ``chatter_app/`` directory):

    pyinstaller Chatter.spec --noconfirm

- Windows  -> a single-file ``dist/Chatter.exe`` (onefile).
- macOS    -> ``dist/Chatter.app`` (onedir bundle), zipped for release via
              ``ditto`` in CI.

Icons (``Chatter.ico`` / ``Chatter.icns``) are generated from
``assets/zebrafinch.png`` by the CI workflow before this spec runs. If they
are missing (e.g. a local build), the spec falls back to no custom icon.
"""

import os
import sys

from kivy.tools.packaging.pyinstaller_hooks import (
    get_deps_minimal,
    hookspath,
    runtime_hooks,
)
from PyInstaller.utils.hooks import collect_all

IS_WIN = sys.platform.startswith('win')
IS_MAC = sys.platform == 'darwin'

# kivy_deps.sdl2 / kivy_deps.glew ship the SDL2/GLEW DLLs and exist only on
# Windows. Import them lazily so the spec still loads on macOS/Linux.
if IS_WIN:
    from kivy_deps import sdl2, glew

# ---------------------------------------------------------------------------
# Resources + native libs that PyInstaller tends to miss
# ---------------------------------------------------------------------------
datas = [('assets', 'assets')]
binaries = []
hiddenimports = [
    # matplotlib renders off-thread with the Agg backend -> Kivy Texture
    'matplotlib.backends.backend_agg',
    # scipy submodules used by the detection pipeline
    'scipy.signal',
    'scipy.ndimage',
    # scikit-learn pieces used for cosine-distance outlier flagging
    'sklearn.metrics.pairwise',
    'sklearn.utils._typedefs',
    'sklearn.utils._heap',
    'sklearn.utils._sorting',
    'sklearn.utils._vector_sentinel',
    # audio + data layer
    'soundfile',
    'duckdb',
    # flat app modules added to sys.path at runtime (PyInstaller can't follow
    # the runtime sys.path inserts in main.py)
    'audio_utils',
    'chatter_controller',
    'chatter_store',
    'chatter_screen',
    'welcome_screen',
    'bout_list',
    'param_input',
    'spectrogram_view',
]

# librosa lazy-imports submodules (and pulls numba); grab everything to be safe.
for pkg in ('librosa',):
    _datas, _binaries, _hidden = collect_all(pkg)
    datas += _datas
    binaries += _binaries
    hiddenimports += _hidden

# soundfile's libsndfile shared library is bundled automatically by
# PyInstaller's contrib hook (hook-soundfile.py); no manual step needed.

excludes = [
    # optional / disabled features
    'tensorflow',
    'birdnetlib',
    # notebook + legacy stacks (not used by the Kivy app)
    'jupyterlab',
    'ipywidgets',
    'ipympl',
    'notebook',
    'IPython',
    # GUI toolkits we don't use (avoid matplotlib dragging in a Tk/Qt backend)
    'tkinter',
    'PyQt5',
    'PyQt6',
    'PySide2',
    'PySide6',
]

# get_deps_minimal() returns a dict that already contains 'binaries', 'datas',
# 'hiddenimports' and 'excludes' (the Kivy core providers it wants in/out). Merge
# those into our own lists rather than passing the dict via **kwargs, which would
# collide with the explicit arguments below.
_kivy_deps = get_deps_minimal(video=None, audio=None)
binaries      += _kivy_deps.pop('binaries', [])
datas         += _kivy_deps.pop('datas', [])
hiddenimports += _kivy_deps.pop('hiddenimports', [])
excludes      += _kivy_deps.pop('excludes', [])

a = Analysis(
    ['main.py'],
    pathex=['core', 'widgets', 'screens'],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=hookspath(),
    runtime_hooks=runtime_hooks(),
    excludes=excludes,
    noarchive=False,
    **_kivy_deps,
)

pyz = PYZ(a.pure, a.zipped_data)

# ---------------------------------------------------------------------------
# Per-platform packaging
# ---------------------------------------------------------------------------
if IS_WIN:
    _icon = 'Chatter.ico' if os.path.exists('Chatter.ico') else None
    # Onefile: fold binaries + datas into a single self-extracting EXE, and
    # bundle Kivy's SDL2/GLEW DLLs via the kivy_deps dependency trees.
    exe = EXE(
        pyz,
        a.scripts,
        a.binaries,
        a.zipfiles,
        a.datas,
        *[Tree(p) for p in (sdl2.dep_bins + glew.dep_bins)],
        name='Chatter',
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        upx=False,
        runtime_tmpdir=None,
        console=False,
        icon=_icon,
    )
else:
    # macOS (and any non-Windows): onedir EXE + COLLECT, wrapped in an .app.
    _icon = 'Chatter.icns' if os.path.exists('Chatter.icns') else None
    exe = EXE(
        pyz,
        a.scripts,
        [],
        exclude_binaries=True,
        name='Chatter',
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        upx=False,
        console=False,
        icon=_icon,
    )
    coll = COLLECT(
        exe,
        a.binaries,
        a.zipfiles,
        a.datas,
        strip=False,
        upx=False,
        name='Chatter',
    )
    if IS_MAC:
        app = BUNDLE(
            coll,
            name='Chatter.app',
            icon=_icon,
            bundle_identifier='org.chatter.app',
            info_plist={
                'CFBundleName': 'Chatter',
                'CFBundleDisplayName': 'Chatter',
                'NSHighResolutionCapable': True,
            },
        )
