# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_data_files
from PyInstaller.utils.hooks import collect_submodules

hidden_imports=[]
hidden_imports+= collect_submodules('llama_stack')
hidden_imports+= collect_submodules('llama_stack.providers.registry')
hidden_imports+= collect_submodules('llama_stack.providers.registry.*')
hidden_imports+= collect_submodules('llama_stack_client')
hidden_imports+= collect_submodules('llama_models')

datas = []
datas += collect_data_files('safehttpx')
datas += collect_data_files('llama_stack')
datas += collect_data_files('llama_stack',subdir='providers',include_py_files=True)
datas += collect_data_files('llama_models')
datas += collect_data_files('llama_stack_client')
datas += collect_data_files('blobfile')
datas += collect_data_files('sqlite-vec')
datas += [('/opt/homebrew/anaconda3/envs/blank/lib/python3.10/site-packages/customtkinter', 'customtkinter/')]



a = Analysis(
    ['app.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='MacQA',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='MacQA',
)
app = BUNDLE(coll,
             name='MacQA.app',
             icon='MacQA.icns',
             bundle_identifier=None)
