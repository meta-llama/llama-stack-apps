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
datas += collect_data_files('gradio_client')
datas += collect_data_files('gradio')
datas += collect_data_files('safehttpx')
datas += collect_data_files('llama_stack')
datas += collect_data_files('llama_models')
datas += collect_data_files('llama_stack_client')
datas += collect_data_files('blobfile')
datas += ( '/Users/kaiwu/work/llama-stack/llama_stack/providers', 'llama_stack/providers' ),

a = Analysis(
    ['MacQA.py'],
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
    module_collection_mode={
        'gradio': 'py',  # Collect gradio package as source .py files
    }
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='MacQA',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
