## MacQA

This is an end-to-end Retrieval Augmented Generation (RAG) App leveraging llama-stack that handles the logic for ingesting documents, storing them in a vector database and providing an inference interface.


### Prerequisite:

**Install Ollama**: This app use ollama to run inference, please follow [ollama's download instruction](https://ollama.com/download) to install Ollama. Make sure there is a binary `/usr/local/bin/ollama`.

### How to run:

1. Open Ollama software.
2. Open the `MacQA.dmg` and move `MacQA.app` to Application folder to have it installed.
3. Double click `MacQA.app` in the Application folder.
4. Open `http://localhost:7861/`, then type the path of data folder and choose the model for the Ollama inference.
5. Wait for the setup to be ready and click `Chat` tab to start chating to this app.

### How to build the app (Optional):

1. Create a new python venv, eg. `conda create -n build_app python=3.10` and then `conda activate build_app` to use it.
2. Run `pip install -r requirements.txt` to install required pypi packages.
3. Run `python MacQA.py` make sure everything works.
4. UPX is a executable packer to reduce the size of our App, we need to download UPX zip corresponding to your machine platform from [UPX website](https://github.com/upx/upx/releases/) to this folder and unzip it.
5. Compile MacQA.py with correct upx path, eg. `pyinstaller --upx-dir ./upx-4.2.4-arm64_linux MacQA.spec`, the one-clickable app should be in `./dist/MacQA.app` (This step may take ~10 mins).
6. Optionally, you can move the MacQA.app to Application folder to have it locally installed.
7. Alternatively, if you want to create a .dmg file for easier distribution. You can follow those steps:

```
1. Copy ./dist/MacQA.app to a new folder.
2. In your Mac, search and open Disk Utility -> File -> New Image -> Image From Folder.
3. Select the folder where you have placed the App. Give a name for the DMG and save. This creates a distributable image for you.
```

