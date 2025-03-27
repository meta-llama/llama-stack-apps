## Computer use agent
**WARNING**: The agent is still under development and is not ready for production use. It will use your computer and may cause unexpected damage. Please monitor the agent while using the app and use it at your own risk.
### Prerequisites:
1. `pip install -r requirements.txt`
2. OmniParser github and model checkpoints are required to be downloaded. See the [OmniParser Setup](#omniparser-setup) section below.
### OmniParser Setup
`git clone git@github.com:microsoft/OmniParser.git`

Navigate to the `OmniParser` directory

```bash
cd OmniParser
```

Download the model checkpoints:

```bash
# Download the model checkpoints to the local directory OmniParser/weights/
mkdir -p weights/icon_detect weights/icon_caption_florence

for file in icon_detect/{train_args.yaml,model.pt,model.yaml} \
            icon_caption/{config.json,generation_config.json,model.safetensors}; do
    huggingface-cli download microsoft/OmniParser-v2.0 "$file" --local-dir weights
done

mv weights/icon_caption weights/icon_caption_florence
```

make sure the weights are downloaded in the `weights` directory and it should be called `icon_detect` and `icon_caption_florence` respectively.

To start the gradio api of `omniparser`, run the following command:

```bash
python gradio_demo.py
```

The gradio api will start at `localhost:<port>` and live sharaing link will be generated. Copy the link and paste it in the `OMNIPARSER_API_URL` in the `utils.py` file.

### Run the agent
Please read [this doc](https://llama-stack.readthedocs.io/en/latest/distributions/index.html) and launch a local llama-stack server at `localhost:8321`.


change the `user_query` in the `app.py` file to your query and run the following command:

```bash
python app.py
```
