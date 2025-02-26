# iOSQuickDemo

iOSQuickDemo is a demo app ([video](https://drive.google.com/file/d/1HnME3VmsYlyeFgsIOMlxZy5c8S2xP4r4/view?usp=sharing)) that shows how to use the Llama Stack Swift SDK ([repo](https://github.com/meta-llama/llama-stack-client-swift)) and its `ChatCompletionRequest` API with a remote Llama Stack server to perform remote inference with Llama 3.1.

## Installation

The quickest way to try out the demo for remote inference is using Together.ai's Llama Stack distro at https://llama-stack.together.ai - you can skip the next section and go to the Build and Run the iOS demo section directly.

## (Optional) Build and Run Own Llama Stack Distro

You need to set up a remote Llama Stack distributions to run this demo. Assuming you have a [Fireworks](https://fireworks.ai/account/api-keys) or [Together](https://api.together.ai/) API key, which you can get easily by clicking the link above:

```
conda create -n llama-stack python=3.10
conda activate llama-stack
pip install --no-cache llama-stack==0.1.4 llama-models==0.1.4 llama-stack-client==0.1.4
```

Then, either:
```
PYPI_VERSION=0.1.4 llama stack build --template fireworks --image-type conda
export FIREWORKS_API_KEY="<your_fireworks_api_key>"
llama stack run fireworks
```
or
```
PYPI_VERSION=0.1.4 llama stack build --template together --image-type conda
export TOGETHER_API_KEY="<your_together_api_key>"
llama stack run together
```

The default port is 5000 for `llama stack run` and you can specify a different port by adding `--port <your_port>` to the end of `llama stack run fireworks|together`.

## Build and Run the iOS demo

1. Double click `iOSQuickDemo/iOSQuickDemo.xcodeproj` to open it in Xcode.

2. Under the iOSQuickDemo project - Package Dependencies, click the + sign, then add `https://github.com/meta-llama/llama-stack-client-swift` at the top right, set "Dependency Rule" to "Up to Next Major Version" and 0.1.4, then click Add Package.

3. (Optional) Replace the `RemoteInference` url string in `ContentView.swift` below with the host IP and port of the remote Llama Stack distro in Build and Run Own Llama Stack Distro:

```
let inference = RemoteInference(url: URL(string: "https://llama-stack.together.ai")!)
```

**Note:** In order for the app to access the remote URL, the app's `Info.plist` needs to have the entry `App Transport Security Settings` with `Allow Arbitrary Loads` set to YES.

4. Build the run the app on an iOS simulator or your device. Then click the Inference button, optionally after entering your own Question, to see the Llama answer. See the demo video [here](https://drive.google.com/file/d/1HnME3VmsYlyeFgsIOMlxZy5c8S2xP4r4/view?usp=sharing).


## Implementation Note

The Llama Stack `chatCompletion` API is used for the inference. Its paramater `request` requires three parameters: a list of messages, the model id, and the stream setting. A `UserMessage`'s `content` contains the user text input inside `TextContentItem`.

Inside the async return of the `chatCompletion`, each returned text chunk is appended to the message as the answer to the user input question.

```swift
for await chunk in try await inference.chatCompletion(
    request:
        Components.Schemas.ChatCompletionRequest(
        model_id: "meta-llama/Llama-3.1-8B-Instruct",
        messages: [
            .user(
            Components.Schemas.UserMessage(
                role: .user,
                content:
                    .InterleavedContentItem(
                        .text(Components.Schemas.TextContentItem(
                            _type: .text,
                            text: userInput
                        )
                    )
                )
            )
        )
        ],
        stream: true)
    ) {
        switch (chunk.event.delta) {
        case .text(let s):
            message += s.text
            break
        case .image(let s):
            print("> \(s)")
            break
        case .tool_call(let s):
            print("> \(s)")
            break
        }
    }
```

For a more advanced demo using the Llama Stack Agent API and custom tool calling feature, see the [iOS Calendar Assistant demo](https://github.com/meta-llama/llama-stack-apps/tree/main/examples/ios_calendar_assistant).
