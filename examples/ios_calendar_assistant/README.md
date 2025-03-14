# iOSCalendarAssistant

iOSCalendarAssistant is a demo app ([video](https://drive.google.com/file/d/1xjdYVm3zDnlxZGi40X_D4IgvmASfG5QZ/view?usp=sharing)) that uses Llama Stack Swift SDK's remote inference and agent APIs to take a meeting transcript, summarizes it, extracts action items, and calls tools to book any followup meetings.

You can also test the create calendar event with a direct ask instead of a detailed meeting note.

## Installation

We also have a demo project for running on-device inference. Checkout the instructions in the section `iOSCalendarAssistantWithLocalInf` below.

We recommend you try the [iOS Quick Demo](../ios_quick_demo) first to confirm the prerequisite and installation - both demos have the same prerequisite and the first two installation steps.

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
PYPI_VERSION=0.1.4 llama stack build --template together --image-type conda
export TOGETHER_API_KEY="<your_together_api_key>"
llama stack run together
```
or
```
PYPI_VERSION=0.1.4 llama stack build --template fireworks --image-type conda
export FIREWORKS_API_KEY="<your_fireworks_api_key>"
llama stack run fireworks
```

The default port is 5000 for `llama stack run` and you can specify a different port by adding `--port <your_port>` to the end of `llama stack run fireworks|together`.

## Build and Run the iOS demo

1. Double click `ios_calendar_assistant/iOSCalendarAssistant.xcodeproj` to open it in Xcode.

2. Either replace "YOUR_TOGETHER_API_KEY" in `ContentView.swift` with your key (you can get a free trial key in seconds at https://api.together.ai):

```
private let agent = RemoteAgents(url: URL(string: "https://llama-stack.together.ai")!, apiKey: "YOUR_TOGETHER_API_KEY")
```

Or replace the line above with the host IP and port of the remote Llama Stack distro (e.g. http://localhost:5000) in Build and Run Own Llama Stack Distro:

```
let agent = RemoteAgents(url: URL(string: "https://localhost:5000")!)
```

**Note:** In order for the app to access the remote URL, the app's `Info.plist` needs to have the entry `App Transport Security Settings` with `Allow Arbitrary Loads` set to YES.

Also, to allow the app to add event to the Calendar app, the `Info.plist` needs to have an entry `Privacy - Calendars Usage Description` and when running the app for the first time, you need to accept the Calendar access request.

4. Build the run the app on an iOS simulator or your device. 

Note: For your first-time build, you may need to Enable and Trust the OpenAPI Generator plugin. A link to enable will be available in the logs. You may need to do a clean build, close Xcode and then restart it again to avoid any cache issues. Otherwise, you may see "Bad Access too URLSession" errors during inference.

5. Once the build is complete, you may try a simple request:

```
Create a calendar event with a meeting title as Llama Stack update for 2-3pm February 19, 2025.
```

Then, a detailed meeting note:
```
Date: February 20, 2025
Time: 10:00 AM - 11:00 AM
Location: Zoom
Attendees:
Sarah (Team Lead)
John (Marketing Manager)
Emily (Product Designer)
Mike (Developer)
Jane (Operations Manager)

Sarah: Good morning, everyone. Thanks for joining the meeting today. Let’s get started. Our main agenda is to review progress on the new product launch and address any blockers.
John: Morning, Sarah. I can kick things off with the marketing update. The campaign assets are 80% ready. We’re working on finalizing the social media calendar and ad creatives.
Sarah: Great progress, John. Do you foresee any risks to meeting the launch deadline?
John: The only concern is securing approvals for ad copy. It’s taking longer than expected.
Sarah: Understood. Let’s prioritize getting those approvals. Emily, can you ensure the design team supports that?
Emily: Absolutely. I’ll follow up with the team today.
Sarah: Perfect. Mike, how’s development coming along?
Mike: We’re on track with the MVP build. There’s one issue with the payment gateway integration, but we expect to resolve it by the end of the week.
Jane: Mike, will that delay testing?
Mike: No, we’re scheduling other tests around it to stay on schedule.
Sarah: Good. Jane, any updates from operations?
Jane: Yes, logistics are sorted, and we’ve confirmed the warehouse availability. The only pending item is training customer support for the new product.
Sarah: Let’s coordinate with the training team to expedite that. Anything else?
Mike: Quick note—can we get feedback on the beta version by Friday?
Sarah: Yes, let’s make that a priority. Anything else? No? Great. Thanks, everyone. Let’s meet again next week from 4-5pm on February 27, 2025 to review progress.
```

You'll see a summary, action items and a Calendar event created, made possible by Llama Stack's custom tool calling API support and Llama 3.1's tool calling capability.


# iOSCalendarAssistantWithLocalInf

iOSCalendarAssistantWithLocalInf is a demo app that uses Llama Stack Swift SDK's local inference and agent APIs and ExecuTorch to run local inference on device.

1. On a Mac terminal, in your top level directory, run commands:
```
git clone https://github.com/meta-llama/llama-stack-apps
git clone https://github.com/meta-llama/llama-stack-client-swift
cd llama-stack-client-swift
git submodule update --init --recursive
```

2. Open `llama-stack-apps/examples/ios_calendar_assistant/iOSCalendarAssistantWithLocalInf.xcodeproj` in Xcode.

3. In the `iOSCalendarAssistantWithLocalInf` project panel, remove `LocalInferenceImpl.xcodeproj` and drag and drop `LocalInferenceImpl.xcodeproj` from `llama-stack-client-swift/local_inference` into the project - in the "Choose options for adding these files" dialog, select "Reference files in place" for Action.

4. Prepare a Llama model file named `llama3_2_spinquant_oct23.pte` by following the steps [here](https://github.com/pytorch/executorch/blob/main/examples/models/llama/README.md#step-2-prepare-model) - you'll also download the `tokenizer.model` file there. Then remove the two missing files from the the project `iOSCalendarAssistantWithLocalInf`, and drag and drop both files to the project, also selecting "Reference files in place" for Action.

5. Build and run the app on an iOS simulator or a real device.

**Note** If you see a build error about cmake not found, you can install cmake by following the instruction [here](https://github.com/pytorch/executorch/blob/main/examples/demo-apps/apple_ios/LLaMA/docs/delegates/xnnpack_README.md#1-install-cmake).
