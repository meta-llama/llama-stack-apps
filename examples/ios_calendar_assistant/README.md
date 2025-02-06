# iOSCalendarAssistant

iOSCalendarAssistant is a demo app ([video](https://drive.google.com/file/d/1xjdYVm3zDnlxZGi40X_D4IgvmASfG5QZ/view?usp=sharing)) that uses Llama Stack Swift SDK's remote inference and agent APIs to take a meeting transcript, summarizes it, extracts action items, and calls tools to book any followup meetings.

You can also test the create calendar event with a direct ask instead of a detailed meeting note.

## Installation

We recommend you try the [iOS Quick Demo](../ios_quick_demo) first to confirm the prerequisite and installation - both demos have the same prerequisite and the first two installation steps.

The quickest way to try out the demo for remote inference is using Together.ai's Llama Stack distro at https://llama-stack.together.ai - you can skip the next section and go to the Build and Run the iOS demo section directly.

## (Optional) Build and Run Own Llama Stack Distro

You need to set up a remote Llama Stack distributions to run this demo. Assuming you have a [Fireworks](https://fireworks.ai/account/api-keys) or [Together](https://api.together.ai/) API key, which you can get easily by clicking the link above:

```
conda create -n llama-stack python=3.10
conda activate llama-stack
pip install --no-cache llama-stack==0.1.0 llama-models==0.1.0 llama-stack-client==0.1.0
```

Then, either:
```
PYPI_VERSION=0.1.0 llama stack build --template fireworks --image-type conda
export FIREWORKS_API_KEY="<your_fireworks_api_key>"
llama stack run fireworks
```
or
```
PYPI_VERSION=0.1.0 llama stack build --template together --image-type conda
export TOGETHER_API_KEY="<your_together_api_key>"
llama stack run together
```

The default port is 5000 for `llama stack run` and you can specify a different port by adding `--port <your_port>` to the end of `llama stack run fireworks|together`.

## Build and Run the iOS demo

1. Double click `ios_calendar_assistant/iOSCalendarAssistant.xcodeproj` to open it in Xcode.

2. Under the iOSCalendarAssistant project - Package Dependencies, click the + sign, then add `https://github.com/meta-llama/llama-stack-client-swift` at the top right and 0.1.0 in the Dependency Rule, then click Add Package.

3. (Optional) Replace the `RemoteInference` url string in `ContentView.swift` below with the host IP and port of the remote Llama Stack distro  in Build and Run Own Llama Stack Distro:

```
private let agent = RemoteAgents(url: URL(string: "https://llama-stack.together.ai")!)
```

**Note:** In order for the app to access the remote URL, the app's `Info.plist` needs to have the entry `App Transport Security Settings` with `Allow Arbitrary Loads` set to YES.

Also, to allow the app to add event to the Calendar app, the `Info.plist` needs to have an entry `Privacy - Calendars Usage Description` and when running the app for the first time, you need to accept the Calendar access request.

4. Build the run the app on an iOS simulator or your device. First you may try a simple request:

```
Create a calendar event with a meeting title as Llama Stack update for 2-3pm February 3, 2025.
```

Then, a detailed meeting note:
```
Date: February 4, 2025
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
Sarah: Yes, let’s make that a priority. Anything else? No? Great. Thanks, everyone. Let’s meet again next week from 4-5pm on February 11, 2025 to review progress.
```

You'll see a summary, action items and a Calendar event created, made possible by Llama Stack's custom tool calling API support and Llama 3.1's tool calling capability.


# iOSCalendarAssistantWithLocalInf

iOSCalendarAssistantWithLocalInf is a demo app that uses Llama Stack Swift SDK's local inference and agent APIs and ExecuTorch to run local inference on device.

1. On a Mac terminal, in your top level directory, run commands:
```
git clone https://github.com/meta-llama/llama-stack-apps
cd llama-stack-apps
git submodule update --init --recursive
```

2. Go back to your top level directory, run commands:

```
git clone https://github.com/meta-llama/llama-stack
cd llama-stack
git submodule update --init --recursive
```

3. Double click `llama-stack-apps/examples/ios_calendar_assistant/iOSCalendarAssistantWithLocalInf.xcodeproj` to open it in Xcode.

4. In the `iOSCalendarAssistantWithLocalInf` project panel, remove `LocalInferenceImpl.xcodeproj` and drag and drop `LocalInferenceImpl.xcodeproj` from `llama-stack/llama_stack/providers/inline/ios/inference` into the `iOSCalendarAssistantWithLocalInf` project.

5. Prepare a Llama model file named `llama3_2_spinquant_oct23.pte` by following the steps [here](https://github.com/pytorch/executorch/blob/main/examples/models/llama/README.md#step-2-prepare-model) - you'll also download the `tokenizer.model` file there. Then drag and drop both files to the project `iOSCalendarAssistantWithLocalInf`.

6. Build and run the app on an iOS simulator or a real device.

**Note** If you see a build error about cmake not found, you can install cmake by following the instruction [here](https://github.com/pytorch/executorch/blob/main/examples/demo-apps/apple_ios/LLaMA/docs/delegates/xnnpack_README.md#1-install-cmake).
