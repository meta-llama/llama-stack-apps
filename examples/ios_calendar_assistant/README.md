# iOSCalendarAssistant

iOSCalendarAssistant is a demo app ([video](https://drive.google.com/file/d/1xjdYVm3zDnlxZGi40X_D4IgvmASfG5QZ/view?usp=sharing)) that takes a meeting transcript, summarizes it, extracts action items, and calls tools to book any followup meetings.

You can also test the create calendar event with a direct ask instead of a detailed meeting note.

We also have a demo project for running on-device inference. Checkout the instructions in the section below.

# Installation

We recommend you try the [iOS Quick Demo](../ios_quick_demo) first to confirm the prerequisite and installation - both demos have the same prerequisite and the first two installation steps.

## Prerequisite

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

3. Replace the `RemoteInference` url string in `ContentView.swift` below with the host IP and port of the remote Llama Stack distro started in Prerequisite:

```
private let agent = RemoteAgents(url: URL(string: "http://127.0.0.1:5000")!)
```
**Note:** In order for the app to access the remote URL, the app's `Info.plist` needs to have the entry `App Transport Security Settings` with `Allow Arbitrary Loads` set to YES.

Also, to allow the app to add event to the Calendar app, the `Info.plist` needs to have an entry `Privacy - Calendars Usage Description` and when running the app for the first time, you need to accept the Calendar access request.

4. Build the run the app on an iOS simulator or your device. First you may try a simple request:

```
Create a calendar event with a meeting title as Llama Stack update for 2-3pm January 27, 2025.
```

Then, a detailed meeting note:
```
Date: January 20, 2025
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
Sarah: Yes, let’s make that a priority. Anything else? No? Great. Thanks, everyone. Let’s meet again next week from 4-5pm on January 27, 2025 to review progress.
```

You'll see a summary, action items and a Calendar event created, made possible by Llama Stack's custom tool calling API support and Llama 3.1's tool calling capability.


# iOSCalendarAssistantWithLocalInf
This project shows you how to run local inference on-device using ExecuTorch in conjunction with Llama Stack Swift SDK. 

1. git clone `https://github.com/meta-llama/llama-stack-apps/tree/main/examples/ios_calendar_assistant`

2. Double click `ios_calendar_assistant/iOSCalendarAssistantWithLocalInf.xcodeproj` to open it in Xcode.

3. If there are already Frameworks in the General section of the TARGETS, remove them.

4. In Package Dependencies, delete all dependencies there and clean the dependencies cache.

5. In Package Dependencies, click the + sign, then add `https://github.com/meta-llama/llama-stack-client-swift`. Select Branch and input `v0.1.0`. This should resolve the package and add necessary dependencies in your project panel. (This should add a LlamaStackClient in your Frameworks)

6. In the same place, add `https://github.com/pytorch/executorch`. Select Branch and input `latest`. This will add ExecuTorch as your dependencies.

7. In the Frameworks for TARGETS, add all ExecuTorch kernels (including debug ones), but not `executorch` one. For example:  
```
backend_coreml
backend_mps
backend_xnnpack
kernels_custom
kernels_optimized
kernels_portable
kernels_quantized
```

8. In your project panel, if there is already a xcode project called `LocalInferenceImpl.xcodeproj`, remove it completely.

9. Then git clone `https://github.com/meta-llama/llama-stack/tree/adecb2a2d3bc5b5fb12280c54096706974e58201/llama_stack/providers/impls/ios/inference/LocalInferenceImpl`

10. In the repo, run `git submodule update --init --recursive` to sync the executorch submodules.

11. Install [Cmake](https://cmake.org/) for the executorch build. Additional [guidance](https://github.com/pytorch/executorch/blob/main/examples/demo-apps/apple_ios/LLaMA/docs/delegates/xnnpack_README.md#1-install-cmake) to install and link cmake

12. Drag `LocalInferenceImpl.xcodeproj` into your `iOSCalendarAssistantWithLocalInf` project. Import it as a reference

13. In LocalInferenceImpl’s Package Dependencies, change `LlamaStackClient package` version to `v0.1.0` matching iOSCalendarAssistantWithLocalInf’s package version. This is important to resolve Stencil dependencies.

14. Add LocalInferenceImpl.framework into the Framework section for TARGETS.

15. In "Build Settings" > "Other Linker Flags" > For both Debug and Release > "Any iOS Simulator SDK", add:
```
-force_load
$(BUILT_PRODUCTS_DIR)/libkernels_optimized-simulator-release.a
-force_load
$(BUILT_PRODUCTS_DIR)/libkernels_custom-simulator-release.a
-force_load
$(BUILT_PRODUCTS_DIR)/libkernels_quantized-simulator-release.a
-force_load
$(BUILT_PRODUCTS_DIR)/libbackend_xnnpack-simulator-release.a
-force_load
$(BUILT_PRODUCTS_DIR)/libbackend_coreml-simulator-release.a
-force_load
$(BUILT_PRODUCTS_DIR)/libbackend_mps-simulator-release.a
```

16. For "Any iOS SDK", add:
```
-force_load
$(BUILT_PRODUCTS_DIR)/libkernels_optimized-ios-release.a
-force_load
$(BUILT_PRODUCTS_DIR)/libkernels_custom-ios-release.a
-force_load
$(BUILT_PRODUCTS_DIR)/libkernels_quantized-ios-release.a
-force_load
$(BUILT_PRODUCTS_DIR)/libbackend_xnnpack-ios-release.a
-force_load
$(BUILT_PRODUCTS_DIR)/libbackend_coreml-ios-release.a
-force_load
$(BUILT_PRODUCTS_DIR)/libbackend_mps-ios-release.a
```

17. Lastly prepare the model: prepare a .pte file following the executorch [docs](https://github.com/pytorch/executorch/blob/main/examples/models/llama/README.md#step-2-prepare-model). Bundle the .pte and tokenizer.model file into Build Phases -> Copy Bundle Resources

18. Build the app for simulator or real device
