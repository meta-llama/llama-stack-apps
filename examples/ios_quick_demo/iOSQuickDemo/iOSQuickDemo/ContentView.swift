/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import SwiftUI
import LlamaStackClient

struct ContentView: View {
  @State private var message: String = ""
  @State private var userInput: String = "Best quotes in Godfather"

  private let runnerQueue = DispatchQueue(label: "org.llamastack.iosquickdemo")

  var body: some View {
    VStack(spacing: 20) {
      Text(message.isEmpty ? "Click Inference to see Llama's answer" : message)
          .font(.headline)
          .foregroundColor(.blue)
          .padding()
          .frame(maxWidth: .infinity)
          .background(Color.gray.opacity(0.2))
          .cornerRadius(8)

      VStack(alignment: .leading, spacing: 10) {
        Text("Question")
            .font(.headline)

        TextField("Enter your question here", text: $userInput)
            .textFieldStyle(RoundedBorderTextFieldStyle())
            .padding()
      }

      Button(action: {
          handleButtonClick(buttonName: "Inference")
      }) {
          Text("Inference")
              .font(.title2)
              .foregroundColor(.white)
              .padding()
              .frame(maxWidth: .infinity)
              .background(Color.green)
              .cornerRadius(8)
      }

      Spacer()
    }
    .padding()
  }

  private func handleButtonClick(buttonName: String) {
    if userInput.isEmpty {
      message = "Please enter a question before clicking 'Inference'."
      return
    }

    message = ""

    let workItem = DispatchWorkItem {
      defer {
        DispatchQueue.main.async {
        }
      }

      Task {

        // replace the URL string if you build and run your own Llama Stack distro as shown in https://github.com/meta-llama/llama-stack-apps/tree/main/examples/ios_quick_demo#optional-build-and-run-own-llama-stack-distro
        let inference = RemoteInference(url: URL(string: "https://llama-stack.together.ai")!)

        do {
          for await chunk in try await inference.chatCompletion(
            request:
              Components.Schemas.ChatCompletionRequest(
                messages: [
                  .user(
                    Components.Schemas.UserMessage(
                      content:
                          .InterleavedContentItem(
                              .text(Components.Schemas.TextContentItem(
                                  text: userInput,
                                  _type: .text
                              )
                          )
                      ),
                      role: .user
                  )
                )
              ],
              model_id: "meta-llama/Llama-3.1-8B-Instruct",
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
        }
        catch {
          print("Error: \(error)")
        }
      }
    }

    runnerQueue.async(execute: workItem)
  }
}
