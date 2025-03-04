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

  let imageUrl = "https://raw.githubusercontent.com/meta-llama/llama-models/refs/heads/main/Llama_Repo.jpeg"
  
  private let runnerQueue = DispatchQueue(label: "org.llamastack.iosquickdemo")
  
  var body: some View {
    VStack(spacing: 20) {
      ScrollView {
        Text(message.isEmpty ? "Click a button to ask Llama" : message)
          .font(.headline)
          .foregroundColor(.blue)
          .padding()
          .frame(maxWidth: .infinity)
          .background(Color.gray.opacity(0.2))
          .cornerRadius(8)
      }
      .frame(maxHeight: 500)

      TextField("Your question or ask", text: $userInput)
          .textFieldStyle(RoundedBorderTextFieldStyle())
          .padding()
      
      AsyncImage(url: URL(string: imageUrl)) { phase in
        switch phase {
        case .empty:
            ProgressView()
        case .success(let image):
            image
                .resizable()
                .aspectRatio(contentMode: .fit)
        case .failure:
            Image(systemName: "llama")
                .resizable()
                .aspectRatio(contentMode: .fit)
        @unknown default:
            EmptyView()
        }
    }
    .frame(height: 200)
      HStack {
        Button(action: {
          handleButtonClick(buttonName: "Text")
        }) {
          Text("Text")
            .font(.title2)
            .foregroundColor(.white)
            .padding()
            .frame(maxWidth: .infinity)
            .background(Color.green)
            .cornerRadius(8)
        }
        
        Button(action: {
          handleButtonClick(buttonName: "Image")
        }) {
          Text("Image")
            .font(.title2)
            .foregroundColor(.white)
            .padding()
            .frame(maxWidth: .infinity)
            .background(Color.green)
            .cornerRadius(8)
        }
        
        Button(action: {
          handleButtonClick(buttonName: "TextImage")
        }) {
          Text("Text Image")
            .font(.title2)
            .foregroundColor(.white)
            .padding()
            .frame(maxWidth: .infinity)
            .background(Color.green)
            .cornerRadius(8)
        }
      }
      Spacer()
    }
    .padding()
  }
  
  private func userMessageWithText(_ text: String) -> Components.Schemas.UserMessage {
    return Components.Schemas.UserMessage(
      role: .user,
      content:
        .case1(text)
    )
  }
  
  private func userMessageToDescribeAnImage(_ imageURL: String) -> Components.Schemas.UserMessage {
    return Components.Schemas.UserMessage(
      role: .user,
      content:
        .InterleavedContentItem(
          .image(Components.Schemas.ImageContentItem(
            _type: .image,
            image: Components.Schemas.ImageContentItem.imagePayload( url: Components.Schemas.URL(uri: imageURL))
            )
          )
      )
    )
  }
  
  private func userMessageWithTextAndImage(_ imageURL: String, _ text: String) -> Components.Schemas.UserMessage {
    return Components.Schemas.UserMessage(
      role: .user,
      content:
        .case3([
          Components.Schemas.InterleavedContentItem.text(
            Components.Schemas.TextContentItem(
              _type: .text,
              text: text
            )
          ),
          Components.Schemas.InterleavedContentItem.image(
            Components.Schemas.ImageContentItem(
              _type: .image,
              image: Components.Schemas.ImageContentItem.imagePayload( url: Components.Schemas.URL(uri: imageURL))
              )
            )
          ])
        )
  }
  
  private func handleButtonClick(buttonName: String) {
    if (buttonName == "Text" || buttonName == "TextImage") {
      if userInput.isEmpty {
        message = "Please enter your question or ask first."
        return
      }
    }
    
    var userMessage : Components.Schemas.UserMessage!
    if (buttonName == "Text") {
      userMessage = userMessageWithText(userInput)
    }
    else if (buttonName == "Image") {
      userMessage = userMessageToDescribeAnImage(imageUrl)
    }
    else if (buttonName == "TextImage") {
      userMessage = userMessageWithTextAndImage(imageUrl, userInput)
    }
    
    UIApplication.shared.sendAction(#selector(UIResponder.resignFirstResponder), to: nil, from: nil, for: nil)
    
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
                model_id:
                  //"meta-llama/Llama-3.1-8B-Instruct", // text-only Llama model
                  "meta-llama/Llama-3.2-11B-Vision-Instruct", // image and text Llama model
                messages: [
                  .user(userMessage)
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
        }
        catch {
          print("Error: \(error)")
        }
      }
    }

    runnerQueue.async(execute: workItem)
  }
}
