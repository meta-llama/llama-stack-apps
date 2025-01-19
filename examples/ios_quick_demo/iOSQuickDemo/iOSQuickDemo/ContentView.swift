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
        let inference = RemoteInference(url: URL(string: "http://54.189.109.3:8501")!)
        do {
          for await chunk in try await inference.chatCompletion(
            request:
              Components.Schemas.ChatCompletionRequest(
                messages: [
                  .UserMessage(Components.Schemas.UserMessage(
                    content: .case1(userInput),
                    role: .user)
                  )
                ], model_id: "meta-llama/Llama-3.1-8B-Instruct",
                stream: true)
          ) {
            switch (chunk.event.delta) {
            case .TextDelta(let s):
              message += s.text
              break
            case .ImageDelta(let s):
              print("> \(s)")
              break
            case .ToolCallDelta(let s):
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
