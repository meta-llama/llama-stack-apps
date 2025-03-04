/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import SwiftUI
import UniformTypeIdentifiers

import LlamaStackClient
import LocalInferenceImpl

import EventKit
import EventKitUI

struct ContentView: View {
  @State private var prompt = ""
  @State private var messages: [Message] = []
  @State private var isGenerating = false
  private let runnerQueue = DispatchQueue(label: "org.llamastack.stacksummary")

  @State var eventStore = EKEventStore()
  @State var presetEvent: EKEvent?

  @State var isShowingEventModal = false

  private let inference: LocalInference
  private let localAgents: LocalAgents
  private let remoteAgents: RemoteAgents
  @State private var isUsingLocalAgents = true

  @State var agentId = ""
  @State var agenticSystemSessionId = ""

  @State private var actionItems = ""

  public init () {
    self.inference = LocalInference(queue: runnerQueue)
    self.localAgents = LocalAgents(inference: self.inference)
    
    // replace the URL string if you build and run your own Llama Stack distro as shown in https://github.com/meta-llama/llama-stack-apps/tree/main/examples/ios_calendar_assistant#optional-build-and-run-own-llama-stack-distro
    self.remoteAgents = RemoteAgents(url: URL(string: "https://llama-stack.together.ai")!)
  }

  var agents: Agents {
    get { isUsingLocalAgents ? self.localAgents : self.remoteAgents }
  }

  private var placeholder: String {
    "Ask Llama to summarize..."
  }

  private var title: String {
    "StackSummary"
  }

  private var isInputEnabled: Bool { return !isGenerating }

  var body: some View {
    NavigationView {
      VStack {
        Picker("Agent Type", selection: $isUsingLocalAgents) {
          Text("Local").tag(true)
          Text("Remote").tag(false)
        }
        .pickerStyle(.segmented)
        .padding()
        MessageListView(messages: $messages)
                  .gesture(
                    DragGesture().onChanged { value in
                      if value.translation.height > 10 {
                        UIApplication.shared.sendAction(#selector(UIResponder.resignFirstResponder), to: nil, from: nil, for: nil)
                      }
                    }
                  )
        Spacer()

        HStack {
          TextField(placeholder, text: $prompt, axis: .vertical)
            .padding(8)
            .background(Color.gray.opacity(0.1))
            .cornerRadius(20)
            .lineLimit(1...10)
            .overlay(
              RoundedRectangle(cornerRadius: 20)
                .stroke(isInputEnabled ? Color.blue : Color.gray, lineWidth: 1)
            )
            .disabled(!isInputEnabled)

          Button(action: generate) {
            Image(systemName: "arrowshape.up.circle.fill")
              .resizable()
              .aspectRatio(contentMode: .fit)
              .frame(height: 28)
          }
          .disabled(isGenerating || (!isInputEnabled || prompt.isEmpty))
        }
        .padding([.leading, .trailing, .bottom], 10)
      }
    }
    .navigationViewStyle(StackNavigationViewStyle())
    .sheet(isPresented:  Binding(
      get: { self.presetEvent != nil },
      set: { if !$0 { self.presetEvent = nil } }
    )) {
      if let event = presetEvent {
        EventEditView(isPresented: $isShowingEventModal,
                      eventStore: self.eventStore,
                      event: event)
      }
    }
  }

  func triggerAddEventToCalendar(title: String, startDate: Date, endDate: Date) {
    eventStore.requestAccess(to: .event) { [self] granted, error in
      if granted {
        let event = EKEvent(eventStore: eventStore)
        event.title = title
        event.startDate = startDate
        event.endDate = endDate
        event.calendar = eventStore.defaultCalendarForNewEvents
        event.notes = self.actionItems
        self.presetEvent = event
      } else {
        print("Calendar access denied")
      }
    }
  }

  func summarizeConversation(prompt: String) async {
    do {
      let request = Components.Schemas.CreateAgentTurnRequest(
        messages: [
          .UserMessage(Components.Schemas.UserMessage(
            role: .user,
            content: .case1("Summarize the following conversation in 1-2 sentences:\n\n \(prompt)")
          ))
        ],
        stream: true
      )

      for try await chunk in try await self.agents.createTurn(agent_id: self.agentId, session_id: self.agenticSystemSessionId, request: request) {
        let payload = chunk.event.payload
        switch (payload) {
        case .step_start(_):
          break
        case .step_progress(let step):
          if (step.delta != nil) {
            DispatchQueue.main.async {
              withAnimation {
                var message = messages.removeLast()
                if case .text(let delta) = step.delta {
                  message.text += "\(delta.text)"
                }
                message.tokenCount += 2
                message.dateUpdated = Date()
                messages.append(message)
              }
            }
          }
        case .step_complete(_):
          break
        case .turn_start(_):
          break
        case .turn_complete(_):
          break
        case .turn_awaiting_input(_):
          break
        }

      }
    } catch {
      print("Summarization failed: \(error)")
    }
  }

  func actionItems(prompt: String) async throws {
    let request = Components.Schemas.CreateAgentTurnRequest(
      messages: [
        .UserMessage(Components.Schemas.UserMessage(
          role: .user,
          content: .case1("List out any action items based on this text:\n\n \(prompt)")
        ))
      ],
      stream: true
    )

    for try await chunk in try await self.agents.createTurn(agent_id: self.agentId, session_id: self.agenticSystemSessionId, request: request) {
      let payload = chunk.event.payload
      switch (payload) {
      case .step_start(_):
        break
      case .step_progress(let step):
        DispatchQueue.main.async(execute: DispatchWorkItem {
          withAnimation {
            var message = messages.removeLast()
            
            if case .text(let delta) = step.delta {
              message.text += "\(delta.text)"
              self.actionItems += "\(delta.text)"
            }
            message.tokenCount += 2
            message.dateUpdated = Date()
            messages.append(message)
          }
        })
      case .step_complete(_):
        break
      case .turn_start(_):
        break
      case .turn_complete(_):
        break
      case .turn_awaiting_input(_):
        break
      }
    }
  }

  func callTools(prompt: String) async throws {
    let request = Components.Schemas.CreateAgentTurnRequest(
      messages: [
        .UserMessage(Components.Schemas.UserMessage(
          role: .user,
          content: .case1("Call functions as needed to handle any actions in the following text:\n\n" + prompt)
        ))
      ],
      stream: true
    )

    for try await chunk in try await self.agents.createTurn(agent_id: self.agentId, session_id: self.agenticSystemSessionId, request: request) {
      let payload = chunk.event.payload
      switch (payload) {
      case .step_start(_):
        break
      case .step_progress(let step):
          switch (step.delta) {
          case .tool_call(let call):
            if call.parse_status == .succeeded {
              switch (call.tool_call) {
              case .ToolCall(let toolCall):
                  var args: [String : String] = [:]
                  for (arg_name, arg) in toolCall.arguments.additionalProperties {
                    switch (arg) {
                    case .case1(let s):
                      args[arg_name] = s
                    case .case2(_), .case3(_), .case4(_), .case5(_), .case6(_):
                      break
                    }
                  }

                  let formatter = DateFormatter()
                  formatter.dateFormat = "yyyy-MM-dd HH:mm"
                  formatter.timeZone = TimeZone.current
                  formatter.locale = Locale.current
                  self.triggerAddEventToCalendar(
                    title: args["event_name"]!,
                    startDate: formatter.date(from: args["start"]!) ?? Date(),
                    endDate: formatter.date(from: args["end"]!) ?? Date()
                  )
              case .case1(_):
                break
              }
            }
          case .text(let text):
            break
          case .image(_):
            break
          }
        break
      case .step_complete(_):
        break
      case .turn_start(_):
        break
      case .turn_complete(_):
        break
      case .turn_awaiting_input(_):
        break
      }
    }
  }

  private func generate() {
    guard !prompt.isEmpty else { return }
    isGenerating = true

    let text = prompt
    prompt = ""
    hideKeyboard()

    runnerQueue.async {
      defer {
        DispatchQueue.main.async {
          isGenerating = false
        }
      }

      let bundle = Bundle.main
      let modelPath = bundle.url(forResource: "llama3_2_spinquant_oct23", withExtension: "pte")?.relativePath
      let tokenizerPath = bundle.url(forResource: "tokenizer", withExtension: "model")?.relativePath
      inference.loadModel(
        modelPath: modelPath!,
        tokenizerPath: tokenizerPath!,
        completion: {_ in }
      )

      Task {
        messages.append(Message(text: text))

        do {
          let createSystemResponse = try await self.agents.create(
            request: Components.Schemas.CreateAgentRequest(
              agent_config: Components.Schemas.AgentConfig(
                client_tools: [ CustomTools.getCreateEventToolForAgent() ],
                max_infer_iters: 1,
                model: "meta-llama/Llama-3.1-8B-Instruct",
                instructions: "You are a helpful assistant",
                enable_session_persistence: false
              )
            )
          )
          self.agentId = createSystemResponse.agent_id

          let createSessionResponse = try await self.agents.createSession(agent_id: self.agentId, request: Components.Schemas.CreateAgentSessionRequest(session_name: "llama-assistant")
          )
          self.agenticSystemSessionId = createSessionResponse.session_id

          messages.append(Message(type: .summary))
          try await summarizeConversation(prompt: text)

         messages.append(Message(type: .actionItems))
         try await actionItems(prompt: text)

         try await callTools(prompt: text)
        } catch {
          print("Error: \(error)")
        }
      }
    }
  }
}

extension View {
  func hideKeyboard() {
    UIApplication.shared.sendAction(#selector(UIResponder.resignFirstResponder), to: nil, from: nil, for: nil)
  }
}
