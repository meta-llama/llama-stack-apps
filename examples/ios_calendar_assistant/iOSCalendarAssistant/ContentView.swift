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

  private let agent = RemoteAgents(url: URL(string: "http://54.189.109.3:8501")!)
//  private let agent = RemoteAgents(url: URL(string: "https://llama-stack.together.ai")!)
  @State var agentId = ""
  @State var agenticSystemSessionId = ""

  @State private var actionItems = ""

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
        MessageListView(messages: $messages)
          .gesture(
            DragGesture().onChanged { value in
              if value.translation.height > 10 {
                UIApplication.shared.sendAction(#selector(UIResponder.resignFirstResponder), to: nil, from: nil, for: nil)
              }
            }
          )
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
            content: .case1("Summarize the following conversation in 1-2 sentences:\n\n \(prompt)"),
            role: .user
          ))
        ],
        stream: true
      )

      for try await chunk in try await self.agent.createTurn(agent_id: self.agentId, session_id: self.agenticSystemSessionId, request: request) {
        let payload = chunk.event.payload
        print(">>>\(payload)")
        switch (payload) {
        case .AgentTurnResponseStepStartPayload(_):
          break
        case .AgentTurnResponseStepProgressPayload(let step):
          if (step.delta != nil) {
            print("==> \(step.delta)")
            DispatchQueue.main.async {
              withAnimation {
                var message = messages.removeLast()
                if case .TextDelta(let textDelta) = step.delta {
                    let text = textDelta.text
                    message.text += "\(text)"
                }
                
                message.tokenCount += 2
                message.dateUpdated = Date()
                messages.append(message)
                print(message.text)
              }
            }
          }
        case .AgentTurnResponseStepCompletePayload(_):
          break
        case .AgentTurnResponseTurnStartPayload(_):
          break
        case .AgentTurnResponseTurnCompletePayload(_):
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
          content: .case1("List out any action items based on this text:\n\n \(prompt)"),
          role: .user
        ))
      ],
      stream: true
    )

    for try await chunk in try await self.agent.createTurn(agent_id: self.agentId, session_id: self.agenticSystemSessionId, request: request) {
      let payload = chunk.event.payload
      switch (payload) {
      case .AgentTurnResponseStepStartPayload(_):
        break
      case .AgentTurnResponseStepProgressPayload(let step):
        if (step.delta != nil) {
          DispatchQueue.main.async {
            withAnimation {
              var message = messages.removeLast()
              
              if case .TextDelta(let textDelta) = step.delta {
                  let text = textDelta.text
                  message.text += "\(text)"
              }
              message.tokenCount += 2
              message.dateUpdated = Date()
              messages.append(message)

              self.actionItems += "\(step.delta)"
            }
          }
        }
      case .AgentTurnResponseStepCompletePayload(_):
        break
      case .AgentTurnResponseTurnStartPayload(_):
        break
      case .AgentTurnResponseTurnCompletePayload(_):
        break
      }
    }
  }

  func callTools(prompt: String) async throws {
    
//    try await self.agent.initAndCreateTurn(messages: [
//      .UserMessage(Components.Schemas.UserMessage(
//        content: .case1("Call functions as needed to handle any actions in the following text:\n\n" + prompt),
//        role: .user
//      ))
//    ])
    
//    let request = Components.Schemas.CreateAgentTurnRequest(
//      messages: [
//        .UserMessage(Components.Schemas.UserMessage(
//          content: .case1("Call functions as needed to handle any actions in the following text:\n\n" + prompt),
//          role: .user
//        ))
//      ],
//      stream: true
//    )

//    for try await chunk in try await self.agent.createTurn(agent_id: self.agentId, session_id: self.agenticSystemSessionId, request: request) {
      
    for try await chunk in try await self.agent.initAndCreateTurn(messages: [
      .UserMessage(Components.Schemas.UserMessage(
        content: .case1("Call functions as needed to handle any actions in the following text:\n\n" + prompt),
        role: .user
      ))
    ]) {
      let payload = chunk.event.payload
      print("**** payload=\(payload)")
      switch (payload) {
      case .AgentTurnResponseStepStartPayload(_):
        break
        
      case .AgentTurnResponseStepProgressPayload(let step):
        if (step.delta != nil) {
          print("!!! \(step.delta)")
          switch (step.delta) {
//          case .case1(_):
//            break
          case .ToolCallDelta(let call):
            switch (call.content) {
            case .ToolCall(let toolName):
              print("***toolName = \(toolName)")
              break
            case .case1(let toolName):
              print("!!!toolName = \(toolName)")
              if (toolName == "create_event") {
                var args: [String : String] = [:]
//                for (arg_name, arg) in call.arguments.additionalProperties {
//                  switch (arg) {
//                  case .case1(let s): // type string
//                    args[arg_name] = s
//                  case .case2(_), .case3(_), .case4(_), .case5(_), .case6(_):
//                    break
//                  }
//                }
//
//                let formatter = DateFormatter()
//                formatter.dateFormat = "yyyy-MM-dd HH:mm"
//                formatter.timeZone = TimeZone.current
//                formatter.locale = Locale.current
//                self.triggerAddEventToCalendar(
//                  title: args["event_name"]!,
//                  startDate: formatter.date(from: args["start"]!) ?? Date(),
//                  endDate: formatter.date(from: args["end"]!) ?? Date()
//                )
              }
            }
          case .TextDelta(_):
            if case .TextDelta(let textDelta) = step.delta {
                let text = textDelta.text
              print(">>>text=\(text)")
            }

            break
          case .ImageDelta(_):
            break
          }
        }
        break
      case .AgentTurnResponseStepCompletePayload(_):
        break
      case .AgentTurnResponseTurnStartPayload(_):
        break
      case .AgentTurnResponseTurnCompletePayload(_):
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

    //runnerQueue.async {
    let workItem = DispatchWorkItem {
      defer {
        DispatchQueue.main.async {
          isGenerating = false
        }
      }

      Task {
        messages.append(Message(text: text))
        messages.append(Message(type: .summary))

        do {
          let createSystemResponse = try await self.agent.create(
            request: Components.Schemas.CreateAgentRequest(
              agent_config: Components.Schemas.AgentConfig(
                client_tools: [ CustomTools.getCreateEventToolForAgent() ],
                enable_session_persistence: false,
                instructions: "You are a helpful assistant",
                max_infer_iters: 1,
                model: "meta-llama/Llama-3.1-8B-Instruct"
                //toolgroups: ["builtin::websearch"],
              )
            )
          )
          self.agentId = createSystemResponse.agent_id

          let createSessionResponse = try await self.agent.createSession(agent_id: self.agentId,
            request: Components.Schemas.CreateAgentSessionRequest(session_name: "llama-assistant")
          )
          self.agenticSystemSessionId = createSessionResponse.session_id

          try await summarizeConversation(prompt: text)

          messages.append(Message(type: .actionItems))
          try await actionItems(prompt: text)

          //try await callTools(prompt: text)
        } catch {
          print("Error: \(error)")
        }
      }
    }
    
    runnerQueue.async(execute: workItem)
  }
}

extension View {
  func hideKeyboard() {
    UIApplication.shared.sendAction(#selector(UIResponder.resignFirstResponder), to: nil, from: nil, for: nil)
  }
}
