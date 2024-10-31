/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import SwiftUI

struct MessageView: View {
  let message: Message

  var body: some View {
    VStack(alignment: .center) {
      if message.type == .info {
        Text(message.text)
          .font(.caption)
          .foregroundColor(.secondary)
          .padding([.leading, .trailing], 10)
      } else {
        VStack(alignment: message.type == .summary || message.type == .actionItems || message.type == .llavagenerated ? .leading : .trailing) {
          if message.type == .summary || message.type == .actionItems || message.type == .llavagenerated || message.type == .prompted {
            Text(message.type == .summary ? "Summary" : (message.type == .actionItems ? "Action Items" : "Prompt"))
              .font(.caption)
              .foregroundColor(.secondary)
              .padding(message.type == .summary || message.type == .actionItems ? .trailing : .leading, 20)
          }
          HStack {
            if message.type != .summary && message.type != .actionItems && message.type != .llavagenerated { Spacer() }
            if message.text.isEmpty {
              if let img = message.image {
                Image(uiImage: img)
                  .resizable()
                  .scaledToFit()
                  .frame(maxWidth: 200, maxHeight: 200)
                  .padding()
                  .background(Color.gray.opacity(0.2))
                  .cornerRadius(8)
                  .padding(.vertical, 2)
              } else {
                ProgressView()
                  .progressViewStyle(CircularProgressViewStyle())
              }
            } else {
              Text(message.text)
                .padding(10)
                .foregroundColor(message.type == .summary || message.type == .actionItems ? .primary : .white)
                .background(message.type == .summary || message.type == .actionItems ? Color(UIColor.secondarySystemBackground) : Color.blue)
                .cornerRadius(20)
                .contextMenu {
                  Button(action: {
                    UIPasteboard.general.string = message.text
                  }) {
                    Text("Copy")
                    Image(systemName: "doc.on.doc")
                  }
                }
            }
            if message.type == .summary || message.type == .actionItems { Spacer() }
          }
          let elapsedTime = message.dateUpdated.timeIntervalSince(message.dateCreated)
        }.padding([.leading, .trailing], message.type == .info ? 0 : 10)
      }
    }
  }
}
