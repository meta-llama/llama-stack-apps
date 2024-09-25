import SwiftUI
import EventKit
import EventKitUI

struct EventEditView: UIViewControllerRepresentable {
  @Binding var isPresented: Bool
  let eventStore: EKEventStore
  let event: EKEvent

  func makeUIViewController(context: Context) -> EKEventEditViewController {
    let controller = EKEventEditViewController()
    controller.eventStore = eventStore
    controller.event = event
    controller.editViewDelegate = context.coordinator
    return controller
  }

  func updateUIViewController(_ uiViewController: EKEventEditViewController, context: Context) {}

  func makeCoordinator() -> Coordinator {
    return Coordinator(isPresented: $isPresented)
  }

  class Coordinator: NSObject, EKEventEditViewDelegate {
    @Binding var isPresented: Bool

    init(isPresented: Binding<Bool>) {
      _isPresented = isPresented
    }

    func eventEditViewController(_ controller: EKEventEditViewController, didCompleteWith action: EKEventEditViewAction) {
      isPresented = false
    }
  }
}
