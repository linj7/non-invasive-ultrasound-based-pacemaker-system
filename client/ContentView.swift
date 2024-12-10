import SwiftUI
import UniformTypeIdentifiers

struct ContentView: View {
    @State private var ejectionFractionText: String = ""
    @State private var coordinates: [String] = []
    @State private var isShowingPicker: Bool = false
    @State private var hasReceivedData: Bool = false  // Flag to track if data has been received

    var body: some View {
        NavigationView {
            ZStack {
                LinearGradient(gradient: Gradient(colors: [Color.black, Color.black.opacity(0.8)]), startPoint: .top, endPoint: .bottom)
                    .edgesIgnoringSafeArea(.all)

                VStack {
                    Spacer()

                    if !hasReceivedData {
                        // Initial UI with dark gray background box
                        ZStack {
                            RoundedRectangle(cornerRadius: 20)
                                .fill(Color.gray.opacity(0.15))
                                .frame(width: 380, height: 260)

                            VStack(spacing: 10) {
                                Text("Non-invasive ultrasound based pacemaker system")
                                    .font(.title2)
                                    .fontWeight(.bold)
                                    .foregroundColor(.white)

                                Text("Use this for real-time cardiac analysis")
                                    .font(.subheadline)
                                    .foregroundColor(.gray)
                                    .padding(.bottom, 10)

                                Button(action: {
                                    isShowingPicker = true
                                }) {
                                    Label("Select and Send Video", systemImage: "paperplane.fill")
                                        .frame(width: 250)
                                        .padding()
                                        .background(Color.blue)
                                        .foregroundColor(.white)
                                        .cornerRadius(12)
                                }
                            }
                            .padding()
                        }
                    } else {
                        // UI after receiving data from the server
                        VStack(spacing: 10) {
                            if !ejectionFractionText.isEmpty {
                                Text(ejectionFractionText)
                                    .padding()
                                    .frame(maxWidth: .infinity)
                                    .background(Color.gray.opacity(0.5))
                                    .foregroundColor(.white)
                                    .cornerRadius(8)
                                    .padding(.horizontal, 20)
                            }

                            ForEach(coordinates.indices, id: \.self) { index in
                                Text("Window\(index + 1): \(coordinates[index])")
                                    .padding()
                                    .frame(maxWidth: .infinity)
                                    .background(Color.gray.opacity(0.5))
                                    .foregroundColor(.white)
                                    .cornerRadius(8)
                                    .padding(.horizontal, 20)
                            }

                            HStack {
                               Button("New Test") {
                                    ejectionFractionText = ""
                                    coordinates = []
                                    hasReceivedData = false // Reset to initial state
                                }
                                .foregroundColor(.white)
                                .padding()
                                .background(Color.blue)
                                .cornerRadius(10)
                            }
                            .padding(.horizontal, 20)
                            .padding(.bottom, 20)
                        }
                    }

                    Spacer()
                }
            }
            .navigationBarHidden(true)
            .sheet(isPresented: $isShowingPicker) {
                VideoPicker(onVideoPicked: { videoUrl in
                    if let videoUrl = videoUrl {
                        FileTransferIOS.shared.sendVideoMultipart(videoUrl: videoUrl) { result in
                            switch result {
                            case .success(let json):
                                parseResponse(json)
                                hasReceivedData = true  // Update on successful data receipt
                            case .failure(let error):
                                ejectionFractionText = "Error: \(error.localizedDescription)"
                                coordinates = []
                            }
                        }
                    }
                })
            }
        }
    }

    func parseResponse(_ response: [String: Any]) {
        if let ejectionFraction = response["ejectionFraction"] as? Double,
           let pacingRequired = response["pacingRequired"] as? Bool {
            ejectionFractionText = "Ejection Fraction: \(ejectionFraction)\(pacingRequired ? "   Pacing needed" : "")"
        }

        if let coordinatesArray = response["coordinates"] as? [[String: String]] {
            coordinates = coordinatesArray.map { coord in
                let x = coord["x"] ?? "0"
                let y = coord["y"] ?? "0"
                return "(\(x), \(y))"
            }
        }
    }
}

struct VideoPicker: UIViewControllerRepresentable {
    var onVideoPicked: (URL?) -> Void

    func makeCoordinator() -> Coordinator {
        Coordinator(onVideoPicked: onVideoPicked)
    }

    func makeUIViewController(context: Context) -> UIImagePickerController {
        let picker = UIImagePickerController()
        picker.sourceType = .photoLibrary
        picker.mediaTypes = [UTType.movie.identifier]
        picker.delegate = context.coordinator
        return picker
    }

    func updateUIViewController(_ uiViewController: UIImagePickerController, context: Context) {
        // This function is intentionally left blank.
    }

    class Coordinator: NSObject, UINavigationControllerDelegate, UIImagePickerControllerDelegate {
        var onVideoPicked: (URL?) -> Void

        init(onVideoPicked: @escaping (URL?) -> Void) {
            self.onVideoPicked = onVideoPicked  // Correct use of 'self' instead of 'this'
        }

        func imagePickerController(_ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey : Any]) {
            picker.dismiss(animated: true, completion: nil)
            let videoURL = info[.mediaURL] as? URL
            onVideoPicked(videoURL)
        }

        func imagePickerControllerDidCancel(_ picker: UIImagePickerController) {
            picker.dismiss(animated: true, completion: nil)
            onVideoPicked(nil)
        }
    }
}


struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
