import UIKit
import UniformTypeIdentifiers

enum FileTransferError: Error {
    case fileIOError(description: String)
    case networkError(description: String)
}

class FileTransferIOS: NSObject, UINavigationControllerDelegate, UIImagePickerControllerDelegate {
    static let shared = FileTransferIOS()
    var completion: ((Result<[String: Any], Error>) -> Void)?

    let serverURL = "http://xxx.xx.x.xx:8080/receive-video"

    func selectVideo(viewController: UIViewController, completion: @escaping (Result<[String: Any], Error>) -> Void) {
        self.completion = completion
        let picker = UIImagePickerController()
        picker.sourceType = .photoLibrary
        picker.mediaTypes = [UTType.movie.identifier]
        picker.delegate = self
        viewController.present(picker, animated: true, completion: nil)
    }

    func imagePickerController(_ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey: Any]) {
        picker.dismiss(animated: true, completion: nil)
        if let videoUrl = info[.mediaURL] as? URL {
            self.completion?(.success(["url": videoUrl]))
        } else {
            self.completion?(.failure(FileTransferError.fileIOError(description: "Failed to select video")))
        }
    }

    func sendVideoMultipart(videoUrl: URL, completion: @escaping (Result<[String: Any], Error>) -> Void) {
        guard let url = URL(string: serverURL) else {
            completion(.failure(FileTransferError.networkError(description: "Invalid URL")))
            return
        }

        let boundary = "Boundary-\(UUID().uuidString)"
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")

        var body = Data()
        let filename = videoUrl.lastPathComponent
        let mimeType = "video/mp4"

        body.append("--\(boundary)\r\n".data(using: .utf8)!)
        body.append("Content-Disposition: form-data; name=\"video\"; filename=\"\(filename)\"\r\n".data(using: .utf8)!)
        body.append("Content-Type: \(mimeType)\r\n\r\n".data(using: .utf8)!)
        do {
            let videoData = try Data(contentsOf: videoUrl)
            body.append(videoData)
        } catch {
            print("Error reading video data:", error)
            completion(.failure(FileTransferError.fileIOError(description: "Failed to read video data")))
            return
        }
        body.append("\r\n".data(using: .utf8)!)
        body.append("--\(boundary)--\r\n".data(using: .utf8)!)

        // Set a custom URLSessionConfiguration to increase the timeout duration
        let configuration = URLSessionConfiguration.default
        configuration.timeoutIntervalForRequest = 600 // Request timeout duration(s)（here is 10 mins）
        configuration.timeoutIntervalForResource = 600 //Resource loading timeout duration

        let session = URLSession(configuration: configuration)

        let task = session.uploadTask(with: request, from: body) { data, response, error in
            if let error = error {
                print("Network error:", error)
                completion(.failure(FileTransferError.networkError(description: error.localizedDescription)))
                return
            }
           
            guard let httpResponse = response as? HTTPURLResponse, httpResponse.statusCode == 200 else {
                print("Server error or invalid status code")
                completion(.failure(FileTransferError.networkError(description: "Server error with status code \((response as? HTTPURLResponse)?.statusCode ?? -1)")))
                return
            }
           
            guard let data = data else {
                print("No data received from server")
                completion(.failure(FileTransferError.networkError(description: "No data received from server")))
                return
            }
           
            do {
                if let json = try JSONSerialization.jsonObject(with: data, options: []) as? [String: Any] {
                    print("Server response JSON:", json)
                    completion(.success(json))
                } else {
                    print("Failed to parse JSON object")
                    completion(.failure(FileTransferError.networkError(description: "Failed to parse server response")))
                }
            } catch {
                print("JSON serialization error:", error)
                completion(.failure(FileTransferError.networkError(description: "Failed to parse server response")))
            }
        }
        task.resume()
    }
}
