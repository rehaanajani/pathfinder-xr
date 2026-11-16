import AVFoundation
let args = CommandLine.arguments
let lines = try! String(contentsOfFile: args[1]).split(separator: "\n").map(String.init)
let outDir = args[2]
let voice = AVSpeechSynthesisVoice(identifier: "com.apple.ttsbundle.gryphon-neural_Simone_en-US_premium")!
let synth = AVSpeechSynthesizer()
for (i, line) in lines.enumerated() {
  let u = AVSpeechUtterance(string: line)
  u.voice = voice
  u.rate = 0.47
  u.postUtteranceDelay = 0
  var file: AVAudioFile? = nil
  let sem = DispatchSemaphore(value: 0)
  var done = false
  synth.write(u) { buf in
    guard let pcm = buf as? AVAudioPCMBuffer else { return }
    if pcm.frameLength == 0 { if !done { done = true; sem.signal() }; return }
    if file == nil {
      file = try! AVAudioFile(forWriting: URL(fileURLWithPath: "\(outDir)/s\(String(format: "%02d", i)).caf"),
                              settings: pcm.format.settings, commonFormat: pcm.format.commonFormat, interleaved: pcm.format.isInterleaved)
    }
    try! file!.write(from: pcm)
  }
  while sem.wait(timeout: .now() + 0.05) == .timedOut { RunLoop.current.run(until: Date().addingTimeInterval(0.05)) }
  file = nil
  print("s\(i) done")
}
