from funasr import AutoModel
import pyaudio
import wave

# 1. 音频文件识别
def file_recognize(audio_path):
    model = AutoModel(model="paraformer-zh", vad_model="fsmn-vad", punc_model="ct-punc")
    res = model.generate(input=audio_path, batch_size_s=300)
    print("文件识别结果：", res[0]['text'])

# 2. 麦克风实时识别
def realtime_recognize():
    model = AutoModel(model="paraformer-zh-streaming")
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=640)
    print("开始实时识别（按Ctrl+C停止）...")
    try:
        while True:
            data = stream.read(640)
            res = model.generate(input=data, cache=list(), is_final=False)
            if res:
                print(f"实时结果：{res[0]['text']}", end="\r")
    except KeyboardInterrupt:
        stream.stop_stream()
        stream.close()

if __name__ == "__main__":
    # 替换为你的音频路径
    file_recognize("配音音频.mp3")
    # 开启实时识别
    # realtime_recognize()