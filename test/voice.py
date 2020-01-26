import pyttsx3


# 播放语音
def sound(contant):
    engine = pyttsx3.init()
    engine.say('  ' + contant)
    engine.runAndWait()
