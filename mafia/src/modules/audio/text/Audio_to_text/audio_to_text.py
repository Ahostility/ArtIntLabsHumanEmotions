
def conver(path: str):
    import speech_recognition as speech_recog
    from subprocess import call
    import os
    import re

    if not '.wav' in path: 
        call(str('ffmpeg -i ' + path + ' ' +  path.split('.')[0] + '.wav'), shell=True)
        os.remove(path)
        path = path.split('.')[0] + '.wav'


    r = speech_recog.Recognizer() 
    sample_audio = speech_recog.AudioFile(path)

    with sample_audio as audio_file:
        audio_content = r.record(audio_file)

    text = r.recognize_google(audio_content,  language="ru-RU")

    return re.sub(r'[^\w\s]+|[\d]+', r'', text).strip()


if __name__ == '__main__': conver()