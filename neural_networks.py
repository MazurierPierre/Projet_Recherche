import speech_recognition as sr


def recognize(file_path, duration=None, offset=None, verbose=False):
    r = sr.Recognizer()

    # Select audio source
    audio_file = sr.AudioFile(file_path)
    with audio_file as source:
        audio = r.record(source, duration=duration, offset=offset)

    # Send data through API
    alternatives = r.recognize_google(audio, show_all=True)

    # If nothing was transcribed by the API
    if not alternatives:
        return '', 0

    # Select result with highest confidence (always first in list)
    result = alternatives["alternative"][0]

    if verbose:
        print("Result     : " + result['transcript'])
        print("Confidence : " + str(result['confidence']))

    return result['transcript'], result['confidence']
