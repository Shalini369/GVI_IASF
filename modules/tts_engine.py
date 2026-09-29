import pyttsx3


def speak_text(text, priority=2):
    """
    Convert detected text or hazard alerts into speech.

    priority=1 → safety/hazard alert
    priority=2 → normal OCR/text alert
    """

    if not text or not text.strip():
        return

    print(f"[TTS] Speaking: {text}")

    engine = pyttsx3.init()

    engine.setProperty("rate", 160)
    engine.setProperty("volume", 1.0)

    engine.say(text)
    engine.runAndWait()

    engine.stop()