from gtts import gTTS


mytext = 'linia autobusowa, 72'
language = 'pl'
myobj = gTTS(text=mytext, lang=language, slow=False)
myobj.save("r_72.mp3")