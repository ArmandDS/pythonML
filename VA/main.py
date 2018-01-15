import sys
import yaml

import speech_recognition as sr

from GreyMatter.SenseCells.tts import tts
from brain import brain

profile = open("profile.yaml.default")
profile_data = yaml.safe_load(profile)
profile.close()

#Variables
name = profile_data['name']
city_name = profile_data['city_name']
city_code = profile_data['city_code']

tts('Welcome '+ name + ', systems are ready to run')

def main():
	r = sr.Recognizer()
	with sr.Microphone(1) as source:
		print('Speak')
		audio = r.listen(source)
	try:
		speech_text = r.recognize_google(audio).lower().replace("'", "")
		print("Vicky thinks you said '" +speech_text + "'")
	except sr.UnknownValueError:
		print("Vicky could not understand you")
	except sr.RequestError as e:
		print("Could not request results from Speech recognition service; {0}".format(e))
	brain(name, speech_text,city_name, city_code)
	
main()