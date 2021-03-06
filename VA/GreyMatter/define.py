import re
import wikipedia

from SenseCells.tts import tts

def define(speech_text):
	words_of_message = speech_text.split()
	words_of_message.remove('define')
	cleaned_message = ' '.join(words_of_message)
	
	try:
		wiki_data = wikipedia.summary(cleaned_message, sentences=5)
		regEx = re.compile(r'([^\(]*)\([^\)]*\) *(.*)')
		m = regEx.match(wiki_data)
		while m:
			wiki_data = m.group(1)+m.group(2)
			m = regEx.match(wiki_data)
			
		wiki_data = wiki_data.replace("'", "")
		tts(wiki_data)
	except wikipedia.exceptions.DisambiguationError as e:
		print('Can you please be more specific?, you may choose something from the following.; {0}'.format(e))
		tts('Can you please be more specific?, you may choose something from the following {0}'.format(e))
		
		