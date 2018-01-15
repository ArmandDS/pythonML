import os
import sys


def tts(message):
	"""Function take an argument 
	and convert it to speech dependeing OS"""
	if sys.platform == "darwin":
		tts_engine = 'say'
		return os.system(tts_engine + ' ' + message)
	elif sys.platform =="linux2" or sys.platform=="linux":
		tts_engine = "espeak"
		return os.system(tts_engine + ' "' + message + '"')
	elif sys.platform =="win32":
		tts_engine = "espeak -v+f3"
		return os.system(tts_engine + ' "' + message + '"')

