import random
from SenseCells.tts import tts


def who_are_u():
	message = ["Iam Vicky, your personal assitant", "Vicky,I already told you", "You ask that so many times, I am Vicky"]
	tts(random.choice(message))

def how_am_i():
	message = ["You are brilliant", "Armand, an Smart guy", "You are so smart"]
	tts(random.choice(message))

def who_am_i(name):
	tts('You are '+name)

def how_are_u():
	tts('I am very good,thank you')
	
def undefined():
	tts('I do not know what do you means')
	