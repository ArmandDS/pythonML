from datetime import datetime

from SenseCells.tts import tts

def what_time_is():
	tts("The time is"+ datetime.strftime(datetime.now(), '%H:%M:%S'))