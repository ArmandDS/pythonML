from GreyMatter import conversation, tell_time, weather, define, business_news_reader

def brain(name, speech_text, city_name, city_code):
	def check_message(check):
		"""
		function that checks messages and present it to the user's input speech
		"""
		word_of_message = speech_text.split()
		if set(check).issubset(set(word_of_message)):
			return True
		else:
			return False
	if check_message(['who', 'are', 'you']):
		conversation.who_are_u()
	elif check_message(['how', 'are', 'you']):
		conversation.how_are_u()
	elif check_message(['how', 'i', 'am']):
		conversation.how_am_i()
	elif check_message(['who', 'am', 'i']):
		conversation.who_am_i(name)
	elif check_message(['time']):
		tell_time.what_time_is()
	elif check_message(['how', 'weather']) or check_message(['hows', 'weather']):
		weather.weather(city_name, city_code)
	elif check_message(['define']):
		define.define(speech_text)
	elif check_message(['business', 'news']):
		business_news_reader.news_reader() 
	else:
		conversation.undefined()