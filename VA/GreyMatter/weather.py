import pywapi

from SenseCells.tts import tts

def weather(city_name, city_code):
	weather_com_result = pywapi.get_weather_from_weather_com(city_code)
	weather_result = "Weather says it is " + weather_com_result['current_conditions']['text'].lower() + " and " +weather_com_result['current_conditions']['temperature'] +\
						 "degree celsius now in " + city_name
	
	tts(weather_result)