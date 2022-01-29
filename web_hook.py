import telebot

bot = telebot.TeleBot("5069703088:AAEA-gfnn-tn9yy6i616eNtiV4sEUJa3x4E")

bot.remove_webhook()
bot.set_webhook("d5dh3ckod1smgtc6ovue.apigw.yandexcloud.net")