import telebot
from telebot import types
import requests, re
from main import predict_model

bot = telebot.TeleBot('6796946783:AAHLotswpGeRSiCuY21xJeXKm5IDBnAXFEI')
welcome_sent = False
print(1)

waiting_for_sentiment = False
sentiment_dict = {True : "positive", False : "negative"}

MODEL_PATH = "model.pkl"

@bot.message_handler(commands=['start'])
def handle_start(message):
    bot.send_message(message.chat.id,
                     "Hello! I'm a sentiment analysis bot. Press send text to analyze.")

@bot.message_handler(func=lambda message: True)
def handle_text_message(message):
    text = message.text
    sentiment = predict_model(text, MODEL_PATH)
    bot.send_message(message.chat.id, f"Text sentiment is: {sentiment_dict[sentiment[0]]}")

bot.polling()