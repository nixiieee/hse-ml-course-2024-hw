from xmlrpc.client import DateTime
from telethon.sync import TelegramClient
from telethon.tl.functions.messages import GetHistoryRequest
from telethon.tl.types import PeerChannel

import csv
import time

file = open('/home/nixiiee/Documents/2course/hse-ml-course-2024-hw/data/api-hash.txt', 'r')
api_id = file.readline()[:-1]
api_hash = file.readline()[:-1]
phone = file.readline()[:-1]
file.close()

client = TelegramClient(phone, api_id, api_hash)
client.start()

dialogs = client.get_dialogs()
for dialog in dialogs:
    if dialog.title == 'Авто-подбор.рф':
        messages = client.iter_messages(dialog)
        for message in messages:
            print(message.text)
            time.sleep(1)