# -*- coding: utf-8 -*-
#HSTW (Hate Speech from Twitter/Whisper)
# - Paper: https://doi.org/10.1145/3078714.3078723
# - Data: https://github.com/Mainack/hatespeech-data-HT-2017

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import pandas as pd
import random
import csv
from utils.data_utils import balance_classes

file = '../../data/HSTW/hatespeech_whisper_released_hypertext_2017.csv'
data = pd.read_csv(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_ALL, encoding='utf-8')


# down sample to 100 per class
data = balance_classes(data, 100, "Hate categories assigned by us")


# Define the instruction templates in 29 different languages
instruction_templates = {
    "Chinese": """
    下面我将向您提供一条包含脏话并表达仇恨的推文。
    请使用以下类别之一对推文进行分类：'behavior', 'race', 'sexual orientation', 'ethnicity', 'other', 'physical', 'class', 'religion', 'gender', 'disability'，以最能描述推文中的仇恨言论。
    推文: "{text}"。
    """,
    "English": """
    Below I will provide you with a tweet that contains swear words and expresses hate.
    Please categorize the tweet using one of the following categories: 'behavior', 'race', 'sexual orientation', 'ethnicity', 'other', 'physical', 'class', 'religion', 'gender', 'disability', that best describes the hate speech in the tweet.
    Tweet: "{text}".
    """,
    "German": """
    Unten werde ich Ihnen einen Tweet zur Verfügung stellen, der Schimpfwörter enthält und Hass ausdrückt.
    Bitte kategorisieren Sie den Tweet mit einer der folgenden Kategorien: 'behavior', 'race', 'sexual orientation', 'ethnicity', 'other', 'physical', 'class', 'religion', 'gender', 'disability', die die Hassrede im Tweet am besten beschreibt.
    Tweet: "{text}".
    """,
    "French": """
    Ci-dessous, je vais vous fournir un tweet contenant des grossièretés et exprimant de la haine.
    Veuillez catégoriser le tweet en utilisant l'une des catégories suivantes : 'behavior', 'race', 'sexual orientation', 'ethnicity', 'other', 'physical', 'class', 'religion', 'gender', 'disability', qui décrit le mieux le discours de haine dans le tweet.
    Tweet : "{text}".
    """,
    "Spanish": """
    A continuación, le proporcionaré un tweet que contiene malas palabras y expresa odio.
    Por favor, clasifique el tweet usando una de las siguientes categorías: 'behavior', 'race', 'sexual orientation', 'ethnicity', 'other', 'physical', 'class', 'religion', 'gender', 'disability', que mejor describa el discurso de odio en el tweet.
    Tweet: "{text}".
    """,
    "Portuguese": """
    Abaixo, vou fornecer-lhe um tweet que contém palavrões e expressa ódio.
    Por favor, categorize o tweet usando uma das seguintes categorias: 'behavior', 'race', 'sexual orientation', 'ethnicity', 'other', 'physical', 'class', 'religion', 'gender', 'disability', que melhor descreve o discurso de ódio no tweet.
    Tweet: "{text}".
    """,
    "Italian": """
    Di seguito ti fornirò un tweet che contiene parolacce ed esprime odio.
    Per favore, classifica il tweet utilizzando una delle seguenti categorie: 'behavior', 'race', 'sexual orientation', 'ethnicity', 'other', 'physical', 'class', 'religion', 'gender', 'disability', che meglio descrive il discorso di odio nel tweet.
    Tweet: "{text}".
    """,
    "Dutch": """
    Hieronder geef ik je een tweet die scheldwoorden bevat en haat uitdrukt.
    Categoriseer de tweet alstublieft met een van de volgende categorieën: 'behavior', 'race', 'sexual orientation', 'ethnicity', 'other', 'physical', 'class', 'religion', 'gender', 'disability', die de haatspraak in de tweet het beste beschrijft.
    Tweet: "{text}".
    """,
    "Russian": """
    Ниже я предоставлю вам твит, содержащий ругательства и выражающий ненависть.
    Пожалуйста, отнесите твит к одной из следующих категорий: 'behavior', 'race', 'sexual orientation', 'ethnicity', 'other', 'physical', 'class', 'religion', 'gender', 'disability', которая лучше всего описывает ненавистническую речь в твите.
    Твит: "{text}".
    """,
    "Czech": """
    Níže vám poskytnu tweet, který obsahuje nadávky a vyjadřuje nenávist.
    Kategorizujte prosím tweet pomocí jedné z následujících kategorií: 'behavior', 'race', 'sexual orientation', 'ethnicity', 'other', 'physical', 'class', 'religion', 'gender', 'disability', která nejlépe popisuje nenávistné projevy v tweetu.
    Tweet: "{text}".
    """,
    "Polish": """
    Poniżej podam Ci tweeta, który zawiera przekleństwa i wyraża nienawiść.
    Proszę sklasyfikować tweeta, używając jednej z następujących kategorii: 'behavior', 'race', 'sexual orientation', 'ethnicity', 'other', 'physical', 'class', 'religion', 'gender', 'disability', która najlepiej opisuje mowę nienawiści w tweecie.
    Tweet: "{text}".
    """,
    "Arabic": """
    أدناه سأقدم لك تغريدة تحتوي على كلمات بذيئة وتعبر عن الكراهية.
    يرجى تصنيف التغريدة باستخدام واحدة من الفئات التالية: 'behavior', 'race', 'sexual orientation', 'ethnicity', 'other', 'physical', 'class', 'religion', 'gender', 'disability' التي تصف خطاب الكراهية في التغريدة بشكل أفضل.
    التغريدة: "{text}".
    """,
    "Persian": """
    در زیر، من یک توییت که حاوی الفاظ توهین‌آمیز است و نفرت را بیان می‌کند به شما ارائه خواهم داد.
    لطفاً توییت را با استفاده از یکی از دسته‌بندی‌های زیر طبقه‌بندی کنید: 'behavior', 'race', 'sexual orientation', 'ethnicity', 'other', 'physical', 'class', 'religion', 'gender', 'disability' که بهترین توصیف را از سخنرانی نفرت در توییت ارائه می‌دهد.
    توییت: "{text}".
    """,
    "Hebrew": """
    להלן אציג בפניך ציוץ המכיל מילים גסות ומביע שנאה.
    אנא סווג את הציוץ באמצעות אחת מהקטגוריות הבאות: 'behavior', 'race', 'sexual orientation', 'ethnicity', 'other', 'physical', 'class', 'religion', 'gender', 'disability' שמתארת בצורה הטובה ביותר את השנאה בציוץ.
    ציוץ: "{text}".
    """,
    "Turkish": """
    Aşağıda küfür içeren ve nefret ifade eden bir tweet sunacağım.
    Tweet'i, tweet'teki nefret söylemini en iyi tanımlayan 'behavior', 'race', 'sexual orientation', 'ethnicity', 'other', 'physical', 'class', 'religion', 'gender', 'disability' kategorilerinden biriyle sınıflandırın.
    Tweet: "{text}".
    """,
    "Japanese": """
    以下に、罵倒語を含み、憎しみを表現するツイートを提供します。
    ツイートに含まれるヘイトスピーチを最もよく表す次のカテゴリーのいずれかを使用して、ツイートを分類してください: 'behavior', 'race', 'sexual orientation', 'ethnicity', 'other', 'physical', 'class', 'religion', 'gender', 'disability'。
    ツイート: "{text}"。
    """,
    "Korean": """
    아래에 욕설이 포함되고 증오를 표현하는 트윗을 제공하겠습니다.
    트윗의 증오 발언을 가장 잘 설명하는 다음 범주 중 하나를 사용하여 트윗을 분류하십시오: 'behavior', 'race', 'sexual orientation', 'ethnicity', 'other', 'physical', 'class', 'religion', 'gender', 'disability'.
    트윗: "{text}".
    """,
    "Vietnamese": """
    Dưới đây tôi sẽ cung cấp cho bạn một tweet chứa từ ngữ tục tĩu và bày tỏ sự căm ghét.
    Vui lòng phân loại tweet bằng cách sử dụng một trong các danh mục sau: 'behavior', 'race', 'sexual orientation', 'ethnicity', 'other', 'physical', 'class', 'religion', 'gender', 'disability' mà mô tả chính xác nhất bài phát biểu thù hận trong tweet.
    Tweet: "{text}".
    """,
    "Thai": """
    ด้านล่างนี้ฉันจะให้ทวีตที่มีคำหยาบคายและแสดงความเกลียดชัง
    โปรดจัดประเภททวีตโดยใช้หมวดหมู่ต่อไปนี้อย่างใดอย่างหนึ่ง: 'behavior', 'race', 'sexual orientation', 'ethnicity', 'other', 'physical', 'class', 'religion', 'gender', 'disability' ที่อธิบายถึงการพูดแสดงความเกลียดชังในทวีตได้ดีที่สุด
    ทวีต: "{text}".
    """,
    "Indonesian": """
    Di bawah ini saya akan memberikan Anda tweet yang mengandung kata-kata kotor dan menyatakan kebencian.
    Silakan kategorikan tweet menggunakan salah satu kategori berikut: 'behavior', 'race', 'sexual orientation', 'ethnicity', 'other', 'physical', 'class', 'religion', 'gender', 'disability' yang paling menggambarkan ujaran kebencian dalam tweet tersebut.
    Tweet: "{text}".
    """,
    "Malay": """
    Di bawah ini saya akan memberikan anda tweet yang mengandungi kata-kata kasar dan menyatakan kebencian.
    Sila kategorikan tweet menggunakan salah satu kategori berikut: 'behavior', 'race', 'sexual orientation', 'ethnicity', 'other', 'physical', 'class', 'religion', 'gender', 'disability' yang paling menggambarkan ucapan kebencian dalam tweet tersebut.
    Tweet: "{text}".
    """,
    "Lao": """
    ຂ້າພະເຈົ້າຈະສະເໜີທະວີດທີ່ມີຄໍາສາບແຊ່ງ ແລະ ປະກາດຄວາມເກຽດຊັງຕໍ່ເຈົ້າ.
    ກະລຸນາແຍກປະເພດທະວີດດ້ວຍໜຶ່ງໃນຫຼາຍປະເພດດັ່ງຕໍ່ໄປນີ້: 'behavior', 'race', 'sexual orientation', 'ethnicity', 'other', 'physical', 'class', 'religion', 'gender', 'disability' ທີ່ອະທິບາຍຄວາມເກຽດຊັງໃນທະວີດໄດ້ດີທີ່ສຸດ.
    ທະວີດ: "{text}".
    """,
    "Burmese": """
    အောက်မှာ ကျွန်တော် သင့်ကို သင်္ချိုင်းစကားတွေ ပါဝင်ပြီး အမုန်းစကားကို ဖော်ပြထားတဲ့ တစ်ခုတည်းသော တိုက်ပွဲမဲတွေကို ပေးပါမယ်။
    Tweet အတွက် အမုန်းစကားကို အကောင်းဆုံးဖော်ပြတဲ့ အမျိုးအစားတစ်ခုဖြစ်တဲ့ 'behavior', 'race', 'sexual orientation', 'ethnicity', 'other', 'physical', 'class', 'religion', 'gender', 'disability' တွေကို အသုံးပြုပြီး အမျိုးအစားခွဲပါ။
    Tweet: "{text}".
    """,
    "Cebuano": """
    Ubos akong ihatag kanimo ang usa ka tweet nga adunay mga pulong nga panumpa ug nagpakita sa kasuko.
    Palihug kategorya ang tweet gamit ang usa sa mosunod nga mga kategorya: 'behavior', 'race', 'sexual orientation', 'ethnicity', 'other', 'physical', 'class', 'religion', 'gender', 'disability' nga labing maayo nga naghulagway sa hate speech sa tweet.
    Tweet: "{text}".
    """,
    "Khmer": """
    ខាងក្រោមខ្ញុំនឹងផ្តល់ឱ្យអ្នកនូវការចាប់អារម្មណ៍ដែលមានពាក្យសម្តីឈ្លើយនិងបង្ហាញពីការរិះគន់។
    សូមធ្វើការតម្រៀបប្រភេទបញ្ជីខាងក្រោមដោយប្រើប្រភេទខាងក្រោមទាំងអស់គ្នា: 'behavior', 'race', 'sexual orientation', 'ethnicity', 'other', 'physical', 'class', 'religion', 'gender', 'disability' ដែលបានពិពណ៌នាការបង្ហាញការរិះគន់ក្នុងការចាប់អារម្មណ៍នេះ។
    Tweet: "{text}".
    """,
    "Tagalog": """
    Sa ibaba ay bibigyan kita ng isang tweet na naglalaman ng mga murang salita at nagpapahayag ng galit.
    Mangyaring uriin ang tweet gamit ang isa sa mga sumusunod na kategorya: 'behavior', 'race', 'sexual orientation', 'ethnicity', 'other', 'physical', 'class', 'religion', 'gender', 'disability' na pinakamagandang naglalarawan ng pahayag ng galit sa tweet.
    Tweet: "{text}".
    """,
    "Hindi": """
    नीचे मैं आपको एक ट्वीट प्रदान करूंगा जिसमें गाली-गलौज के शब्द शामिल हैं और नफरत व्यक्त की गई है।
    कृपया ट्वीट को निम्नलिखित श्रेणियों में से किसी एक का उपयोग करके वर्गीकृत करें: 'behavior', 'race', 'sexual orientation', 'ethnicity', 'other', 'physical', 'class', 'religion', 'gender', 'disability' जो ट्वीट में घृणा भाषण का सबसे अच्छा वर्णन करता है।
    ट्वीट: "{text}".
    """,
    "Bengali": """
    নীচে আমি আপনাকে একটি টুইট সরবরাহ করব যাতে গালিগালাজ এবং ঘৃণা প্রকাশ করা হয়েছে।
    দয়া করে টুইটটিকে নিম্নলিখিত বিভাগের মধ্যে একটি ব্যবহার করে শ্রেণীবদ্ধ করুন: 'behavior', 'race', 'sexual orientation', 'ethnicity', 'other', 'physical', 'class', 'religion', 'gender', 'disability' যা টুইটের ঘৃণ্য বক্তব্যকে সবচেয়ে ভালোভাবে বর্ণনা করে।
    টুইট: "{text}".
    """,
    "Urdu": """
    نیچے میں آپ کو ایک ٹویٹ فراہم کروں گا جس میں گالی گلوچ کے الفاظ شامل ہیں اور نفرت کا اظہار کیا گیا ہے۔
    براہ کرم ٹویٹ کو درج ذیل زمروں میں سے کسی ایک کا استعمال کرتے ہوئے درجہ بندی کریں: 'behavior', 'race', 'sexual orientation', 'ethnicity', 'other', 'physical', 'class', 'religion', 'gender', 'disability' جو ٹویٹ میں نفرت انگیز تقریر کی بہترین وضاحت کرتا ہے۔
    ٹویٹ: "{text}".
    """
}


# Function to generate instructions based on the language templates
def get_instruction(row):
    template = random.choice(list(instruction_templates.values()))
    return template.format(text=row["text"])

# Create a new DataFrame to store instructions and outputs
inst_data = pd.DataFrame(columns=['instruction', 'output'])

# Generate the instructions and corresponding outputs
for _, row in data.iterrows():
    instruction = get_instruction(row)
    output = row["Hate categories assigned by us"].replace("_", " ")
    inst_data = pd.concat([inst_data, pd.DataFrame({'instruction': [instruction], 'output': [output]})], ignore_index=True)

# Save the data to a Parquet file
inst_data.to_parquet("../../data/HSTW/HSW.parquet", index=False)

