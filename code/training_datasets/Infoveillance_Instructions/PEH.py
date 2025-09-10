# -*- coding: utf-8 -*-
# PEH (Perceived Emotions in Hurricane)
# - Paper: https://doi.org/10.48550/arXiv.2004.14299
# - Data: https://github.com/shreydesai/hurricane

import pandas as pd
import random

# Define all emotions
all_emotions = ["aggressiveness", "optimism", "love", "submission", "awe", "disapproval", "remorse", "contempt"]

# File path to the datasets
folderpath1 = "../../data/PEH/datasets_binary/"

# Define multilingual templates for emotion detection
emotion_templates = [
    # Chinese
    """请确定以下推文中是否包含任何形式的{emotion}。如果包含，请回答 'yes'。如果不包含，请回答 'no'。\n推文内容: "{tweet}"。\n推文到此结束。\n请用 'yes' 或 'no' 作答。""",

    # English
    """Please determine if the provided tweet below contains any form of {emotion}. If it does, please respond 'yes'. If it does not, please respond 'no'.\nTweet: "{tweet}". Now the tweet ends.\nPlease respond with 'yes' or 'no'.""",

    # German
    """Bitte bestimmen Sie, ob der folgende Tweet eine Form von {emotion} enthält. Wenn ja, antworten Sie bitte mit 'yes'. Wenn nein, antworten Sie bitte mit 'no'.\nTweet: "{tweet}". Der Tweet endet hier.\nBitte antworten Sie mit 'yes' oder 'no'.""",

    # French
    """Veuillez déterminer si le tweet ci-dessous contient une forme de {emotion}. S'il en contient, répondez 'yes'. S'il n'en contient pas, répondez 'no'.\nTweet: "{tweet}". Le tweet se termine ici.\nVeuillez répondre par 'yes' ou 'no'.""",

    # Spanish
    """Por favor, determine si el siguiente tweet contiene alguna forma de {emotion}. Si es así, responda 'yes'. Si no, responda 'no'.\nTweet: "{tweet}". El tweet termina aquí.\nResponda con 'yes' o 'no'.""",

    # Portuguese
    """Por favor, determine se o tweet abaixo contém alguma forma de {emotion}. Se contiver, responda 'yes'. Se não contiver, responda 'no'.\nTweet: "{tweet}". O tweet termina aqui.\nResponda com 'yes' ou 'no'.""",

    # Italian
    """Si prega di determinare se il seguente tweet contiene qualche forma di {emotion}. Se lo fa, rispondi 'yes'. Se non lo fa, rispondi 'no'.\nTweet: "{tweet}". Il tweet finisce qui.\nRispondi con 'yes' o 'no'.""",

    # Dutch
    """Bepaal of de onderstaande tweet enige vorm van {emotion} bevat. Als dat zo is, antwoord dan met 'yes'. Zo niet, antwoord dan met 'no'.\nTweet: "{tweet}". De tweet eindigt hier.\nAntwoord met 'yes' of 'no'.""",

    # Russian
    """Определите, содержит ли следующий твит какую-либо форму {emotion}. Если да, ответьте 'yes'. Если нет, ответьте 'no'.\nТвит: "{tweet}". Твит заканчивается здесь.\nОтветьте 'yes' или 'no'.""",

    # Czech
    """Určete, zda následující tweet obsahuje nějakou formu {emotion}. Pokud ano, odpovězte 'yes'. Pokud ne, odpovězte 'no'.\nTweet: "{tweet}". Tweet zde končí.\nOdpovězte 'yes' nebo 'no'.""",

    # Polish
    """Proszę określić, czy poniższy tweet zawiera jakąkolwiek formę {emotion}. Jeśli tak, odpowiedz 'yes'. Jeśli nie, odpowiedz 'no'.\nTweet: "{tweet}". Tweet kończy się tutaj.\nOdpowiedz 'yes' lub 'no'.""",

    # Arabic
    """يرجى تحديد ما إذا كانت التغريدة أدناه تحتوي على أي شكل من أشكال {emotion}. إذا كانت تحتوي، يرجى الرد بـ 'yes'. إذا لم تكن كذلك، يرجى الرد بـ 'no'.\nتغريدة: "{tweet}". تنتهي التغريدة هنا.\nيرجى الرد بـ 'yes' أو 'no'.""",

    # Persian
    """لطفاً تعیین کنید که آیا توییت زیر حاوی هر گونه {emotion} است یا خیر. اگر هست، پاسخ دهید 'yes'. اگر نیست، پاسخ دهید 'no'.\nتوییت: "{tweet}". توییت در اینجا به پایان می‌رسد.\nلطفاً با 'yes' یا 'no' پاسخ دهید.""",

    # Hebrew
    """אנא קבעו אם הציוץ הבא מכיל כל סוג של {emotion}. אם כן, אנא השיבו 'yes'. אם לא, השיבו 'no'.\nציוץ: "{tweet}". הציוץ מסתיים כאן.\nאנא השיבו ב-'yes' או 'no'.""",

    # Turkish
    """Lütfen aşağıdaki tweetin herhangi bir {emotion} içerip içermediğini belirleyin. Eğer içeriyorsa 'yes' yanıtını verin. Eğer içermiyorsa 'no' yanıtını verin.\nTweet: "{tweet}". Tweet burada bitiyor.\nLütfen 'yes' veya 'no' olarak yanıtlayın.""",

    # Japanese
    """以下のツイートに {emotion} のいずれかの形態が含まれているかどうかを判断してください。含まれている場合は「yes」と答えてください。含まれていない場合は「no」と答えてください。\nツイート: "{tweet}"。これでツイートは終了します。\n「yes」または「no」で回答してください。""",

    # Korean
    """다음 트윗에 {emotion} 형태가 포함되어 있는지 확인하십시오. 포함되어 있다면 'yes'라고 응답하십시오. 포함되어 있지 않다면 'no'라고 응답하십시오.\n트윗: "{tweet}". 트윗은 여기서 끝납니다.\n'yes' 또는 'no'로 응답해 주십시오.""",

    # Vietnamese
    """Vui lòng xác định xem tweet dưới đây có chứa bất kỳ dạng {emotion} nào không. Nếu có, vui lòng trả lời 'yes'. Nếu không, vui lòng trả lời 'no'.\nTweet: "{tweet}". Tweet kết thúc ở đây.\nVui lòng trả lời 'yes' hoặc 'no'.""",

    # Thai
    """โปรดระบุว่าทวีตด้านล่างนี้มีรูปแบบใด ๆ ของ {emotion} หรือไม่ หากมี โปรดตอบกลับ 'yes' หากไม่มี โปรดตอบกลับ 'no'\nทวีต: "{tweet}" ทวีตสิ้นสุดที่นี่\nโปรดตอบกลับด้วย 'yes' หรือ 'no'.""",

    # Indonesian
    """Tolong tentukan apakah tweet di bawah ini mengandung bentuk {emotion} apa pun. Jika ya, silakan jawab 'yes'. Jika tidak, silakan jawab 'no'.\nTweet: "{tweet}". Tweet berakhir di sini.\nSilakan jawab dengan 'yes' atau 'no'.""",

    # Malay
    """Sila tentukan sama ada tweet di bawah mengandungi sebarang bentuk {emotion}. Jika ada, sila jawab 'yes'. Jika tidak, sila jawab 'no'.\nTweet: "{tweet}". Tweet tamat di sini.\nSila jawab dengan 'yes' atau 'no'.""",

    # Lao
    """ກະລຸນາກວດສອບວ່າເນື້ອໃນທີ່ສົນໄຊຂອງແບບຟອມຂອງ {emotion} ມີໃນໃດຫຼືບໍ່ມີໃນບົດວິຈານທີ່ສົນໄຊນີ້. ຖ້າມີກະລຸນາຕອບກັບ 'yes'. ຖ້າບໍ່ມີກະລຸນາຕອບກັບ 'no'.\nຂໍ້ຄວາມ: "{tweet}". ນີ້ແມ່ນສີ່ແຫ່ງທີ່ຈະສິ້ນສຸດ.\nກະລຸນາຕອບກັບ 'yes' ຫຼື 'no'.""",

    # Burmese
    """အောက်ပါတွစ်တာတွင် {emotion} ၏ မည်သည့်အမျိုးအစားမျှ ပါဝင်ကြောင်းသေချာအောင် ဆန်းစစ်ပါ။ ပါဝင်ပါက 'yes' ဟုပြန်လည်ဖြေကြပါ။ ပါဝင်မရှိပါက 'no' ဟုပြန်လည်ဖြေကြပါ။\nတွစ်တာ: "{tweet}"။ အဆိုပါတွစ်တာသည် ဤနေရာတွင်ဆုံးပြီးပါပြီ။\n'yes' သို့မဟုတ် 'no' ဟုပြန်လည်ဖြေကြပါ။""",

    # Cebuano
    """Palihug tukma ang kung ang gipakete nga tweet sa ubos adunay bisan unsang porma sa {emotion}. Kung naa, palihug itubag 'yes'. Kung wala, palihug itubag 'no'.\nTweet: "{tweet}". Dinhi nagtakop ang tweet.\nPalihug itubag 'yes' o 'no'.""",

    # Khmer
    """សូមកំណត់ថាតើប្រកាសដែលបានផ្ដល់នៅខាងក្រោមមានរាងអារម្មណ៍ {emotion} ណាមួយទេ។ បើមាន សូមឆ្លើយតប 'yes' បើគ្មាន សូមឆ្លើយតប 'no'\nប្រកាស: "{tweet}"។ ប្រកាសនេះបានបញ្ចប់នៅទីនេះ។\nសូមឆ្លើយតបដោយ 'yes' ឬ 'no'""",

    # Tagalog
    """Pakisuri kung ang ibinigay na tweet sa ibaba ay naglalaman ng anumang anyo ng {emotion}. Kung oo, mangyaring tumugon ng 'yes'. Kung hindi, mangyaring tumugon ng 'no'.\nTweet: "{tweet}". Nagtatapos na ang tweet.\nPakisagot ng 'yes' o 'no'.""",

    # Hindi
    """कृपया यह निर्धारित करें कि नीचे दिए गए ट्वीट में {emotion} का कोई रूप शामिल है या नहीं। यदि हां, तो कृपया 'yes' का उत्तर दें। यदि नहीं, तो कृपया 'no' का उत्तर दें।\nट्वीट: "{tweet}"। अब ट्वीट समाप्त होता है।\nकृपया 'yes' या 'no' के साथ उत्तर दें।""",

    # Bengali
    """দয়া করে পরীক্ষা করুন যে নীচে দেওয়া টুইটে {emotion} এর কোনও রূপ রয়েছে কিনা। যদি থাকে, অনুগ্রহ করে 'yes' বলে উত্তর দিন। যদি না থাকে, অনুগ্রহ করে 'no' বলে উত্তর দিন।\nটুইট: "{tweet}"। টুইটটি এখানে শেষ হয়।\nঅনুগ্রহ করে 'yes' বা 'no' দিয়ে উত্তর দিন।""",

    # Urdu
    """براہ کرم یہ تعین کریں کہ آیا نیچے فراہم کردہ ٹویٹ میں {emotion} کی کوئی شکل موجود ہے۔ اگر ایسا ہے تو، براہ کرم 'yes' کا جواب دیں۔ اگر نہیں، تو براہ کرم 'no' کا جواب دیں۔\nٹویٹ: "{tweet}"۔ اب ٹویٹ ختم ہوتا ہے۔\nبراہ کرم 'yes' یا 'no' کے ساتھ جواب دیں۔"""
]


# Function to create instruction based on the selected template
def create_instruction(tweet, emotion):
    instruction_template = random.choice(emotion_templates)
    instruction = instruction_template.format(tweet=tweet, emotion=emotion)
    return instruction

# get a blank dataframe
inst_data = pd.DataFrame(columns=['instruction', 'output'])

for emotion in all_emotions:
    train_datapath = folderpath1 + emotion + "_train.csv"
    valid_datapath = folderpath1 + emotion + "_valid.csv"
    test_datapath = folderpath1 + emotion + "_test.csv"

    train_data = pd.read_csv(train_datapath)
    valid_data = pd.read_csv(valid_datapath)
    test_data = pd.read_csv(test_datapath)

    data = pd.concat([train_data, valid_data, test_data], ignore_index=True)

    for i, row in data.iterrows():
        output = 'yes' if row[emotion] == 1 else 'no'
        instruction = create_instruction(row["text"], emotion)
        inst_data = pd.concat([inst_data, pd.DataFrame({'instruction': [instruction], 'output': [output]})], ignore_index=True)

# collect a sample of n=10000
inst_data = inst_data.sample(n=10000, random_state=42)

# Save the DataFrame to a parquet file
inst_data.to_parquet("../../data/PEH/hurricane_multilingual.parquet", index=False)