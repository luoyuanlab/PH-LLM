# - coding: utf-8 -*-
#  SemEval-2020 Task 12 (Subtask B, C)
# - Paper: https://doi.org/10.18653/v1/2023.acl-short.66
# - Data: https://zenodo.org/records/3950379#.XxZ-aFVKipp

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import pandas as pd
import random
from utils.data_utils import balance_classes

fileA_path = "../../data/SemEval2020Task12/extended_test/test_a_tweets_all.tsv"
fileA_labels_path = "../../data/SemEval2020Task12/extended_test/test_a_labels_all.csv"
fileB_path = "../../data/SemEval2020Task12/extended_test/test_b_tweets_all.tsv"
fileB_labels_path = "../../data/SemEval2020Task12/extended_test/test_b_labels_all.csv"
fileC_path = "../../data/SemEval2020Task12/extended_test/test_c_tweets_all.tsv"
fileC_labels_path = "../../data/SemEval2020Task12/extended_test/test_c_labels_all.csv"
dataA = pd.read_csv (fileA_path, sep = '\t')
dataB = pd.read_csv (fileB_path, sep = '\t')
dataC = pd.read_csv (fileC_path, sep = '\t')

labelsA = pd.read_csv (fileA_labels_path, names = ["id", "label"])
labelsB = pd.read_csv (fileB_labels_path, names = ["id", "label"])
labelsC = pd.read_csv (fileC_labels_path, names = ["id", "label"])
#add labels to data based on id
dataA = dataA.set_index('id')
dataA = dataA.join(labelsA.set_index('id'))
dataB = dataB.set_index('id')
dataB = dataB.join(labelsB.set_index('id'))
dataC = dataC.set_index('id')
dataC = dataC.join(labelsC.set_index('id'))
# Sub B
# Function to create instruction in the target language
def get_instruction(tweet, templates):
    # Randomly select one of the multilingual templates
    instruction_template = random.choice(templates)
    instruction = instruction_template.format(tweet=tweet)
    return instruction

# Function to create output based on the label
def get_output(label):
    return "yes" if label == "TIN" else "no"

# Define multilingual templates for the instruction
multilingual_templates = [
    # Chinese
    "请判断以下提供的冒犯性推文是否针对特定个人、群体或其他对象，如果是，请回答'yes'。如果推文中的冒犯性语言没有特定的目标，请回答'no'。推文: \"{tweet}\"。现在推文结束。请用'yes'或'no'作答。",

    # English
    "Please determine if the provided offensive tweet below is targeted at a specific individual, a group, or others. If so, respond 'yes'. If the tweet's offensive language is not targeted, respond 'no'. Tweet: \"{tweet}\". Now the tweet ends. Please respond with 'yes' or 'no'.",

    # German
    "Bitte bestimmen Sie, ob der unten bereitgestellte beleidigende Tweet auf eine bestimmte Person, Gruppe oder andere gerichtet ist. Wenn ja, antworten Sie mit 'yes'. Wenn die beleidigende Sprache des Tweets nicht auf ein Ziel gerichtet ist, antworten Sie mit 'no'. Tweet: \"{tweet}\". Jetzt endet der Tweet. Bitte antworten Sie mit 'yes' oder 'no'.",

    # French
    "Veuillez déterminer si le tweet offensant fourni ci-dessous est dirigé contre une personne, un groupe ou d'autres cibles spécifiques. Si c'est le cas, répondez 'yes'. Si le langage offensant du tweet n'est pas ciblé, répondez 'no'. Tweet : \"{tweet}\". Maintenant, le tweet se termine. Veuillez répondre par 'yes' ou 'no'.",

    # Spanish
    "Por favor, determine si el tweet ofensivo proporcionado a continuación está dirigido a una persona específica, un grupo u otros. Si es así, responda 'yes'. Si el lenguaje ofensivo del tweet no está dirigido, responda 'no'. Tweet: \"{tweet}\". Ahora el tweet termina. Responda con 'yes' o 'no'.",

    # Portuguese
    "Por favor, determine se o tweet ofensivo fornecido abaixo é direcionado a um indivíduo específico, um grupo ou outros. Se sim, responda 'yes'. Se a linguagem ofensiva do tweet não for direcionada, responda 'no'. Tweet: \"{tweet}\". Agora o tweet termina. Por favor, responda com 'yes' ou 'no'.",

    # Italian
    "Si prega di determinare se il tweet offensivo fornito di seguito è rivolto a un individuo specifico, a un gruppo o ad altri. In tal caso, rispondere 'yes'. Se il linguaggio offensivo del tweet non è mirato, rispondere 'no'. Tweet: \"{tweet}\". Ora il tweet finisce. Rispondere con 'yes' o 'no'.",

    # Dutch
    "Bepaal of de onderstaande beledigende tweet gericht is op een specifieke persoon, groep of anderen. Zo ja, antwoord dan met 'yes'. Als het beledigende taalgebruik van de tweet niet gericht is, antwoord dan met 'no'. Tweet: \"{tweet}\". Nu eindigt de tweet. Antwoord alstublieft met 'yes' of 'no'.",

    # Russian
    "Пожалуйста, определите, направлен ли приведенный ниже оскорбительный твит на конкретного человека, группу или других лиц. Если да, ответьте 'yes'. Если оскорбительный язык твита не направлен на цель, ответьте 'no'. Твит: \"{tweet}\". Теперь твит заканчивается. Пожалуйста, ответьте 'yes' или 'no'.",

    # Czech
    "Prosím, určete, zda je níže uvedený urážlivý tweet zaměřen na konkrétní osobu, skupinu nebo jiné. Pokud ano, odpovězte 'yes'. Pokud urážlivý jazyk tweetu není zaměřen na cíl, odpovězte 'no'. Tweet: \"{tweet}\". Nyní tweet končí. Odpovězte prosím 'yes' nebo 'no'.",

    # Polish
    "Proszę określić, czy poniższy obraźliwy tweet jest skierowany do konkretnej osoby, grupy lub innych. Jeśli tak, odpowiedz 'yes'. Jeśli obraźliwy język tweeta nie jest skierowany, odpowiedz 'no'. Tweet: \"{tweet}\". Teraz tweet się kończy. Odpowiedz proszę 'yes' lub 'no'.",

    # Arabic
    "يرجى تحديد ما إذا كانت التغريدة الهجومية المقدمة أدناه موجهة إلى شخص معين أو مجموعة أو غيرهم. إذا كان الأمر كذلك، استجب بـ 'yes'. إذا لم تكن لغة التغريدة الهجومية موجهة، استجب بـ 'no'. التغريدة: \"{tweet}\". الآن انتهت التغريدة. يرجى الرد بـ 'yes' أو 'no'.",

    # Persian
    "لطفاً تعیین کنید که آیا توییت توهین‌آمیز زیر به یک فرد خاص، گروه یا دیگران هدف‌گیری شده است یا خیر. اگر چنین است، پاسخ 'yes' دهید. اگر زبان توهین‌آمیز توییت هدف‌گیری نشده باشد، پاسخ 'no' دهید. توییت: \"{tweet}\". اکنون توییت به پایان می‌رسد. لطفاً با 'yes' یا 'no' پاسخ دهید.",

    # Hebrew
    "אנא קבע אם הציוץ הפוגעני המסופק למטה מכוון לאדם ספציפי, לקבוצה או לאחרים. אם כן, השב 'yes'. אם שפת הציוץ הפוגענית אינה מכוונת, השב 'no'. ציוץ: \"{tweet}\". הציוץ נגמר כעת. אנא השב 'yes' או 'no'.",

    # Turkish
    "Lütfen aşağıdaki saldırgan tweet'in belirli bir bireyi, grubu veya diğerlerini hedef alıp almadığını belirleyin. Eğer öyleyse, 'yes' yanıtını verin. Tweet'in saldırgan dili hedef alınmamışsa, 'no' yanıtını verin. Tweet: \"{tweet}\". Tweet şimdi sona erdi. Lütfen 'yes' veya 'no' yanıtını verin.",

    # Japanese
    "以下の攻撃的なツイートが特定の個人、グループ、または他の対象をターゲットにしているかどうかを判断してください。そうであれば、「yes」と答えてください。ツイートの攻撃的な言葉がターゲットではない場合は、「no」と答えてください。ツイート：「{tweet}」。これでツイートは終了です。「yes」または「no」で答えてください。",

    # Korean
    "아래에 제공된 공격적인 트윗이 특정 개인, 그룹 또는 다른 사람들을 대상으로 하는지 판단하세요. 그렇다면 'yes'라고 답변하세요. 트윗의 공격적인 언어가 타겟팅되지 않았다면 'no'라고 답변하세요. 트윗: \"{tweet}\". 이제 트윗이 끝났습니다. 'yes' 또는 'no'로 답변해 주세요.",

    # Vietnamese
    "Vui lòng xác định xem tweet xúc phạm được cung cấp bên dưới có nhằm vào một cá nhân cụ thể, một nhóm hoặc những người khác không. Nếu có, hãy trả lời 'yes'. Nếu ngôn từ xúc phạm trong tweet không có mục tiêu, hãy trả lời 'no'. Tweet: \"{tweet}\". Bây giờ tweet kết thúc. Vui lòng trả lời bằng 'yes' hoặc 'no'.",

    # Thai
    "โปรดตรวจสอบว่าทวีตที่ไม่เหมาะสมที่ให้ไว้ด้านล่างมีเป้าหมายไปที่บุคคลเฉพาะ กลุ่ม หรือคนอื่นๆ หรือไม่ หากเป็นเช่นนั้น ให้ตอบ 'yes' หากภาษาที่ไม่เหมาะสมของทวีตไม่ได้ถูกกำหนดเป้าหมาย โปรดตอบ 'no' ทวีต: \"{tweet}\" ตอนนี้ทวีตสิ้นสุดแล้ว โปรดตอบ 'yes' หรือ 'no'",

    # Indonesian
    "Silakan tentukan apakah tweet ofensif yang diberikan di bawah ini ditujukan pada individu tertentu, grup, atau lainnya. Jika demikian, jawab 'yes'. Jika bahasa ofensif tweet tidak ditargetkan, jawab 'no'. Tweet: \"{tweet}\". Sekarang tweet berakhir. Silakan jawab dengan 'yes' atau 'no'.",

    # Malay
    "Sila tentukan sama ada tweet ofensif yang diberikan di bawah ini disasarkan kepada individu tertentu, kumpulan, atau lain-lain. Jika ya, jawab 'yes'. Jika bahasa ofensif tweet tidak disasarkan, jawab 'no'. Tweet: \"{tweet}\". Sekarang tweet tamat. Sila jawab dengan 'yes' atau 'no'.",

    # Lao
    "ກະລຸນາກຳນົດວ່າຂໍ້ຄວາມ tweet ທີ່ມີຄວາມບໍ່ຖືກຕ້ອງທີ່ໃຫ້ມາດ້ານລຸ່ມນີ້ມີຕົ້ນຕໍທີ່ຈະຊັກຈູງເຂົ້າເຖິງບຸກຄົນທີ່ກໍານົດ, ກຸ່ມ, ຫຼືຄົນອື່ນໆ ຫຼືບໍ່. ຖ້າວ່າແມ່ນ, ກະລຸນາຕອບ 'yes'. ຖ້າວ່າພາສາທີ່ບໍ່ຖືກຕ້ອງຂອງ tweet ບໍ່ໄດ້ຖືກກໍານົດເປົ້າໝາຍ, ກະລຸນາຕອບ 'no'. Tweet: \"{tweet}\". ຂໍ້ຄວາມຫມາຍ tweet ໄດ້ສິ້ນສຸດລົງແລ້ວ. ກະລຸນາຕອບດ້ວຍ 'yes' ຫຼື 'no'.",

    # Burmese
    "အောက်တွင်ပေးထားသော အပြစ်ရှိသော တူဿ်သည် တစ်ဦးချင်းနှင့် သတ်မှတ်ထားသော ပစ်မှတ်၊ အုပ်စု သို့မဟုတ် အခြားသူများကို ပစ်မှတ်ထားခြင်းဖြစ်သည်ဟု သတ်မှတ်ပါ။ အကယ်၍ သတ်မှတ်ထားပါက 'yes' ဖြင့် ဖြေကြားပါ။ တူဿ်၏ အပြစ်ရှိသော ဘာသာစကားသည် ပစ်မှတ်မထားပါက 'no' ဖြင့် ဖြေကြားပါ။ တူဿ်: \"{tweet}\". ယခုတွင် တူဿ် ပြီးဆုံးပါပြီ။ 'yes' သို့မဟုတ် 'no' ဖြင့် ပြန်ကြားပါ။",

    # Cebuano
    "Palihug pag-determinar kung ang gihatag nga malisyoso nga tweet sa ubos gilantaw sa usa ka piho nga indibidwal, usa ka grupo, o uban pa. Kung mao, tubaga ang 'yes'. Kung ang dili maayo nga pinulongan sa tweet wala maglambigit, tubaga ang 'no'. Tweet: \"{tweet}\". Karon ang tweet mohuman na. Palihug motubag og 'yes' o 'no'.",

    # Khmer
    "សូមកំណត់ថាតើការប្រកាសខាងក្រោមដែលអាក្រក់នេះត្រូវបានផ្តោតលើជនបុគ្គលជាក់លាក់ក្រុមមួយឬអ្នកដទៃ។ ប្រសិនបើមាន សូមឆ្លើយថា 'yes' ។ ប្រសិនបើភាសាអាក្រក់នៃការប្រកាសមិនត្រូវបានផ្តោតសូមឆ្លើយថា 'no' ។ Tweet: \"{tweet}\" ។ ឥឡូវនេះការប្រកាសបានបញ្ចប់ហើយ សូមឆ្លើយថា 'yes' ឬ 'no'.",

    # Tagalog
    "Pakitukoy kung ang ibinigay na bastos na tweet sa ibaba ay nakatuon sa isang partikular na indibidwal, isang grupo, o iba pa. Kung oo, sumagot ng 'yes'. Kung ang bastos na wika ng tweet ay hindi nakatuon, sumagot ng 'no'. Tweet: \"{tweet}\". Ngayon natapos na ang tweet. Pakisagot ng 'yes' o 'no'.",

    # Hindi
    "कृपया यह निर्धारित करें कि नीचे दिया गया आपत्तिजनक ट्वीट किसी विशिष्ट व्यक्ति, समूह या अन्य लोगों को लक्षित करता है या नहीं। यदि हां, तो 'yes' के साथ उत्तर दें। यदि ट्वीट की आपत्तिजनक भाषा लक्षित नहीं है, तो 'no' के साथ उत्तर दें। ट्वीट: \"{tweet}\"। अब ट्वीट समाप्त हो गया है। कृपया 'yes' या 'no' के साथ उत्तर दें।",

    # Bengali
    "নীচে দেওয়া আপত্তিকর টুইটটি কোনও নির্দিষ্ট ব্যক্তি, গোষ্ঠী বা অন্যদের লক্ষ্য করে কিনা তা নির্ধারণ করুন। যদি তা হয়, তাহলে 'yes' দিয়ে উত্তর দিন। যদি টুইটের আপত্তিকর ভাষা লক্ষ্যবস্তু না হয়, তাহলে 'no' দিয়ে উত্তর দিন। টুইট: \"{tweet}\"। এখন টুইট শেষ। অনুগ্রহ করে 'yes' বা 'no' দিয়ে উত্তর দিন।",

    # Urdu
    "براہ کرم تعین کریں کہ آیا نیچے دیا گیا توہین آمیز ٹویٹ کسی خاص فرد، گروپ، یا دوسروں کو نشانہ بنا رہا ہے۔ اگر ایسا ہے تو، 'yes' کے ساتھ جواب دیں۔ اگر ٹویٹ کی توہین آمیز زبان کو نشانہ نہیں بنایا گیا ہے، تو 'no' کے ساتھ جواب دیں۔ ٹویٹ: \"{tweet}\"۔ اب ٹویٹ ختم ہو گیا ہے۔ براہ کرم 'yes' یا 'no' کے ساتھ جواب دیں۔"
]


# Create a blank dataframe
inst_data = pd.DataFrame(columns=['instruction', 'output'])

for i, row in enumerate(dataB.iterrows()):
    tweet = row[1]['tweet']
    output = get_output(row[1]["label"])
    instruction = get_instruction(tweet, multilingual_templates)
    inst_data = pd.concat([inst_data, pd.DataFrame({'instruction': [instruction], 'output': [output]})], ignore_index=True)

# collect a sample of n=2100
inst_data = inst_data.sample(n=2100, random_state=42).reset_index(drop=True)

# Save the dataframe to a parquet file
inst_data.to_parquet("../../data/SemEval2020Task12/SemEval2020Task12subB_multilingual.parquet", index=False)

# Sub C

# Define multilingual templates for the instruction
multilingual_templates = [
    # Chinese
    "请判断以下提供的攻击性推文是否针对个人、群体或其他人。如果是个人，请回答'IND'。如果是群体，请回答'GRP'。如果是其他人，请回答'OTH'。推文: \"{tweet}\"。现在推文结束。请用'IND'、'GRP'或'OTH'作答。",

    # English
    "Please determine if the provided offensive tweet below targets an individual, a group, or others. If individual, respond 'IND'. If Group, respond 'GRP'. If others, respond 'OTH'. Tweet: \"{tweet}\". Now the tweet ends. Please respond with 'IND', 'GRP', or 'OTH'.",

    # German
    "Bitte bestimmen Sie, ob der unten bereitgestellte beleidigende Tweet auf eine Einzelperson, eine Gruppe oder andere abzielt. Wenn es sich um eine Einzelperson handelt, antworten Sie mit 'IND'. Wenn es sich um eine Gruppe handelt, antworten Sie mit 'GRP'. Wenn es sich um andere handelt, antworten Sie mit 'OTH'. Tweet: \"{tweet}\". Jetzt endet der Tweet. Bitte antworten Sie mit 'IND', 'GRP' oder 'OTH'.",

    # French
    "Veuillez déterminer si le tweet offensant ci-dessous cible un individu, un groupe ou d'autres. Si c'est un individu, répondez 'IND'. Si c'est un groupe, répondez 'GRP'. Si c'est d'autres, répondez 'OTH'. Tweet : \"{tweet}\". Maintenant, le tweet se termine. Veuillez répondre par 'IND', 'GRP' ou 'OTH'.",

    # Spanish
    "Por favor, determine si el tweet ofensivo proporcionado a continuación está dirigido a un individuo, un grupo u otros. Si es un individuo, responda 'IND'. Si es un grupo, responda 'GRP'. Si es otros, responda 'OTH'. Tweet: \"{tweet}\". Ahora el tweet termina. Responda con 'IND', 'GRP' o 'OTH'.",

    # Portuguese
    "Por favor, determine se o tweet ofensivo fornecido abaixo tem como alvo um indivíduo, um grupo ou outros. Se for um indivíduo, responda 'IND'. Se for um grupo, responda 'GRP'. Se for outros, responda 'OTH'. Tweet: \"{tweet}\". Agora o tweet termina. Por favor, responda com 'IND', 'GRP' ou 'OTH'.",

    # Italian
    "Si prega di determinare se il tweet offensivo fornito di seguito è mirato a un individuo, un gruppo o altri. Se è un individuo, rispondere 'IND'. Se è un gruppo, rispondere 'GRP'. Se è altri, rispondere 'OTH'. Tweet: \"{tweet}\". Ora il tweet finisce. Rispondere con 'IND', 'GRP' o 'OTH'.",

    # Dutch
    "Bepaal of de onderstaande beledigende tweet gericht is op een individu, een groep of anderen. Als het een individu is, antwoord dan met 'IND'. Als het een groep is, antwoord dan met 'GRP'. Als het anderen betreft, antwoord dan met 'OTH'. Tweet: \"{tweet}\". Nu eindigt de tweet. Antwoord alstublieft met 'IND', 'GRP' of 'OTH'.",

    # Russian
    "Пожалуйста, определите, направлен ли приведенный ниже оскорбительный твит на конкретного человека, группу или других лиц. Если на конкретного человека, ответьте 'IND'. Если на группу, ответьте 'GRP'. Если на других, ответьте 'OTH'. Твит: \"{tweet}\". Теперь твит заканчивается. Пожалуйста, ответьте 'IND', 'GRP' или 'OTH'.",

    # Czech
    "Prosím, určete, zda následující urážlivý tweet cílí na jednotlivce, skupinu nebo jiné. Pokud na jednotlivce, odpovězte 'IND'. Pokud na skupinu, odpovězte 'GRP'. Pokud na jiné, odpovězte 'OTH'. Tweet: \"{tweet}\". Nyní tweet končí. Odpovězte prosím 'IND', 'GRP' nebo 'OTH'.",

    # Polish
    "Proszę określić, czy poniższy obraźliwy tweet jest skierowany na osobę, grupę czy inne osoby. Jeśli na osobę, odpowiedz 'IND'. Jeśli na grupę, odpowiedz 'GRP'. Jeśli na inne osoby, odpowiedz 'OTH'. Tweet: \"{tweet}\". Teraz tweet się kończy. Odpowiedz proszę 'IND', 'GRP' lub 'OTH'.",

    # Arabic
    "يرجى تحديد ما إذا كانت التغريدة المسيئة أدناه تستهدف فردًا أو مجموعة أو غيرهم. إذا كان فردًا، فاستجب بـ 'IND'. إذا كانت مجموعة، فاستجب بـ 'GRP'. إذا كان آخرون، فاستجب بـ 'OTH'. التغريدة: \"{tweet}\". الآن انتهت التغريدة. يرجى الرد بـ 'IND' أو 'GRP' أو 'OTH'.",

    # Persian
    "لطفاً تعیین کنید که آیا توییت توهین آمیز زیر به یک فرد، یک گروه یا دیگران هدف می‌گیرد. اگر فرد است، پاسخ 'IND' دهید. اگر گروه است، پاسخ 'GRP' دهید. اگر دیگران، پاسخ 'OTH' دهید. توییت: \"{tweet}\". اکنون توییت به پایان می‌رسد. لطفاً با 'IND'، 'GRP' یا 'OTH' پاسخ دهید.",

    # Hebrew
    "אנא קבע אם הציוץ הפוגעני שלהלן מכוון לאדם יחיד, קבוצה או אחרים. אם מדובר באדם יחיד, השב 'IND'. אם מדובר בקבוצה, השב 'GRP'. אם מדובר באחרים, השב 'OTH'. ציוץ: \"{tweet}\". הציוץ נגמר כעת. אנא השב 'IND', 'GRP' או 'OTH'.",

    # Turkish
    "Lütfen aşağıdaki saldırgan tweet'in bir bireyi, bir grubu veya diğerlerini hedef alıp almadığını belirleyin. Eğer bireyse, 'IND' yanıtını verin. Eğer grup ise, 'GRP' yanıtını verin. Diğerleri ise, 'OTH' yanıtını verin. Tweet: \"{tweet}\". Tweet şimdi sona erdi. Lütfen 'IND', 'GRP' veya 'OTH' yanıtını verin.",

    # Japanese
    "以下の攻撃的なツイートが個人、グループ、または他の人を対象としているかどうかを判断してください。個人の場合は「IND」と答えてください。グループの場合は「GRP」と答えてください。他の場合は「OTH」と答えてください。ツイート：「{tweet}」。これでツイートは終了です。「IND」、「GRP」、または「OTH」で答えてください。",

    # Korean
    "아래 제공된 공격적인 트윗이 개인, 그룹 또는 다른 사람을 대상으로 하는지 확인해 주세요. 개인이라면 'IND'라고 답변하세요. 그룹이라면 'GRP'라고 답변하세요. 다른 경우에는 'OTH'라고 답변하세요. 트윗: \"{tweet}\". 이제 트윗이 끝났습니다. 'IND', 'GRP' 또는 'OTH'로 답변해 주세요.",

    # Vietnamese
    "Vui lòng xác định xem tweet xúc phạm được cung cấp dưới đây có nhắm mục tiêu vào một cá nhân, một nhóm hay những người khác không. Nếu là cá nhân, hãy trả lời 'IND'. Nếu là nhóm, hãy trả lời 'GRP'. Nếu là người khác, hãy trả lời 'OTH'. Tweet: \"{tweet}\". Bây giờ tweet kết thúc. Vui lòng trả lời bằng 'IND', 'GRP' hoặc 'OTH'.",

    # Thai
    "โปรดตรวจสอบว่าทวีตที่ให้ไว้ด้านล่างนี้มุ่งเป้าไปที่บุคคล กลุ่ม หรือคนอื่นๆ หรือไม่ หากเป็นบุคคล ให้ตอบ 'IND' หากเป็นกลุ่ม ให้ตอบ 'GRP' หากเป็นคนอื่นๆ ให้ตอบ 'OTH' ทวีต: \"{tweet}\" ตอนนี้ทวีตสิ้นสุดแล้ว โปรดตอบ 'IND' 'GRP' หรือ 'OTH'",

    # Indonesian
    "Silakan tentukan apakah tweet ofensif yang disediakan di bawah ini menargetkan individu, kelompok, atau lainnya. Jika individu, jawab 'IND'. Jika grup, jawab 'GRP'. Jika lainnya, jawab 'OTH'. Tweet: \"{tweet}\". Sekarang tweet berakhir. Silakan jawab dengan 'IND', 'GRP', atau 'OTH'.",

    # Malay
    "Sila tentukan sama ada tweet ofensif yang diberikan di bawah ini menyasarkan individu, kumpulan atau orang lain. Jika individu, jawab 'IND'. Jika Kumpulan, jawab 'GRP'. Jika lain-lain, jawab 'OTH'. Tweet: \"{tweet}\". Sekarang tweet tamat. Sila jawab dengan 'IND', 'GRP', atau 'OTH'.",

    # Lao
    "ກະລຸນາກຳນົດວ່າຂໍ້ຄວາມທີ່ບໍ່ສຸພາບທີ່ສະເໜີຂ້າງລຸ່ມນີ້ເປັນການຕັ້ງເປົ້າໝາຍເອົາບຸກຄົນ, ກຸ່ມ, ຫຼື ອື່ນໆ. ຖ້າວ່າເປັນບຸກຄົນ, ກະລຸນາຕອບ 'IND'. ຖ້າວ່າເປັນກຸ່ມ, ກະລຸນາຕອບ 'GRP'. ຖ້າວ່າເປັນອື່ນໆ, ກະລຸນາຕອບ 'OTH'. Tweet: \"{tweet}\". ຂໍ້ຄວາມນີ້ສິ້ນສຸດລົງແລ້ວ. ກະລຸນາຕອບດ້ວຍ 'IND', 'GRP' ຫຼື 'OTH'.",

    # Burmese
    "အောက်တွင်ပေးထားသော စော်ကားမှု ပြုလုပ်သော တူဿ်သည် ပစ်မှတ်ထားသောအနေဖြင့် ဖြစ်ပါက၊ ဥပမာ အပစ်မှတ်ထားသော သီးခြား၊ အဖွဲ့အစည်း သို့မဟုတ် အခြားသူများကို ဖြည့်ရန်ဖြစ်ပါက 'IND' ဖြင့် ဖြေကြားပါ။ အဖွဲ့ကို ဖြည့်ရန်ဖြစ်ပါက 'GRP' ဖြင့် ဖြေကြားပါ။ အခြားများကို ဖြည့်ရန်ဖြစ်ပါက 'OTH' ဖြင့် ပြန်ကြားပါ။ တူဿ်: \"{tweet}\". ယခုတွင် တူဿ် ပြီးဆုံးပါပြီ။ 'IND' 'GRP' သို့မဟုတ် 'OTH' ဖြင့် ပြန်ကြားပါ။",

    # Cebuano
    "Palihug pag-determinar kung ang gihatag nga malisyoso nga tweet sa ubos gipuntirya ba ang usa ka indibidwal, usa ka grupo, o uban pa. Kung indibidwal, tubaga ang 'IND'. Kung Grupo, tubaga ang 'GRP'. Kung uban pa, tubaga ang 'OTH'. Tweet: \"{tweet}\". Karon ang tweet mohuman na. Palihug motubag og 'IND', 'GRP' o 'OTH'.",

    # Khmer
    "សូមកំណត់ថាតើការប្រកាសផ្អែកលើការលោភលើបុគ្គល ក្រុមឬផ្សេងទៀតដែលផ្តល់ដោយស្វ័យប្រវត្តិ ឬដោយផ្ទាល់។ ប្រសិនបើលោភលើបុគ្គល សូមឆ្លើយថា 'IND'។ ប្រសិនបើលោភលើក្រុម សូមឆ្លើយថា 'GRP'។ ប្រសិនបើលោភលើផ្សេងទៀត សូមឆ្លើយថា 'OTH'។ ការប្រកាស: \"{tweet}\"។ ឥឡូវនេះការប្រកាសបានបញ្ចប់ហើយ សូមឆ្លើយថា 'IND' 'GRP' ឬ 'OTH'.",

    # Tagalog
    "Pakitukoy kung ang ibinigay na tweet sa ibaba ay nagta-target sa isang indibidwal, isang grupo, o iba pa. Kung indibidwal, sumagot ng 'IND'. Kung Grupo, sumagot ng 'GRP'. Kung iba pa, sumagot ng 'OTH'. Tweet: \"{tweet}\". Ngayon natapos na ang tweet. Pakisagot ng 'IND', 'GRP' o 'OTH'.",

    # Hindi
    "कृपया यह निर्धारित करें कि नीचे दिया गया आपत्तिजनक ट्वीट किसी व्यक्ति, समूह, या अन्य को लक्षित करता है। यदि व्यक्ति को लक्षित करता है, तो 'IND' के साथ उत्तर दें। यदि समूह को लक्षित करता है, तो 'GRP' के साथ उत्तर दें। यदि अन्य को लक्षित करता है, तो 'OTH' के साथ उत्तर दें। ट्वीट: \"{tweet}\"। अब ट्वीट समाप्त हो गया है। कृपया 'IND', 'GRP' या 'OTH' के साथ उत्तर दें।",

    # Bengali
    "নীচে দেওয়া অপমানজনক টুইটটি কোনও ব্যক্তি, একটি গোষ্ঠী বা অন্যদের লক্ষ্য করে কিনা তা নির্ধারণ করুন। যদি এটি একজন ব্যক্তির লক্ষ্য হয়, তাহলে 'IND' দিয়ে উত্তর দিন। যদি এটি একটি গোষ্ঠীর লক্ষ্য হয়, তাহলে 'GRP' দিয়ে উত্তর দিন। যদি এটি অন্যদের লক্ষ্য হয়, তাহলে 'OTH' দিয়ে উত্তর দিন। টুইট: \"{tweet}\"। এখন টুইট শেষ। অনুগ্রহ করে 'IND', 'GRP' বা 'OTH' দিয়ে উত্তর দিন।",

    # Urdu
    "براہ کرم تعین کریں کہ آیا نیچے دیا گیا توہین آمیز ٹویٹ کسی فرد، گروپ یا دوسروں کو نشانہ بناتا ہے۔ اگر فرد کو نشانہ بناتا ہے، تو 'IND' کے ساتھ جواب دیں۔ اگر گروپ کو نشانہ بناتا ہے، تو 'GRP' کے ساتھ جواب دیں۔ اگر دوسروں کو نشانہ بناتا ہے، تو 'OTH' کے ساتھ جواب دیں۔ ٹویٹ: \"{tweet}\"۔ اب ٹویٹ ختم ہو گیا ہے۔ براہ کرم 'IND'، 'GRP' یا 'OTH' کے ساتھ جواب دیں۔"
]

# Create a blank dataframe
inst_data = pd.DataFrame(columns=['instruction', 'output'])

for i, row in enumerate(dataC.iterrows()):
    tweet = row[1]['tweet']
    output = row[1]["label"]
    instruction = get_instruction(tweet, multilingual_templates)
    inst_data = pd.concat([inst_data, pd.DataFrame({'instruction': [instruction], 'output': [output]})], ignore_index=True)

inst_data = balance_classes(inst_data, 300, 'output')

inst_data.to_parquet("../../data/SemEval2020Task12/SemEval2020Task12subC_multilingual.parquet", index=False)