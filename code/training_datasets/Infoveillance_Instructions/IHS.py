# -*- coding: utf-8 -*-
# IHS (Indonesian Hate Speech)  
# - Paper: https://doi.org/10.18653/v1/W19-3506
# - Data: https://github.com/okkyibrohim/id-multi-label-hate-speech-and-abusive-language-detection

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import pandas as pd
import random
from utils.data_utils import downsample_to_minority_class

raw_data = pd.read_csv("../../data/IHS/re_dataset.csv", encoding = 'cp1252')

# List of prompts for all languages
prompts = [
    # Chinese
    "请确定这条推文是否包含仇恨言论。仇恨言论是针对某人或群体的直接或间接言论，包含基于该人或群体固有特征的仇恨。推文内容：{}。推文结束。请回答‘yes’如果推文包含仇恨言论。否则，请回答‘no’（用英文）。",

    # English
    "Please determine if the tweet contains hate speech. Hate speech is a direct or indirect speech toward a person or group containing hatred based on something inherent to that person or group. Tweet: {}. Now the tweet ends. Please respond 'yes' if the tweet contains hate speech. Otherwise, respond 'no'.",

    # German
    "Bitte bestimmen Sie, ob der Tweet Hassrede enthält. Hassrede ist eine direkte oder indirekte Äußerung gegenüber einer Person oder Gruppe, die aufgrund von etwas, das dieser Person oder Gruppe innewohnt, Hass enthält. Tweet: {}. Der Tweet endet jetzt. Bitte antworten Sie mit 'yes', wenn der Tweet Hassrede enthält. Andernfalls antworten Sie mit 'no' (auf Englisch).",

    # French
    "Veuillez déterminer si le tweet contient un discours de haine. Le discours de haine est une expression directe ou indirecte envers une personne ou un groupe contenant de la haine basée sur quelque chose d'intrinsèque à cette personne ou à ce groupe. Tweet: {}. Le tweet se termine maintenant. Veuillez répondre 'yes' si le tweet contient un discours de haine. Sinon, répondez 'no' (en anglais).",

    # Spanish
    "Por favor, determine si el tweet contiene discurso de odio. El discurso de odio es un discurso directo o indirecto hacia una persona o grupo que contiene odio basado en algo inherente a esa persona o grupo. Tweet: {}. El tweet termina ahora. Por favor, responda 'yes' si el tweet contiene discurso de odio. De lo contrario, responda 'no' (en inglés).",

    # Portuguese
    "Por favor, determine se o tweet contém discurso de ódio. O discurso de ódio é uma expressão direta ou indireta dirigida a uma pessoa ou grupo que contém ódio com base em algo inerente a essa pessoa ou grupo. Tweet: {}. O tweet termina agora. Por favor, responda 'yes' se o tweet contém discurso de ódio. Caso contrário, responda 'no' (em inglês).",

    # Italian
    "Si prega di determinare se il tweet contiene discorsi di odio. Il discorso di odio è un discorso diretto o indiretto verso una persona o gruppo che contiene odio basato su qualcosa di inerente a quella persona o gruppo. Tweet: {}. Il tweet termina ora. Si prega di rispondere 'yes' se il tweet contiene discorsi di odio. Altrimenti, rispondere 'no' (in inglese).",

    # Dutch
    "Bepaal of de tweet haatdragende taal bevat. Haatspraak is een directe of indirecte uitlating jegens een persoon of groep die haat bevat op basis van iets dat inherent is aan die persoon of groep. Tweet: {}. Nu eindigt de tweet. Reageer alsjeblieft met 'yes' als de tweet haatdragende taal bevat. Zo niet, antwoord 'no' (in het Engels).",

    # Russian
    "Определите, содержит ли твит ненавистническую речь. Ненавистническая речь — это прямая или косвенная речь в адрес человека или группы, содержащая ненависть на основе чего-то присущего этому человеку или группе. Твит: {}. Теперь твит заканчивается. Пожалуйста, ответьте 'yes', если твит содержит ненавистническую речь. В противном случае ответьте 'no' (на английском).",

    # Czech
    "Určete prosím, zda tweet obsahuje nenávistný projev. Nenávistný projev je přímý nebo nepřímý projev vůči osobě nebo skupině obsahující nenávist na základě něčeho inherentního této osobě nebo skupině. Tweet: {}. Tweet nyní končí. Prosím, odpovězte 'yes', pokud tweet obsahuje nenávistný projev. V opačném případě odpovězte 'no' (v angličtině).",

    # Polish
    "Proszę określić, czy tweet zawiera mowę nienawiści. Mowa nienawiści to bezpośrednia lub pośrednia wypowiedź skierowana do osoby lub grupy, zawierająca nienawiść opartą na czymś, co jest inherentne dla tej osoby lub grupy. Tweet: {}. Tweet teraz się kończy. Proszę odpowiedzieć 'yes', jeśli tweet zawiera mowę nienawiści. W przeciwnym razie odpowiedz 'no' (po angielsku).",

    # Arabic
    "يرجى تحديد ما إذا كانت التغريدة تحتوي على خطاب كراهية. خطاب الكراهية هو خطاب مباشر أو غير مباشر تجاه شخص أو مجموعة يحتوي على كراهية بناءً على شيء متأصل في هذا الشخص أو المجموعة. التغريدة: {} . الآن تنتهي التغريدة. يرجى الرد بـ 'yes' إذا كانت التغريدة تحتوي على خطاب كراهية. خلاف ذلك، الرد بـ 'no' (بالإنجليزية).",

    # Persian
    "لطفاً تعیین کنید که آیا توییت حاوی سخنان نفرت‌آمیز است یا خیر. سخنان نفرت‌آمیز به سخنانی مستقیم یا غیرمستقیم در مورد یک فرد یا گروه اطلاق می‌شود که حاوی نفرت بر اساس چیزی ذاتی در آن فرد یا گروه است. توییت: {} . اکنون توییت به پایان می‌رسد. لطفاً اگر توییت حاوی سخنان نفرت‌آمیز است، با 'yes' پاسخ دهید. در غیر این صورت، با 'no' پاسخ دهید (به انگلیسی).",

    # Hebrew
    "אנא קבע אם הציוץ מכיל דיבור שנאה. דיבור שנאה הוא דיבור ישיר או עקיף כלפי אדם או קבוצה הכולל שנאה על בסיס משהו שטבוע באדם או בקבוצה הזו. ציוץ: {} . הציוץ מסתיים כעת. אנא השיבו 'yes' אם הציוץ מכיל דיבור שנאה. אחרת, השיבו 'no' (באנגלית).",

    # Turkish
    "Lütfen tweet'in nefret söylemi içerip içermediğini belirleyin. Nefret söylemi, bir kişi veya gruba yönelik, o kişiye veya gruba özgü bir şeye dayanan nefret içeren doğrudan veya dolaylı bir söylemdir. Tweet: {} . Tweet şimdi sona eriyor. Tweet nefret söylemi içeriyorsa lütfen 'yes' yanıtını verin. Aksi takdirde, 'no' yanıtını verin (İngilizce olarak).",

    # Japanese
    "ツイートにヘイトスピーチが含まれているかどうかを判断してください。ヘイトスピーチとは、その人やグループに固有のものに基づいて、個人またはグループに向けられた憎悪を含む直接的または間接的な言葉です。ツイート内容: {} 。ツイートはこれで終わります。ツイートにヘイトスピーチが含まれている場合は 'yes' と回答してください。それ以外の場合は 'no' と回答してください（英語で）。",

    # Korean
    "트윗에 혐오 발언이 포함되어 있는지 확인해 주세요. 혐오 발언은 특정 개인 또는 그룹에 고유한 무언가에 대한 증오를 포함하는 직접적이거나 간접적인 발언입니다. 트윗 내용: {} . 이제 트윗이 끝납니다. 트윗에 혐오 발언이 포함되어 있다면 'yes'로 응답해 주세요. 그렇지 않다면 'no'로 응답해 주세요 (영어로).",

    # Vietnamese
    "Vui lòng xác định liệu tweet có chứa ngôn từ kích động thù địch hay không. Ngôn từ kích động thù địch là lời nói trực tiếp hoặc gián tiếp nhằm vào một cá nhân hoặc nhóm chứa đựng sự thù ghét dựa trên điều gì đó vốn có của cá nhân hoặc nhóm đó. Nội dung tweet: {} . Bây giờ tweet kết thúc. Vui lòng trả lời 'yes' nếu tweet chứa ngôn từ kích động thù địch. Nếu không, trả lời 'no' (bằng tiếng Anh).",

    # Thai
    "โปรดพิจารณาว่าทวีตนี้มีเนื้อหาที่เป็นคำพูดที่สร้างความเกลียดชังหรือไม่ คำพูดที่สร้างความเกลียดชังคือคำพูดที่แสดงออกโดยตรงหรือโดยอ้อมไปยังบุคคลหรือกลุ่มที่แสดงความเกลียดชังซึ่งมีพื้นฐานมาจากสิ่งที่มีอยู่ในตัวบุคคลหรือกลุ่มนั้นๆ เนื้อหาทวีต: {} . ขณะนี้ทวีตจบลงแล้ว โปรดตอบ 'yes' หากทวีตมีเนื้อหาที่สร้างความเกลียดชัง หากไม่ใช่ ให้ตอบ 'no' (ภาษาอังกฤษ).",

    # Indonesian
    "Silakan tentukan apakah tweet tersebut mengandung ujaran kebencian. Ujaran kebencian adalah ucapan langsung atau tidak langsung terhadap seseorang atau kelompok yang mengandung kebencian berdasarkan sesuatu yang melekat pada orang atau kelompok tersebut. Tweet: {} . Tweet sekarang berakhir. Silakan jawab 'yes' jika tweet tersebut mengandung ujaran kebencian. Jika tidak, jawab 'no' (dalam bahasa Inggris).",

    # Malay
    "Sila tentukan sama ada tweet tersebut mengandungi ucapan kebencian. Ucapan kebencian adalah ucapan langsung atau tidak langsung terhadap seseorang atau kumpulan yang mengandungi kebencian berdasarkan sesuatu yang wujud pada orang atau kumpulan itu. Tweet: {} . Tweet kini berakhir. Sila jawab 'yes' jika tweet tersebut mengandungi ucapan kebencian. Jika tidak, jawab 'no' (dalam bahasa Inggeris).",

    # Lao
    "ກະລຸນາກຳນົດວ່າທະວີດນີ້ມີຄຳເວົ້າທີ່ເປັນຄວາມເກຽດຊັງຫຼືບໍ່. ຄຳເວົ້າທີ່ເປັນຄວາມເກຽດຊັງແມ່ນຄຳເວົ້າທີ່ພາສາສົ່ງໄປຫາບຸກຄົນຫຼືກຸ່ມຄົນທີ່ມີຄວາມເກຽດຊັງທີ່ມີພື້ນຖານມາຈາກບາງຢ່າງທີ່ມີຢູ່ໃນບຸກຄົນຫຼືກຸ່ມນັ້ນ. ຂໍ້ຄວາມທະວີດ: {} . ທະວີດນີ້ຈະສິ້ນສຸດໃນຕອນນີ້. ກະລຸນາຕອບ 'yes' ຖ້າທະວີດມີຄຳເວົ້າທີ່ເປັນຄວາມເກຽດຊັງ. ຖ້າບໍ່, ຕອບ 'no' (ໃນພາສາອັງກິດ).",

    # Burmese
    "တစ်ကိုယ်ရည်စွမ်းဆောင်ချက်ဖြစ်သည်မှာ ဤတွစ်က ဟိတ်စပီချ်ပါဝင်မပါသောကိုစစ်ဆေးပါသည်. ဟိတ်စပီချ်သည် တစ်ဦးတစ်ယောက်နှင့် ဦးတစ်ဦးလူအုပ်စုတစ်စုတစ်စုကဲ့သို့ တိုက်ရိုက်ပြောဆိုသောစကားလုံးဖြစ်သည်. တွစ်မှာရှိသောစကားလုံးမှာ {} . အဆိုပါစကားလုံးတွေကိုသက်သက်ရှိသည်နှင့်ဆိုင်သောဟိတ်စပီချ်သည် တိုက်ရိုက်ပြောဆိုသောစကားလုံးဖြစ်သည်. ဤတွစ်သည်ယခုဆုံးသည်. အကယ်၍ တွစ်တွင် ဟိတ်စပီချ်ပါဝင်လျှင် 'yes' ဖြင့်ဖြေပေးပါ. ဟုဆက်၍မပြောပါလျှင် 'no' ဖြင့်ဖြေပေးပါ. (အင်္ဂလိပ်စကားလုံးဖြင့်).",

    # Cebuano
    "Palihug tukma kung ang tweet adunay sulod nga pagpanghimaraot nga pulong. Ang pagpanghimaraot nga pulong mao ang direkta o dili direkta nga pagpahayag ngadto sa usa ka tawo o grupo nga naglangkob og kapungot base sa usa ka butang nga sulod sa mao nga tawo o grupo. Tweet: {} . Karon ang tweet natapos na. Palihug tubaga og 'yes' kung ang tweet adunay pagpanghimaraot nga pulong. Kung dili, tubaga og 'no' (sa English).",

    # Khmer
    "សូមកំណត់ថាតើតើតើផ្ទៃក្នុងនេះមានមាតិកាចៅក្រមក្តីសាសនាឬសិទ្ធិព្រួញមាននៅលើផ្ទៃក្នុងនេះ។ ចៅក្រមក្តីសាសនាឬសិទ្ធិព្រួញជាសម្តីផ្ទាល់ ឬអន់ភាពអន់ភាពអន់ភាពដែលសម្រាប់បុគ្គលរឺក្រុមដែលមានសេដ្ឋកិច្ចបែបអក្សរមួយចំនួនដែលត្រូវបានគេសម្រេចចេញនៅលើរបស់នោះ។ ផ្ទៃក្នុងនេះមានអត្ថបទ៖ {} . បច្ចុប្បន្ននេះសារនេះបានបញ្ចប់។ សូមឆ្លើយតបដោយ 'yes' ប្រសិនបើសារមានមាតិកាចៅក្រមក្តីសាសនាឬសិទ្ធិព្រួញរឺទេក្នុងអត្ថបទ (ជាភាសាអង់គ្លេស)។",

    # Tagalog
    "Pakisuri kung ang tweet ay naglalaman ng hate speech. Ang hate speech ay direktang o hindi direktang pananalita patungkol sa isang tao o grupo na naglalaman ng galit na nakabatay sa isang bagay na likas sa taong iyon o grupo. Tweet: {} . Ngayon, natapos na ang tweet. Pakisagot ng 'yes' kung ang tweet ay naglalaman ng hate speech. Kung hindi, sagutin ng 'no' (sa Ingles).",

    # Hindi
    "कृपया यह निर्धारित करें कि ट्वीट में घृणा भाषण शामिल है या नहीं। घृणा भाषण एक व्यक्ति या समूह के प्रति सीधे या परोक्ष रूप से कही गई बात है जिसमें उस व्यक्ति या समूह के प्रति नफरत होती है। ट्वीट: {} . अब ट्वीट समाप्त होता है। कृपया 'yes' उत्तर दें यदि ट्वीट में घृणा भाषण शामिल है। अन्यथा, 'no' उत्तर दें (अंग्रेज़ी में)।",

    # Bengali
    "দয়া করে নির্ধারণ করুন যে টুইটটিতে ঘৃণাসূচক বক্তব্য অন্তর্ভুক্ত রয়েছে কিনা। ঘৃণাসূচক বক্তব্য হল কোনো ব্যক্তি বা গোষ্ঠীর বিরুদ্ধে সরাসরি বা পরোক্ষভাবে বলা কথা, যা ঐ ব্যক্তি বা গোষ্ঠীর প্রতি অন্তর্নিহিত ঘৃণার উপর ভিত্তি করে থাকে। টুইট: {} . এখন টুইট শেষ হয়েছে। দয়া করে 'yes' দিয়ে উত্তর দিন যদি টুইটটিতে ঘৃণাসূচক বক্তব্য অন্তর্ভুক্ত থাকে। অন্যথায়, 'no' দিয়ে উত্তর দিন (ইংরেজিতে)।",

    # Urdu
    "براہ کرم طے کریں کہ آیا ٹویٹ میں نفرت انگیز تقریر شامل ہے۔ نفرت انگیز تقریر ایک شخص یا گروپ کی طرف براہ راست یا بالواسطہ تقریر ہے جس میں اس شخص یا گروپ کے بارے میں نفرت شامل ہوتی ہے۔ ٹویٹ: {} . اب ٹویٹ ختم ہوتا ہے۔ براہ کرم 'yes' کا جواب دیں اگر ٹویٹ میں نفرت انگیز تقریر شامل ہے۔ ورنہ، 'no' کا جواب دیں (انگریزی میں)۔"
]


# Function to construct the output column
def get_output(result):
    return 'yes' if result == 1 else 'no'

# Function to construct the instruction column
def get_instruction(text):
    instruction_template = random.choice(prompts)
    instruction = instruction_template.format(text)
    return instruction

# Applying the functions to create the dataset
inst_data = pd.DataFrame(columns=['instruction', 'output'])

for i, row in raw_data.iterrows():
    # print(row['Tweet'])
    instruction = get_instruction(row['Tweet'])
    output = get_output(row['HS'])
    inst_data = pd.concat([inst_data, pd.DataFrame({'instruction': [instruction], 'output': [output]})], ignore_index=True)

inst_data.to_parquet("../../data/IHS/IHS.parquet", index=False)

instruction_df_balanced = downsample_to_minority_class(inst_data, 'output')
# sample n=10000 and save to the same parquet
instruction_df_balanced.sample(n=10000).to_parquet("../../data/IHS/IHS.parquet", index=False)