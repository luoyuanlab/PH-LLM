# -*- coding: utf-8 -*-
# UEC (Urdu Emotion Classification) 
# - Paper: https://doi.org/10.1145/3574318.3574327
# - Data: https://sites.google.com/view/multi-label-emotionsfire-task/dataset

import pandas as pd
import random

# Combine the training and test data
taskB_train_file = '../../data/UTT/taskB_Training Dataset.xlsx'
taskB_test_file = "../../data/UTT/taskB_Test_data_with_truth_labels.xlsx"

# UTT
# Define multilingual templates for detecting threatening content
threat_templates = [
    # Chinese
    """请确定以下推文是否包含任何威胁性内容。如果是，请回答 "yes"。如果不是，请回答 "no"。
我们遵循推特的定义来描述针对个人或群体的威胁性帖子，包括威胁使用暴力行为、杀害或造成严重身体伤害、恐吓和使用暴力语言。
推文: "{tweet}"。推文到此结束。
请回答 "yes" 或 "no"。""",

    # English
    """Please determine if the provided tweet below contains any threatening content. If so, respond "yes". If not, respond "no".
We followed Twitter's definition to describe threatening posts toward individuals or groups, including threats of violent acts, killing or inflicting serious physical harm, intimidation, and the use of violent language.
Tweet: "{tweet}". Now the tweet ends.
Please respond with 'yes' or 'no'.""",

    # German
    """Bitte bestimmen Sie, ob der unten stehende Tweet bedrohende Inhalte enthält. Wenn ja, antworten Sie mit "yes". Wenn nicht, antworten Sie mit "no".
Wir folgen der Definition von Twitter, um bedrohliche Posts zu beschreiben, die sich gegen Einzelpersonen oder Gruppen richten, einschließlich Drohungen mit Gewalttaten, Mord oder schwerer Körperverletzung, Einschüchterung und der Verwendung gewalttätiger Sprache.
Tweet: "{tweet}". Der Tweet endet hier.
Bitte antworten Sie mit "yes" oder "no".""",

    # French
    """Veuillez déterminer si le tweet ci-dessous contient des contenus menaçants. Si c'est le cas, répondez "yes". Sinon, répondez "no".
Nous avons suivi la définition de Twitter pour décrire les posts menaçants envers une personne ou un groupe, y compris les menaces d'actes violents, de tuer ou d'infliger des dommages physiques graves, d'intimider et d'utiliser un langage violent.
Tweet: "{tweet}". Le tweet se termine ici.
Veuillez répondre par "yes" ou "no".""",

    # Spanish
    """Por favor, determine si el siguiente tweet contiene contenido amenazante. Si es así, responda "yes". Si no, responda "no".
Seguimos la definición de Twitter para describir las publicaciones amenazantes hacia un individuo o grupo, incluyendo amenazas de actos violentos, matar o infligir daños físicos graves, intimidar y usar lenguaje violento.
Tweet: "{tweet}". El tweet termina aquí.
Por favor, responda con "yes" o "no".""",

    # Portuguese
    """Por favor, determine se o tweet fornecido abaixo contém algum conteúdo ameaçador. Se sim, responda "yes". Se não, responda "no".
Seguimos a definição do Twitter para descrever postagens ameaçadoras em relação a indivíduos ou grupos, incluindo ameaças de atos violentos, matar ou infligir graves danos físicos, intimidar e usar linguagem violenta.
Tweet: "{tweet}". Agora o tweet termina.
Por favor, responda com "yes" ou "no".""",

    # Italian
    """Si prega di determinare se il tweet fornito di seguito contiene contenuti minacciosi. In tal caso, rispondere "yes". In caso contrario, rispondere "no".
Abbiamo seguito la definizione di Twitter per descrivere i post minacciosi verso un individuo o gruppi, incluse minacce di atti violenti, uccisioni o gravi danni fisici, intimidazione e uso di linguaggio violento.
Tweet: "{tweet}". Ora il tweet termina.
Si prega di rispondere con "yes" o "no".""",

    # Dutch
    """Bepaal of de onderstaande tweet bedreigende inhoud bevat. Zo ja, antwoord dan met "yes". Zo niet, antwoord dan met "no".
We volgden Twitters definitie om bedreigende berichten te beschrijven die gericht zijn op individuen of groepen, inclusief bedreigingen met gewelddadige handelingen, moord of ernstig lichamelijk letsel, intimidatie en het gebruik van gewelddadige taal.
Tweet: "{tweet}". Nu eindigt de tweet.
Beantwoord met "yes" of "no".""",

    # Russian
    """Определите, содержит ли приведенный ниже твит угрозы. Если да, ответьте "yes". Если нет, ответьте "no".
Мы следуем определению Twitter, чтобы описать угрожающие сообщения, направленные против людей или групп, включая угрозы насильственными действиями, убийствами или нанесением серьезных физических повреждений, запугиванием и использованием жестокого языка.
Твит: "{tweet}". Теперь твит заканчивается.
Ответьте "yes" или "no".""",

    # Czech
    """Určete, zda níže uvedený tweet obsahuje jakýkoli výhružný obsah. Pokud ano, odpovězte "yes". Pokud ne, odpovězte "no".
Řídíme se definicí Twitteru pro popis výhružných příspěvků, směřovaných proti jednotlivcům nebo skupinám, včetně výhrůžek násilnými činy, vraždou nebo způsobením vážné tělesné újmy, zastrašováním a používáním násilného jazyka.
Tweet: "{tweet}". Nyní tweet končí.
Odpovězte "yes" nebo "no".""",

    # Polish
    """Określ, czy poniższy tweet zawiera jakąkolwiek treść zagrażającą. Jeśli tak, odpowiedz "yes". Jeśli nie, odpowiedz "no".
Kierowaliśmy się definicją Twittera dotyczącą opisania zagrażających postów skierowanych przeciwko osobom lub grupom, w tym gróźb aktów przemocy, morderstwa lub wyrządzenia poważnych szkód fizycznych, zastraszania oraz użycia brutalnego języka.
Tweet: "{tweet}". Teraz tweet się kończy.
Odpowiedz "yes" lub "no".""",

    # Arabic
    """يرجى تحديد ما إذا كانت التغريدة المقدمة أدناه تحتوي على أي محتوى تهديدي. إذا كان الأمر كذلك، فاستجب بـ "yes". إذا لم يكن كذلك، فاستجب بـ "no".
لقد اتبعنا تعريف تويتر لوصف المنشورات المهددة تجاه الأفراد أو الجماعات، بما في ذلك التهديدات بالأفعال العنيفة، أو القتل أو إلحاق ضرر جسدي خطير، أو التخويف، أو استخدام لغة عنيفة.
التغريدة: "{tweet}". تنتهي التغريدة الآن.
يرجى الرد بـ "yes" أو "no".""",

    # Persian
    """لطفاً مشخص کنید که آیا توییت زیر شامل هرگونه محتوای تهدیدآمیز است یا خیر. اگر چنین است، با "yes" پاسخ دهید. اگر نه، با "no" پاسخ دهید.
ما از تعریف توییتر برای توصیف پست‌های تهدیدآمیز علیه افراد یا گروه‌ها استفاده کردیم، از جمله تهدید به اعمال خشونت‌آمیز، قتل یا آسیب جدی جسمی، ارعاب و استفاده از زبان خشونت‌آمیز.
توییت: "{tweet}". اکنون توییت به پایان می‌رسد.
لطفاً با "yes" یا "no" پاسخ دهید.""",

    # Hebrew
    """נא לקבוע אם הציוץ המסופק למטה מכיל תוכן מאיים. אם כן, השב "yes". אם לא, השב "no".
אנו עוקבים אחר ההגדרה של טוויטר לתיאור פוסטים מאיימים כלפי אנשים או קבוצות, כולל איומים במעשי אלימות, הרג או גרימת נזק פיזי חמור, הפחדה ושימוש בשפה אלימה.
ציוץ: "{tweet}". הציוץ מסתיים כאן.
נא להשיב "yes" או "no".""",

    # Turkish
    """Lütfen aşağıda verilen tweet'in herhangi bir tehdit edici içerik içerip içermediğini belirleyin. Eğer öyleyse, "yes" ile yanıtlayın. Değilse, "no" ile yanıtlayın.
Twitter'ın tanımına göre, bireylere veya gruplara yönelik şiddet eylemleriyle tehdit, öldürme veya ciddi fiziksel zarar verme, korkutma ve şiddet içeren dil kullanma gibi tehdit edici içerikleri açıklıyoruz.
Tweet: "{tweet}". Şimdi tweet sona eriyor.
Lütfen "yes" veya "no" ile yanıtlayın.""",

    # Japanese
    """以下の提供されたツイートに脅迫的な内容が含まれているかどうかを判断してください。そうであれば、「yes」と答えてください。そうでなければ、「no」と答えてください。
Twitterの定義に従い、個人やグループに対して暴力行為で脅す、殺害する、深刻な身体的危害を加える、脅迫する、または暴力的な言葉を使用する脅迫的な投稿を記述します。
ツイート: "{tweet}"。ツイートはここで終了します。
「yes」または「no」で回答してください。""",

    # Korean
    """아래 제공된 트윗이 위협적인 내용을 포함하고 있는지 확인해 주세요. 그렇다면 "yes"라고 응답하세요. 그렇지 않다면 "no"라고 응답하세요.
우리는 Twitter의 정의에 따라 개인이나 그룹에 대한 폭력적 행동, 살해, 심각한 신체적 손해를 위협하거나 위협적인 언어를 사용하는 게시물을 설명했습니다.
트윗: "{tweet}". 이제 트윗이 끝납니다.
"yes" 또는 "no"로 응답해 주세요.""",

    # Vietnamese
    """Vui lòng xác định xem tweet được cung cấp bên dưới có chứa nội dung đe dọa nào không. Nếu có, hãy trả lời "yes". Nếu không, hãy trả lời "no".
Chúng tôi đã tuân theo định nghĩa của Twitter để mô tả các bài đăng đe dọa đối với cá nhân hoặc nhóm, bao gồm đe dọa bằng hành động bạo lực, giết hoặc gây tổn hại nghiêm trọng về thể chất, đe dọa và sử dụng ngôn ngữ bạo lực.
Tweet: "{tweet}". Bây giờ tweet kết thúc.
Vui lòng trả lời "yes" hoặc "no".""",

    # Thai
    """โปรดระบุว่าทวีตที่ให้มาด้านล่างมีเนื้อหาที่เป็นการคุกคามหรือไม่ หากเป็นเช่นนั้น ให้ตอบว่า "yes" หากไม่เป็นเช่นนั้น ให้ตอบว่า "no"
เราปฏิบัติตามคำจำกัดความของ Twitter เพื่ออธิบายโพสต์คุกคามต่อบุคคลหรือกลุ่มต่างๆ รวมถึงการข่มขู่ด้วยการกระทำที่รุนแรง สังหารหรือทำร้ายร่างกายอย่างรุนแรง และการใช้ภาษารุนแรง
ทวีต: "{tweet}" ทวีตสิ้นสุดที่นี่
โปรดตอบว่า "yes" หรือ "no""",

    # Indonesian
    """Tentukan apakah tweet yang disediakan di bawah ini mengandung konten yang mengancam. Jika ya, balas dengan "yes". Jika tidak, balas dengan "no".
Kami mengikuti definisi Twitter untuk menggambarkan postingan yang mengancam terhadap individu atau kelompok, termasuk ancaman dengan tindakan kekerasan, membunuh atau melukai secara fisik, intimidasi, dan penggunaan bahasa kekerasan.
Tweet: "{tweet}". Sekarang tweet berakhir.
Silakan balas dengan "yes" atau "no".""",

    # Malay
    """Sila tentukan sama ada tweet yang diberikan di bawah mengandungi sebarang kandungan yang mengancam. Jika ya, balas dengan "yes". Jika tidak, balas dengan "no".
Kami mengikuti definisi Twitter untuk menggambarkan siaran yang mengancam terhadap individu atau kumpulan, termasuk ancaman dengan tindakan ganas, membunuh atau mencederakan secara fizikal yang serius, menakutkan, dan menggunakan bahasa ganas.
Tweet: "{tweet}". Kini tweet berakhir.
Sila balas dengan "yes" atau "no".""",

    # Lao
    """ກະລຸນາລະບຸວ່າທະວິດທີ່ມີຢູ່ດ້ານລຸ່ມນີ້ມີເນື້ອຫາຂູ່ຂົນຫຼືບໍ່. ຖ້າມີ, ກະລຸນາຕອບກັບ "yes". ຖ້າບໍ່ມີ, ກະລຸນາຕອບກັບ "no".
ພວກເຮົາໄດ້ປະຕິບັດຕາມຄຳຈຳກັດຄວາມຂອງ Twitter ເພື່ອອະທິບາຍໂພສທີ່ມີການຂູ່ຂົນຕໍ່ບຸກຄົນຫຼືກຸ່ມບຸກຄົນ, ຮ່ວມທັງການຂູ່ຂົນດ້ວຍການໃຊ້ຄວາມຮຸນແຮງ, ການຂ້າຫຼືການເຮັດໃຫ້ເກີດຄວາມເສຍຫາຍຢ່າງຮ້າຍແຮງຕໍ່ຮ່າງກາຍ, ການຂູ່ຂົນ, ແລະການໃຊ້ພາສາຮຸນແຮງ.
ທະວິດ: "{tweet}". ບັດນີ້ທະວິດໄດ້ສິ້ນສຸດແລ້ວ.
ກະລຸນາຕອບກັບ "yes" ຫຼື "no".""",

    # Burmese
    """ကျေးဇူးပြုပြီးအောက်တွင်ပံ့ပိုးထားသည့်တွစ်တာတွင် အချက်အလက်အကြောင်းအရာပါရှိသောမဟုတ်သောအကြောင်းကိုအတည်ပြုပါ။ ထို့ပါက "yes" ဖြင့်ဖြေကြပါ။ မဟုတ်ပါက "no" ဖြင့်ဖြေပါ။
ကျနော်တို့Twitter၏အဓိပ္ပါယ်ကိုလိုက်နာခဲ့သည် ၊ သည်ဖော်ပြရန်ကိုယ်ခန္ဓာကို ထိခိုက်မှုများဖြစ်စေနိုင်သော အမူအရာများ၊ သတ်ဖြတ်ခြင်း၊ကြောက်ရွံ့စေခြင်းနှင့်အကြမ်းဖက်စကားများကိုအသုံးပြု၍ ပို့စ်များကိုဖော်ပြရန်Twitter၏အဓိပ္ပါယ်ဆိုခြင်းကိုလိုက်နာခဲ့သည်။
Tweet: "{tweet}" Tweet သည်ယခုတွင်ပိတ်သိမ်းပါပြီ။
ကျေးဇူးပြုပြီး "yes" သို့မဟုတ် "no" ဖြင့်ဖြေကြပါ။""",

    # Cebuano
    """Palihug pagtino kung ang gihatag nga tweet sa ubos adunay bisan unsang hulga nga sulod. Kung oo, tubaga ang "yes". Kung dili, tubaga ang "no".
Gisunod namo ang kahulugan sa Twitter sa paghulagway sa mga naghulga nga mga post ngadto sa usa ka indibidwal o mga grupo aron hulgaon sila sa marahas nga mga buhat, pagpatay o paghatag og seryoso nga pisikal nga kadaot, pagpanghadlok, ug paggamit sa marahas nga pinulongan.
Tweet: "{tweet}". Karon ang tweet natapos.
Palihug pagtubag og "yes" o "no".""",

    # Khmer
    """សូមកំណត់មើលថាតើអត្ថបទ Tweet ដែលផ្តល់អោយខាងក្រោមនេះមានខ្លឹមសារដែលគំរាមកំហែង ឬអត់។ ប្រសិនបើមាន សូមឆ្លើយថា "yes"។ ប្រសិនបើមិនមាន សូមឆ្លើយថា "no"។
យើងបានតាមដានការបញ្ជាក់នៃ Twitter ដើម្បីពិពណ៌នាអំពីអត្ថបទដែលគំរាមកំហែងទៅលើមនុស្សម្នាក់ឬក្រុមមនុស្ស ដោយគំរាមកំហែងប្រើអំពើហឹង្សា សម្លាប់ ឬធ្វើអោយរងរបួសធ្ងន់ធ្ងរដល់រាងកាយ គំរាមកំហែង និងប្រើភាសាហឹង្សា។
Tweet: "{tweet}"។ ឥឡូវនេះអត្ថបទ Tweet បញ្ចប់ហើយ។
សូមឆ្លើយថា "yes" ឬ "no"។""",

    # Tagalog
    """Pakisuri kung ang ibinigay na tweet sa ibaba ay naglalaman ng anumang mapanlinlang na nilalaman. Kung oo, sumagot ng "yes". Kung hindi, sumagot ng "no".
Sinunod namin ang kahulugan ng Twitter upang ilarawan ang mga mapanlinlang na post na patungkol sa isang indibidwal o grupo upang magbanta ng marahas na aksyon, pumatay o magdulot ng malubhang pinsala sa katawan, upang takutin, at gumamit ng marahas na wika.
Tweet: "{tweet}". Ngayon natapos na ang tweet.
Pakisagot ng "yes" o "no".""",

    # Hindi
    """कृपया निर्धारित करें कि नीचे दिया गया ट्वीट किसी भी तरह की धमकी वाली सामग्री है या नहीं। यदि हां, तो "yes" के साथ उत्तर दें। यदि नहीं, तो "no" के साथ उत्तर दें।
हमने ट्विटर की परिभाषा का पालन किया है ताकि किसी व्यक्ति या समूह के खिलाफ धमकी भरे पोस्ट का वर्णन किया जा सके, जिसमें हिंसक कृत्यों की धमकी देना, मार डालना या गंभीर शारीरिक नुकसान पहुँचाना, डराना और हिंसक भाषा का प्रयोग करना शामिल है।
ट्वीट: "{tweet}"। अब ट्वीट समाप्त होता है।
कृपया "yes" या "no" के साथ उत्तर दें।""",

    # Bengali
    """নীচের দেওয়া টুইটটি কোনো হুমকিস্বরূপ বিষয়বস্তু রয়েছে কিনা তা নির্ধারণ করুন। যদি থাকে, তাহলে "yes" দিয়ে উত্তর দিন। যদি না থাকে, তাহলে "no" দিয়ে উত্তর দিন।
কোনও ব্যক্তি বা গোষ্ঠীর প্রতি সহিংস কাজ, হত্যা বা গুরুতর শারীরিক ক্ষতি করার হুমকি, ভীতি প্রদর্শন এবং সহিংস ভাষা ব্যবহার করে হুমকিস্বরূপ পোস্টগুলি বর্ণনা করতে আমরা টুইটারের সংজ্ঞা অনুসরণ করেছি।
টুইট: "{tweet}"। এখন টুইট শেষ হয়েছে।
"yes" বা "no" দিয়ে উত্তর দিন।""",

    # Urdu
    """براہ کرم یہ تعین کریں کہ آیا نیچے فراہم کردہ ٹویٹ میں کوئی دھمکی آمیز مواد شامل ہے۔ اگر ایسا ہے تو، "yes" کے ساتھ جواب دیں۔ اگر نہیں، تو "no" کے ساتھ جواب دیں۔
ہم نے کسی فرد یا گروہوں کے خلاف دھمکی آمیز پوسٹس کو بیان کرنے کے لیے ٹویٹر کی تعریف کی پیروی کی، تاکہ پرتشدد کارروائیوں کی دھمکی دی جائے، قتل کیا جائے یا شدید جسمانی نقصان پہنچایا جائے، ڈرایا جائے، اور پرتشدد زبان استعمال کی جائے۔
ٹویٹ: "{tweet}"۔ اب ٹویٹ ختم ہوتا ہے۔
براہ کرم "yes" یا "no" کے ساتھ جواب دیں۔"""
]


# Function to create instruction based on the selected template
def create_instruction(row):
    instruction_template = random.choice(threat_templates)
    instruction = instruction_template.format(tweet=row["Tweets"])
    return instruction

# Function to get the output based on the label
def get_output(row):
    if row["label"] == 1:
        return "yes"
    else:
        return "no"

# get a blank dataframe
inst_dataB = pd.DataFrame(columns=['instruction', 'output'])
taskB_train_data = pd.read_excel(taskB_train_file)
taskB_test_data = pd.read_excel(taskB_test_file)
taskB_data = pd.concat([taskB_train_data, taskB_test_data])

# Generate instructions and outputs
for i, row in taskB_data.iterrows():
    output = get_output(row)
    instruction = create_instruction(row)
    inst_dataB = pd.concat([inst_dataB, pd.DataFrame({'instruction': [instruction], 'output': [output]})], ignore_index=True)

inst_dataB = inst_dataB.sample(n=4000, random_state=42)
# Save the DataFrame to a parquet file
inst_dataB.to_parquet("../../data/UTT/emothreat_taskB_multilingual.parquet", index=False)

