# - coding: utf-8 -*-
#  SemEval-2020 Task 9 
# - Paper: https://arxiv.org/abs/2008.04277
# - Data: https://github.com/rsgoss/NLP_finalproj/tree/main/data 

import pandas as pd
import random
# files to process
train_file = '../../data/SemEval2020Task9/hindi-english/train_14k_split.csv'
val_file = '../../data/SemEval2020Task9/hindi-english/val_3k_split.csv'
test_file = '../../data/SemEval2020Task9/hindi-english/test_unalbelled.csv'
train_data = pd.read_csv(train_file, header=0)
val_data = pd.read_csv(val_file, header=0)
test_data = pd.read_csv(test_file, header=0)
test_data_labels = pd.read_csv('../../data/SemEval2020Task9/hindi-english/test_labels_hinglish.txt', sep=',')
#rename test data columns
test_data_labels.columns = ['id', 'sentiment']
#merge test data with labels
test_data = pd.merge(test_data, test_data_labels, on='id')
train_val_data = pd.concat([train_data, val_data], ignore_index=True)
train_val_data = train_val_data[['sentence', 'sentiment']]
test_data = test_data[['sentence', 'sentiment']]
#merge train, val and test data
data = pd.concat([train_val_data, test_data], ignore_index=True)

# Function to create instruction in the target language
def get_instruction(row, templates):
    # Randomly select one of the multilingual templates
    instruction_template = random.choice(templates)
    instruction = instruction_template.format(sentence=row['sentence'])
    return instruction

# Define multilingual templates for the instruction
multilingual_templates = [
    # Chinese
    "请判断以下提供的推文的态度。如果态度是积极的，请回答'positive'。如果是消极的，请回答'negative'。如果是中立的，您可以回答'neutral'。推文: \"{sentence}\"。现在推文结束。请用'positive'、'negative'或'neutral'作答。",

    # English
    "Please determine the attitude of the provided Hinglish tweet below. If the attitude is positive, respond 'positive'. If negative, respond 'negative'. If neutral, you may respond 'neutral'. Tweet: \"{sentence}\". Now the tweet ends. Please respond with 'positive', 'negative', or 'neutral'.",

    # German
    "Bitte bestimmen Sie die Haltung des unten angegebenen Hinglish-Tweets. Wenn die Haltung positiv ist, antworten Sie mit 'positive'. Wenn negativ, antworten Sie mit 'negative'. Wenn neutral, können Sie mit 'neutral' antworten. Tweet: \"{sentence}\". Jetzt endet der Tweet. Bitte antworten Sie mit 'positive', 'negative' oder 'neutral'.",

    # French
    "Veuillez déterminer l'attitude du tweet Hinglish fourni ci-dessous. Si l'attitude est positive, répondez 'positive'. Si elle est négative, répondez 'negative'. Si elle est neutre, vous pouvez répondre 'neutral'. Tweet : \"{sentence}\". Maintenant, le tweet se termine. Veuillez répondre par 'positive', 'negative' ou 'neutral'.",

    # Spanish
    "Por favor, determine la actitud del tweet Hinglish proporcionado a continuación. Si la actitud es positiva, responda 'positive'. Si es negativa, responda 'negative'. Si es neutral, puede responder 'neutral'. Tweet: \"{sentence}\". Ahora el tweet termina. Responda con 'positive', 'negative' o 'neutral'.",

    # Portuguese
    "Por favor, determine a atitude do tweet Hinglish fornecido abaixo. Se a atitude for positiva, responda 'positive'. Se for negativa, responda 'negative'. Se for neutra, você pode responder 'neutral'. Tweet: \"{sentence}\". Agora o tweet termina. Por favor, responda com 'positive', 'negative' ou 'neutral'.",

    # Italian
    "Si prega di determinare l'atteggiamento del tweet Hinglish fornito di seguito. Se l'atteggiamento è positivo, rispondere 'positive'. Se è negativo, rispondere 'negative'. Se è neutrale, puoi rispondere 'neutral'. Tweet: \"{sentence}\". Ora il tweet finisce. Rispondere con 'positive', 'negative' o 'neutral'.",

    # Dutch
    "Bepaal de houding van de onderstaande Hinglish-tweet. Als de houding positief is, antwoord dan met 'positive'. Als het negatief is, antwoord dan met 'negative'. Als het neutraal is, kunt u antwoorden met 'neutral'. Tweet: \"{sentence}\". Nu eindigt de tweet. Antwoord alstublieft met 'positive', 'negative' of 'neutral'.",

    # Russian
    "Пожалуйста, определите отношение в приведенном ниже твите на Hinglish. Если отношение положительное, ответьте 'positive'. Если отрицательное, ответьте 'negative'. Если нейтральное, вы можете ответить 'neutral'. Твит: \"{sentence}\". Теперь твит заканчивается. Пожалуйста, ответьте 'positive', 'negative' или 'neutral'.",

    # Czech
    "Prosím, určete postoj v následujícím Hinglish tweetu. Pokud je postoj pozitivní, odpovězte 'positive'. Pokud negativní, odpovězte 'negative'. Pokud neutrální, můžete odpovědět 'neutral'. Tweet: \"{sentence}\". Nyní tweet končí. Odpovězte prosím 'positive', 'negative' nebo 'neutral'.",

    # Polish
    "Proszę określić postawę w poniższym Hinglish tweecie. Jeśli postawa jest pozytywna, odpowiedz 'positive'. Jeśli negatywna, odpowiedz 'negative'. Jeśli neutralna, możesz odpowiedzieć 'neutral'. Tweet: \"{sentence}\". Teraz tweet się kończy. Odpowiedz proszę 'positive', 'negative' lub 'neutral'.",

    # Arabic
    "يرجى تحديد الموقف في التغريدة باللغة الهندية الإنجليزية المقدمة أدناه. إذا كان الموقف إيجابيًا، فاستجب بـ 'positive'. إذا كان سلبيًا، فاستجب بـ 'negative'. إذا كان محايدًا، يمكنك الرد بـ 'neutral'. التغريدة: \"{sentence}\". الآن انتهت التغريدة. يرجى الرد بـ 'positive' أو 'negative' أو 'neutral'.",

    # Persian
    "لطفاً نگرش موجود در توییت هینگلیش زیر را تعیین کنید. اگر نگرش مثبت است، پاسخ 'positive' دهید. اگر منفی است، پاسخ 'negative' دهید. اگر خنثی است، می‌توانید پاسخ 'neutral' دهید. توییت: \"{sentence}\". اکنون توییت به پایان می‌رسد. لطفاً با 'positive'، 'negative' یا 'neutral' پاسخ دهید.",

    # Hebrew
    "אנא קבע את העמדה בציוץ ההינגליש שלהלן. אם העמדה חיובית, השב 'positive'. אם שלילית, השב 'negative'. אם ניטרלית, אתה יכול להשיב 'neutral'. ציוץ: \"{sentence}\". הציוץ נגמר כעת. אנא השב 'positive', 'negative' או 'neutral'.",

    # Turkish
    "Aşağıda verilen Hinglish tweet'in tutumunu belirleyin. Eğer tutum olumluysa, 'positive' yanıtını verin. Eğer olumsuzsa, 'negative' yanıtını verin. Eğer nötrse, 'neutral' yanıtını verebilirsiniz. Tweet: \"{sentence}\". Tweet şimdi sona erdi. Lütfen 'positive', 'negative' veya 'neutral' yanıtını verin.",

    # Japanese
    "以下のヒングリッシュのツイートの態度を判断してください。態度がポジティブであれば、「positive」と答えてください。ネガティブであれば、「negative」と答えてください。中立的であれば、「neutral」と答えることができます。ツイート: \"{sentence}\"。これでツイートは終了です。「positive」、「negative」、または「neutral」で答えてください。",

    # Korean
    "아래 제공된 힌글리시 트윗의 태도를 결정해 주세요. 태도가 긍정적이라면 'positive'라고 답변하세요. 부정적이라면 'negative'라고 답변하세요. 중립적이라면 'neutral'이라고 답변하셔도 됩니다. 트윗: \"{sentence}\". 이제 트윗이 끝났습니다. 'positive', 'negative' 또는 'neutral'로 답변해 주세요.",

    # Vietnamese
    "Vui lòng xác định thái độ trong tweet Hinglish được cung cấp dưới đây. Nếu thái độ là tích cực, hãy trả lời 'positive'. Nếu tiêu cực, hãy trả lời 'negative'. Nếu trung lập, bạn có thể trả lời 'neutral'. Tweet: \"{sentence}\". Bây giờ tweet kết thúc. Vui lòng trả lời bằng 'positive', 'negative' hoặc 'neutral'.",

    # Thai
    "โปรดระบุทัศนคติของทวีต Hinglish ที่ให้ไว้ด้านล่าง หากทัศนคติเชิงบวก ตอบว่า 'positive' หากเชิงลบ ตอบว่า 'negative' หากเป็นกลาง คุณอาจตอบว่า 'neutral' ทวีต: \"{sentence}\" ตอนนี้ทวีตสิ้นสุดแล้ว โปรดตอบด้วย 'positive', 'negative' หรือ 'neutral'。",

    # Indonesian
    "Silakan tentukan sikap dari tweet Hinglish yang diberikan di bawah ini. Jika sikapnya positif, jawab 'positive'. Jika negatif, jawab 'negative'. Jika netral, Anda dapat menjawab 'neutral'. Tweet: \"{sentence}\". Sekarang tweet berakhir. Silakan jawab dengan 'positive', 'negative', atau 'neutral'.",

    # Malay
    "Sila tentukan sikap dalam tweet Hinglish yang disediakan di bawah ini. Jika sikap itu positif, jawab 'positive'. Jika negatif, jawab 'negative'. Jika neutral, anda boleh menjawab 'neutral'. Tweet: \"{sentence}\". Sekarang tweet tamat. Sila jawab dengan 'positive', 'negative', atau 'neutral'.",

    # Lao
    "ກະລຸນາກຳນົດທ່າທີ່ໃນທວິດ Hinglish ທີ່ໄດ້ຮັບດ້ານລຸ່ມນີ້. ຖ້າວ່າທ່າທີ່ເປັນບວກ, ກະລຸນາຕອບ 'positive'. ຖ້າວ່າເປັນລົບ, ກະລຸນາຕອບ 'negative'. ຖ້າວ່າເປັນກາງ, ເຈົ້າສາມາດຕອບ 'neutral'. Tweet: \"{sentence}\". ຂໍ້ຄວາມນີ້ສິ້ນສຸດລົງແລ້ວ. ກະລຸນາຕອບດ້ວຍ 'positive', 'negative' ຫຼື 'neutral'.",

    # Burmese
    "အောက်တွင်ပေးထားသော Hinglish တူဿ်၏ thái độကို သတ်မှတ်ပါ။ thái độသည် လိင်ဖြင့်အကောင်းဖြစ်ပါက, 'positive' ဖြင့် ဖြေကြားပါ။ အကောင်းဖြစ်ပါက, 'negative' ဖြင့် ပြန်ကြားပါ။ အကောင်းဖြစ်ပါက, 'neutral' ဖြင့် ပြန်ကြားနိုင်ပါသည်။ တူဿ်: \"{sentence}\". ယခုတွင် တူဿ် ပြီးဆုံးပါပြီ။ 'positive', 'negative' သို့မဟုတ် 'neutral' ဖြင့် ပြန်ကြားပါ။",

    # Cebuano
    "Palihug pag-determinar sa hiyas sa Hinglish nga tweet sa ubos. Kung ang hiyas positibo, tubaga ang 'positive'. Kung negatibo, tubaga ang 'negative'. Kung neutral, tubaga ang 'neutral'. Tweet: \"{sentence}\". Karon ang tweet mohuman na. Palihug motubag og 'positive', 'negative', o 'neutral'.",

    # Khmer
    "សូមកំណត់អាកប្បកិរិយានៃពាក្យអនាមិកក្នុងការបង្ហាញក្នុងការបង្ហាញលើកអប្បកិរិយានេះ ។ ប្រសិនបើពាក្យនេះមានអាកប្បកិរិយាអំណោយផល សូមឆ្លើយថា 'positive'។ ប្រសិនបើមានអាកប្បកិរិយាអវិជ្ជមាន សូមឆ្លើយថា 'negative'។ ប្រសិនបើមានអាកប្បកិរិយាមធ្យម អ្នកអាចឆ្លើយថា 'neutral'។ ការបង្ហាញ: \"{sentence}\"។ ឥឡូវនេះការបង្ហាញបានបញ្ចប់ហើយ សូមឆ្លើយថា 'positive', 'negative' ឬ 'neutral'។",

    # Tagalog
    "Pakitukoy ang saloobin sa ibinigay na Hinglish na tweet sa ibaba. Kung positibo ang saloobin, sumagot ng 'positive'. Kung negatibo, sumagot ng 'negative'. Kung neutral, maaari kang sumagot ng 'neutral'. Tweet: \"{sentence}\". Ngayon natapos na ang tweet. Pakisagot ng 'positive', 'negative', o 'neutral'.",

    # Hindi
    "कृपया नीचे दिए गए Hinglish ट्वीट में दृष्टिकोण का निर्धारण करें। यदि दृष्टिकोण सकारात्मक है, तो 'positive' के साथ उत्तर दें। यदि नकारात्मक है, तो 'negative' के साथ उत्तर दें। यदि तटस्थ है, तो आप 'neutral' के साथ उत्तर दे सकते हैं। ट्वीट: \"{sentence}\"। अब ट्वीट समाप्त हो गया है। कृपया 'positive', 'negative', या 'neutral' के साथ उत्तर दें।",

    # Bengali
    "নীচে দেওয়া হিংলিশ টুইটের মনোভাব নির্ধারণ করুন। যদি মনোভাবটি ইতিবাচক হয়, তাহলে 'positive' দিয়ে উত্তর দিন। যদি নেতিবাচক হয়, তাহলে 'negative' দিয়ে উত্তর দিন। যদি নিরপেক্ষ হয়, আপনি 'neutral' দিয়ে উত্তর দিতে পারেন। টুইট: \"{sentence}\"। এখন টুইট শেষ। অনুগ্রহ করে 'positive', 'negative' বা 'neutral' দিয়ে উত্তর দিন।",

    # Urdu
    "براہ کرم نیچے دیے گئے ہنگلش ٹویٹ کے رویے کا تعین کریں۔ اگر رویہ مثبت ہے تو 'positive' کے ساتھ جواب دیں۔ اگر منفی ہے تو 'negative' کے ساتھ جواب دیں۔ اگر غیر جانبدار ہے تو آپ 'neutral' کے ساتھ جواب دے سکتے ہیں۔ ٹویٹ: \"{sentence}\"۔ اب ٹویٹ ختم ہو گیا ہے۔ براہ کرم 'positive', 'negative', یا 'neutral' کے ساتھ جواب دیں۔"
]


# Create a blank dataframe
inst_data = pd.DataFrame(columns=['instruction', 'output'])

# Iterate over each row in the data and create the instruction/output pair
for i, row in data.iterrows():
    instruction = get_instruction(row, multilingual_templates)
    output = row["sentiment"]
    inst_data = pd.concat([inst_data, pd.DataFrame({'instruction': [instruction], 'output': [output]})], ignore_index=True)

# collect a sample of n=500
inst_data = inst_data.sample(n=12000, random_state=42)

# Save the dataframe to a parquet file
inst_data.to_parquet("../../data/SemEval2020Task9/Hinglish_SemEva2020t9.parquet", index=False)