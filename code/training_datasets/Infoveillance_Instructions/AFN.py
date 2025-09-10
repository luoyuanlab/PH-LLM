# -*- coding: utf-8 -*-
# AFN (Arabic Fake News)
# - Paper: https://doi.org/10.14569/IJACSA.2021.0120691
# - Data: https://github.com/yemen2016/FakeNewsDetection/tree/main
import pandas as pd
import random

raw_data = pd.read_csv("../../data/AFN/ManualAnnotatedFakeNewsDataset.txt", sep='\t', names = ["label", "tweet"])
# List of prompts for all languages
prompts = [
    # Chinese
    "请确定这条推文是否是假新闻。推文：{text}。推文到此结束。如果是假新闻，请回答 'yes'。否则，请回答 'no'。",
    "请判断这条推文是否是假新闻。推文内容：{text}。推文结束。如果是假新闻，请回答 'yes'。否则，请回答 'no'。",

    # English
    "Please determine if the tweet is fake news or not fake news. Tweet: {text}. Now the tweet ends. If it is fake news, respond 'yes'. Otherwise, respond 'no'.",
    "Please assess if the following tweet is fake news. Tweet: {text}. The tweet ends here. If it is fake news, respond with 'yes'. Otherwise, respond with 'no'.",

    # German
    "Bitte bestimmen Sie, ob der Tweet eine Falschnachricht ist oder nicht. Tweet: {text}. Der Tweet endet hier. Wenn es eine Falschnachricht ist, antworten Sie mit 'yes'. Andernfalls antworten Sie mit 'no'.",
    "Bitte prüfen Sie, ob der folgende Tweet eine Falschnachricht ist. Tweet: {text}. Der Tweet endet jetzt. Wenn es sich um eine Falschnachricht handelt, antworten Sie mit 'yes', andernfalls mit 'no'.",

    # French
    "Veuillez déterminer si le tweet est une fausse nouvelle ou non. Tweet : {text}. Le tweet se termine ici. Si c'est une fausse nouvelle, répondez 'yes'. Sinon, répondez 'no'.",
    "Veuillez évaluer si le tweet suivant est une fausse nouvelle. Tweet : {text}. Le tweet se termine maintenant. Si c'est une fausse nouvelle, répondez par 'yes'. Sinon, répondez par 'no'.",

    # Spanish
    "Por favor, determine si el tweet es una noticia falsa o no. Tweet: {text}. Ahora el tweet termina. Si es una noticia falsa, responda 'yes'. De lo contrario, responda 'no'.",
    "Por favor, evalúe si el siguiente tweet es una noticia falsa. Tweet: {text}. El tweet termina aquí. Si es una noticia falsa, responda con 'yes'. De lo contrario, responda con 'no'.",

    # Portuguese
    "Por favor, determine se o tweet é uma notícia falsa ou não. Tweet: {text}. Agora o tweet termina. Se for uma notícia falsa, responda 'yes'. Caso contrário, responda 'no'.",
    "Por favor, avalie se o seguinte tweet é uma notícia falsa. Tweet: {text}. O tweet termina aqui. Se for uma notícia falsa, responda 'yes'. Caso contrário, responda 'no'.",

    # Italian
    "Per favore, determina se il tweet è una notizia falsa o meno. Tweet: {text}. Ora il tweet termina. Se è una notizia falsa, rispondi 'yes'. Altrimenti, rispondi 'no'.",
    "Per favore, valuta se il seguente tweet è una notizia falsa. Tweet: {text}. Il tweet termina qui. Se è una notizia falsa, rispondi con 'yes'. Altrimenti, rispondi con 'no'.",

    # Dutch
    "Bepaal of de tweet nepnieuws is of niet. Tweet: {text}. Nu eindigt de tweet. Als het nepnieuws is, reageer dan met 'yes'. Anders reageer met 'no'.",
    "Beoordeel of de volgende tweet nepnieuws is. Tweet: {text}. De tweet eindigt hier. Als het nepnieuws is, reageer dan met 'yes'. Anders reageer met 'no'.",

    # Russian
    "Пожалуйста, определите, является ли этот твит фейковой новостью или нет. Твит: {text}. Твит на этом заканчивается. Если это фейк, ответьте 'yes'. В противном случае ответьте 'no'.",
    "Оцените, является ли следующий твит фейковой новостью. Твит: {text}. Твит заканчивается здесь. Если это фейк, ответьте 'yes'. В противном случае ответьте 'no'.",

    # Czech
    "Prosím, určete, zda je tento tweet falešná zpráva nebo ne. Tweet: {text}. Tweet zde končí. Pokud je to falešná zpráva, odpovězte 'yes'. Jinak odpovězte 'no'.",
    "Ohodnoťte, zda je následující tweet falešná zpráva. Tweet: {text}. Tweet končí zde. Pokud je to falešná zpráva, odpovězte 'yes'. Jinak odpovězte 'no'.",

    # Polish
    "Proszę określić, czy ten tweet jest fałszywą wiadomością, czy nie. Tweet: {text}. Tweet kończy się tutaj. Jeśli to fałszywa wiadomość, odpowiedz 'yes'. W przeciwnym razie odpowiedz 'no'.",
    "Proszę ocenić, czy następujący tweet jest fałszywą wiadomością. Tweet: {text}. Tweet kończy się tutaj. Jeśli to fałszywa wiadomość, odpowiedz 'yes'. W przeciwnym razie odpowiedz 'no'.",

    # Arabic
    "يرجى تحديد ما إذا كانت التغريدة أخبارًا مزيفة أم لا. التغريدة: {text}. تنتهي التغريدة هنا. إذا كانت أخبارًا مزيفة، أجب بـ 'yes'. خلاف ذلك، أجب بـ 'no'.",
    "يرجى تقييم ما إذا كانت التغريدة التالية أخبارًا مزيفة. التغريدة: {text}. تنتهي التغريدة هنا. إذا كانت أخبارًا مزيفة، أجب بـ 'yes'. خلاف ذلك، أجب بـ 'no'.",

    # Persian
    "لطفاً تعیین کنید که آیا این توییت خبر جعلی است یا خیر. توییت: {text}. توییت در اینجا به پایان می‌رسد. اگر خبر جعلی است، پاسخ دهید 'yes'. در غیر این صورت، پاسخ دهید 'no'.",
    "لطفاً ارزیابی کنید که آیا توییت زیر خبر جعلی است. توییت: {text}. توییت در اینجا به پایان می‌رسد. اگر خبر جعلی است، پاسخ دهید 'yes'. در غیر این صورت، پاسخ دهید 'no'.",

    # Hebrew
    "אנא קבע אם הציוץ הוא חדשות מזויפות או לא. ציוץ: {text}. הציוץ מסתיים כאן. אם זה חדשות מזויפות, השב 'yes'. אחרת, השב 'no'.",
    "אנא הערך אם הציוץ הבא הוא חדשות מזויפות. ציוץ: {text}. הציוץ מסתיים כאן. אם זה חדשות מזויפות, השב 'yes'. אחרת, השב 'no'.",

    # Turkish
    "Lütfen tweetin sahte haber olup olmadığını belirleyin. Tweet: {text}. Tweet burada sona eriyor. Eğer sahte haberse, 'yes' yanıtını verin. Aksi takdirde, 'no' yanıtını verin.",
    "Lütfen aşağıdaki tweetin sahte haber olup olmadığını değerlendirin. Tweet: {text}. Tweet burada sona eriyor. Eğer sahte haberse, 'yes' yanıtını verin. Aksi takdirde, 'no' yanıtını verin.",

    # Japanese
    "このツイートがフェイクニュースかどうかを判断してください。ツイート: {text}。ツイートはここで終了します。フェイクニュースの場合は 'yes' と答えてください。そうでない場合は 'no' と答えてください。",
    "次のツイートがフェイクニュースかどうかを評価してください。ツイート: {text}。ツイートはここで終了します。フェイクニュースの場合は 'yes' と答えてください。そうでない場合は 'no' と答えてください。",

    # Korean
    "이 트윗이 가짜 뉴스인지 여부를 결정하십시오. 트윗: {text}. 트윗은 여기서 끝납니다. 가짜 뉴스라면 'yes'로 응답하십시오. 그렇지 않으면 'no'로 응답하십시오.",
    "다음 트윗이 가짜 뉴스인지 평가하십시오. 트윗: {text}. 트윗은 여기서 끝납니다. 가짜 뉴스라면 'yes'로 응답하십시오. 그렇지 않으면 'no'로 응답하십시오.",

    # Vietnamese
    "Vui lòng xác định liệu tweet này có phải là tin giả hay không. Tweet: {text}. Tweet kết thúc ở đây. Nếu đó là tin giả, hãy trả lời 'yes'. Nếu không, hãy trả lời 'no'.",
    "Vui lòng đánh giá xem tweet sau có phải là tin giả hay không. Tweet: {text}. Tweet kết thúc ở đây. Nếu đó là tin giả, hãy trả lời 'yes'. Nếu không, hãy trả lời 'no'.",

    # Thai
    "โปรดระบุว่าทวีตนี้เป็นข่าวปลอมหรือไม่ ทวีต: {text} ทวีตจบที่นี่ หากเป็นข่าวปลอม โปรดตอบ 'yes' หากไม่ใช่ โปรดตอบ 'no'",
    "โปรดประเมินว่าทวีตต่อไปนี้เป็นข่าวปลอมหรือไม่ ทวีต: {text} ทวีตจบที่นี่ หากเป็นข่าวปลอม โปรดตอบ 'yes' หากไม่ใช่ โปรดตอบ 'no'",

    # Indonesian
    "Tolong tentukan apakah tweet ini adalah berita palsu atau tidak. Tweet: {text}. Tweet berakhir di sini. Jika itu berita palsu, balas dengan 'yes'. Jika tidak, balas dengan 'no'.",
    "Tolong nilai apakah tweet berikut adalah berita palsu. Tweet: {text}. Tweet berakhir di sini. Jika itu berita palsu, balas dengan 'yes'. Jika tidak, balas dengan 'no'.",

    # Malay
    "Sila tentukan sama ada tweet ini adalah berita palsu atau tidak. Tweet: {text}. Tweet berakhir di sini. Jika itu berita palsu, balas dengan 'yes'. Jika tidak, balas dengan 'no'.",
    "Sila nilai sama ada tweet berikut adalah berita palsu. Tweet: {text}. Tweet berakhir di sini. Jika itu berita palsu, balas dengan 'yes'. Jika tidak, balas dengan 'no'.",

    # Lao
    "ກະລຸນາກຳນົດວ່າທວິດນີ້ເປັນຂ່າວປອມຫຼືບໍ່. ທວິດ: {text}. ທວິດຈະສິ້ນສຸດທີ່ນີ້. ຖ້າມັນເປັນຂ່າວປອມ, ກະລຸນາຕອບ 'yes'. ຖ້າບໍ່, ກະລຸນາຕອບ 'no'.",
    "ກະລຸນາປະເມີນວ່າທວິດຕໍ່ໄປນີ້ເປັນຂ່າວປອມຫຼືບໍ່. ທວິດ: {text}. ທວິດຈະສິ້ນສຸດທີ່ນີ້. ຖ້າມັນເປັນຂ່າວປອມ, ກະລຸນາຕອບ 'yes'. ຖ້າບໍ່, ກະລຸນາຕອບ 'no'.",

    # Burmese
    "ကျေးဇူးပြု၍ ဤTweet သည်အတုဖြစ်သည် သို့မဟုတ် မဟုတ်ပါ။ Tweet: {text}။ Tweet သည်ဒီမှာအဆုံးသတ်ပါသည်။ ဤသည်အတုဖြစ်ပါက 'yes' ဟုပြန်ကြားပါ။ မဟုတ်ပါက 'no' ဟုပြန်ကြားပါ။",
    "ကျေးဇူးပြု၍ နောက်ထပ် Tweet သည်အတုဖြစ်သည် သို့မဟုတ် မဟုတ်ပါ။ Tweet: {text}။ Tweet သည်ဒီမှာအဆုံးသတ်ပါသည်။ ဤသည်အတုဖြစ်ပါက 'yes' ဟုပြန်ကြားပါ။ မဟုတ်ပါက 'no' ဟုပြန်ကြားပါ။",

    # Cebuano
    "Palihug tukma-a kung kini nga tweet bakak nga balita o dili. Tweet: {text}. Ang tweet magtapos dinhi. Kung kini bakak nga balita, tubaga og 'yes'. Kung dili, tubaga og 'no'.",
    "Palihug e-assess kung ang mosunod nga tweet bakak nga balita. Tweet: {text}. Ang tweet magtapos dinhi. Kung kini bakak nga balita, tubaga og 'yes'. Kung dili, tubaga og 'no'.",

    # Khmer
    "សូមកំណត់ថាតើទំនាក់ទំនងនេះជាព័ត៌មានក្លែងក្លាយឬអត់។ ទំនាក់ទំនង: {text}។ ទំនាក់ទំនងបញ្ចប់នៅទីនេះ។ ប្រសិនបើវាជាព័ត៌មានក្លែងក្លាយ សូមឆ្លើយតប 'yes'។ បើអត់ សូមឆ្លើយតប 'no'។",
    "សូមវាយតម្លៃថាតើទំនាក់ទំនងបន្ទាប់ជាព័ត៌មានក្លែងក្លាយឬអត់។ ទំនាក់ទំនង: {text}។ ទំនាក់ទំនងបញ្ចប់នៅទីនេះ។ ប្រសិនបើវាជាព័ត៌មានក្លែងក្លាយ សូមឆ្លើយតប 'yes'។ បើអត់ សូមឆ្លើយតប 'no'។",

    # Tagalog
    "Pakisuri kung ang tweet na ito ay pekeng balita o hindi. Tweet: {text}. Dito nagtatapos ang tweet. Kung ito ay pekeng balita, sumagot ng 'yes'. Kung hindi, sumagot ng 'no'.",
    "Pakisuri kung ang susunod na tweet ay pekeng balita. Tweet: {text}. Dito nagtatapos ang tweet. Kung ito ay pekeng balita, sumagot ng 'yes'. Kung hindi, sumagot ng 'no'.",

    # Hindi
    "कृपया जांच करें कि यह ट्वीट फेक न्यूज है या नहीं। ट्वीट: {text}। अब ट्वीट समाप्त होता है। यदि यह फेक न्यूज है, तो 'yes' उत्तर दें। अन्यथा, 'no' उत्तर दें।",
    "कृपया यह आकलन करें कि निम्नलिखित ट्वीट फेक न्यूज है या नहीं। ट्वीट: {text}। ट्वीट यहां समाप्त होता है। यदि यह फेक न्यूज है, तो 'yes' उत्तर दें। अन्यथा, 'no' उत्तर दें।",

    # Bengali
    "দয়া করে নির্ধারণ করুন যে এই টুইটটি ভুয়া খবর কিনা। টুইট: {text}। টুইটটি এখানেই শেষ। যদি এটি ভুয়া খবর হয়, তবে 'yes' দিয়ে উত্তর দিন। অন্যথায়, 'no' দিয়ে উত্তর দিন।",
    "দয়া করে মূল্যায়ন করুন যে নিম্নলিখিত টুইটটি ভুয়া খবর কিনা। টুইট: {text}। টুইটটি এখানেই শেষ। যদি এটি ভুয়া খবর হয়, তবে 'yes' দিয়ে উত্তর দিন। অন্যথায়, 'no' দিয়ে উত্তর দিন।",

    # Urdu
    "براہ کرم تعین کریں کہ یہ ٹویٹ جعلی خبر ہے یا نہیں۔ ٹویٹ: {text}۔ اب ٹویٹ ختم ہو گیا ہے۔ اگر یہ جعلی خبر ہے، تو 'yes' کا جواب دیں۔ بصورت دیگر، 'no' کا جواب دیں۔",
    "براہ کرم اندازہ کریں کہ آیا مندرجہ ذیل ٹویٹ جعلی خبر ہے۔ ٹویٹ: {text}۔ ٹویٹ یہاں ختم ہو جاتا ہے۔ اگر یہ جعلی خبر ہے، تو 'yes' کا جواب دیں۔ بصورت دیگر، 'no' کا جواب دیں۔"
]


# Define the functions
def get_instruction(row):
    instruction_template = random.choice(prompts)
    instruction = instruction_template.format(text=row["tweet"])
    return instruction

def get_output(row):
    return 'yes' if row['label'] == "__label__Fake" else 'no'

# Generate instructions and outputs
inst_data = pd.DataFrame(columns=['instruction', 'output'])

for i, row in raw_data.iterrows():
    instruction = get_instruction(row)
    output = get_output(row)
    inst_data = pd.concat([inst_data, pd.DataFrame({'instruction': [instruction], 'output': [output]})], ignore_index=True)

# Display the final instruction dataset
inst_data.to_parquet('../../data/AFN/afn.parquet', index= False)
# sample n=1500 and save to the same parquet file
sampled_data = inst_data.sample(n=1500, random_state=42).to_parquet('../../data/AFN/afn.parquet', index=False)