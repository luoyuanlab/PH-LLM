# -*- coding: utf-8 -*-
# ToLD-BR (Brazilian Portuguese Toxic Lang.)
# - Paper: https://arxiv.org/abs/2010.04543
# - Data: https://github.com/JAugusto97/ToLD-Br

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import pandas as pd
import random
from utils.data_utils import balance_classes

df = pd.read_csv('../../data/ToLD-BR/ToLD-BR.csv')


# Assuming df is already loaded with your data

# Define the instruction templates in different languages
instruction_templates = {
    "Chinese": """
    请预测以下推文的注释。
    共计 3 名注释者根据他们对推文是否包含恐同的理解对每条推文进行了注释。
    如果您认为至少 2 名注释者认为该推文包含恐同，请回答 'yes'。否则，请回答 'no'。

    推文：{}""",

    "English": """
    Please predict the annotation of the following tweet.
    A total of 3 annotators have annotated each tweet according to their understanding of whether the tweet contains homophobia or not.
    If you think at least 2 annotators believe that the tweet contains homophobia, please respond with 'yes'. Otherwise, please respond with 'no'.

    Tweet: {}""",

    "German": """
    Bitte sagen Sie die Annotation des folgenden Tweets voraus.
    Insgesamt haben 3 Annotatoren jeden Tweet gemäß ihrem Verständnis darüber annotiert, ob der Tweet Homophobie enthält oder nicht.
    Wenn Sie glauben, dass mindestens 2 Annotatoren der Meinung sind, dass der Tweet Homophobie enthält, antworten Sie bitte mit 'yes'. Andernfalls antworten Sie bitte mit 'no'.

    Tweet: {}""",

    "French": """
    Veuillez prédire l'annotation du tweet suivant.
    Un total de 3 annotateurs ont annoté chaque tweet selon leur compréhension de la question de savoir si le tweet contient de l'homophobie ou non.
    Si vous pensez qu'au moins 2 annotateurs pensent que le tweet contient de l'homophobie, veuillez répondre par 'yes'. Sinon, veuillez répondre par 'no'.

    Tweet : {}""",

    "Spanish": """
    Por favor, prediga la anotación del siguiente tweet.
    Un total de 3 anotadores han anotado cada tweet según su comprensión de si el tweet contiene homofobia o no.
    Si cree que al menos 2 anotadores creen que el tweet contiene homofobia, responda con 'yes'. De lo contrario, responda con 'no'.

    Tweet: {}""",

    "Portuguese": """
    Por favor, preveja a anotação do seguinte tweet.
    Um total de 3 anotadores anotaram cada tweet de acordo com seu entendimento de se o tweet contém homofobia ou não.
    Se você acha que pelo menos 2 anotadores acreditam que o tweet contém homofobia, responda com 'yes'. Caso contrário, responda com 'no'.

    Tweet: {}""",

    "Italian": """
    Si prega di prevedere l'annotazione del seguente tweet.
    Un totale di 3 annotatori hanno annotato ogni tweet secondo la loro comprensione di se il tweet contiene omofobia o meno.
    Se pensi che almeno 2 annotatori credano che il tweet contenga omofobia, rispondi con 'yes'. Altrimenti, rispondi con 'no'.

    Tweet: {}""",

    "Dutch": """
    Voorspel alstublieft de annotatie van de volgende tweet.
    In totaal hebben 3 annotatoren elke tweet geannoteerd volgens hun begrip van of de tweet homofobie bevat of niet.
    Als u denkt dat ten minste 2 annotatoren geloven dat de tweet homofobie bevat, antwoord dan met 'yes'. Zo niet, antwoord dan met 'no'.

    Tweet: {}""",

    "Russian": """
    Пожалуйста, предскажите аннотацию следующего твита.
    В общей сложности 3 аннотатора аннотировали каждый твит в соответствии с их пониманием того, содержит ли твит гомофобию или нет.
    Если вы считаете, что как минимум 2 аннотатора считают, что твит содержит гомофобию, ответьте 'yes'. В противном случае ответьте 'no'.

    Твит: {}""",

    "Czech": """
    Předpovězte prosím anotaci následujícího tweetu.
    Celkem 3 anotátoři anotovali každý tweet podle toho, jak chápou, zda tweet obsahuje homofobii nebo ne.
    Pokud si myslíte, že alespoň 2 anotátoři věří, že tweet obsahuje homofobii, odpovězte prosím 'yes'. V opačném případě odpovězte 'no'.

    Tweet: {}""",

    "Polish": """
    Proszę przewidzieć adnotację następującego tweeta.
    Łącznie 3 adnotatorzy oznaczyli każdy tweet zgodnie z ich zrozumieniem, czy tweet zawiera homofobię, czy nie.
    Jeśli uważasz, że co najmniej 2 adnotatorów uważa, że tweet zawiera homofobię, odpowiedz 'yes'. W przeciwnym razie odpowiedz 'no'.

    Tweet: {}""",

    "Arabic": """
    يرجى التنبؤ بتعليق التغريدة التالية.
    قام 3 ملاحظين بتعليق كل تغريدة وفقًا لفهمهم لما إذا كانت التغريدة تحتوي على رهاب المثلية أم لا.
    إذا كنت تعتقد أن ما لا يقل عن 2 من الملاحظين يعتقدون أن التغريدة تحتوي على رهاب المثلية، فيرجى الرد بـ 'yes'. خلاف ذلك، يرجى الرد بـ 'no'.

    التغريدة: {}""",

    "Persian": """
    لطفاً پیش‌بینی کنید که حاشیه‌نویسی توییت زیر چگونه خواهد بود.
    مجموعاً ۳ حاشیه‌نویس، هر توییت را بر اساس فهم خود از اینکه آیا توییت حاوی همجنس‌گراهراسی است یا نه، حاشیه‌نویسی کرده‌اند.
    اگر فکر می‌کنید که حداقل ۲ حاشیه‌نویس بر این باورند که توییت حاوی همجنس‌گراهراسی است، لطفاً با 'yes' پاسخ دهید. در غیر این صورت، لطفاً با 'no' پاسخ دهید.

    توییت: {}""",

    "Hebrew": """
    נא לחזות את ההערה של הציוץ הבא.
    בסך הכל 3 מסמנים סימנו כל ציוץ על פי הבנתם האם הציוץ מכיל הומופוביה או לא.
    אם אתה חושב שלפחות 2 מסמנים מאמינים שהציוץ מכיל הומופוביה, נא להשיב 'yes'. אחרת, נא להשיב 'no'.

    ציוץ: {}""",

    "Turkish": """
    Lütfen aşağıdaki tweetin anotasyonunu tahmin edin.
    Toplam 3 anotatör, her tweeti tweetin homofobi içerip içermediğini anlama şekline göre anotasyon yaptı.
    En az 2 anotatörün tweetin homofobi içerdiğine inandığını düşünüyorsanız, lütfen 'yes' ile yanıtlayın. Aksi takdirde 'no' ile yanıtlayın.

    Tweet: {}""",

    "Japanese": """
    次のツイートの注釈を予測してください。
    合計3人のアノテーターが、それぞれのツイートがホモフォビアを含んでいるかどうかについて理解に基づいて注釈を付けました。
    2人以上のアノテーターがそのツイートがホモフォビアを含んでいると考えている場合、「yes」と回答してください。 それ以外の場合は、「no」と回答してください。

    ツイート: {}""",

    "Korean": """
    다음 트윗의 주석을 예측하세요.
    총 3명의 주석자가 트윗에 대해 트윗이 동성애 혐오를 포함하는지 여부에 따라 주석을 달았습니다.
    최소 2명의 주석자가 트윗에 동성애 혐오가 포함되어 있다고 생각한다면 'yes'라고 응답하세요. 그렇지 않으면 'no'라고 응답하세요.

    트윗: {}""",

    "Vietnamese": """
    Vui lòng dự đoán chú thích của tweet sau.
    Tổng cộng có 3 người chú thích đã chú thích từng tweet theo cách hiểu của họ về việc liệu tweet đó có chứa homophobia hay không.
    Nếu bạn nghĩ rằng ít nhất 2 người chú thích tin rằng tweet chứa homophobia, vui lòng trả lời 'yes'. Nếu không, vui lòng trả lời 'no'.

    Tweet: {}""",

    "Thai": """
    โปรดทำนายคำอธิบายของทวีตต่อไปนี้
    ผู้ตรวจสอบทั้งหมด 3 คนได้อธิบายทวีตแต่ละรายการตามความเข้าใจของพวกเขาว่าทวีตนั้นมีการเกลียดกลัวเพศที่สามหรือไม่
    หากคุณคิดว่ามีผู้ตรวจสอบอย่างน้อย 2 คนเชื่อว่าทวีตนั้นมีการเกลียดกลัวเพศที่สาม โปรดตอบกลับด้วย 'yes' มิฉะนั้น โปรดตอบกลับด้วย 'no'

    ทวีต: {}""",

    "Indonesian": """
    Silakan prediksi anotasi dari tweet berikut ini.
    Sebanyak 3 anotator telah mengomentari setiap tweet berdasarkan pemahaman mereka tentang apakah tweet tersebut mengandung homofobia atau tidak.
    Jika Anda berpikir setidaknya 2 anotator percaya bahwa tweet tersebut mengandung homofobia, harap balas dengan 'yes'. Jika tidak, harap balas dengan 'no'.

    Tweet: {}""",

    "Malay": """
    Sila ramalkan anotasi tweet berikut.
    Sebanyak 3 penanda telah menandakan setiap tweet mengikut kefahaman mereka sama ada tweet itu mengandungi homofobia atau tidak.
    Jika anda fikir sekurang-kurangnya 2 penanda percaya bahawa tweet itu mengandungi homofobia, sila balas dengan 'yes'. Jika tidak, sila balas dengan 'no'.

    Tweet: {}""",

    "Lao": """
    ກະລຸນາທຳນາຍການອະທິບາຍຂອງທະວີດຕໍາຕໍ່ໄປນີ້.
    ຜູ້ທຳອະທິບາຍທັງຫມົດ 3 ຄົນໄດ້ທຳອະທິບາຍແຕ່ລະທະວີດຕາມຄວາມເຂົ້າໃຈຂອງພວກເຂົາວ່າທະວີດມີຄວາມກົດລັງເພດຫລືບໍ່.
    ຖ້າທ່ານຄິດວ່າຢ່າງຫນ້ອຍ 2 ຄົນເຊື່ອວ່າທະວີດນີ້ມີຄວາມກົດລັງເພດ, ກະລຸນາຕອບກັບ 'yes'. ຖ້າບໍ່ແມ່ນ, ກະລຸນາຕອບກັບ 'no'.

    ທະວີດ: {}""",

    "Burmese": """
    ကျေးဇူးပြု၍ အောက်ပါ တွစ်အတွက် မှတ်ချက်ကို ခန့်မှန်းပါ။
    တွစ်တိုင်းကို ၎င်းတို့၏ နားလည်မှုအရ ၃ ဦးမှတ်ချက်ပြုခဲ့ကြသည်၊ တစ်တွစ်တွင် အထက်ပါသဘောထားများပါဝင်မရှိ မေးမြန်းထားပါသည်။
    တစ်တွစ်တွင် အထက်ပါသဘောထားများပါဝင်ကြောင်း ယုံကြည်ချက်ရှိရင် 'yes' နှင့်တုံ့ပြန်ပါ။ မဟုတ်ပါက 'no' နှင့်တုံ့ပြန်ပါ။

    တြစ်: {}""",

    "Cebuano": """
    Palihug tag-ani ang anotasyon sa musunod nga tweet.
    Sa kinatibuk-an, adunay 3 ka mga anotator nga nag-anotato sa matag tweet sumala sa ilang pagsabot kung ang tweet naglambigit ba sa homophobia o dili.
    Kung sa imong hunahuna adunay labing menos 2 ka mga anotator nga nagtuo nga ang tweet naglambigit sa homophobia, palihug tubaga kini og 'yes'. Kung dili, palihug tubaga kini og 'no'.

    Tweet: {}""",

    "Khmer": """
    សូមទាយមើលពីការបញ្ជាក់នៃការបញ្ចូលខាងក្រោម។
    សរុបអ្នកមើល 3 នាក់បានបញ្ចេញមតិជាមួយការបញ្ជាក់ខាងក្រោមដោយផ្អែកលើការយល់ចិត្តអំពីមើលឃើញកំហុសជាមួយភាពមិនទៀងទាត់ឬទេ។
    ប្រសិនបើអ្នកយល់ថានៃអ្នកមើល 2 នាក់បានសម្រេចថាខាងក្រោមមានភាពមិនទៀងទាត់ សូមទទួលយកជាមួយនឹងការពិត 'yes'. បើមិនមែន សូមទទួលយកជាមួយនឹង 'no'.

    Tweet: {}""",

    "Tagalog": """
    Pakihulaan ang anotasyon ng sumusunod na tweet.
    Isang kabuuan ng 3 anotador ang nag-anotasyon ng bawat tweet ayon sa kanilang pagkakaintindi kung naglalaman ng homophobia ang tweet o hindi.
    Kung sa tingin mo ay naniniwala ang hindi bababa sa 2 anotador na naglalaman ng homophobia ang tweet, mangyaring tumugon ng 'yes'. Kung hindi, mangyaring tumugon ng 'no'.

    Tweet: {}""",

    "Hindi": """
    कृपया निम्नलिखित ट्वीट की एनोटेशन का अनुमान लगाएं।
    कुल 3 एनोटेटरों ने प्रत्येक ट्वीट को एनोटेट किया है कि उनके समझ के अनुसार ट्वीट में समलैंगिकता विरोधी भावनाएँ हैं या नहीं।
    यदि आपको लगता है कि कम से कम 2 एनोटेटर मानते हैं कि ट्वीट में समलैंगिकता विरोधी भावनाएँ हैं, तो कृपया 'yes' के साथ उत्तर दें। अन्यथा, कृपया 'no' के साथ उत्तर दें।

    ट्वीट: {}""",

    "Bengali": """
    অনুগ্রহ করে নিম্নলিখিত টুইটের ব্যাখ্যা অনুমান করুন।
    মোট ৩ জন ব্যাখ্যাকারী প্রতিটি টুইটকে তাদের বোঝার উপর ভিত্তি করে ব্যাখ্যা করেছেন যে টুইটটিতে সমকামিতা বিরোধীতা রয়েছে কিনা।
    আপনি যদি মনে করেন অন্তত ২ জন ব্যাখ্যাকারী বিশ্বাস করেন যে টুইটটিতে সমকামিতা বিরোধীতা রয়েছে, অনুগ্রহ করে 'yes' দিয়ে উত্তর দিন। অন্যথায়, 'no' দিয়ে উত্তর দিন।

    টুইট: {}""",

    "Urdu": """
    براہ کرم درج ذیل ٹویٹ کی تشریح کی پیش گوئی کریں۔
    3 تشریح کاروں نے کل ملا کر ہر ٹویٹ کی تشریح کی ہے کہ آیا ٹویٹ میں ہم جنس پرستی کی مخالفت شامل ہے یا نہیں۔
    اگر آپ کو لگتا ہے کہ کم از کم 2 تشریح کار یہ سمجھتے ہیں کہ ٹویٹ میں ہم جنس پرستی کی مخالفت شامل ہے، تو براہ کرم 'yes' کے ساتھ جواب دیں۔ بصورت دیگر، براہ کرم 'no' کے ساتھ جواب دیں۔

    ٹویٹ: {}"""
}


# Create a new DataFrame with the required columns
df1 = df[['text', 'homophobia']].copy()

# Generate the instructions using randomly selected templates
df1['instruction'] = df1['text'].apply(
    lambda x: random.choice(list(instruction_templates.values())).format(x)
)

# Generate the outputs with English labels "yes" or "no"
df1['output'] = df1['homophobia'].apply(lambda x: 'yes' if x >= 2 else 'no')

# Keep only the 'instruction' and 'label' columns
df1 = df1[['instruction', 'output']]

# Shuffle the DataFrame and save to Parquet
df1 = df1.sample(frac=1).reset_index(drop=True)


df1 = balance_classes(df1, 200, 'output')
df1.to_parquet('../../data/ToLD-BR/ToLD-BR-homophobia.parquet', index=False)


# Assuming df is already loaded with your data

# Define the instruction templates in different languages
instruction_templates = {
    "Chinese": """
    请预测以下推文的注释。
    共计 3 名注释者根据他们对推文是否包含淫秽语言的理解对每条推文进行了注释。
    如果您认为至少 2 名注释者认为该推文包含淫秽语言，请回答 'yes'。否则，请回答 'no'。

    推文：{}""",

    "English": """
    Please predict the annotation of the following tweet.
    A total of 3 annotators have annotated each tweet according to their understanding of whether the tweet contains obscene language or not.
    If you think at least 2 annotators believe that the tweet contains obscene language, please respond with 'yes'. Otherwise, please respond with 'no'.

    Tweet: {}""",

    "German": """
    Bitte sagen Sie die Annotation des folgenden Tweets voraus.
    Insgesamt haben 3 Annotatoren jeden Tweet gemäß ihrem Verständnis darüber annotiert, ob der Tweet obszöne Sprache enthält oder nicht.
    Wenn Sie glauben, dass mindestens 2 Annotatoren der Meinung sind, dass der Tweet obszöne Sprache enthält, antworten Sie bitte mit 'yes'. Andernfalls antworten Sie bitte mit 'no'.

    Tweet: {}""",

    "French": """
    Veuillez prédire l'annotation du tweet suivant.
    Un total de 3 annotateurs ont annoté chaque tweet selon leur compréhension de la question de savoir si le tweet contient un langage obscène ou non.
    Si vous pensez qu'au moins 2 annotateurs pensent que le tweet contient un langage obscène, veuillez répondre par 'yes'. Sinon, veuillez répondre par 'no'.

    Tweet : {}""",

    "Spanish": """
    Por favor, prediga la anotación del siguiente tweet.
    Un total de 3 anotadores han anotado cada tweet según su comprensión de si el tweet contiene lenguaje obsceno o no.
    Si cree que al menos 2 anotadores creen que el tweet contiene lenguaje obsceno, responda con 'yes'. De lo contrario, responda con 'no'.

    Tweet: {}""",

    "Portuguese": """
    Por favor, preveja a anotação do seguinte tweet.
    Um total de 3 anotadores anotaram cada tweet de acordo com seu entendimento de se o tweet contém linguagem obscena ou não.
    Se você acha que pelo menos 2 anotadores acreditam que o tweet contém linguagem obscena, responda com 'yes'. Caso contrário, responda com 'no'.

    Tweet: {}""",

    "Italian": """
    Si prega di prevedere l'annotazione del seguente tweet.
    Un totale di 3 annotatori hanno annotato ogni tweet secondo la loro comprensione di se il tweet contiene linguaggio osceno o meno.
    Se pensi che almeno 2 annotatori credano che il tweet contenga linguaggio osceno, rispondi con 'yes'. Altrimenti, rispondi con 'no'.

    Tweet: {}""",

    "Dutch": """
    Voorspel alstublieft de annotatie van de volgende tweet.
    In totaal hebben 3 annotatoren elke tweet geannoteerd volgens hun begrip van of de tweet obsceen taalgebruik bevat of niet.
    Als u denkt dat ten minste 2 annotatoren geloven dat de tweet obsceen taalgebruik bevat, antwoord dan met 'yes'. Zo niet, antwoord dan met 'no'.

    Tweet: {}""",

    "Russian": """
    Пожалуйста, предскажите аннотацию следующего твита.
    В общей сложности 3 аннотатора аннотировали каждый твит в соответствии с их пониманием того, содержит ли твит непристойные выражения или нет.
    Если вы считаете, что как минимум 2 аннотатора считают, что твит содержит непристойные выражения, ответьте 'yes'. В противном случае ответьте 'no'.

    Твит: {}""",

    "Czech": """
    Předpovězte prosím anotaci následujícího tweetu.
    Celkem 3 anotátoři anotovali každý tweet podle toho, jak chápou, zda tweet obsahuje obscénní jazyk nebo ne.
    Pokud si myslíte, že alespoň 2 anotátoři věří, že tweet obsahuje obscénní jazyk, odpovězte prosím 'yes'. V opačném případě odpovězte 'no'.

    Tweet: {}""",

    "Polish": """
    Proszę przewidzieć adnotację następującego tweeta.
    Łącznie 3 adnotatorzy oznaczyli każdy tweet zgodnie z ich zrozumieniem, czy tweet zawiera język obsceniczny, czy nie.
    Jeśli uważasz, że co najmniej 2 adnotatorów uważa, że tweet zawiera język obsceniczny, odpowiedz 'yes'. W przeciwnym razie odpowiedz 'no'.

    Tweet: {}""",

    "Arabic": """
    يرجى التنبؤ بتعليق التغريدة التالية.
    قام 3 ملاحظين بتعليق كل تغريدة وفقًا لفهمهم لما إذا كانت التغريدة تحتوي على لغة فاحشة أم لا.
    إذا كنت تعتقد أن ما لا يقل عن 2 من الملاحظين يعتقدون أن التغريدة تحتوي على لغة فاحشة، فيرجى الرد بـ 'yes'. خلاف ذلك، يرجى الرد بـ 'no'.

    التغريدة: {}""",

    "Persian": """
    لطفاً پیش‌بینی کنید که حاشیه‌نویسی توییت زیر چگونه خواهد بود.
    مجموعاً ۳ حاشیه‌نویس، هر توییت را بر اساس فهم خود از اینکه آیا توییت حاوی زبان زشت است یا نه، حاشیه‌نویسی کرده‌اند.
    اگر فکر می‌کنید که حداقل ۲ حاشیه‌نویس بر این باورند که توییت حاوی زبان زشت است، لطفاً با 'yes' پاسخ دهید. در غیر این صورت، لطفاً با 'no' پاسخ دهید.

    توییت: {}""",

    "Hebrew": """
    נא לחזות את ההערה של הציוץ הבא.
    בסך הכל 3 מסמנים סימנו כל ציוץ על פי הבנתם האם הציוץ מכיל שפה גסה או לא.
    אם אתה חושב שלפחות 2 מסמנים מאמינים שהציוץ מכיל שפה גסה, נא להשיב 'yes'. אחרת, נא להשיב 'no'.

    ציוץ: {}""",

    "Turkish": """
    Lütfen aşağıdaki tweetin anotasyonunu tahmin edin.
    Toplam 3 anotatör, her tweeti tweetin müstehcen dil içerip içermediğini anlama şekline göre anotasyon yaptı.
    En az 2 anotatörün tweetin müstehcen dil içerdiğine inandığını düşünüyorsanız, lütfen 'yes' ile yanıtlayın. Aksi takdirde 'no' ile yanıtlayın.

    Tweet: {}""",

    "Japanese": """
    次のツイートの注釈を予測してください。
    合計3人のアノテーターが、それぞれのツイートが猥褻な言葉を含んでいるかどうかについて理解に基づいて注釈を付けました。
    2人以上のアノテーターがそのツイートが猥褻な言葉を含んでいると考えている場合、「yes」と回答してください。 それ以外の場合は、「no」と回答してください。

    ツイート: {}""",

    "Korean": """
    다음 트윗의 주석을 예측하세요.
    총 3명의 주석자가 트윗에 대해 트윗이 음란한 언어를 포함하는지 여부에 따라 주석을 달았습니다.
    최소 2명의 주석자가 트윗에 음란한 언어가 포함되어 있다고 생각한다면 'yes'라고 응답하세요. 그렇지 않으면 'no'라고 응답하세요.

    트윗: {}""",

    "Vietnamese": """
    Vui lòng dự đoán chú thích của tweet sau.
    Tổng cộng có 3 người chú thích đã chú thích từng tweet theo cách hiểu của họ về việc liệu tweet đó có chứa ngôn ngữ tục tĩu hay không.
    Nếu bạn nghĩ rằng ít nhất 2 người chú thích tin rằng tweet chứa ngôn ngữ tục tĩu, vui lòng trả lời 'yes'. Nếu không, vui lòng trả lời 'no'.

    Tweet: {}""",

    "Thai": """
    โปรดทำนายคำอธิบายของทวีตต่อไปนี้
    ผู้ตรวจสอบทั้งหมด 3 คนได้อธิบายทวีตแต่ละรายการตามความเข้าใจของพวกเขาว่าทวีตนั้นมีภาษาหยาบคายหรือไม่
    หากคุณคิดว่ามีผู้ตรวจสอบอย่างน้อย 2 คนเชื่อว่าทวีตนั้นมีภาษาหยาบคาย โปรดตอบกลับด้วย 'yes' มิฉะนั้น โปรดตอบกลับด้วย 'no'

    ทวีต: {}""",

    "Indonesian": """
    Silakan prediksi anotasi dari tweet berikut ini.
    Sebanyak 3 anotator telah mengomentari setiap tweet berdasarkan pemahaman mereka tentang apakah tweet tersebut mengandung bahasa cabul atau tidak.
    Jika Anda berpikir setidaknya 2 anotator percaya bahwa tweet tersebut mengandung bahasa cabul, harap balas dengan 'yes'. Jika tidak, harap balas dengan 'no'.

    Tweet: {}""",

    "Malay": """
    Sila ramalkan anotasi tweet berikut.
    Sebanyak 3 penanda telah menandakan setiap tweet mengikut kefahaman mereka sama ada tweet itu mengandungi bahasa lucah atau tidak.
    Jika anda fikir sekurang-kurangnya 2 penanda percaya bahawa tweet itu mengandungi bahasa lucah, sila balas dengan 'yes'. Jika tidak, sila balas dengan 'no'.

    Tweet: {}""",

    "Lao": """
    ກະລຸນາທຳນາຍການອະທິບາຍຂອງທະວີດຕໍາຕໍ່ໄປນີ້.
    ຜູ້ທຳອະທິບາຍທັງຫມົດ 3 ຄົນໄດ້ທຳອະທິບາຍແຕະລະທະວີດຕາມຄວາມເຂົ້າໃຈຂອງພວກເຂົາວ່າທະວີດນີ້ມີພາສາທີ່ຫຍາບຄາຍຫລືບໍ່.
    ຖ້າທ່ານຄິດວ່າຢ່າງຫນ້ອຍ 2 ຄົນເຊື່ອວ່າທະວີດນີ້ມີພາສາທີ່ຫຍາບຄາຍ, ກະລຸນາຕອບກັບ 'yes'. ຖ້າບໍ່ແມ່ນ, ກະລຸນາຕອບກັບ 'no'.

    ທະວີດ: {}""",

    "Burmese": """
    ကျေးဇူးပြု၍ အောက်ပါ တွစ်အတွက် မှတ်ချက်ကို ခန့်မှန်းပါ။
    တွစ်တိုင်းကို ၎င်းတို့၏ နားလည်မှုအရ ၃ ဦးမှတ်ချက်ပြုခဲ့ကြသည်၊ တစ်တွစ်တွင် အထက်ပါသဘောထားများပါဝင်မရှိ မေးမြန်းထားပါသည်။
    တစ်တွစ်တွင် အထက်ပါသဘောထားများပါဝင်ကြောင်း ယုံကြည်ချက်ရှိရင် 'yes' နှင့်တုံ့ပြန်ပါ။ မဟုတ်ပါက 'no' နှင့်တုံ့ပြန်ပါ။

    တြစ်: {}""",

    "Cebuano": """
    Palihug tag-ani ang anotasyon sa musunod nga tweet.
    Sa kinatibuk-an, adunay 3 ka mga anotator nga nag-anotato sa matag tweet sumala sa ilang pagsabot kung ang tweet naglambigit ba sa malaw-ay nga pinulongan o dili.
    Kung sa imong hunahuna adunay labing menos 2 ka mga anotator nga nagtuo nga ang tweet naglambigit sa malaw-ay nga pinulongan, palihug tubaga kini og 'yes'. Kung dili, palihug tubaga kini og 'no'.

    Tweet: {}""",

    "Khmer": """
    សូមទាយមើលពីការបញ្ជាក់នៃការបញ្ចូលខាងក្រោម។
    សរុបអ្នកមើល 3 នាក់បានបញ្ចេញមតិជាមួយការបញ្ជាក់ខាងក្រោមដោយផ្អែកលើការយល់ចិត្តអំពីមើលឃើញកំហុសជាមួយភាសាដែលមិនត្រឹមត្រូវ ឬទេ។
    ប្រសិនបើអ្នកយល់ថានៃអ្នកមើល 2 នាក់បានសម្រេចថាខាងក្រោមមានភាសាដែលមិនត្រឹមត្រូវ សូមទទួលយកជាមួយនឹងការពិត 'yes'. បើមិនមែន សូមទទួលយកជាមួយនឹង 'no'.

    Tweet: {}""",

    "Tagalog": """
    Pakihulaan ang anotasyon ng sumusunod na tweet.
    Isang kabuuan ng 3 anotador ang nag-anotasyon ng bawat tweet ayon sa kanilang pagkakaintindi kung naglalaman ng bastos na salita ang tweet o hindi.
    Kung sa tingin mo ay naniniwala ang hindi bababa sa 2 anotador na naglalaman ng bastos na salita ang tweet, mangyaring tumugon ng 'yes'. Kung hindi, mangyaring tumugon ng 'no'.

    Tweet: {}""",

    "Hindi": """
    कृपया निम्नलिखित ट्वीट की एनोटेशन का अनुमान लगाएं।
    कुल 3 एनोटेटरों ने प्रत्येक ट्वीट को एनोटेट किया है कि उनके समझ के अनुसार ट्वीट में अश्लील भाषा है या नहीं।
    यदि आपको लगता है कि कम से कम 2 एनोटेटर मानते हैं कि ट्वीट में अश्लील भाषा है, तो कृपया 'yes' के साथ उत्तर दें। अन्यथा, कृपया 'no' के साथ उत्तर दें।

    ट्वीट: {}""",

    "Bengali": """
    অনুগ্রহ করে নিম্নলিখিত টুইটের ব্যাখ্যা অনুমান করুন।
    মোট ৩ জন ব্যাখ্যাকারী প্রতিটি টুইটকে তাদের বোঝার উপর ভিত্তি করে ব্যাখ্যা করেছেন যে টুইটটিতে অশ্লীল ভাষা রয়েছে কিনা।
    আপনি যদি মনে করেন অন্তত ২ জন ব্যাখ্যাকারী বিশ্বাস করেন যে টুইটটিতে অশ্লীল ভাষা রয়েছে, অনুগ্রহ করে 'yes' দিয়ে উত্তর দিন। অন্যথায়, 'no' দিয়ে উত্তর দিন।

    টুইট: {}""",

    "Urdu": """
    براہ کرم درج ذیل ٹویٹ کی تشریح کی پیش گوئی کریں۔
    3 تشریح کاروں نے کل ملا کر ہر ٹویٹ کی تشریح کی ہے کہ آیا ٹویٹ میں فحش زبان شامل ہے یا نہیں۔
    اگر آپ کو لگتا ہے کہ کم از کم 2 تشریح کار یہ سمجھتے ہیں کہ ٹویٹ میں فحش زبان شامل ہے، تو براہ کرم 'yes' کے ساتھ جواب دیں۔ بصورت دیگر، براہ کرم 'no' کے ساتھ جواب دیں۔

    ٹویٹ: {}"""
}


# Create a new DataFrame with the required columns
df2 = df[['text', 'obscene']].copy()

# Generate the instructions using randomly selected templates
df2['instruction'] = df2['text'].apply(
    lambda x: random.choice(list(instruction_templates.values())).format(x)
)

# Generate the outputs with English labels "yes" or "no"
df2['output'] = df2['obscene'].apply(lambda x: 'yes' if x >= 2 else 'no')

# Keep only the 'instruction' and 'label' columns
df2 = df2[['instruction', 'output']]


df2 = balance_classes(df2, 2500, 'output')

# Shuffle the DataFrame and save to Parquet
df2 = df2.sample(frac=1).reset_index(drop=True)
df2.to_parquet('../../data/ToLD-BR/ToLD-BR-obscene.parquet', index=False)

# Define the instruction templates in different languages
instruction_templates = {
    "Chinese": """
    请预测以下推文的注释。
    共计 3 名注释者根据他们对推文是否包含侮辱的理解对每条推文进行了注释。
    如果您认为至少 2 名注释者认为该推文包含侮辱，请回答 'yes'。否则，请回答 'no'。

    推文：{}""",

    "English": """
    Please predict the annotation of the following tweet.
    A total of 3 annotators have annotated each tweet according to their understanding of whether the tweet contains insult or not.
    If you think at least 2 annotators believe that the tweet contains insult, please respond with 'yes'. Otherwise, please respond with 'no'.

    Tweet: {}""",

    "German": """
    Bitte sagen Sie die Annotation des folgenden Tweets voraus.
    Insgesamt haben 3 Annotatoren jeden Tweet gemäß ihrem Verständnis darüber annotiert, ob der Tweet eine Beleidigung enthält oder nicht.
    Wenn Sie glauben, dass mindestens 2 Annotatoren der Meinung sind, dass der Tweet eine Beleidigung enthält, antworten Sie bitte mit 'yes'. Andernfalls antworten Sie bitte mit 'no'.

    Tweet: {}""",

    "French": """
    Veuillez prédire l'annotation du tweet suivant.
    Un total de 3 annotateurs ont annoté chaque tweet selon leur compréhension de la question de savoir si le tweet contient une insulte ou non.
    Si vous pensez qu'au moins 2 annotateurs pensent que le tweet contient une insulte, veuillez répondre par 'yes'. Sinon, veuillez répondre par 'no'.

    Tweet : {}""",

    "Spanish": """
    Por favor, prediga la anotación del siguiente tweet.
    Un total de 3 anotadores han anotado cada tweet según su comprensión de si el tweet contiene un insulto o no.
    Si cree que al menos 2 anotadores creen que el tweet contiene un insulto, responda con 'yes'. De lo contrario, responda con 'no'.

    Tweet: {}""",

    "Portuguese": """
    Por favor, preveja a anotação do seguinte tweet.
    Um total de 3 anotadores anotaram cada tweet de acordo com seu entendimento de se o tweet contém insulto ou não.
    Se você acha que pelo menos 2 anotadores acreditam que o tweet contém insulto, responda com 'yes'. Caso contrário, responda com 'no'.

    Tweet: {}""",

    "Italian": """
    Si prega di prevedere l'annotazione del seguente tweet.
    Un totale di 3 annotatori hanno annotato ogni tweet secondo la loro comprensione di se il tweet contiene insulti o meno.
    Se pensi che almeno 2 annotatori credano che il tweet contenga insulti, rispondi con 'yes'. Altrimenti, rispondi con 'no'.

    Tweet: {}""",

    "Dutch": """
    Voorspel alstublieft de annotatie van de volgende tweet.
    In totaal hebben 3 annotatoren elke tweet geannoteerd volgens hun begrip van of de tweet beledigingen bevat of niet.
    Als u denkt dat ten minste 2 annotatoren geloven dat de tweet beledigingen bevat, antwoord dan met 'yes'. Zo niet, antwoord dan met 'no'.

    Tweet: {}""",

    "Russian": """
    Пожалуйста, предскажите аннотацию следующего твита.
    В общей сложности 3 аннотатора аннотировали каждый твит в соответствии с их пониманием того, содержит ли твит оскорбления или нет.
    Если вы считаете, что как минимум 2 аннотатора считают, что твит содержит оскорбления, ответьте 'yes'. В противном случае ответьте 'no'.

    Твит: {}""",

    "Czech": """
    Předpovězte prosím anotaci následujícího tweetu.
    Celkem 3 anotátoři anotovali každý tweet podle toho, jak chápou, zda tweet obsahuje urážku nebo ne.
    Pokud si myslíte, že alespoň 2 anotátoři věří, že tweet obsahuje urážku, odpovězte prosím 'yes'. V opačném případě odpovězte 'no'.

    Tweet: {}""",

    "Polish": """
    Proszę przewidzieć adnotację następującego tweeta.
    Łącznie 3 adnotatorzy oznaczyli każdy tweet zgodnie z ich zrozumieniem, czy tweet zawiera obelgi, czy nie.
    Jeśli uważasz, że co najmniej 2 adnotatorów uważa, że tweet zawiera obelgi, odpowiedz 'yes'. W przeciwnym razie odpowiedz 'no'.

    Tweet: {}""",

    "Arabic": """
    يرجى التنبؤ بتعليق التغريدة التالية.
    قام 3 ملاحظين بتعليق كل تغريدة وفقًا لفهمهم لما إذا كانت التغريدة تحتوي على إهانة أم لا.
    إذا كنت تعتقد أن ما لا يقل عن 2 من الملاحظين يعتقدون أن التغريدة تحتوي على إهانة، فيرجى الرد بـ 'yes'. خلاف ذلك، يرجى الرد بـ 'no'.

    التغريدة: {}""",

    "Persian": """
    لطفاً پیش‌بینی کنید که حاشیه‌نویسی توییت زیر چگونه خواهد بود.
    مجموعاً ۳ حاشیه‌نویس، هر توییت را بر اساس فهم خود از اینکه آیا توییت حاوی توهین است یا نه، حاشیه‌نویسی کرده‌اند.
    اگر فکر می‌کنید که حداقل ۲ حاشیه‌نویس بر این باورند که توییت حاوی توهین است، لطفاً با 'yes' پاسخ دهید. در غیر این صورت، لطفاً با 'no' پاسخ دهید.

    توییت: {}""",

    "Hebrew": """
    נא לחזות את ההערה של הציוץ הבא.
    בסך הכל 3 מסמנים סימנו כל ציוץ על פי הבנתם האם הציוץ מכיל עלבון או לא.
    אם אתה חושב שלפחות 2 מסמנים מאמינים שהציוץ מכיל עלבון, נא להשיב 'yes'. אחרת, נא להשיב 'no'.

    ציוץ: {}""",

    "Turkish": """
    Lütfen aşağıdaki tweetin anotasyonunu tahmin edin.
    Toplam 3 anotatör, her tweeti tweetin hakaret içerip içermediğini anlama şekline göre anotasyon yaptı.
    En az 2 anotatörün tweetin hakaret içerdiğine inandığını düşünüyorsanız, lütfen 'yes' ile yanıtlayın. Aksi takdirde 'no' ile yanıtlayın.

    Tweet: {}""",

    "Japanese": """
    次のツイートの注釈を予測してください。
    合計3人のアノテーターが、それぞれのツイートが侮辱を含んでいるかどうかについて理解に基づいて注釈を付けました。
    2人以上のアノテーターがそのツイートが侮辱を含んでいると考えている場合、「yes」と回答してください。 それ以外の場合は、「no」と回答してください。

    ツイート: {}""",

    "Korean": """
    다음 트윗의 주석을 예측하세요.
    총 3명의 주석자가 트윗에 대해 트윗이 모욕을 포함하는지 여부에 따라 주석을 달았습니다.
    최소 2명의 주석자가 트윗에 모욕이 포함되어 있다고 생각한다면 'yes'라고 응답하세요. 그렇지 않으면 'no'라고 응답하세요.

    트윗: {}""",

    "Vietnamese": """
    Vui lòng dự đoán chú thích của tweet sau.
    Tổng cộng có 3 người chú thích đã chú thích từng tweet theo cách hiểu của họ về việc liệu tweet đó có chứa lời lăng mạ hay không.
    Nếu bạn nghĩ rằng ít nhất 2 người chú thích tin rằng tweet chứa lời lăng mạ, vui lòng trả lời 'yes'. Nếu không, vui lòng trả lời 'no'.

    Tweet: {}""",

    "Thai": """
    โปรดทำนายคำอธิบายของทวีตต่อไปนี้
    ผู้ตรวจสอบทั้งหมด 3 คนได้อธิบายทวีตแต่ละรายการตามความเข้าใจของพวกเขาว่าทวีตนั้นมีคำดูหมิ่นหรือไม่
    หากคุณคิดว่ามีผู้ตรวจสอบอย่างน้อย 2 คนเชื่อว่าทวีตนั้นมีคำดูหมิ่น โปรดตอบกลับด้วย 'yes' มิฉะนั้น โปรดตอบกลับด้วย 'no'

    ทวีต: {}""",

    "Indonesian": """
    Silakan prediksi anotasi dari tweet berikut ini.
    Sebanyak 3 anotator telah mengomentari setiap tweet berdasarkan pemahaman mereka tentang apakah tweet tersebut mengandung penghinaan atau tidak.
    Jika Anda berpikir setidaknya 2 anotator percaya bahwa tweet tersebut mengandung penghinaan, harap balas dengan 'yes'. Jika tidak, harap balas dengan 'no'.

    Tweet: {}""",

    "Malay": """
    Sila ramalkan anotasi tweet berikut.
    Sebanyak 3 penanda telah menandakan setiap tweet mengikut kefahaman mereka sama ada tweet itu mengandungi penghinaan atau tidak.
    Jika anda fikir sekurang-kurangnya 2 penanda percaya bahawa tweet itu mengandungi penghinaan, sila balas dengan 'yes'. Jika tidak, sila balas dengan 'no'.

    Tweet: {}""",

    "Lao": """
    ກະລຸນາທຳນາຍການອະທິບາຍຂອງທະວີດຕໍາຕໍ່ໄປນີ້.
    ຜູ້ທຳອະທິບາຍທັງຫມົດ 3 ຄົນໄດ້ທຳອະທິບາຍແຕ່ລະທະວີດຕາມຄວາມເຂົ້າໃຈຂອງພວກເຂົາວ່າທະວີດນີ້ມີການຫຍາບຄາຍຫລືບໍ່.
    ຖ້າທ່ານຄິດວ່າຢ່າງຫນ້ອຍ 2 ຄົນເຊື່ອວ່າທະວີດນີ້ມີການຫຍາບຄາຍ, ກະລຸນາຕອບກັບ 'yes'. ຖ້າບໍ່ແມ່ນ, ກະລຸນາຕອບກັບ 'no'.

    ທະວີດ: {}""",

    "Burmese": """
    ကျေးဇူးပြု၍ အောက်ပါ တွစ်အတွက် မှတ်ချက်ကို ခန့်မှန်းပါ။
    တွစ်တိုင်းကို ၎င်းတို့၏ နားလည်မှုအရ ၃ ဦးမှတ်ချက်ပြုခဲ့ကြသည်၊ တစ်တွစ်တွင် အထက်ပါသဘောထားများပါဝင်မရှိ မေးမြန်းထားပါသည်။
    တစ်တွစ်တွင် အထက်ပါသဘောထားများပါဝင်ကြောင်း ယုံကြည်ချက်ရှိရင် 'yes' နှင့်တုံ့ပြန်ပါ။ မဟုတ်ပါက 'no' နှင့်တုံ့ပြန်ပါ။

    တြစ်: {}""",

    "Cebuano": """
    Palihug tag-ani ang anotasyon sa musunod nga tweet.
    Sa kinatibuk-an, adunay 3 ka mga anotator nga nag-anotato sa matag tweet sumala sa ilang pagsabot kung ang tweet naglambigit ba sa insulto o dili.
    Kung sa imong hunahuna adunay labing menos 2 ka mga anotator nga nagtuo nga ang tweet naglambigit sa insulto, palihug tubaga kini og 'yes'. Kung dili, palihug tubaga kini og 'no'.

    Tweet: {}""",

    "Khmer": """
    សូមទាយមើលពីការបញ្ជាក់នៃការបញ្ចូលខាងក្រោម។
    សរុបអ្នកមើល 3 នាក់បានបញ្ចេញមតិជាមួយការបញ្ជាក់ខាងក្រោមដោយផ្អែកលើការយល់ចិត្តអំពីមើលឃើញកំហុសជាមួយការរិះគន់ឬទេ។
    ប្រសិនបើអ្នកយល់ថានៃអ្នកមើល 2 នាក់បានសម្រេចថាខាងក្រោមមានការរិះគន់ សូមទទួលយកជាមួយនឹងការពិត 'yes'. បើមិនមែន សូមទទួលយកជាមួយនឹង 'no'.

    Tweet: {}""",

    "Tagalog": """
    Pakihulaan ang anotasyon ng sumusunod na tweet.
    Isang kabuuan ng 3 anotador ang nag-anotasyon ng bawat tweet ayon sa kanilang pagkakaintindi kung naglalaman ng insulto ang tweet o hindi.
    Kung sa tingin mo ay naniniwala ang hindi bababa sa 2 anotador na naglalaman ng insulto ang tweet, mangyaring tumugon ng 'yes'. Kung hindi, mangyaring tumugon ng 'no'.

    Tweet: {}""",

    "Hindi": """
    कृपया निम्नलिखित ट्वीट की एनोटेशन का अनुमान लगाएं।
    कुल 3 एनोटेटरों ने प्रत्येक ट्वीट को एनोटेट किया है कि उनके समझ के अनुसार ट्वीट में अपमान है या नहीं।
    यदि आपको लगता है कि कम से कम 2 एनोटेटर मानते हैं कि ट्वीट में अपमान है, तो कृपया 'yes' के साथ उत्तर दें। अन्यथा, कृपया 'no' के साथ उत्तर दें।

    ट्वीट: {}""",

    "Bengali": """
    অনুগ্রহ করে নিম্নলিখিত টুইটের ব্যাখ্যা অনুমান করুন।
    মোট ৩ জন ব্যাখ্যাকারী প্রতিটি টুইটকে তাদের বোঝার উপর ভিত্তি করে ব্যাখ্যা করেছেন যে টুইটটিতে অপমান রয়েছে কিনা।
    আপনি যদি মনে করেন অন্তত ২ জন ব্যাখ্যাকারী বিশ্বাস করেন যে টুইটটিতে অপমান রয়েছে, অনুগ্রহ করে 'yes' দিয়ে উত্তর দিন। অন্যথায়, 'no' দিয়ে উত্তর দিন।

    টুইট: {}""",

    "Urdu": """
    براہ کرم درج ذیل ٹویٹ کی تشریح کی پیش گوئی کریں۔
    3 تشریح کاروں نے کل ملا کر ہر ٹویٹ کی تشریح کی ہے کہ آیا ٹویٹ میں توہین شامل ہے یا نہیں۔
    اگر آپ کو لگتا ہے کہ کم از کم 2 تشریح کار یہ سمجھتے ہیں کہ ٹویٹ میں توہین شامل ہے، تو براہ کرم 'yes' کے ساتھ جواب دیں۔ بصورت دیگر، براہ کرم 'no' کے ساتھ جواب دیں۔

    ٹویٹ: {}"""
}


# Create a new DataFrame with the required columns
df3 = df[['text', 'insult']].copy()

# Generate the instructions using randomly selected templates
df3['instruction'] = df3['text'].apply(
    lambda x: random.choice(list(instruction_templates.values())).format(x)
)

# Generate the outputs with English labels "yes" or "no"
df3['output'] = df3['insult'].apply(lambda x: 'yes' if x >= 2 else 'no')

# Keep only the 'instruction' and 'label' columns
df3 = df3[['instruction', 'output']]
df3 = balance_classes(df3, 2500, 'output')
# Shuffle the DataFrame and save to Parquet
df3 = df3.sample(frac=1).reset_index(drop=True)
df3.to_parquet('../../data/ToLD-BR/ToLD-BR-insult.parquet', index=False)

# Define the instruction templates in different languages
instruction_templates = {
    "Chinese": """
    请预测以下推文的注释。
    共计 3 名注释者根据他们对推文是否包含种族主义的理解对每条推文进行了注释。
    如果您认为至少 2 名注释者认为该推文包含种族主义，请回答 'yes'。否则，请回答 'no'。

    推文：{}""",
    "English": """
    Please predict the annotation of the following tweet.
    A total of 3 annotators have annotated each tweet according to their understanding of whether the tweet contains racism or not.
    If you think at least 2 annotators believe that the tweet contains racism, please respond with 'yes'. Otherwise, please respond with 'no'.

    Tweet: {}""",
    "German": """
    Bitte sagen Sie die Annotation des folgenden Tweets voraus.
    Insgesamt haben 3 Annotatoren jeden Tweet gemäß ihrem Verständnis darüber annotiert, ob der Tweet Rassismus enthält oder nicht.
    Wenn Sie glauben, dass mindestens 2 Annotatoren der Meinung sind, dass der Tweet Rassismus enthält, antworten Sie bitte mit 'yes'. Andernfalls antworten Sie bitte mit 'no'.

    Tweet: {}""",
    "French": """
    Veuillez prédire l'annotation du tweet suivant.
    Un total de 3 annotateurs ont annoté chaque tweet selon leur compréhension de la question de savoir si le tweet contient du racisme ou non.
    Si vous pensez qu'au moins 2 annotateurs pensent que le tweet contient du racisme, veuillez répondre par 'yes'. Sinon, veuillez répondre par 'no'.

    Tweet : {}""",
    "Spanish": """
    Por favor, prediga la anotación del siguiente tweet.
    Un total de 3 anotadores han anotado cada tweet según su comprensión de si el tweet contiene racismo o no.
    Si cree que al menos 2 anotadores creen que el tweet contiene racismo, responda con 'yes'. De lo contrario, responda con 'no'.

    Tweet: {}""",
    "Portuguese": """
    Por favor, preveja a anotação do seguinte tweet.
    Um total de 3 anotadores anotaram cada tweet de acordo com seu entendimento de se o tweet contém racismo ou não.
    Se você acha que pelo menos 2 anotadores acreditam que o tweet contém racismo, responda com 'yes'. Caso contrário, responda com 'no'.

    Tweet: {}""",
    "Italian": """
    Si prega di prevedere l'annotazione del seguente tweet.
    Un totale di 3 annotatori hanno annotato ogni tweet secondo la loro comprensione di se il tweet contiene razzismo o meno.
    Se pensi che almeno 2 annotatori credano che il tweet contenga razzismo, rispondi con 'yes'. Altrimenti, rispondi con 'no'.

    Tweet: {}""",
    "Dutch": """
    Voorspel alstublieft de annotatie van de volgende tweet.
    In totaal hebben 3 annotatoren elke tweet geannoteerd volgens hun begrip van of de tweet racisme bevat of niet.
    Als u denkt dat ten minste 2 annotatoren geloven dat de tweet racisme bevat, antwoord dan met 'yes'. Zo niet, antwoord dan met 'no'.

    Tweet: {}""",
    "Russian": """
    Пожалуйста, предскажите аннотацию следующего твита.
    В общей сложности 3 аннотатора аннотировали каждый твит в соответствии с их пониманием того, содержит ли твит расизм или нет.
    Если вы считаете, что как минимум 2 аннотатора считают, что твит содержит расизм, ответьте 'yes'. В противном случае ответьте 'no'.

    Твит: {}""",
    "Czech": """
    Předpovězte prosím anotaci následujícího tweetu.
    Celkem 3 anotátoři anotovali každý tweet podle toho, jak chápou, zda tweet obsahuje rasismus nebo ne.
    Pokud si myslíte, že alespoň 2 anotátoři věří, že tweet obsahuje rasismus, odpovězte prosím 'yes'. V opačném případě odpovězte 'no'.

    Tweet: {}""",
    "Polish": """
    Proszę przewidzieć adnotację następującego tweeta.
    Łącznie 3 adnotatorzy oznaczyli każdy tweet zgodnie z ich zrozumieniem, czy tweet zawiera rasizm, czy nie.
    Jeśli uważasz, że co najmniej 2 adnotatorów uważa, że tweet zawiera rasizm, odpowiedz 'yes'. W przeciwnym razie odpowiedz 'no'.

    Tweet: {}""",
    "Arabic": """
    يرجى التنبؤ بتعليق التغريدة التالية.
    قام 3 ملاحظين بتعليق كل تغريدة وفقًا لفهمهم لما إذا كانت التغريدة تحتوي على عنصرية أم لا.
    إذا كنت تعتقد أن ما لا يقل عن 2 من الملاحظين يعتقدون أن التغريدة تحتوي على عنصرية، فيرجى الرد بـ 'yes'. خلاف ذلك، يرجى الرد بـ 'no'.

    التغريدة: {}""",
    "Persian": """
    لطفاً پیش‌بینی کنید که حاشیه‌نویسی توییت زیر چگونه خواهد بود.
    مجموعاً ۳ حاشیه‌نویس، هر توییت را بر اساس فهم خود از اینکه آیا توییت حاوی نژادپرستی است یا نه، حاشیه‌نویسی کرده‌اند.
    اگر فکر می‌کنید که حداقل ۲ حاشیه‌نویس بر این باورند که توییت حاوی نژادپرستی است، لطفاً با 'yes' پاسخ دهید. در غیر این صورت، لطفاً با 'no' پاسخ دهید.

    توییت: {}""",
    "Hebrew": """
    נא לחזות את ההערה של הציוץ הבא.
    בסך הכל 3 מסמנים סימנו כל ציוץ על פי הבנתם האם הציוץ מכיל גזענות או לא.
    אם אתה חושב שלפחות 2 מסמנים מאמינים שהציוץ מכיל גזענות, נא להשיב 'yes'. אחרת, נא להשיב 'no'.

    ציוץ: {}""",
    "Turkish": """
    Lütfen aşağıdaki tweetin anotasyonunu tahmin edin.
    Toplam 3 anotatör, her tweeti tweetin ırkçılık içerip içermediğini anlama şekline göre anotasyon yaptı.
    En az 2 anotatörün tweetin ırkçılık içerdiğine inandığını düşünüyorsanız, lütfen 'yes' ile yanıtlayın. Aksi takdirde 'no' ile yanıtlayın.

    Tweet: {}""",
    "Japanese": """
    次のツイートの注釈を予測してください。
    合計3人のアノテーターが、それぞれのツイートが人種差別を含んでいるかどうかについて理解に基づいて注釈を付けました。
    2人以上のアノテーターがそのツイートが人種差別を含んでいると考えている場合、「yes」と回答してください。 それ以外の場合は、「no」と回答してください。

    ツイート: {}""",
    "Korean": """
    다음 트윗의 주석을 예측하세요.
    총 3명의 주석자가 트윗에 대해 트윗이 인종 차별을 포함하는지 여부에 따라 주석을 달았습니다.
    최소 2명의 주석자가 트윗에 인종 차별이 포함되어 있다고 생각한다면 'yes'라고 응답하세요. 그렇지 않으면 'no'라고 응답하세요.

    트윗: {}""",
    "Vietnamese": """
    Vui lòng dự đoán chú thích của tweet sau.
    Tổng cộng có 3 người chú thích đã chú thích từng tweet theo cách hiểu của họ về việc liệu tweet đó có chứa phân biệt chủng tộc hay không.
    Nếu bạn nghĩ rằng ít nhất 2 người chú thích tin rằng tweet chứa phân biệt chủng tộc, vui lòng trả lời 'yes'. Nếu không, vui lòng trả lời 'no'.

    Tweet: {}""",
    "Thai": """
    โปรดทำนายคำอธิบายของทวีตต่อไปนี้
    ผู้ตรวจสอบทั้งหมด 3 คนได้อธิบายทวีตแต่ละรายการตามความเข้าใจของพวกเขาว่าทวีตนั้นมีการเหยียดเชื้อชาติหรือไม่
    หากคุณคิดว่ามีผู้ตรวจสอบอย่างน้อย 2 คนเชื่อว่าทวีตนั้นมีการเหยียดเชื้อชาติ โปรดตอบกลับด้วย 'yes' มิฉะนั้น โปรดตอบกลับด้วย 'no'

    ทวีต: {}""",
    "Indonesian": """
    Silakan prediksi anotasi dari tweet berikut ini.
    Sebanyak 3 anotator telah mengomentari setiap tweet berdasarkan pemahaman mereka tentang apakah tweet tersebut mengandung rasisme atau tidak.
    Jika Anda berpikir setidaknya 2 anotator percaya bahwa tweet tersebut mengandung rasisme, harap balas dengan 'yes'. Jika tidak, harap balas dengan 'no'.

    Tweet: {}""",
    "Malay": """
    Sila ramalkan anotasi tweet berikut.
    Sebanyak 3 penanda telah menandakan setiap tweet mengikut kefahaman mereka sama ada tweet itu mengandungi perkauman atau tidak.
    Jika anda fikir sekurang-kurangnya 2 penanda percaya bahawa tweet itu mengandungi perkauman, sila balas dengan 'yes'. Jika tidak, sila balas dengan 'no'.

    Tweet: {}""",
    "Lao": """
    ກະລຸນາທຳນາຍການອະທິບາຍຂອງທະວີດຕໍາຕໍ່ໄປນີ້.
    ຜູ້ທຳອະທິບາຍທັງຫມົດ 3 ຄົນໄດ້ທຳອະທິບາຍແຕ່ລະທະວີດຕາມຄວາມເຂົ້າໃຈຂອງພວກເຂົາວ່າທະວີດນີ້ມີການເຫຍີດເຊື້ອຊາດຫລືບໍ່.
    ຖ້າທ່ານຄິດວ່າຢ່າງຫນ້ອຍ 2 ຄົນເຊື່ອວ່າທະວີດນີ້ມີການເຫຍີດເຊື້ອຊາດ, ກະລຸນາຕອບກັບ 'yes'. ຖ້າບໍ່ແມ່ນ, ກະລຸນາຕອບກັບ 'no'.

    ທະວີດ: {}""",
    "Burmese": """
    ကျေးဇူးပြု၍ အောက်ပါ တွစ်အတွက် မှတ်ချက်ကို ခန့်မှန်းပါ။
    တွစ်တိုင်းကို ၎င်းတို့၏ နားလည်မှုအရ ၃ ဦးမှတ်ချက်ပြုခဲ့ကြသည်၊ တစ်တွစ်တွင် အထက်ပါသဘောထားများပါဝင်မရှိ မေးမြန်းထားပါသည်။
    တစ်တွစ်တွင် အထက်ပါသဘောထားများပါဝင်ကြောင်း ယုံကြည်ချက်ရှိရင် 'yes' နှင့်တုံ့ပြန်ပါ။ မဟုတ်ပါက 'no' နှင့်တုံ့ပြန်ပါ။

    တြစ်: {}""",
    "Cebuano": """
    Palihug tag-ani ang anotasyon sa musunod nga tweet.
    Sa kinatibuk-an, adunay 3 ka mga anotator nga nag-anotato sa matag tweet sumala sa ilang pagsabot kung ang tweet naglambigit ba sa racism o dili.
    Kung sa imong hunahuna adunay labing menos 2 ka mga anotator nga nagtuo nga ang tweet naglambigit sa racism, palihug tubaga kini og 'yes'. Kung dili, palihug tubaga kini og 'no'.

    Tweet: {}""",
    "Khmer": """
    សូមទាយមើលពីការបញ្ជាក់នៃការបញ្ចូលខាងក្រោម។
    សរុបអ្នកមើល 3 នាក់បានបញ្ចេញមតិជាមួយការបញ្ជាក់ខាងក្រោមដោយផ្អែកលើការយល់ចិត្តអំពីមើលឃើញកំហុសជាមួយពូជសាសន៍ឬទេ។
    ប្រសិនបើអ្នកយល់ថានៃអ្នកមើល 2 នាក់បានសម្រេចថាខាងក្រោមមានពូជសាសន៍ សូមទទួលយកជាមួយនឹងការពិត 'yes'. បើមិនមែន សូមទទួលយកជាមួយនឹង 'no'.

    Tweet: {}""",
    "Tagalog": """
    Pakihulaan ang anotasyon ng sumusunod na tweet.
    Isang kabuuan ng 3 anotador ang nag-anotasyon ng bawat tweet ayon sa kanilang pagkakaintindi kung naglalaman ng rasismo ang tweet o hindi.
    Kung sa tingin mo ay naniniwala ang hindi bababa sa 2 anotador na naglalaman ng rasismo ang tweet, mangyaring tumugon ng 'yes'. Kung hindi, mangyaring tumugon ng 'no'.

    Tweet: {}""",
    "Hindi": """
    कृपया निम्नलिखित ट्वीट की एनोटेशन का अनुमान लगाएं।
    कुल 3 एनोटेटरों ने प्रत्येक ट्वीट को एनोटेट किया है कि उनके समझ के अनुसार ट्वीट में नस्लवाद है या नहीं।
    यदि आपको लगता है कि कम से कम 2 एनोटेटर मानते हैं कि ट्वीट में नस्लवाद है, तो कृपया 'yes' के साथ उत्तर दें। अन्यथा, कृपया 'no' के साथ उत्तर दें।

    ट्वीट: {}""",
    "Bengali": """
    অনুগ্রহ করে নিম্নলিখিত টুইটের ব্যাখ্যা অনুমান করুন।
    মোট ৩ জন ব্যাখ্যাকারী প্রতিটি টুইটকে তাদের বোঝার উপর ভিত্তি করে ব্যাখ্যা করেছেন যে টুইটটিতে বর্ণবাদ রয়েছে কিনা।
    আপনি যদি মনে করেন অন্তত ২ জন ব্যাখ্যাকারী বিশ্বাস করেন যে টুইটটিতে বর্ণবাদ রয়েছে, অনুগ্রহ করে 'yes' দিয়ে উত্তর দিন। অন্যথায়, 'no' দিয়ে উত্তর দিন।

    টুইট: {}""",
    "Urdu": """
    براہ کرم درج ذیل ٹویٹ کی تشریح کی پیش گوئی کریں۔
    3 تشریح کاروں نے کل ملا کر ہر ٹویٹ کی تشریح کی ہے کہ آیا ٹویٹ میں نسل پرستی شامل ہے یا نہیں۔
    اگر آپ کو لگتا ہے کہ کم از کم 2 تشریح کار یہ سمجھتے ہیں کہ ٹویٹ میں نسل پرستی شامل ہے، تو براہ کرم 'yes' کے ساتھ جواب دیں۔ بصورت دیگر، براہ کرم 'no' کے ساتھ جواب دیں۔

    ٹویٹ: {}"""
}

# Create a new DataFrame with the required columns
df4 = df[['text', 'racism']].copy()

# Generate the instructions using randomly selected templates
df4['instruction'] = df4['text'].apply(
    lambda x: random.choice(list(instruction_templates.values())).format(x)
)

# Generate the outputs with English labels "yes" or "no"
df4['output'] = df4['racism'].apply(lambda x: 'yes' if x >= 2 else 'no')

# Keep only the 'instruction' and 'label' columns
df4 = df4[['instruction', 'output']]
df4 = balance_classes(df4, 50, 'output')

# Shuffle the DataFrame and save to Parquet
df4 = df4.sample(frac=1).reset_index(drop=True)
df4.to_parquet('../../data/ToLD-BR/ToLD-BR-racism.parquet', index=False)

# Define the instruction templates in different languages
instruction_templates = {
    "Chinese": """
    请预测以下推文的注释。
    共计 3 名注释者根据他们对推文是否包含厌女症的理解对每条推文进行了注释。
    如果您认为至少 2 名注释者认为该推文包含厌女症，请回答 'yes'。否则，请回答 'no'。

    推文：{}""",

    "English": """
    Please predict the annotation of the following tweet.
    A total of 3 annotators have annotated each tweet according to their understanding of whether the tweet contains misogyny or not.
    If you think at least 2 annotators believe that the tweet contains misogyny, please respond with 'yes'. Otherwise, please respond with 'no'.

    Tweet: {}""",

    "German": """
    Bitte sagen Sie die Annotation des folgenden Tweets voraus.
    Insgesamt haben 3 Annotatoren jeden Tweet gemäß ihrem Verständnis darüber annotiert, ob der Tweet Frauenfeindlichkeit enthält oder nicht.
    Wenn Sie glauben, dass mindestens 2 Annotatoren der Meinung sind, dass der Tweet Frauenfeindlichkeit enthält, antworten Sie bitte mit 'yes'. Andernfalls antworten Sie bitte mit 'no'.

    Tweet: {}""",

    "French": """
    Veuillez prédire l'annotation du tweet suivant.
    Un total de 3 annotateurs ont annoté chaque tweet selon leur compréhension de la question de savoir si le tweet contient de la misogynie ou non.
    Si vous pensez qu'au moins 2 annotateurs pensent que le tweet contient de la misogynie, veuillez répondre par 'yes'. Sinon, veuillez répondre par 'no'.

    Tweet : {}""",

    "Spanish": """
    Por favor, prediga la anotación del siguiente tweet.
    Un total de 3 anotadores han anotado cada tweet según su comprensión de si el tweet contiene misoginia o no.
    Si cree que al menos 2 anotadores creen que el tweet contiene misoginia, responda con 'yes'. De lo contrario, responda con 'no'.

    Tweet: {}""",

    "Portuguese": """
    Por favor, preveja a anotação do seguinte tweet.
    Um total de 3 anotadores anotaram cada tweet de acordo com seu entendimento de se o tweet contém misoginia ou não.
    Se você acha que pelo menos 2 anotadores acreditam que o tweet contém misoginia, responda com 'yes'. Caso contrário, responda com 'no'.

    Tweet: {}""",

    "Italian": """
    Si prega di prevedere l'annotazione del seguente tweet.
    Un totale di 3 annotatori hanno annotato ogni tweet secondo la loro comprensione di se il tweet contiene misoginia o meno.
    Se pensi che almeno 2 annotatori credano che il tweet contenga misoginia, rispondi con 'yes'. Altrimenti, rispondi con 'no'.

    Tweet: {}""",

    "Dutch": """
    Voorspel alstublieft de annotatie van de volgende tweet.
    In totaal hebben 3 annotatoren elke tweet geannoteerd volgens hun begrip van of de tweet misogynie bevat of niet.
    Als u denkt dat ten minste 2 annotatoren geloven dat de tweet misogynie bevat, antwoord dan met 'yes'. Zo niet, antwoord dan met 'no'.

    Tweet: {}""",

    "Russian": """
    Пожалуйста, предскажите аннотацию следующего твита.
    В общей сложности 3 аннотатора аннотировали каждый твит в соответствии с их пониманием того, содержит ли твит мизогинию или нет.
    Если вы считаете, что как минимум 2 аннотатора считают, что твит содержит мизогинию, ответьте 'yes'. В противном случае ответьте 'no'.

    Твит: {}""",

    "Czech": """
    Předpovězte prosím anotaci následujícího tweetu.
    Celkem 3 anotátoři anotovali každý tweet podle toho, jak chápou, zda tweet obsahuje misogynii nebo ne.
    Pokud si myslíte, že alespoň 2 anotátoři věří, že tweet obsahuje misogynii, odpovězte prosím 'yes'. V opačném případě odpovězte 'no'.

    Tweet: {}""",

    "Polish": """
    Proszę przewidzieć adnotację następującego tweeta.
    Łącznie 3 adnotatorzy oznaczyli każdy tweet zgodnie z ich zrozumieniem, czy tweet zawiera mizoginię, czy nie.
    Jeśli uważasz, że co najmniej 2 adnotatorów uważa, że tweet zawiera mizoginię, odpowiedz 'yes'. W przeciwnym razie odpowiedz 'no'.

    Tweet: {}""",

    "Arabic": """
    يرجى التنبؤ بتعليق التغريدة التالية.
    قام 3 ملاحظين بتعليق كل تغريدة وفقًا لفهمهم لما إذا كانت التغريدة تحتوي على كراهية النساء أم لا.
    إذا كنت تعتقد أن ما لا يقل عن 2 من الملاحظين يعتقدون أن التغريدة تحتوي على كراهية النساء، فيرجى الرد بـ 'yes'. خلاف ذلك، يرجى الرد بـ 'no'.

    التغريدة: {}""",

    "Persian": """
    لطفاً پیش‌بینی کنید که حاشیه‌نویسی توییت زیر چگونه خواهد بود.
    مجموعاً ۳ حاشیه‌نویس، هر توییت را بر اساس فهم خود از اینکه آیا توییت حاوی زن‌ستیزی است یا نه، حاشیه‌نویسی کرده‌اند.
    اگر فکر می‌کنید که حداقل ۲ حاشیه‌نویس بر این باورند که توییت حاوی زن‌ستیزی است، لطفاً با 'yes' پاسخ دهید. در غیر این صورت، لطفاً با 'no' پاسخ دهید.

    توییت: {}""",

    "Hebrew": """
    נא לחזות את ההערה של הציוץ הבא.
    בסך הכל 3 מסמנים סימנו כל ציוץ על פי הבנתם האם הציוץ מכיל מיזוגיניה או לא.
    אם אתה חושב שלפחות 2 מסמנים מאמינים שהציוץ מכיל מיזוגיניה, נא להשיב 'yes'. אחרת, נא להשיב 'no'.

    ציוץ: {}""",

    "Turkish": """
    Lütfen aşağıdaki tweetin anotasyonunu tahmin edin.
    Toplam 3 anotatör, her tweeti tweetin kadın düşmanlığı içerip içermediğini anlama şekline göre anotasyon yaptı.
    En az 2 anotatörün tweetin kadın düşmanlığı içerdiğine inandığını düşünüyorsanız, lütfen 'yes' ile yanıtlayın. Aksi takdirde 'no' ile yanıtlayın.

    Tweet: {}""",

    "Japanese": """
    次のツイートの注釈を予測してください。
    合計3人のアノテーターが、それぞれのツイートが女性嫌悪を含んでいるかどうかについて理解に基づいて注釈を付けました。
    2人以上のアノテーターがそのツイートが女性嫌悪を含んでいると考えている場合、「yes」と回答してください。 それ以外の場合は、「no」と回答してください。

    ツイート: {}""",

    "Korean": """
    다음 트윗의 주석을 예측하세요.
    총 3명의 주석자가 트윗에 대해 트윗이 여성 혐오를 포함하는지 여부에 따라 주석을 달았습니다.
    최소 2명의 주석자가 트윗에 여성 혐오가 포함되어 있다고 생각한다면 'yes'라고 응답하세요. 그렇지 않으면 'no'라고 응답하세요.

    트윗: {}""",

    "Vietnamese": """
    Vui lòng dự đoán chú thích của tweet sau.
    Tổng cộng có 3 người chú thích đã chú thích từng tweet theo cách hiểu của họ về việc liệu tweet đó có chứa sự thù ghét phụ nữ hay không.
    Nếu bạn nghĩ rằng ít nhất 2 người chú thích tin rằng tweet chứa sự thù ghét phụ nữ, vui lòng trả lời 'yes'. Nếu không, vui lòng trả lời 'no'.

    Tweet: {}""",

    "Thai": """
    โปรดทำนายคำอธิบายของทวีตต่อไปนี้
    ผู้ตรวจสอบทั้งหมด 3 คนได้อธิบายทวีตแต่ละรายการตามความเข้าใจของพวกเขาว่าทวีตนั้นมีการเกลียดชังผู้หญิงหรือไม่
    หากคุณคิดว่ามีผู้ตรวจสอบอย่างน้อย 2 คนเชื่อว่าทวีตนั้นมีการเกลียดชังผู้หญิง โปรดตอบกลับด้วย 'yes' มิฉะนั้น โปรดตอบกลับด้วย 'no'

    ทวีต: {}""",

    "Indonesian": """
    Silakan prediksi anotasi dari tweet berikut ini.
    Sebanyak 3 anotator telah mengomentari setiap tweet berdasarkan pemahaman mereka tentang apakah tweet tersebut mengandung kebencian terhadap perempuan atau tidak.
    Jika Anda berpikir setidaknya 2 anotator percaya bahwa tweet tersebut mengandung kebencian terhadap perempuan, harap balas dengan 'yes'. Jika tidak, harap balas dengan 'no'.

    Tweet: {}""",

    "Malay": """
    Sila ramalkan anotasi tweet berikut.
    Sebanyak 3 penanda telah menandakan setiap tweet mengikut kefahaman mereka sama ada tweet itu mengandungi kebencian terhadap wanita atau tidak.
    Jika anda fikir sekurang-kurangnya 2 penanda percaya bahawa tweet itu mengandungi kebencian terhadap wanita, sila balas dengan 'yes'. Jika tidak, sila balas dengan 'no'.

    Tweet: {}""",

    "Lao": """
    ກະລຸນາທຳນາຍການອະທິບາຍຂອງທະວີດຕໍາຕໍ່ໄປນີ້.
    ຜູ້ທຳອະທິບາຍທັງຫມົດ 3 ຄົນໄດ້ທຳອະທິບາຍແຕ່ລະທະວີດຕາມຄວາມເຂົ້າໃຈຂອງພວກເຂົາວ່າທະວີດນີ້ມີຄວາມກຽຍຊັງຜູ້ຍິງຫລືບໍ່.
    ຖ້າທ່ານຄິດວ່າຢ່າງຫນ້ອຍ 2 ຄົນເຊື່ອວ່າທະວີດນີ້ມີຄວາມກຽຍຊັງຜູ້ຍິງ, ກະລຸນາຕອບກັບ 'yes'. ຖ້າບໍ່ແມ່ນ, ກະລຸນາຕອບກັບ 'no'.

    ທະວີດ: {}""",

    "Burmese": """
    ကျေးဇူးပြု၍ အောက်ပါ တွစ်အတွက် မှတ်ချက်ကို ခန့်မှန်းပါ။
    တွစ်တိုင်းကို ၎င်းတို့၏ နားလည်မှုအရ ၃ ဦးမှတ်ချက်ပြုခဲ့ကြသည်၊ တစ်တွစ်တွင် အထက်ပါသဘောထားများပါဝင်မရှိ မေးမြန်းထားပါသည်။
    တစ်တွစ်တွင် အထက်ပါသဘောထားများပါဝင်ကြောင်း ယုံကြည်ချက်ရှိရင် 'yes' နှင့်တုံ့ပြန်ပါ။ မဟုတ်ပါက 'no' နှင့်တုံ့ပြန်ပါ။

    တြစ်: {}""",

    "Cebuano": """
    Palihug tag-ani ang anotasyon sa musunod nga tweet.
    Sa kinatibuk-an, adunay 3 ka mga anotator nga nag-anotato sa matag tweet sumala sa ilang pagsabot kung ang tweet naglambigit ba sa misogyny o dili.
    Kung sa imong hunahuna adunay labing menos 2 ka mga anotator nga nagtuo nga ang tweet naglambigit sa misogyny, palihug tubaga kini og 'yes'. Kung dili, palihug tubaga kini og 'no'.

    Tweet: {}""",

    "Khmer": """
    សូមទាយមើលពីការបញ្ជាក់នៃការបញ្ចូលខាងក្រោម។
    សរុបអ្នកមើល 3 នាក់បានបញ្ចេញមតិជាមួយការបញ្ជាក់ខាងក្រោមដោយផ្អែកលើការយល់ចិត្តអំពីមើលឃើញកំហុសជាមួយនិងការប្រកាន់ខ្ជាប់បុរសឬទេ។
    ប្រសិនបើអ្នកយល់ថានៃអ្នកមើល 2 នាក់បានសម្រេចថាខាងក្រោមមានការប្រកាន់ខ្ជាប់បុរស សូមទទួលយកជាមួយនឹងការពិត 'yes'. បើមិនមែន សូមទទួលយកជាមួយនឹង 'no'.

    Tweet: {}""",

    "Tagalog": """
    Pakihulaan ang anotasyon ng sumusunod na tweet.
    Isang kabuuan ng 3 anotador ang nag-anotasyon ng bawat tweet ayon sa kanilang pagkakaintindi kung naglalaman ng misogynya ang tweet o hindi.
    Kung sa tingin mo ay naniniwala ang hindi bababa sa 2 anotador na naglalaman ng misogynya ang tweet, mangyaring tumugon ng 'yes'. Kung hindi, mangyaring tumugon ng 'no'.

    Tweet: {}""",

    "Hindi": """
    कृपया निम्नलिखित ट्वीट की एनोटेशन का अनुमान लगाएं।
    कुल 3 एनोटेटरों ने प्रत्येक ट्वीट को एनोटेट किया है कि उनके समझ के अनुसार ट्वीट में स्त्री द्वेष है या नहीं।
    यदि आपको लगता है कि कम से कम 2 एनोटेटर मानते हैं कि ट्वीट में स्त्री द्वेष है, तो कृपया 'yes' के साथ उत्तर दें। अन्यथा, कृपया 'no' के साथ उत्तर दें।

    ट्वीट: {}""",

    "Bengali": """
    অনুগ্রহ করে নিম্নলিখিত টুইটের ব্যাখ্যা অনুমান করুন।
    মোট ৩ জন ব্যাখ্যাকারী প্রতিটি টুইটকে তাদের বোঝার উপর ভিত্তি করে ব্যাখ্যা করেছেন যে টুইটটিতে নারী বিদ্বেষ রয়েছে কিনা।
    আপনি যদি মনে করেন অন্তত ২ জন ব্যাখ্যাকারী বিশ্বাস করেন যে টুইটটিতে নারী বিদ্বেষ রয়েছে, অনুগ্রহ করে 'yes' দিয়ে উত্তর দিন। অন্যথায়, 'no' দিয়ে উত্তর দিন।

    টুইট: {}""",

    "Urdu": """
    براہ کرم درج ذیل ٹویٹ کی تشریح کی پیش گوئی کریں۔
    3 تشریح کاروں نے کل ملا کر ہر ٹویٹ کی تشریح کی ہے کہ آیا ٹویٹ میں عورت دشمنی شامل ہے یا نہیں۔
    اگر آپ کو لگتا ہے کہ کم از کم 2 تشریح کار یہ سمجھتے ہیں کہ ٹویٹ میں عورت دشمنی شامل ہے، تو براہ کرم 'yes' کے ساتھ جواب دیں۔ بصورت دیگر، براہ کرم 'no' کے ساتھ جواب دیں۔

    ٹویٹ: {}"""
}


# Create a new DataFrame with the required columns
df5 = df[['text', 'misogyny']].copy()

# Generate the instructions using randomly selected templates
df5['instruction'] = df5['text'].apply(
    lambda x: random.choice(list(instruction_templates.values())).format(x)
)

# Generate the outputs with English labels "yes" or "no"
df5['output'] = df5['misogyny'].apply(lambda x: 'yes' if x >= 2 else 'no')

# Keep only the 'instruction' and 'label' columns
df5 = df5[['instruction', 'output']]
df5 = balance_classes(df5, 200, 'output')

# Shuffle the DataFrame and save to Parquet
df5 = df5.sample(frac=1).reset_index(drop=True)
df5.to_parquet('../../data/ToLD-BR/ToLD-BR-misogyny.parquet', index=False)

# Define the instruction templates in different languages
instruction_templates = {
    "Chinese": """
    请预测以下推文的注释。
    共计 3 名注释者根据他们对推文是否包含仇外心理的理解对每条推文进行了注释。
    如果您认为至少 2 名注释者认为该推文包含仇外心理，请回答 'yes'。否则，请回答 'no'。

    推文：{}""",

    "English": """
    Please predict the annotation of the following tweet.
    A total of 3 annotators have annotated each tweet according to their understanding of whether the tweet contains xenophobia or not.
    If you think at least 2 annotators believe that the tweet contains xenophobia, please respond with 'yes'. Otherwise, please respond with 'no'.

    Tweet: {}""",

    "German": """
    Bitte sagen Sie die Annotation des folgenden Tweets voraus.
    Insgesamt haben 3 Annotatoren jeden Tweet gemäß ihrem Verständnis darüber annotiert, ob der Tweet Fremdenfeindlichkeit enthält oder nicht.
    Wenn Sie glauben, dass mindestens 2 Annotatoren der Meinung sind, dass der Tweet Fremdenfeindlichkeit enthält, antworten Sie bitte mit 'yes'. Andernfalls antworten Sie bitte mit 'no'.

    Tweet: {}""",

    "French": """
    Veuillez prédire l'annotation du tweet suivant.
    Un total de 3 annotateurs ont annoté chaque tweet selon leur compréhension de la question de savoir si le tweet contient de la xénophobie ou non.
    Si vous pensez qu'au moins 2 annotateurs pensent que le tweet contient de la xénophobie, veuillez répondre par 'yes'. Sinon, veuillez répondre par 'no'.

    Tweet : {}""",

    "Spanish": """
    Por favor, prediga la anotación del siguiente tweet.
    Un total de 3 anotadores han anotado cada tweet según su comprensión de si el tweet contiene xenofobia o no.
    Si cree que al menos 2 anotadores creen que el tweet contiene xenofobia, responda con 'yes'. De lo contrario, responda con 'no'.

    Tweet: {}""",

    "Portuguese": """
    Por favor, preveja a anotação do seguinte tweet.
    Um total de 3 anotadores anotaram cada tweet de acordo com seu entendimento de se o tweet contém xenofobia ou não.
    Se você acha que pelo menos 2 anotadores acreditam que o tweet contém xenofobia, responda com 'yes'. Caso contrário, responda com 'no'.

    Tweet: {}""",

    "Italian": """
    Si prega di prevedere l'annotazione del seguente tweet.
    Un totale di 3 annotatori hanno annotato ogni tweet secondo la loro comprensione di se il tweet contiene xenofobia o meno.
    Se pensi che almeno 2 annotatori credano che il tweet contenga xenofobia, rispondi con 'yes'. Altrimenti, rispondi con 'no'.

    Tweet: {}""",

    "Dutch": """
    Voorspel alstublieft de annotatie van de volgende tweet.
    In totaal hebben 3 annotatoren elke tweet geannoteerd volgens hun begrip van of de tweet vreemdelingenhaat bevat of niet.
    Als u denkt dat ten minste 2 annotatoren geloven dat de tweet vreemdelingenhaat bevat, antwoord dan met 'yes'. Zo niet, antwoord dan met 'no'.

    Tweet: {}""",

    "Russian": """
    Пожалуйста, предскажите аннотацию следующего твита.
    В общей сложности 3 аннотатора аннотировали каждый твит в соответствии с их пониманием того, содержит ли твит ксенофобию или нет.
    Если вы считаете, что как минимум 2 аннотатора считают, что твит содержит ксенофобию, ответьте 'yes'. В противном случае ответьте 'no'.

    Твит: {}""",

    "Czech": """
    Předpovězte prosím anotaci následujícího tweetu.
    Celkem 3 anotátoři anotovali každý tweet podle toho, jak chápou, zda tweet obsahuje xenofobii nebo ne.
    Pokud si myslíte, že alespoň 2 anotátoři věří, že tweet obsahuje xenofobii, odpovězte prosím 'yes'. V opačném případě odpovězte 'no'.

    Tweet: {}""",

    "Polish": """
    Proszę przewidzieć adnotację następującego tweeta.
    Łącznie 3 adnotatorzy oznaczyli każdy tweet zgodnie z ich zrozumieniem, czy tweet zawiera ksenofobię, czy nie.
    Jeśli uważasz, że co najmniej 2 adnotatorów uważa, że tweet zawiera ksenofobię, odpowiedz 'yes'. W przeciwnym razie odpowiedz 'no'.

    Tweet: {}""",

    "Arabic": """
    يرجى التنبؤ بتعليق التغريدة التالية.
    قام 3 ملاحظين بتعليق كل تغريدة وفقًا لفهمهم لما إذا كانت التغريدة تحتوي على كراهية الأجانب أم لا.
    إذا كنت تعتقد أن ما لا يقل عن 2 من الملاحظين يعتقدون أن التغريدة تحتوي على كراهية الأجانب، فيرجى الرد بـ 'yes'. خلاف ذلك، يرجى الرد بـ 'no'.

    التغريدة: {}""",

    "Persian": """
    لطفاً پیش‌بینی کنید که حاشیه‌نویسی توییت زیر چگونه خواهد بود.
    مجموعاً ۳ حاشیه‌نویس، هر توییت را بر اساس فهم خود از اینکه آیا توییت حاوی بیگانه‌هراسی است یا نه، حاشیه‌نویسی کرده‌اند.
    اگر فکر می‌کنید که حداقل ۲ حاشیه‌نویس بر این باورند که توییت حاوی بیگانه‌هراسی است، لطفاً با 'yes' پاسخ دهید. در غیر این صورت، لطفاً با 'no' پاسخ دهید.

    توییت: {}""",

    "Hebrew": """
    נא לחזות את ההערה של הציוץ הבא.
    בסך הכל 3 מסמנים סימנו כל ציוץ על פי הבנתם האם הציוץ מכיל קסנופוביה או לא.
    אם אתה חושב שלפחות 2 מסמנים מאמינים שהציוץ מכיל קסנופוביה, נא להשיב 'yes'. אחרת, נא להשיב 'no'.

    ציוץ: {}""",

    "Turkish": """
    Lütfen aşağıdaki tweetin anotasyonunu tahmin edin.
    Toplam 3 anotatör, her tweeti tweetin yabancı düşmanlığı içerip içermediğini anlama şekline göre anotasyon yaptı.
    En az 2 anotatörün tweetin yabancı düşmanlığı içerdiğine inandığını düşünüyorsanız, lütfen 'yes' ile yanıtlayın. Aksi takdirde 'no' ile yanıtlayın.

    Tweet: {}""",

    "Japanese": """
    次のツイートの注釈を予測してください。
    合計3人のアノテーターが、それぞれのツイートが外国人恐怖症を含んでいるかどうかについて理解に基づいて注釈を付けました。
    2人以上のアノテーターがそのツイートが外国人恐怖症を含んでいると考えている場合、「yes」と回答してください。 それ以外の場合は、「no」と回答してください。

    ツイート: {}""",

    "Korean": """
    다음 트윗의 주석을 예측하세요.
    총 3명의 주석자가 트윗에 대해 트윗이 외국인 혐오를 포함하는지 여부에 따라 주석을 달았습니다.
    최소 2명의 주석자가 트윗에 외국인 혐오가 포함되어 있다고 생각한다면 'yes'라고 응답하세요. 그렇지 않으면 'no'라고 응답하세요.

    트윗: {}""",

    "Vietnamese": """
    Vui lòng dự đoán chú thích của tweet sau.
    Tổng cộng có 3 người chú thích đã chú thích từng tweet theo cách hiểu của họ về việc liệu tweet đó có chứa sự bài ngoại hay không.
    Nếu bạn nghĩ rằng ít nhất 2 người chú thích tin rằng tweet chứa sự bài ngoại, vui lòng trả lời 'yes'. Nếu không, vui lòng trả lời 'no'.

    Tweet: {}""",

    "Thai": """
    โปรดทำนายคำอธิบายของทวีตต่อไปนี้
    ผู้ตรวจสอบทั้งหมด 3 คนได้อธิบายทวีตแต่ละรายการตามความเข้าใจของพวกเขาว่าทวีตนั้นมีความเกลียดชังชาวต่างชาติหรือไม่
    หากคุณคิดว่ามีผู้ตรวจสอบอย่างน้อย 2 คนเชื่อว่าทวีตนั้นมีความเกลียดชังชาวต่างชาติ โปรดตอบกลับด้วย 'yes' มิฉะนั้น โปรดตอบกลับด้วย 'no'

    ทวีต: {}""",

    "Indonesian": """
    Silakan prediksi anotasi dari tweet berikut ini.
    Sebanyak 3 anotator telah mengomentari setiap tweet berdasarkan pemahaman mereka tentang apakah tweet tersebut mengandung xenophobia atau tidak.
    Jika Anda berpikir setidaknya 2 anotator percaya bahwa tweet tersebut mengandung xenophobia, harap balas dengan 'yes'. Jika tidak, harap balas dengan 'no'.

    Tweet: {}""",

    "Malay": """
    Sila ramalkan anotasi tweet berikut.
    Sebanyak 3 penanda telah menandakan setiap tweet mengikut kefahaman mereka sama ada tweet itu mengandungi xenofobia atau tidak.
    Jika anda fikir sekurang-kurangnya 2 penanda percaya bahawa tweet itu mengandungi xenofobia, sila balas dengan 'yes'. Jika tidak, sila balas dengan 'no'.

    Tweet: {}""",

    "Lao": """
    ກະລຸນາທຳນາຍການອະທິບາຍຂອງທະວີດຕໍາຕໍ່ໄປນີ້.
    ຜູ້ທຳອະທິບາຍທັງຫມົດ 3 ຄົນໄດ້ທຳອະທິບາຍແຕ່ລະທະວີດຕາມຄວາມເຂົ້າໃຈຂອງພວກເຂົາວ່າທະວີດນີ້ມີຄວາມກຽຍຊັງຕ່າງຊາດຫລືບໍ່.
    ຖ້າທ່ານຄິດວ່າຢ່າງຫນ້ອຍ 2 ຄົນເຊື່ອວ່າທະວີດນີ້ມີຄວາມກຽຍຊັງຕ່າງຊາດ, ກະລຸນາຕອບກັບ 'yes'. ຖ້າບໍ່ແມ່ນ, ກະລຸນາຕອບກັບ 'no'.

    ທະວີດ: {}""",

    "Burmese": """
    ကျေးဇူးပြု၍ အောက်ပါ တွစ်အတွက် မှတ်ချက်ကို ခန့်မှန်းပါ။
    တွစ်တိုင်းကို ၎င်းတို့၏ နားလည်မှုအရ ၃ ဦးမှတ်ချက်ပြုခဲ့ကြသည်၊ တစ်တွစ်တွင် အထက်ပါသဘောထားများပါဝင်မရှိ မေးမြန်းထားပါသည်။
    တစ်တွစ်တွင် အထက်ပါသဘောထားများပါဝင်ကြောင်း ယုံကြည်ချက်ရှိရင် 'yes' နှင့်တုံ့ပြန်ပါ။ မဟုတ်ပါက 'no' နှင့်တုံ့ပြန်ပါ။

    တြစ်: {}""",

    "Cebuano": """
    Palihug tag-ani ang anotasyon sa musunod nga tweet.
    Sa kinatibuk-an, adunay 3 ka mga anotator nga nag-anotato sa matag tweet sumala sa ilang pagsabot kung ang tweet naglambigit ba sa xenophobia o dili.
    Kung sa imong hunahuna adunay labing menos 2 ka mga anotator nga nagtuo nga ang tweet naglambigit sa xenophobia, palihug tubaga kini og 'yes'. Kung dili, palihug tubaga kini og 'no'.

    Tweet: {}""",

    "Khmer": """
    សូមទាយមើលពីការបញ្ជាក់នៃការបញ្ចូលខាងក្រោម។
    សរុបអ្នកមើល 3 នាក់បានបញ្ចេញមតិជាមួយការបញ្ជាក់ខាងក្រោមដោយផ្អែកលើការយល់ចិត្តអំពីមើលឃើញកំហុសជាមួយនិងការស្អប់ខ្ពើមករណីជនបរទេស ឬទេ។
    ប្រសិនបើអ្នកយល់ថានៃអ្នកមើល 2 នាក់បានសម្រេចថាខាងក្រោមមានការស្អប់ខ្ពើមករណីជនបរទេស សូមទទួលយកជាមួយនឹងការពិត 'yes'. បើមិនមែន សូមទទួលយកជាមួយនឹង 'no'.

    Tweet: {}""",

    "Tagalog": """
    Pakihulaan ang anotasyon ng sumusunod na tweet.
    Isang kabuuan ng 3 anotador ang nag-anotasyon ng bawat tweet ayon sa kanilang pagkakaintindi kung naglalaman ng xenophobia ang tweet o hindi.
    Kung sa tingin mo ay naniniwala ang hindi bababa sa 2 anotador na naglalaman ng xenophobia ang tweet, mangyaring tumugon ng 'yes'. Kung hindi, mangyaring tumugon ng 'no'.

    Tweet: {}""",

    "Hindi": """
    कृपया निम्नलिखित ट्वीट की एनोटेशन का अनुमान लगाएं।
    कुल 3 एनोटेटरों ने प्रत्येक ट्वीट को एनोटेट किया है कि उनके समझ के अनुसार ट्वीट में विदेशी-विरोधीता है या नहीं।
    यदि आपको लगता है कि कम से कम 2 एनोटेटर मानते हैं कि ट्वीट में विदेशी-विरोधीता है, तो कृपया 'yes' के साथ उत्तर दें। अन्यथा, कृपया 'no' के साथ उत्तर दें।

    ट्वीट: {}""",

    "Bengali": """
    অনুগ্রহ করে নিম্নলিখিত টুইটের ব্যাখ্যা অনুমান করুন।
    মোট ৩ জন ব্যাখ্যাকারী প্রতিটি টুইটকে তাদের বোঝার উপর ভিত্তি করে ব্যাখ্যা করেছেন যে টুইটটিতে বিদেশী বিরোধিতা রয়েছে কিনা।
    আপনি যদি মনে করেন অন্তত ২ জন ব্যাখ্যাকারী বিশ্বাস করেন যে টুইটটিতে বিদেশী বিরোধিতা রয়েছে, অনুগ্রহ করে 'yes' দিয়ে উত্তর দিন। অন্যথায়, 'no' দিয়ে উত্তর দিন।

    টুইট: {}""",

    "Urdu": """
    براہ کرم درج ذیل ٹویٹ کی تشریح کی پیش گوئی کریں۔
    3 تشریح کاروں نے کل ملا کر ہر ٹویٹ کی تشریح کی ہے کہ آیا ٹویٹ میں غیر ملکیوں سے نفرت شامل ہے یا نہیں۔
    اگر آپ کو لگتا ہے کہ کم از کم 2 تشریح کار یہ سمجھتے ہیں کہ ٹویٹ میں غیر ملکیوں سے نفرت شامل ہے، تو براہ کرم 'yes' کے ساتھ جواب دیں۔ بصورت دیگر، براہ کرم 'no' کے ساتھ جواب دیں۔

    ٹویٹ: {}"""
}


# Create a new DataFrame with the required columns
df6 = df[['text', 'xenophobia']].copy()

# Generate the instructions using randomly selected templates
df6['instruction'] = df6['text'].apply(
    lambda x: random.choice(list(instruction_templates.values())).format(x)
)

# Generate the outputs with English labels "yes" or "no"
df6['output'] = df6['xenophobia'].apply(lambda x: 'yes' if x >= 2 else 'no')

# Keep only the 'instruction' and 'label' columns
df6 = df6[['instruction', 'output']]
df6 = balance_classes(df6, 50, 'output')

# Shuffle the DataFrame and save to Parquet
df6 = df6.sample(frac=1).reset_index(drop=True)
df6.to_parquet('../../data/ToLD-BR/ToLD-BR-xenophobia.parquet', index=False)