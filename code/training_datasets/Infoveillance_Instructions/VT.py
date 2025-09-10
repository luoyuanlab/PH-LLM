# -*- coding: utf-8 -*-
# VT (Vulgarity on Twitter) 
# - Paper: https://aclanthology.org/C18-1248
# - Data: https://github.com/ericholgate/vulgartwitter/tree/master/data

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import pandas as pd
import random
from utils.data_utils import balance_classes

# Load data
df = pd.read_csv('../../data/VT/cleaned_data.tsv', sep='\t')

# Define multilingual templates for the sentiment analysis prompt
sentiment_templates = [
    # Chinese
    """以下是一条社交媒体推文: {}。\n请根据以下五级评分标准对该推文的情感进行评价。\n1: 非常负面\n2: 稍微负面\n3: 中立\n4: 稍微正面\n5: 非常正面。\n请输出相应的数字，不需要解释。""",

    # English
    """Here's a social media tweet: {}.\nHow would you evaluate the sentiment of this tweet based on a five-point scale below.\n1: very negative\n2: somewhat negative\n3: neutral\n4: somewhat positive\n5: very positive.\nOutput the number without any explanation.""",

    # German
    """Hier ist ein Social-Media-Tweet: {}.\nWie würden Sie die Stimmung dieses Tweets auf einer Fünf-Punkte-Skala bewerten?\n1: sehr negativ\n2: etwas negativ\n3: neutral\n4: etwas positiv\n5: sehr positiv.\nGeben Sie nur die Zahl ohne Erklärung aus.""",

    # French
    """Voici un tweet des médias sociaux: {}.\nComment évalueriez-vous le sentiment de ce tweet sur une échelle de cinq points ci-dessous?\n1: très négatif\n2: quelque peu négatif\n3: neutre\n4: quelque peu positif\n5: très positif.\nSaisissez le numéro sans aucune explication.""",

    # Spanish
    """Aquí hay un tweet de redes sociales: {}.\n¿Cómo evaluarías el sentimiento de este tweet en una escala de cinco puntos a continuación?\n1: muy negativo\n2: algo negativo\n3: neutral\n4: algo positivo\n5: muy positivo.\nEscriba el número sin ninguna explicación.""",

    # Portuguese
    """Aqui está um tweet de mídia social: {}.\nComo você avaliaria o sentimento deste tweet com base em uma escala de cinco pontos abaixo?\n1: muito negativo\n2: um pouco negativo\n3: neutro\n4: um pouco positivo\n5: muito positivo.\nForneça o número sem nenhuma explicação.""",

    # Italian
    """Ecco un tweet sui social media: {}.\nCome valuteresti il sentimento di questo tweet su una scala a cinque punti di seguito?\n1: molto negativo\n2: piuttosto negativo\n3: neutro\n4: piuttosto positivo\n5: molto positivo.\nFornisci solo il numero senza spiegazioni.""",

    # Dutch
    """Hier is een social media-tweet: {}.\nHoe zou u het sentiment van deze tweet evalueren op basis van een vijfpuntsschaal hieronder?\n1: zeer negatief\n2: enigszins negatief\n3: neutraal\n4: enigszins positief\n5: zeer positief.\nGeef alleen het nummer zonder uitleg.""",

    # Russian
    """Вот твит из социальных сетей: {}.\nКак бы вы оценили настроение этого твита по пятибалльной шкале ниже?\n1: очень негативный\n2: несколько негативный\n3: нейтральный\n4: несколько положительный\n5: очень положительный.\nВыведите только число без объяснений.""",

    # Czech
    """Zde je tweet ze sociálních médií: {}.\nJak byste ohodnotili sentiment tohoto tweetu na pětibodové škále níže?\n1: velmi negativní\n2: poněkud negativní\n3: neutrální\n4: poněkud pozitivní\n5: velmi pozitivní.\nZadejte číslo bez jakéhokoli vysvětlení.""",

    # Polish
    """Oto tweet z mediów społecznościowych: {}.\nJak oceniłbyś nastrój tego tweeta w oparciu o pięciopunktową skalę poniżej?\n1: bardzo negatywny\n2: nieco negatywny\n3: neutralny\n4: nieco pozytywny\n5: bardzo pozytywny.\nPodaj tylko numer bez żadnych wyjaśnień.""",

    # Arabic
    """إليك تغريدة من وسائل التواصل الاجتماعي: {}.\nكيف تقيم مشاعر هذه التغريدة بناءً على مقياس من خمس نقاط أدناه؟\n1: سلبي جدًا\n2: سلبي إلى حد ما\n3: محايد\n4: إيجابي إلى حد ما\n5: إيجابي جدًا.\nاكتب الرقم فقط بدون أي تفسير.""",

    # Persian
    """در اینجا یک توییت از شبکه‌های اجتماعی است: {}.\nچگونه احساسات این توییت را بر اساس یک مقیاس پنج نقطه‌ای ارزیابی می‌کنید؟\n1: بسیار منفی\n2: تا حدی منفی\n3: خنثی\n4: تا حدی مثبت\n5: بسیار مثبت.\nفقط شماره را بدون هیچ توضیحی بنویسید.""",

    # Hebrew
    """הנה ציוץ מהרשתות החברתיות: {}.\nאיך היית מעריך את הרגש של הציוץ הזה על בסיס סולם של חמש נקודות למטה?\n1: שלילי מאוד\n2: שלילי במידה מסוימת\n3: נייטרלי\n4: חיובי במידה מסוימת\n5: חיובי מאוד.\nציין את המספר בלבד ללא כל הסבר.""",

    # Turkish
    """İşte bir sosyal medya tweeti: {}.\nBu tweetin duygusunu aşağıdaki beş puanlık ölçeğe göre nasıl değerlendirirsiniz?\n1: çok olumsuz\n2: biraz olumsuz\n3: nötr\n4: biraz olumlu\n5: çok olumlu.\nAçıklama yapmadan yalnızca sayıyı belirtin.""",

    # Japanese
    """以下はソーシャルメディアのツイートです: {}。\nこのツイートの感情を以下の5段階評価でどのように評価しますか?\n1: 非常にネガティブ\n2: ややネガティブ\n3: 中立\n4: ややポジティブ\n5: 非常にポジティブ。\n説明なしで番号を出力してください。""",

    # Korean
    """다음은 소셜 미디어의 트윗입니다: {}.\n이 트윗의 감정을 아래의 5점 척도에 따라 어떻게 평가하시겠습니까?\n1: 매우 부정적\n2: 다소 부정적\n3: 중립\n4: 다소 긍정적\n5: 매우 긍정적.\n설명 없이 숫자만 출력하십시오.""",

    # Vietnamese
    """Đây là một tweet trên mạng xã hội: {}.\nBạn sẽ đánh giá cảm xúc của tweet này như thế nào dựa trên thang điểm năm dưới đây?\n1: rất tiêu cực\n2: hơi tiêu cực\n3: trung tính\n4: hơi tích cực\n5: rất tích cực.\nChỉ xuất ra số mà không cần giải thích.""",

    # Thai
    """นี่คือตัวอย่างทวีตจากสื่อสังคมออนไลน์: {}。\nคุณจะประเมินความรู้สึกของทวีตนี้อย่างไรตามมาตราส่วนห้าจุดด้านล่าง?\n1: ลบมาก\n2: ลบบางส่วน\n3: เป็นกลาง\n4: บวกบางส่วน\n5: บวกมาก。\nพิมพ์เฉพาะตัวเลขโดยไม่ต้องมีคำอธิบายใด ๆ""",

    # Indonesian
    """Berikut adalah tweet dari media sosial: {}.\nBagaimana Anda menilai sentimen dari tweet ini berdasarkan skala lima poin di bawah ini?\n1: sangat negatif\n2: agak negatif\n3: netral\n4: agak positif\n5: sangat positif.\nKeluarkan nomornya saja tanpa penjelasan.""",

    # Malay
    """Ini adalah tweet dari media sosial: {}.\nBagaimana anda menilai sentimen tweet ini berdasarkan skala lima mata di bawah?\n1: sangat negatif\n2: agak negatif\n3: neutral\n4: agak positif\n5: sangat positif.\nNyatakan nombor tersebut tanpa sebarang penjelasan.""",

    # Lao
    """นี่คือทวีตจากสื่อสังคมออนไลน์: {}.\nคุณจะประเมินความรู้สึกของทวีตนี้ตามมาตราส่วนห้าจุดด้านล่างได้อย่างไร?\n1: เชิงลบมาก\n2: เชิงลบบางส่วน\n3: เป็นกลาง\n4: เชิงบวกบางส่วน\n5: เชิงบวกมาก.\nพิมพ์เฉพาะตัวเลขโดยไม่ต้องมีคำอธิบายใดๆ""",

    # Burmese
    """ဤတွင်ဆိုရှယ်မီဒီယာမှတွစ်တာဖြစ်ပါသည်: {}。\nဒီတွစ်တာ၏ခံစားချက်ကို အောက်တွင် ဖော်ပြထားသော ၅ အဆင့်သတ်မှတ်ချက်အရ ဘယ်လိုသတ်မှတ်မလဲ?\n1: အလွန် 부정적\n2: 약간 부정적\n3: 중립\n4: 약간 긍정적\n5: 매우 긍정적\n설명 없이 숫자만 출력하십시오""",

    # Cebuano
    """Aniay usa ka tweet gikan sa social media: {}.\nUnsaon nimo pag-evaluate sa sentiment niini nga tweet base sa lima ka punto nga scale sa ubos?\n1: labing negatibo\n2: medyo negatibo\n3: neutral\n4: medyo positibo\n5: labing positibo.\nIbutang ang numero nga walay bisan unsang pagpasabot.""",

    # Khmer
    """នេះគឺជាអត្ថបទប្រព័ន្ធផ្សព្វផ្សាយសង្គមមួយ: {}。\nតើអ្នកនឹងវាយតម្លៃអារម្មណ៍នៃការប្រកាសនេះយ៉ាងដូចម្ដេចដោយផ្អែកលើជម្រើស 5 លេខខាងក្រោម?\n1: អវិជ្ជមានខ្លាំង\n2: អវិជ្ជមានបន្តិច\n3: អព្យាក្រឹត\n4: អវិជ្ជមានបន្តិច\n5: អវិជ្ជមានខ្លាំង.\nសូមចុចលើលេខបន្តិចនិងមិនត្រូវបញ្ជាក់ពិពណ៌នា។""",

    # Tagalog
    """Narito ang isang tweet mula sa social media: {}.\nPaano mo i-evaluate ang sentiment ng tweet na ito batay sa limang puntos na iskala sa ibaba?\n1: napakanegatibo\n2: medyo negatibo\n3: neutral\n4: medyo positibo\n5: napakapositibo.\nIlagay ang numero nang walang anumang paliwanag.""",

    # Hindi
    """यहाँ एक सोशल मीडिया ट्वीट है: {}。\nआप इस ट्वीट की भावना का मूल्यांकन पाँच-बिंदु पैमाने के आधार पर कैसे करेंगे?\n1: बहुत नकारात्मक\n2: कुछ हद तक नकारात्मक\n3: तटस्थ\n4: कुछ हद तक सकारात्मक\n5: बहुत सकारात्मक।\nकोई भी स्पष्टीकरण के बिना केवल संख्या दर्ज करें।""",

    # Bengali
    """এখানে একটি সামাজিক মিডিয়া টুইট রয়েছে: {}。\nএই টুইটের অনুভূতির মূল্যায়ন পাঁচ-পয়েন্টের স্কেলে কীভাবে করবেন?\n1: খুব নেতিবাচক\n2: কিছুটা নেতিবাচক\n3: নিরপেক্ষ\n4: কিছুটা ইতিবাচক\n5: খুব ইতিবাচক।\nসংখ্যাটি লিখুন কোনও ব্যাখ্যা ছাড়াই।""",

    # Urdu
    """یہاں ایک سوشل میڈیا ٹویٹ ہے: {}.\nآپ اس ٹویٹ کے جذبات کو کس طرح پانچ نکاتی پیمانے پر جانچیں گے؟\n1: بہت منفی\n2: کچھ منفی\n3: غیر جانبدار\n4: کچھ مثبت\n5: بہت مثبت۔\nکوئی بھی وضاحت کے بغیر صرف نمبر لکھیں۔"""
]


# Function to create instruction based on the selected template
def create_instruction(text):
    instruction_template = random.choice(sentiment_templates)
    instruction = instruction_template.format(text)
    return instruction

# Create instructions for each tweet
df['instruction'] = df["Tweet"].apply(create_instruction)
df['output'] = df['Majority']

# Select the required columns
df = df[['instruction', 'output']]


df = balance_classes(df, 500, 'output')

# Save the DataFrame to a parquet file
df.to_parquet("../../data/VT/Vulgarity_Tweets.parquet", index=False)