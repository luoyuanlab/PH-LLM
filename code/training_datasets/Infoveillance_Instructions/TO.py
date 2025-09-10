# -*- coding: utf-8 -*-
# TO (Twitter Optimists)
# - Paper: https://web.eecs.umich.edu/~mihalcea/papers/ruan.acl16.pdf
# - Data: http://lit.eecs.umich.edu/downloads.html#Twitter%20Optimism%20Dataset

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import pandas as pd
import random
from utils.data_utils import balance_classes

# Load data
df = pd.read_csv('../../data/TO/tweets_annotation.csv')
# Define multilingual templates for the optimism/pessimism classification prompt
templates = [
    # Chinese
    """请分析下面的社交媒体帖子，并将其感知的乐观或悲观程度分类为以下三个类别之一：“optimistic”（乐观）、“pessimistic”（悲观）、“neutral”（中立）。
中立意味着介于两者之间。
帖子内容:
{}
哪个类别最能分类该帖子？""",

    # English
    """Please analyze the social media post below and classify its perceived level of optimism or pessimism into ONE of the following categories: "optimistic", "pessimistic", "neutral".
Neutral means somewhere in between.
Post:
{}
Which category best classifies the post?""",

    # German
    """Bitte analysieren Sie den folgenden Social-Media-Beitrag und klassifizieren Sie das wahrgenommene Maß an Optimismus oder Pessimismus in EINE der folgenden Kategorien: "optimistic" (optimistisch), "pessimistic" (pessimistisch), "neutral" (neutral).
Neutral bedeutet irgendwo dazwischen.
Beitrag:
{}
Welche Kategorie beschreibt den Beitrag am besten?""",

    # French
    """Veuillez analyser le post de réseau social ci-dessous et classer son niveau perçu d'optimisme ou de pessimisme dans l'UNE des catégories suivantes : "optimistic" (optimiste), "pessimistic" (pessimiste), "neutral" (neutre).
Neutre signifie quelque part entre les deux.
Poste:
{}
Quelle catégorie décrit le mieux le post ?""",

    # Spanish
    """Por favor, analice la publicación en redes sociales a continuación y clasifique su nivel percibido de optimismo o pesimismo en UNA de las siguientes categorías: "optimistic" (optimista), "pessimistic" (pesimista), "neutral" (neutral).
Neutral significa en algún punto intermedio.
Publicación:
{}
¿Qué categoría describe mejor la publicación?""",

    # Portuguese
    """Por favor, analise a publicação nas redes sociais abaixo e classifique seu nível percebido de otimismo ou pessimismo em UMA das seguintes categorias: "optimistic" (otimista), "pessimistic" (pessimista), "neutral" (neutro).
Neutro significa em algum ponto intermediário.
Postagem:
{}
Qual categoria melhor classifica a postagem?""",

    # Italian
    """Si prega di analizzare il post sui social media di seguito e di classificare il suo livello percepito di ottimismo o pessimismo in UNA delle seguenti categorie: "optimistic" (ottimista), "pessimistic" (pessimista), "neutral" (neutro).
Neutro significa da qualche parte nel mezzo.
Post:
{}
Quale categoria descrive meglio il post?""",

    # Dutch
    """Analyseer alstublieft de onderstaande sociale mediapost en classificeer het waargenomen niveau van optimisme of pessimisme in EEN van de volgende categorieën: "optimistic" (optimistisch), "pessimistic" (pessimistisch), "neutral" (neutraal).
Neutraal betekent ergens tussenin.
Bericht:
{}
Welke categorie beschrijft het bericht het beste?""",

    # Russian
    """Пожалуйста, проанализируйте нижеуказанный пост в социальных сетях и классифицируйте его предполагаемый уровень оптимизма или пессимизма в ОДНУ из следующих категорий: "optimistic" (оптимистичный), "pessimistic" (пессимистичный), "neutral" (нейтральный).
Нейтральный означает что-то среднее.
Пост:
{}
Какая категория лучше всего описывает пост?""",

    # Czech
    """Analyzujte prosím níže uvedený příspěvek na sociálních sítích a zařaďte jeho vnímanou úroveň optimismu nebo pesimismu do JEDNÉ z následujících kategorií: "optimistic" (optimistický), "pessimistic" (pesimistický), "neutral" (neutrální).
Neutrální znamená něco mezi.
Příspěvek:
{}
Která kategorie nejlépe klasifikuje příspěvek?""",

    # Polish
    """Proszę przeanalizować poniższy post w mediach społecznościowych i sklasyfikować jego postrzegany poziom optymizmu lub pesymizmu do JEDNEJ z następujących kategorii: "optimistic" (optymistyczny), "pessimistic" (pesymistyczny), "neutral" (neutralny).
Neutralny oznacza coś pomiędzy.
Post:
{}
Która kategoria najlepiej opisuje post?""",

    # Arabic
    """يرجى تحليل المنشور أدناه على وسائل التواصل الاجتماعي وتصنيف مستوى التفاؤل أو التشاؤم المدرك إلى واحدة من الفئات التالية: "optimistic" (متفائل)، "pessimistic" (متشائم)، "neutral" (محايد).
المحايد يعني في مكان ما بينهما.
المنشور:
{}
أي فئة تصف المنشور بشكل أفضل؟""",

    # Persian
    """لطفاً پست زیر در شبکه‌های اجتماعی را تحلیل کنید و سطح درک شده‌ی خوش‌بینی یا بدبینی آن را در یکی از دسته‌های زیر قرار دهید: "optimistic" (خوش‌بینانه)، "pessimistic" (بدبینانه)، "neutral" (خنثی).
خنثی یعنی جایی در بین این دو.
پست:
{}
کدام دسته‌بندی بهترین توصیف را از پست ارائه می‌دهد؟""",

    # Hebrew
    """אנא נתח את הפוסט הבא במדיה החברתית וסווג את רמת האופטימיות או הפסימיות הנתפסת לאחת מהקטגוריות הבאות: "optimistic" (אופטימי), "pessimistic" (פסימי), "neutral" (נייטרלי).
נייטרלי פירושו משהו באמצע.
פוסט:
{}
איזו קטגוריה מתארת את הפוסט בצורה הטובה ביותר?""",

    # Turkish
    """Lütfen aşağıdaki sosyal medya gönderisini analiz edin ve algılanan iyimserlik veya kötümserlik düzeyini aşağıdaki kategorilerden BİRİ olarak sınıflandırın: "optimistic" (iyimser), "pessimistic" (kötümser), "neutral" (nötr).
Nötr, bu ikisi arasında bir şey demektir.
Gönderi:
{}
Gönderiyi en iyi hangi kategori sınıflandırır?""",

    # Japanese
    """以下のソーシャルメディアの投稿を分析し、その楽観的または悲観的なレベルを次のカテゴリのいずれかに分類してください：「optimistic」（楽観的）、「pessimistic」（悲観的）、「neutral」（中立）。
中立とは、その中間のどこかです。
投稿:
{}
この投稿を最もよく分類するカテゴリはどれですか？""",

    # Korean
    """다음 소셜 미디어 게시물을 분석하고 낙관적 또는 비관적인 것으로 인식된 수준을 다음 범주 중 하나로 분류하세요: "optimistic" (낙관적), "pessimistic" (비관적), "neutral" (중립적).
중립적은 그 중간 어디쯤입니다.
게시물:
{}
어떤 범주가 게시물을 가장 잘 분류합니까?""",

    # Vietnamese
    """Vui lòng phân tích bài đăng trên mạng xã hội dưới đây và phân loại mức độ lạc quan hay bi quan của nó vào MỘT trong các danh mục sau: "optimistic" (lạc quan), "pessimistic" (bi quan), "neutral" (trung lập).
Trung lập có nghĩa là ở đâu đó ở giữa.
Bài đăng:
{}
Danh mục nào phân loại bài đăng tốt nhất?""",

    # Thai
    """โปรดวิเคราะห์โพสต์โซเชียลมีเดียด้านล่างและจัดหมวดหมู่ระดับการมองโลกในแง่ดีหรือแง่ร้ายตามที่รับรู้เป็นหมวดหมู่ต่อไปนี้ "optimistic" (มองในแง่ดี), "pessimistic" (มองในแง่ร้าย), "neutral" (เป็นกลาง)
ความเป็นกลางหมายถึงบางสิ่งที่อยู่ระหว่างกลาง
โพสต์:
{}
หมวดหมู่ใดที่จำแนกโพสต์ได้ดีที่สุด?""",

    # Indonesian
    """Harap analisis kiriman media sosial di bawah ini dan klasifikasikan tingkat optimisme atau pesimisme yang dirasakan ke dalam SATU dari kategori berikut: "optimistic" (optimis), "pessimistic" (pesimis), "neutral" (netral).
Netral berarti di antara keduanya.
Kiriman:
{}
Kategori mana yang paling mengklasifikasikan kiriman ini?""",

    # Malay
    """Sila analisis siaran media sosial di bawah dan klasifikasikan tahap optimisme atau pesimisme yang dilihat ke dalam SATU daripada kategori berikut: "optimistic" (optimistik), "pessimistic" (pesimistik), "neutral" (neutral).
Neutral bermaksud di antara keduanya.
Siaran:
{}
Kategori manakah yang paling baik mengklasifikasikan siaran itu?""",

    # Lao
    """โปรดวิเคราะห์โพสต์โซเชียลมีเดียด้านล่างและจัดหมวดหมู่ระดับของความมองโลกในแง่ดีหรือมองในแง่ร้ายตามที่รับรู้เป็นหมวดหมู่ต่อไปนี้: "optimistic" (มองในแง่ดี), "pessimistic" (มองในแง่ร้าย), "neutral" (เป็นกลาง)
ความเป็นกลางหมายถึงสิ่งที่อยู่ระหว่างสองสิ่งนี้
โพสต์:
{}
หมวดหมู่ใดที่จัดประเภทโพสต์ได้ดีที่สุด?""",

    # Burmese
    """အောက်ပါ လူမှုမီဒီယာစာပို့ကို လေ့လာပြီး ၎င်း၏ အားသာချက် သို့မဟုတ် အားနည်းချက်ကို အောက်ပါ အမျိုးအစားများထဲမှ "optimistic" (သတ္တိရှိခြင်း), "pessimistic" (စိတ်ဆင်းရဲခြင်း), "neutral" (အတွင်းရေး) အမျိုးအစားတစ်ခုသို့ သတ်မှတ်ပါ။
အတွင်းရေးသည် ၎င်းအကြားက အနေအထားဖြစ်သည်။
စာပို့:
{}
ဤစာပို့ကို အကောင်းဆုံး သတ်မှတ်နိုင်သော အမျိုးအစားမှာ မည်သည့်အမျိုးအစားဖြစ်ပါသနည်း?""",

    # Cebuano
    """Palihug i-analyze ang social media post sa ubos ug i-classify ang nakita nga lebel sa optimism o pessimism ngadto sa USA sa mosunod nga mga kategorya: "optimistic", "pessimistic", "neutral".
Ang neutral nagpasabot sa tunga-tunga.
Post:
{}
Asa nga kategorya ang labing maayo nga nagklasipikar sa post?""",

    # Khmer
    """សូមវិភាគអត្ថបទបណ្តាញសង្គមខាងក្រោម ហើយចាត់ថ្នាក់កម្រិតនៃការមើលឃើញកថាភាពឬការមើលឃើញកថាភាពជាក្រុមមួយក្នុងចំណោមក្រុមដូចខាងក្រោម: "optimistic" (សង្ឃឹមថាមានអនាគតល្អ), "pessimistic" (សង្ឃឹមថាអនាគតអាក្រក់), "neutral" (មធ្យម)
មធ្យមមានន័យថាខាងលើគ្នា
អត្ថបទ:
{}
ក្រុមណាដែលមានចំណាត់ថ្នាក់ល្អបំផុត?""",

    # Tagalog
    """Pakianalisa ang post sa social media sa ibaba at uriin ang antas ng optimismo o pessimismo nito sa ISA sa mga sumusunod na kategorya: "optimistic" (optimista), "pessimistic" (pesimista), "neutral" (neutral).
Ang neutral ay nangangahulugang isang lugar sa pagitan.
Post:
{}
Aling kategorya ang pinakamahusay na nag-uuri ng post?""",

    # Hindi
    """कृपया नीचे दिए गए सोशल मीडिया पोस्ट का विश्लेषण करें और इसके अनुभव किए गए आशावाद या निराशावाद के स्तर को निम्नलिखित श्रेणियों में से किसी एक में वर्गीकृत करें: "optimistic" (आशावादी), "pessimistic" (निराशावादी), "neutral" (तटस्थ)।
तटस्थ का अर्थ है कहीं बीच में।
पोस्ट:
{}
कौन सा श्रेणी इस पोस्ट को सबसे अच्छी तरह से वर्गीकृत करता है?""",

    # Bengali
    """দয়া করে নীচের সোশ্যাল মিডিয়া পোস্টটি বিশ্লেষণ করুন এবং এর উপলব্ধি করা আশাবাদ বা হতাশার স্তরকে নিম্নলিখিত বিভাগগুলির মধ্যে একটি হিসাবে শ্রেণীবদ্ধ করুন: "optimistic" (আশাবাদী), "pessimistic" (নিরাশাবাদী), "neutral" (নিরপেক্ষ)।
নিরপেক্ষ মানে এর মধ্যবর্তী কোথাও।
পোস্ট:
{}
কোন বিভাগটি পোস্টটি সবচেয়ে ভাল শ্রেণীবদ্ধ করে?""",

    # Urdu
    """براہ کرم نیچے دی گئی سوشل میڈیا پوسٹ کا تجزیہ کریں اور اس کی سطح پر perceived optimism یا pessimism کو درج ذیل میں سے کسی ایک زمرے میں درجہ بندی کریں: "optimistic" (پرامید), "pessimistic" (مایوس), "neutral" (غیر جانبدار)۔
غیر جانبدار کا مطلب ہے کہیں درمیان میں۔
پوسٹ:
{}
کون سی قسم پوسٹ کو سب سے بہتر درجہ بندی کرتی ہے؟"""
]

# Function to create instruction based on the selected template
def create_instruction(text):
    instruction_template = random.choice(templates)
    instruction = instruction_template.format(text)
    return instruction

# Create instructions for each tweet
df['instruction'] = df["Tweet"].apply(create_instruction)

# Function to classify the tweet
def classify_tweet(score):
    if score >= 1:
        return 'optimistic'
    elif score <= -1:
        return 'pessimistic'
    else:
        return 'neutral'

# Apply classification
df['output'] = df['AverageAnnotation'].apply(classify_tweet)

# Select the required columns
instructions_df = df[['instruction', 'output']]
# downsampling and upsampling
instructions_df = balance_classes(instructions_df, 2000, 'output')
# Save the DataFrame to a parquet file
instructions_df.to_parquet("../../data/TO/Multilingual_Optimism_Pessimism_Classification.parquet", index=False)
instructions_df.output.value_counts()