# -*- coding: utf-8 -*-
# AD (Aggression Detection)
# - Paper: http://aclweb.org/anthology/W18-5106
# - Data: https://github.com/SilentFlame/AggressionDetection/blob/master/DataPre-Processing/processedDataWithoutID.txt

import pandas as pd
import random
from sklearn.utils import resample

# Load data
df = pd.read_csv('../../data/AD/processedDataWithoutID.txt', sep="\t", header=None)

# Define the multilingual templates for each category
CAG_templates = [
    # Chinese
    """以下是一个社交媒体文本: {text}\n\n 如果这些文本是隐蔽的攻击 (CAG)，即针对受害者的间接攻击，通常表现为（不真诚的）礼貌表达（通过使用约定俗成的礼貌结构），如讽刺、反问等，则回答 'yes'。否则，回答 'no'。""",

    # English
    """Here's a social media text: {text}\n\n Respond 'yes' if the texts are Covertly-Aggressive (CAG), an indirect attack against the victim and is often packaged as (insincere) polite expressions (through the use of conventionalized polite structures), such as satire, rhetorical questions, etc. Otherwise, respond 'no'.""",

    # German
    """Hier ist ein Social-Media-Text: {text}\n\n Antworten Sie mit 'yes', wenn die Texte verdeckt-aggressiv (CAG) sind, ein indirekter Angriff gegen das Opfer, der oft als (unaufrichtige) höfliche Ausdrücke verpackt ist (durch die Verwendung konventioneller höflicher Strukturen), wie Satire, rhetorische Fragen usw. Andernfalls antworten Sie mit 'no'.""",

    # French
    """Voici un texte des médias sociaux: {text}\n\n Répondez 'yes' si les textes sont agressifs de manière cachée (CAG), une attaque indirecte contre la victime, souvent déguisée en expressions polies (insincères) (par l'utilisation de structures polies conventionnelles), comme des satires, des questions rhétoriques, etc. Sinon, répondez 'no'.""",

    # Spanish
    """Aquí hay un texto de redes sociales: {text}\n\n Responda 'yes' si los textos son Agresivos Encubiertos (CAG), un ataque indirecto contra la víctima que a menudo se presenta como expresiones educadas (insinceras) (a través del uso de estructuras educadas convencionales), como sátiras, preguntas retóricas, etc. De lo contrario, responda 'no'.""",

    # Portuguese
    """Aqui está um texto de mídia social: {text}\n\n Responda 'yes' se os textos forem Agressivos de Forma Oculta (CAG), um ataque indireto contra a vítima que muitas vezes é embalado como expressões polidas (insinceras) (através do uso de estruturas polidas convencionais), como sátiras, perguntas retóricas, etc. Caso contrário, responda 'no'.""",

    # Italian
    """Ecco un testo dei social media: {text}\n\n Rispondi 'yes' se i testi sono Aggressivi in Modo Nascosto (CAG), un attacco indiretto contro la vittima spesso confezionato come espressioni educate (insincere) (attraverso l'uso di strutture educate convenzionali), come satira, domande retoriche, ecc. Altrimenti, rispondi 'no'.""",

    # Dutch
    """Hier is een social media tekst: {text}\n\n Antwoord 'yes' als de teksten Covert-Aggressief (CAG) zijn, een indirecte aanval tegen het slachtoffer, vaak verpakt als (onoprechte) beleefde uitdrukkingen (door het gebruik van conventionele beleefde structuren), zoals satire, retorische vragen, enz. Anders antwoord 'no'.""",

    # Russian
    """Вот текст из социальных сетей: {text}\n\n Ответьте 'yes', если тексты являются скрыто-агрессивными (CAG), это косвенная атака на жертву, часто представленная в виде (неискренних) вежливых выражений (через использование общепринятых вежливых структур), таких как сатира, риторические вопросы и т.д. В противном случае ответьте 'no'.""",

    # Czech
    """Zde je text ze sociálních médií: {text}\n\n Odpovězte 'yes', pokud jsou texty skrytě agresivní (CAG), což je nepřímý útok proti oběti, často zabalený jako (neupřímné) zdvořilé výrazy (použitím konvenčních zdvořilých struktur), jako je satira, rétorické otázky atd. Jinak odpovězte 'no'.""",

    # Polish
    """Oto tekst z mediów społecznościowych: {text}\n\n Odpowiedz 'yes', jeśli teksty są ukrycie agresywne (CAG), czyli pośredni atak na ofiarę, często zapakowany jako (nieszczere) uprzejme wyrażenia (poprzez użycie konwencjonalnych uprzejmych struktur), takie jak satyra, pytania retoryczne itp. W przeciwnym razie odpowiedz 'no'.""",

    # Arabic
    """إليك نص من وسائل التواصل الاجتماعي: {text}\n\n أجب بـ 'yes' إذا كانت النصوص عدوانية بشكل غير مباشر (CAG)، وهي هجوم غير مباشر على الضحية وغالبًا ما تكون مغلفة بتعبيرات مهذبة (غير صادقة) (من خلال استخدام هياكل مهذبة تقليدية)، مثل السخرية، الأسئلة البلاغية، إلخ. وإلا، أجب بـ 'no'.""",

    # Persian
    """در اینجا یک متن رسانه اجتماعی است: {text}\n\n پاسخ 'yes' بدهید اگر متون به طور مخفیانه تهاجمی هستند (CAG)، حمله غیرمستقیم به قربانی است که اغلب به عنوان بیان‌های مودبانه (غیر صادقانه) (از طریق استفاده از ساختارهای مودبانه متعارف)، مانند طنز، سوالات بلاغی و غیره بسته‌بندی می‌شود. در غیر این صورت، پاسخ 'no' بدهید.""",

    # Hebrew
    """הנה טקסט מהרשתות החברתיות: {text}\n\n השב 'yes' אם הטקסטים הם תוקפניים בצורה סמויה (CAG), התקפה עקיפה נגד הקורבן, שלעתים קרובות ארוזה כהבעות מנומסות (לא כנות) (באמצעות שימוש במבנים מנומסים קונבנציונליים), כמו סאטירה, שאלות רטוריות וכו'. אחרת, השב 'no'.""",

    # Turkish
    """İşte bir sosyal medya metni: {text}\n\n Metinler Gizli-Saldırgan (CAG) ise, yani mağdura yönelik dolaylı bir saldırı olup genellikle (samimi olmayan) kibar ifadeler (geleneksel kibar yapıların kullanımıyla) olarak paketlenmişse 'yes' yanıtını verin. Aksi takdirde, 'no' yanıtını verin.""",

    # Japanese
    """以下はソーシャルメディアのテキストです: {text}\n\n テキストが覆い隠された攻撃 (CAG)、つまり被害者への間接的な攻撃であり、多くの場合、（不誠実な）丁寧な表現（一般的な丁寧な構造を使用して）としてパッケージ化されている場合は「yes」と答えてください。 さもなければ、「no」と答えてください。""",

    # Korean
    """다음은 소셜 미디어 텍스트입니다: {text}\n\n 텍스트가 은밀한 공격 (CAG)인 경우, 즉 피해자에 대한 간접 공격이며 종종 (불성실한) 예의 표현 (일반적인 예의 구조를 사용하여)으로 포장된 경우 'yes'라고 답하십시오. 그렇지 않으면 'no'라고 답하십시오.""",

    # Vietnamese
    """Đây là một văn bản truyền thông xã hội: {text}\n\n Trả lời 'yes' nếu các văn bản là Tấn công Ngầm (CAG), một cuộc tấn công gián tiếp chống lại nạn nhân và thường được đóng gói dưới dạng các biểu hiện lịch sự (không chân thành) (thông qua việc sử dụng các cấu trúc lịch sự thông thường), như châm biếm, câu hỏi tu từ, v.v. Ngược lại, trả lời 'no'.""",

    # Thai
    """นี่คือตัวอย่างข้อความจากสื่อสังคมออนไลน์: {text}\n\n ตอบ 'yes' หากข้อความเป็นการโจมตีโดยปริยาย (CAG) ซึ่งเป็นการโจมตีทางอ้อมต่อเหยื่อและมักถูกบรรจุเป็นคำพูดที่สุภาพ (ไม่จริงใจ) (โดยใช้โครงสร้างคำสุภาพตามที่กำหนด) เช่น การประชดประชัน คำถามเชิงวาทศิลป์ ฯลฯ หากไม่ใช่ ให้ตอบว่า 'no'.""",

    # Indonesian
    """Berikut adalah teks media sosial: {text}\n\n Jawab 'yes' jika teks tersebut termasuk Covertly-Aggressive (CAG), serangan tidak langsung terhadap korban yang sering kali dikemas sebagai ekspresi sopan (tidak tulus) (melalui penggunaan struktur sopan konvensional), seperti sindiran, pertanyaan retoris, dll. Jika tidak, jawab 'no'.""",

    # Malay
    """Ini adalah teks media sosial: {text}\n\n Jawab 'yes' jika teks adalah Agresif Terselubung (CAG), serangan tidak langsung terhadap korban yang sering kali disajikan sebagai ekspresi sopan (tidak tulus) (melalui penggunaan struktur sopan konvensional), seperti sindiran, soalan retorik, dll. Jika tidak, jawab 'no'.""",

    # Lao
    """นี่คือตัวอย่างข้อความจากสื่อสังคมออนไลน์: {text}\n\n ตอบ 'yes' หากข้อความเหล่านี้เป็นการโจมตีแบบปกปิด (CAG) ซึ่งเป็นการโจมตีทางอ้อมต่อเหยื่อและมักจะปรากฏในรูปแบบการแสดงความสุภาพ (ไม่จริงใจ) (ผ่านการใช้โครงสร้างสุภาพที่เป็นมาตรฐาน) เช่น การเสียดสี, คำถามเชิงวาทศิลป์ เป็นต้น มิฉะนั้น ให้ตอบว่า 'no'.""",

    # Burmese
    """ဒီမှာ ဆိုရှယ်မီဒီယာစာသားပါ။ {text}\n\n အကယ်၍ စာသားများသည် Covertly-Aggressive (CAG) ဖြစ်ပြီး ဒါဟာ သင့်ရဲ႕ ထိခိုက်မှုကို တိုက်ရိုက်မဟုတ်ဘဲ ကာတွန်း၊ ချီးမွမ်းစကားစသည့် ပုံမှန်ပေးသည့် အတက်မပါသော polite expression ဖြင့် ထိခိုက်စေခြင်းဖြစ်ပါက 'yes' ဟုဖြေပါ။ ဒါမဟုတ်ပါက 'no' ဟု ဖြေပါ။""",

    # Cebuano
    """Aniay usa ka teksto sa social media: {text}\n\n Tubaga og 'yes' kung ang mga teksto kay Covertly-Aggressive (CAG), usa ka dili direktang pag-atake batok sa biktima ug kasagaran giputos isip (dili tinud-anay) nga polite expressions (pinaagi sa paggamit og conventionalized polite structures), sama sa satire, rhetorical questions, ug uban pa. Kung dili, tubaga og 'no'.""",

    # Khmer
    """នេះគឺជាអត្ថបទប្រព័ន្ធផ្សព្វផ្សាយសង្គមមួយ: {text}\n\n សូមឆ្លើយថា 'yes' ប្រសិនបើអត្ថបទទាំងនេះគឺជាការវាយប្រហារដោយឆាប់រហ័ស (CAG) ដែលជាការវាយប្រហារក្រៅពីជVictims និងជាញឹកញាប់បង្ហាញខ្លួនក្នុងអាការៈ polite (insincere) (ដោយការប្រើប្រាស់សំណុំ polite ដែលគេបានចាត់ត្រា), រួមជាមួយការរិះគន់, សំណួរបែបសំណួរមិនច្បាស់, ។ វិញមួយឆ្លើយថា 'no' ។""",

    # Tagalog
    """Narito ang isang teksto sa social media: {text}\n\n Sagutin ng 'yes' kung ang mga teksto ay Covertly-Aggressive (CAG), isang hindi direktang pag-atake laban sa biktima at kadalasang naka-pack bilang (hindi taos-pusong) magalang na mga expression (sa pamamagitan ng paggamit ng mga kinaugalian na magalang na mga istruktura), tulad ng satire, mga tanong na retorikal, atbp. Kung hindi, sagutin ng 'no'.""",

    # Hindi
    """यहां एक सोशल मीडिया पाठ है: {text}\n\n 'yes' का उत्तर दें यदि पाठ छुपे हुए आक्रामक (CAG) हैं, एक अप्रत्यक्ष हमला जो पीड़ित के खिलाफ होता है और अक्सर इसे (असत्य) शिष्ट अभिव्यक्तियों के रूप में पैक किया जाता है (सामान्य शिष्ट संरचनाओं के उपयोग के माध्यम से), जैसे व्यंग्य, व्यंग्यात्मक प्रश्न आदि। अन्यथा, 'no' का उत्तर दें।""",

    # Bengali
    """এখানে একটি সামাজিক মিডিয়া টেক্সট রয়েছে: {text}\n\n 'yes' উত্তর দিন যদি টেক্সটগুলি গোপনে-আক্রমণাত্মক (CAG) হয়, যা প্রায়ই শিকারকে একটি পরোক্ষ আক্রমণ হিসাবে প্যাক করা হয় (অসৎ) ভদ্র অভিব্যক্তি হিসাবে (পারম্পরিক ভদ্র গঠনের মাধ্যমে), যেমন বিদ্রূপ, তুচ্ছ প্রশ্ন ইত্যাদি। অন্যথায়, 'no' উত্তর দিন।""",

    # Urdu
    """یہاں ایک سوشل میڈیا متن ہے: {text}\n\n جواب 'yes' دیں اگر متون خفیہ طور پر جارحانہ (CAG) ہیں، جو متاثرہ کے خلاف ایک بالواسطہ حملہ ہے اور اکثر (غیر مخلص) شائستہ اظہار کے طور پر پیک کیا جاتا ہے (روایتی شائستہ ڈھانچوں کے استعمال کے ذریعے)، جیسے طنز، بلاغی سوالات وغیرہ۔ ورنہ، جواب 'no' دیں۔"""
]


OAG_templates = [
    # Chinese
    """以下是一个社交媒体文本: {text}\n\n 如果这些文本是公开攻击性 (OAG)，即通过使用被认为是攻击性和/或特定句法结构的特定词汇项目或词汇特征，公开表达攻击性，则回答 'yes'。 否则，回答 'no'。""",

    # English
    """Here's a social media text: {text}\n\n Respond 'yes' if the texts are Overtly-Aggressive (OAG), in which aggression is overtly expressed either through the use of specific kinds of lexical items or lexical features which are considered aggressive and/or certain syntactic structures. Otherwise, respond 'no'.""",

    # German
    """Hier ist ein Social-Media-Text: {text}\n\n Antworten Sie mit 'yes', wenn die Texte offen aggressiv (OAG) sind, bei denen die Aggression entweder durch die Verwendung bestimmter Arten von lexikalischen Elementen oder lexikalischen Merkmalen, die als aggressiv gelten, und/oder durch bestimmte syntaktische Strukturen offen zum Ausdruck gebracht wird. Andernfalls antworten Sie mit 'no'.""",

    # French
    """Voici un texte des médias sociaux: {text}\n\n Répondez 'yes' si les textes sont ouvertement agressifs (OAG), où l'agression est exprimée ouvertement soit par l'utilisation de certains types d'éléments lexicaux ou de caractéristiques lexicales considérées comme agressives et/ou de certaines structures syntaxiques. Sinon, répondez 'no'.""",

    # Spanish
    """Aquí hay un texto de redes sociales: {text}\n\n Responda 'yes' si los textos son abiertamente agresivos (OAG), en los que la agresión se expresa abiertamente ya sea mediante el uso de ciertos tipos de elementos léxicos o características léxicas que se consideran agresivos y/o ciertas estructuras sintácticas. De lo contrario, responda 'no'.""",

    # Portuguese
    """Aqui está um texto de mídia social: {text}\n\n Responda 'yes' se os textos forem Agressivos de Forma Aberta (OAG), em que a agressão é expressa abertamente através do uso de certos tipos de itens lexicais ou características lexicais consideradas agressivas e/ou certas estruturas sintáticas. Caso contrário, responda 'no'.""",

    # Italian
    """Ecco un testo dei social media: {text}\n\n Rispondi 'yes' se i testi sono apertamente aggressivi (OAG), in cui l'aggressione è espressa apertamente sia attraverso l'uso di specifici tipi di elementi lessicali o caratteristiche lessicali considerate aggressive e/o determinate strutture sintattiche. Altrimenti, rispondi 'no'.""",

    # Dutch
    """Hier is een social media tekst: {text}\n\n Antwoord 'yes' als de teksten Overtly-Aggressief (OAG) zijn, waarbij de agressie openlijk wordt uitgedrukt door het gebruik van specifieke soorten lexicale items of lexicale kenmerken die als agressief worden beschouwd en/of bepaalde syntactische structuren. Anders antwoord 'no'.""",

    # Russian
    """Вот текст из социальных сетей: {text}\n\n Ответьте 'yes', если тексты являются открыто-агрессивными (OAG), в которых агрессия открыто выражается либо через использование определенных видов лексических единиц или лексических признаков, которые считаются агрессивными, и/или определенных синтаксических структур. В противном случае ответьте 'no'.""",

    # Czech
    """Zde je text ze sociálních médií: {text}\n\n Odpovězte 'yes', pokud jsou texty otevřeně agresivní (OAG), ve kterých je agrese otevřeně vyjádřena buď použitím určitých druhů lexikálních prvků nebo lexikálních vlastností, které jsou považovány za agresivní a/nebo určitých syntaktických struktur. Jinak odpovězte 'no'.""",

    # Polish
    """Oto tekst z mediów społecznościowych: {text}\n\n Odpowiedz 'yes', jeśli teksty są jawnie agresywne (OAG), w których agresja jest wyrażana otwarcie poprzez użycie określonych rodzajów jednostek leksykalnych lub cech leksykalnych uznanych za agresywne i/lub określonych struktur składniowych. W przeciwnym razie odpowiedz 'no'.""",

    # Arabic
    """إليك نص من وسائل التواصل الاجتماعي: {text}\n\n أجب بـ 'yes' إذا كانت النصوص عدوانية بشكل علني (OAG)، حيث يتم التعبير عن العدوانية بشكل علني إما من خلال استخدام أنواع معينة من العناصر أو الميزات اللغوية التي تُعتبر عدوانية و/أو تراكيب نحوية معينة. وإلا، أجب بـ 'no'.""",

    # Persian
    """در اینجا یک متن رسانه اجتماعی است: {text}\n\n پاسخ 'yes' بدهید اگر متون به طور آشکار تهاجمی هستند (OAG)، در جایی که تهاجم به طور آشکار از طریق استفاده از انواع خاصی از موارد واژگانی یا ویژگی‌های واژگانی که به عنوان تهاجمی شناخته می‌شوند و/یا ساختارهای نحوی خاص ابراز می‌شود. در غیر این صورت، پاسخ 'no' بدهید.""",

    # Hebrew
    """הנה טקסט מהרשתות החברתיות: {text}\n\n השב 'yes' אם הטקסטים הם תוקפניים בצורה גלויה (OAG), בה האגרסיה מתבטאת בגלוי או דרך השימוש בסוגים מסוימים של פריטים או מאפיינים לשוניים הנחשבים אגרסיביים ו/או במבנים תחביריים מסוימים. אחרת, השב 'no'.""",

    # Turkish
    """İşte bir sosyal medya metni: {text}\n\n Metinler Açıkça Saldırgan (OAG) ise, yani saldırganlık ya belirli türde sözcük ögeleri veya saldırgan kabul edilen sözcük özellikleri ve/veya belirli sözdizimsel yapılar yoluyla açıkça ifade ediliyorsa 'yes' yanıtını verin. Aksi takdirde, 'no' yanıtını verin.""",

    # Japanese
    """以下はソーシャルメディアのテキストです: {text}\n\n テキストが公開的に攻撃的 (OAG) である場合、つまり攻撃性が特定の種類の語彙アイテムや攻撃的と見なされる語彙の特徴、および/または特定の構文構造の使用を通じて明示的に表現されている場合は「yes」と答えてください。さもなければ、「no」と答えてください。""",

    # Korean
    """다음은 소셜 미디어 텍스트입니다: {text}\n\n 텍스트가 명백히 공격적인 경우 (OAG), 즉 공격성이 특정 종류의 어휘 항목 또는 공격적으로 간주되는 어휘 특징 및/또는 특정 구문 구조를 통해 명백히 표현되는 경우 'yes'라고 답하십시오. 그렇지 않으면 'no'라고 답하십시오.""",

    # Vietnamese
    """Đây là một văn bản truyền thông xã hội: {text}\n\n Trả lời 'yes' nếu các văn bản là Công khai Tấn công (OAG), trong đó sự gây hấn được biểu hiện công khai thông qua việc sử dụng các mục từ vựng cụ thể hoặc các tính năng từ vựng được coi là gây hấn và/hoặc các cấu trúc cú pháp nhất định. Ngược lại, trả lời 'no'.""",

    # Thai
    """นี่คือตัวอย่างข้อความจากสื่อสังคมออนไลน์: {text}\n\n ตอบ 'yes' หากข้อความเป็นการโจมตีโดยตรง (OAG) โดยการแสดงความรุนแรงที่เห็นได้ชัดผ่านการใช้คำศัพท์หรือโครงสร้างทางไวยากรณ์เฉพาะที่ถือว่าก้าวร้าว. มิฉะนั้นให้ตอบ 'no'.""",

    # Indonesian
    """Berikut adalah teks media sosial: {text}\n\n Jawab 'yes' jika teks tersebut termasuk Overtly-Aggressive (OAG), di mana agresi diekspresikan secara terbuka baik melalui penggunaan item leksikal tertentu atau fitur leksikal yang dianggap agresif dan/atau struktur sintaksis tertentu. Jika tidak, jawab 'no'.""",

    # Malay
    """Ini adalah teks media sosial: {text}\n\n Jawab 'yes' jika teks adalah Agresif Terbuka (OAG), di mana pencerobohan dinyatakan secara terbuka melalui penggunaan jenis item leksikal tertentu atau ciri leksikal yang dianggap agresif dan/atau struktur sintaksis tertentu. Jika tidak, jawab 'no'.""",

    # Lao
    """นี่คือตัวอย่างข้อความจากสื่อสังคมออนไลน์: {text}\n\n ตอบ 'yes' หากข้อความเหล่านี้เป็นการโจมตีโดยตรง (OAG) ซึ่งเป็นการแสดงความรุนแรงที่ชัดเจนผ่านการใช้คำศัพท์หรือโครงสร้างทางไวยากรณ์เฉพาะที่ถือว่ารุนแรง. หากไม่ใช่, ตอบว่า 'no'.""",

    # Burmese
    """ဒီမှာ ဆိုရှယ်မီဒီယာစာသားပါ။ {text}\n\n အကယ်၍ စာသားများသည် ပေါ်ပြီကျန်သေးသောရုပ်သိမ်းစေသော Overtly-Aggressive (OAG) ဖြစ်ပြီး ကျန်ခဲ့သောကာယကဏ်မရှိဘဲတန်ခိုးဖို့ရဲ့လက်ရှိလုံးဝမလိုအပ်သော တန်ခိုးပြောသည်။ 'yes' ဟုဖြေပါ။ ဖြစ်ခဲ့ပါက 'no' ဟုဖြေပါ။""",

    # Cebuano
    """Aniay usa ka teksto sa social media: {text}\n\n Tubaga og 'yes' kung ang mga teksto kay Overtly-Aggressive (OAG), diin ang agresyon gipahayag og tuyo pinaagi sa paggamit sa mga espesipikong klase sa lexical items o lexical features nga giisip nga agresibo ug/o espesipikong syntactic structures. Kung dili, tubaga og 'no'.""",

    # Khmer
    """នេះគឺជាអត្ថបទប្រព័ន្ធផ្សព្វផ្សាយសង្គមមួយ: {text}\n\n សូមឆ្លើយថា 'yes' ប្រសិនបើអត្ថបទទាំងនេះគឺជាការវាយប្រហារដោយឆាប់រហ័ស (OAG) ដែលជាការវាយប្រហារត្រង់ៗនិងត្រូវបានបង្ហាញនៅក្នុងពាក្យសម្ងាត់ដែលត្រូវបានគិតថាគឺជាការវាយប្រហារគួរប្រឆាំងនិង/ឬច្រើនកូតស្តង់ដា។ វិញមួយឆ្លើយថា 'no' ។""",

    # Tagalog
    """Narito ang isang teksto sa social media: {text}\n\n Sagutin ng 'yes' kung ang mga teksto ay Overtly-Aggressive (OAG), kung saan ang agresyon ay malinaw na ipinahayag sa pamamagitan ng paggamit ng mga tiyak na uri ng mga item na lexicon o mga tampok na lexicon na itinuturing na agresibo at/o ilang mga istrukturang syntactic. Kung hindi, sagutin ng 'no'.""",

    # Hindi
    """यहां एक सोशल मीडिया पाठ है: {text}\n\n 'yes' का उत्तर दें यदि पाठ खुलेआम आक्रामक (OAG) हैं, जिसमें आक्रामकता को स्पष्ट रूप से व्यक्त किया जाता है, जैसे कि आक्रामक मानी जाने वाली विशिष्ट प्रकार की शब्दावली या व्याकरणिक संरचनाओं का उपयोग। अन्यथा, 'no' का उत्तर दें।""",

    # Bengali
    """এখানে একটি সামাজিক মিডিয়া টেক্সট রয়েছে: {text}\n\n 'yes' উত্তর দিন যদি টেক্সটগুলি খোলাখুলিভাবে আক্রমণাত্মক (OAG) হয়, যেখানে আক্রমণাত্মকতা খোলাখুলিভাবে প্রকাশ করা হয় যা বিশেষ ধরণের শব্দ বা বিশেষ্য গঠনগুলি ব্যবহারের মাধ্যমে, যা আক্রমণাত্মক বলে মনে করা হয়। অন্যথায়, 'no' উত্তর দিন।""",

    # Urdu
    """یہاں ایک سوشل میڈیا متن ہے: {text}\n\n جواب 'yes' دیں اگر متون کھل کر جارحانہ ہیں (OAG)، جس میں جارحیت کو کھلے عام اظہار کیا جاتا ہے چاہے یہ مخصوص قسم کے لسانی عناصر یا لسانی خصوصیات کے استعمال کے ذریعے ہو، جو جارحانہ سمجھی جاتی ہیں، اور/یا مخصوص نحوی ڈھانچوں کا استعمال۔ ورنہ، جواب 'no' دیں۔"""
]


NAG_templates = [
    # Chinese
    """以下是一个社交媒体文本: {text}\n\n 如果这些文本是非攻击性 (NAG)，即这些文本既不是隐蔽攻击性 (CAG) 也不是公开攻击性 (OAG)，则回答 'yes'。 否则，回答 'no'。""",

    # English
    """Here's a social media text: {text}\n\n Respond 'yes' if the texts are Non-Aggressive (NAG), meaning the texts are neither Covertly-Aggressive (CAG) nor Overtly-Aggressive (OAG). Otherwise, respond 'no'.""",

    # German
    """Hier ist ein Social-Media-Text: {text}\n\n Antworten Sie mit 'yes', wenn die Texte nicht aggressiv (NAG) sind, das heißt, die Texte sind weder verdeckt-aggressiv (CAG) noch offen-aggressiv (OAG). Andernfalls antworten Sie mit 'no'.""",

    # French
    """Voici un texte des médias sociaux: {text}\n\n Répondez 'yes' si les textes sont non agressifs (NAG), ce qui signifie que les textes ne sont ni agressifs de manière cachée (CAG) ni ouvertement agressifs (OAG). Sinon, répondez 'no'.""",

    # Spanish
    """Aquí hay un texto de redes sociales: {text}\n\n Responda 'yes' si los textos no son agresivos (NAG), es decir, los textos no son ni Agresivos Encubiertos (CAG) ni Agresivos Abiertos (OAG). De lo contrario, responda 'no'.""",

    # Portuguese
    """Aqui está um texto de mídia social: {text}\n\n Responda 'yes' se os textos forem Não Agressivos (NAG), ou seja, os textos não são nem Agressivos de Forma Oculta (CAG) nem Agressivos de Forma Aberta (OAG). Caso contrário, responda 'no'.""",

    # Italian
    """Ecco un testo dei social media: {text}\n\n Rispondi 'yes' se i testi sono Non Aggressivi (NAG), ovvero i testi non sono né Aggressivi in Modo Nascosto (CAG) né Aggressivi in Modo Aperto (OAG). Altrimenti, rispondi 'no'.""",

    # Dutch
    """Hier is een social media tekst: {text}\n\n Antwoord 'yes' als de teksten Niet-Agressief (NAG) zijn, wat betekent dat de teksten noch Covert-Aggressief (CAG) noch Overt-Aggressief (OAG) zijn. Anders antwoord 'no'.""",

    # Russian
    """Вот текст из социальных сетей: {text}\n\n Ответьте 'yes', если тексты являются неагрессивными (NAG), что означает, что тексты не являются ни скрыто-агрессивными (CAG), ни открыто-агрессивными (OAG). В противном случае ответьте 'no'.""",

    # Czech
    """Zde je text ze sociálních médií: {text}\n\n Odpovězte 'yes', pokud jsou texty neagresivní (NAG), což znamená, že texty nejsou ani skrytě agresivní (CAG), ani otevřeně agresivní (OAG). Jinak odpovězte 'no'.""",

    # Polish
    """Oto tekst z mediów społecznościowych: {text}\n\n Odpowiedz 'yes', jeśli teksty są nieagresywne (NAG), co oznacza, że teksty nie są ani ukrycie agresywne (CAG), ani jawnie agresywne (OAG). W przeciwnym razie odpowiedz 'no'.""",

    # Arabic
    """إليك نص من وسائل التواصل الاجتماعي: {text}\n\n أجب بـ 'yes' إذا كانت النصوص غير عدوانية (NAG)، مما يعني أن النصوص ليست عدوانية بشكل غير مباشر (CAG) ولا عدوانية بشكل علني (OAG). وإلا، أجب بـ 'no'.""",

    # Persian
    """در اینجا یک متن رسانه اجتماعی است: {text}\n\n پاسخ 'yes' بدهید اگر متون غیرتهاجمی (NAG) هستند، به این معنی که متون نه به صورت مخفیانه تهاجمی (CAG) هستند و نه به صورت آشکارا تهاجمی (OAG). در غیر این صورت، پاسخ 'no' بدهید.""",

    # Hebrew
    """הנה טקסט מהרשתות החברתיות: {text}\n\n השב 'yes' אם הטקסטים אינם תוקפניים (NAG), כלומר הטקסטים אינם תוקפניים בצורה סמויה (CAG) ולא תוקפניים בצורה גלויה (OAG). אחרת, השב 'no'.""",

    # Turkish
    """İşte bir sosyal medya metni: {text}\n\n Metinler Saldırgan Değilse (NAG), yani metinler ne Gizli-Saldırgan (CAG) ne de Açıkça-Saldırgan (OAG) değilse 'yes' yanıtını verin. Aksi takdirde, 'no' yanıtını verin.""",

    # Japanese
    """以下はソーシャルメディアのテキストです: {text}\n\n テキストが非攻撃的 (NAG) である場合、つまり、テキストが隠れている攻撃 (CAG) または公開されている攻撃 (OAG) ではない場合は「yes」と答えてください。さもなければ、「no」と答えてください。""",

    # Korean
    """다음은 소셜 미디어 텍스트입니다: {text}\n\n 텍스트가 비공격적인 경우 (NAG), 즉 텍스트가 은밀한 공격 (CAG)도 아니고 공개적인 공격 (OAG)도 아닌 경우 'yes'라고 답하십시오. 그렇지 않으면 'no'라고 답하십시오.""",

    # Vietnamese
    """Đây là một văn bản truyền thông xã hội: {text}\n\n Trả lời 'yes' nếu các văn bản là Không Gây Hấn (NAG), nghĩa là các văn bản không phải là Tấn công Ngầm (CAG) cũng như không phải là Tấn công Công khai (OAG). Ngược lại, trả lời 'no'.""",

    # Thai
    """นี่คือตัวอย่างข้อความจากสื่อสังคมออนไลน์: {text}\n\n ตอบ 'yes' หากข้อความไม่ใช่การโจมตี (NAG) หมายความว่าข้อความไม่ใช่การโจมตีทางอ้อม (CAG) และไม่ใช่การโจมตีโดยตรง (OAG) หากไม่ใช่ ให้ตอบว่า 'no'.""",

    # Indonesian
    """Berikut adalah teks media sosial: {text}\n\n Jawab 'yes' jika teks tersebut termasuk Non-Agresif (NAG), artinya teks tersebut bukan Covertly-Aggressive (CAG) maupun Overtly-Aggressive (OAG). Jika tidak, jawab 'no'.""",

    # Malay
    """Ini adalah teks media sosial: {text}\n\n Jawab 'yes' jika teks adalah Tidak Agresif (NAG), yang bermaksud teks tersebut bukan Covertly-Aggressive (CAG) dan bukan juga Overtly-Aggressive (OAG). Jika tidak, jawab 'no'.""",

    # Lao
    """นี่คือตัวอย่างข้อความจากสื่อสังคมออนไลน์: {text}\n\n ตอบ 'yes' หากข้อความเหล่านี้ไม่ใช่การโจมตี (NAG) ซึ่งหมายความว่าข้อความเหล่านี้ไม่ใช่การโจมตีโดยปริยาย (CAG) หรือการโจมตีโดยตรง (OAG). หากไม่ใช่ ให้ตอบว่า 'no'.""",

    # Burmese
    """ဒီမှာ ဆိုရှယ်မီဒီယာစာသားပါ။ {text}\n\n အကယ်၍ စာသားများသည် တိုးလျှောက်မဖြစ်၊ တနောက်လေ၏ Covertly-Aggressive (CAG) ဟု ချီးမြှောက်ခြင်းမရှိဘဲ၊ အသိအမှတ်ပြုရန်လျက်ရှိသဖြင့် (Non-Aggressive, NAG) ဖြစ်ပါက 'yes' ဟု ဖြေပါ။ မဟုတ်ပါက 'no' ဟု ဖြေပါ။""",

    # Cebuano
    """Aniay usa ka teksto sa social media: {text}\n\n Tubaga og 'yes' kung ang mga teksto kay Non-Aggressive (NAG), nga nagpasabot nga ang mga teksto kay dili Covertly-Aggressive (CAG) o Overtly-Aggressive (OAG). Kung dili, tubaga og 'no'.""",

    # Khmer
    """នេះគឺជាអត្ថបទប្រព័ន្ធផ្សព្វផ្សាយសង្គមមួយ: {text}\n\n សូមឆ្លើយថា 'yes' ប្រសិនបើអត្ថបទទាំងនេះគឺជាការការពារប្រឆាំង (NAG) គឺជាពាក្យសម្ងាត់ដែលមិនមែនជាការការពារប្រឆាំងចំពោះរូបភាពផ្សេងៗទៀតទេ។ យើងគ្រាន់តែបង្វឺនគោលការណ៍សូមឆ្លើយថា 'no' ។""",

    # Tagalog
    """Narito ang isang teksto sa social media: {text}\n\n Sagutin ng 'yes' kung ang mga teksto ay Hindi-Aggressive (NAG), ibig sabihin, ang mga teksto ay hindi Covertly-Aggressive (CAG) o Overtly-Aggressive (OAG). Kung hindi, sagutin ng 'no'.""",

    # Hindi
    """यहां एक सोशल मीडिया पाठ है: {text}\n\n 'yes' का उत्तर दें यदि पाठ गैर-आक्रामक (NAG) हैं, जिसका अर्थ है कि पाठ न तो छुपे हुए आक्रामक (CAG) हैं और न ही खुले आक्रामक (OAG) हैं। अन्यथा, 'no' का उत्तर दें।""",

    # Bengali
    """এখানে একটি সামাজিক মিডিয়া টেক্সট রয়েছে: {text}\n\n 'yes' উত্তর দিন যদি টেক্সটগুলি অ-আক্রমণাত্মক (NAG) হয়, অর্থাৎ টেক্সটগুলি না গোপনভাবে আক্রমণাত্মক (CAG) এবং না স্পষ্টভাবে আক্রমণাত্মক (OAG)। অন্যথায়, 'no' উত্তর দিন।""",

    # Urdu
    """یہاں ایک سوشل میڈیا متن ہے: {text}\n\n جواب 'yes' دیں اگر متون غیر جارحانہ ہیں (NAG)، اس کا مطلب ہے کہ متون نہ خفیہ طور پر جارحانہ ہیں (CAG) اور نہ ہی کھل کر جارحانہ ہیں (OAG)۔ ورنہ، جواب 'no' دیں۔"""
]


# Function to create instruction based on the selected category and randomly chosen template
def create_instruction(text, category):
    if category == 'CAG':
        instruction_template = random.choice(CAG_templates)
    elif category == 'OAG':
        instruction_template = random.choice(OAG_templates)
    else:
        instruction_template = random.choice(NAG_templates)

    instruction = instruction_template.format(text=text)
    return instruction

# Create a blank DataFrame
instructions_df = pd.DataFrame(columns=['instruction', 'output'])
# Iterate over each row and create instructions for each category
for index, row in df.iterrows():
    text = row[0]
    for category in ['CAG', 'OAG', 'NAG']:
        instruction = create_instruction(text, category)
        output = 'yes' if row[1] == category else 'no'
        instructions_df = pd.concat([instructions_df, pd.DataFrame({'instruction': [instruction], 'output': [output]})], ignore_index=True)

# # Save the DataFrame to a parquet file
# instructions_df.to_parquet("../../data/AD/Multilingual_AggreIden.parquet", index=False)

# Separate the majority and minority classes
df_majority = instructions_df[instructions_df.output == 'no']
df_minority = instructions_df[instructions_df.output == 'yes']

# Downsample the majority class
df_majority_downsampled = resample(df_majority,
                                   replace=False,    # sample without replacement
                                   n_samples=len(df_minority), # to match minority class
                                   random_state=42) # reproducible results

# Combine the minority class with the downsampled majority class
instructions_df_balanced = pd.concat([df_majority_downsampled, df_minority])
# instructions_df_balanced.to_parquet("../../data/AD/Multilingual_AggreIden.parquet", index=False)
# sample n=23000 and save to the same parquet file
instructions_df_balanced.sample(n=8000).to_parquet("../../data/AD/Multilingual_AggreIden.parquet", index=False)