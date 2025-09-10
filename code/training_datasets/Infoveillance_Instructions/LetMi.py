# -*- coding: utf-8 -*-
# Let-Mi: An Arabic Levantine Twitter Dataset for Misogynistic Language
# - Paper: https://github.com/leondz/hatespeechdata?tab=readme-ov-file
# - Data: https://drive.google.com/file/d/1mM2vnjsy7QfUmdVUpKqHRJjZyQobhTrW/view
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import pandas as pd
import random
from utils.data_utils import balance_classes

# Load the data
df = pd.read_csv('../../data/LetMi/let-mi_train_part.csv')


# Define the instruction templates in different languages
instruction_templates = {
    "Chinese": """
    这里有一条来自知名女性记者/主持人时间线的推文回复：{}
    判断这条推文回复是否具有厌女倾向。
    厌女推文是指那些表达对女性的仇恨、侮辱或言语攻击的情况。
    如果是，请回答“yes”。否则，请回答“no”。
    """,

    "English": """
    Here is a tweet reply from the timelines of popular female journalists/reporters: {}
    Determine if the tweet reply is misogynistic or not.
    Misogynistic tweets are those instances that express any hatred, insulting or verbal abuse towards women.
    If so, respond 'yes'. Otherwise, respond 'no'.
    """,

    "German": """
    Hier ist eine Antwort auf einen Tweet aus den Timelines beliebter Journalistinnen/Reporterinnen: {}
    Bestimmen Sie, ob die Antwort frauenfeindlich ist oder nicht.
    Frauenfeindliche Tweets sind solche, die Hass, Beleidigung oder verbale Angriffe auf Frauen ausdrücken.
    Wenn ja, antworten Sie mit 'yes'. Andernfalls antworten Sie mit 'no'.
    """,

    "French": """
    Voici une réponse à un tweet provenant des timelines de journalistes/reporters populaires : {}
    Déterminez si la réponse au tweet est misogyne ou non.
    Les tweets misogynes sont ceux qui expriment toute haine, insulte ou abus verbal envers les femmes.
    Si c'est le cas, répondez 'yes'. Sinon, répondez 'no'.
    """,

    "Spanish": """
    Aquí tienes una respuesta a un tuit de las líneas de tiempo de periodistas/reporteras populares: {}
    Determina si la respuesta al tuit es misógina o no.
    Los tuits misóginos son aquellos que expresan cualquier odio, insulto o abuso verbal hacia las mujeres.
    Si es así, responde 'yes'. De lo contrario, responde 'no'.
    """,

    "Portuguese": """
    Aqui está uma resposta a um tweet das timelines de jornalistas/repórteres populares: {}
    Determine se a resposta ao tweet é misógina ou não.
    Tweets misóginos são aqueles que expressam qualquer ódio, insulto ou abuso verbal contra as mulheres.
    Se for o caso, responda 'yes'. Caso contrário, responda 'no'.
    """,

    "Italian": """
    Ecco una risposta a un tweet dalle timeline di giornaliste/reporter popolari: {}
    Determina se la risposta al tweet è misogina o no.
    I tweet misogini sono quelli che esprimono odio, insulti o abusi verbali verso le donne.
    Se è così, rispondi 'yes'. Altrimenti, rispondi 'no'.
    """,

    "Dutch": """
    Hier is een reactie op een tweet uit de tijdlijnen van populaire vrouwelijke journalisten/reporters: {}
    Bepaal of de tweetreactie misogynistisch is of niet.
    Misogynistische tweets zijn die gevallen waarin haat, beledigingen of verbale mishandeling tegen vrouwen worden geuit.
    Als dat zo is, reageer dan met 'yes'. Anders reageer met 'no'.
    """,

    "Russian": """
    Вот ответ на твит из лент популярных журналисток/репортеров: {}
    Определите, является ли ответ на твит женоненавистническим.
    Женоненавистнические твиты - это те случаи, когда выражается ненависть, оскорбление или словесное оскорбление в адрес женщин.
    Если да, ответьте 'yes'. В противном случае ответьте 'no'.
    """,

    "Czech": """
    Zde je odpověď na tweet z časové osy oblíbených novinářek/reportrérek: {}
    Určete, zda je odpověď na tweet misogynistická, nebo ne.
    Misogynistické tweety jsou ty, které vyjadřují jakoukoli nenávist, urážku nebo slovní útok vůči ženám.
    Pokud ano, odpovězte 'yes'. V opačném případě odpovězte 'no'.
    """,

    "Polish": """
    Oto odpowiedź na tweet z linii czasu popularnych dziennikarek/reporterek: {}
    Określ, czy odpowiedź na tweet jest mizoginistyczna, czy nie.
    Mizoginistyczne tweety to te, które wyrażają jakąkolwiek nienawiść, zniewagę lub werbalne nadużycia wobec kobiet.
    Jeśli tak, odpowiedz 'yes'. W przeciwnym razie odpowiedz 'no'.
    """,

    "Arabic": """
    إليك رد على تغريدة من جداول زمنية للصحفيات/المراسلات المشهورات: {}
    حدد ما إذا كان الرد على التغريدة معادياً للنساء أم لا.
    التغريدات المعادية للنساء هي تلك التي تعبر عن أي كراهية أو إهانة أو إساءة لفظية تجاه النساء.
    إذا كان الأمر كذلك، أجب بـ 'yes'. خلاف ذلك، أجب بـ 'no'.
    """,

    "Persian": """
    در اینجا پاسخی به یک توییت از جداول زمانی خبرنگاران/گزارشگران زن محبوب آمده است: {}
    تعیین کنید که آیا پاسخ به توییت ضد زن است یا خیر.
    توییت‌های ضد زن آن‌هایی هستند که هر گونه نفرت، توهین یا سوء استفاده لفظی نسبت به زنان را بیان می‌کنند.
    اگر چنین است، پاسخ دهید 'yes'. در غیر این صورت، پاسخ دهید 'no'.
    """,

    "Hebrew": """
    הנה תגובה לציוץ מציוץ בטוויטר מהציר הזמן של עיתונאיות/כתבות פופולריות: {}
    קבע אם התגובה לציוץ היא מיזוגנית או לא.
    ציוצים מיזוגניים הם אלה שמביעים שנאה, עלבון או התעללות מילולית כלפי נשים.
    אם כן, השב 'yes'. אחרת, השב 'no'.
    """,

    "Turkish": """
    İşte popüler kadın gazetecilerin/raportörlerin zaman çizelgelerinden bir tweet yanıtı: {}
    Tweet yanıtının kadın düşmanlığı içerip içermediğini belirleyin.
    Kadın düşmanı tweetler, kadınlara yönelik herhangi bir nefret, hakaret veya sözlü taciz ifade eden durumlardır.
    Eğer öyleyse, 'yes' diye yanıtlayın. Aksi takdirde 'no' diye yanıtlayın.
    """,

    "Japanese": """
    人気のある女性ジャーナリスト/レポーターのタイムラインからのツイート返信はこちらです: {}
    ツイートの返信が女性嫌悪的かどうかを判断してください。
    女性嫌悪的なツイートは、女性に対する憎悪、侮辱、または言葉による虐待を表すものです。
    もしそうなら、「yes」と答えてください。そうでない場合は、「no」と答えてください。
    """,

    "Korean": """
    인기 있는 여성 기자/리포터의 타임라인에서 온 트윗 답글입니다: {}
    트윗 답글이 여성혐오적인지 여부를 판단하세요.
    여성혐오적인 트윗은 여성에 대한 증오, 모욕 또는 언어적 학대를 표현하는 경우입니다.
    그렇다면 'yes'라고 응답하십시오. 그렇지 않으면 'no'라고 응답하십시오.
    """,

    "Vietnamese": """
    Đây là một phản hồi trên tweet từ dòng thời gian của các nhà báo/phóng viên nữ nổi tiếng: {}
    Xác định xem phản hồi tweet có phải là kỳ thị phụ nữ hay không.
    Những tweet kỳ thị phụ nữ là những trường hợp thể hiện bất kỳ sự căm ghét, xúc phạm hoặc lăng mạ phụ nữ nào.
    Nếu đúng, hãy trả lời 'yes'. Nếu không, hãy trả lời 'no'.
    """,

    "Thai": """
    นี่คือการตอบกลับทวีตจากไทม์ไลน์ของนักข่าว/ผู้สื่อข่าวหญิงที่มีชื่อเสียง: {}
    กำหนดว่าการตอบกลับทวีตนี้มีการแสดงออกถึงความเกลียดชังทางเพศหรือไม่
    ทวีตที่เกลียดชังทางเพศคือกรณีที่แสดงความเกลียดชัง การดูหมิ่น หรือการใช้ถ้อยคำหยาบคายต่อผู้หญิง
    หากเป็นเช่นนั้น ให้ตอบว่า 'yes' มิฉะนั้น ให้ตอบว่า 'no'
    """,

    "Indonesian": """
    Berikut adalah balasan tweet dari timeline jurnalis/reporter wanita populer: {}
    Tentukan apakah balasan tweet tersebut misoginis atau tidak.
    Tweet misoginis adalah tweet yang mengekspresikan kebencian, penghinaan, atau pelecehan verbal terhadap wanita.
    Jika iya, jawab 'yes'. Jika tidak, jawab 'no'.
    """,

    "Malay": """
    Berikut adalah balasan tweet daripada garis masa wartawan/wartawan wanita popular: {}
    Tentukan sama ada balasan tweet itu bersifat misoginis atau tidak.
    Tweet misoginis adalah tweet yang menyatakan kebencian, penghinaan atau penyalahgunaan lisan terhadap wanita.
    Jika ya, jawab 'yes'. Jika tidak, jawab 'no'.
    """,

    "Lao": """
    ນີ້ແມ່ນການຕອບກັບທິດຊອບທາງທະນາມສຳລັບນັກຂ່າວ/ຜູ້ລາວທີ່ມີຊື່ສຽງ: {}
    ກຳນົດວ່າການຕອບກັບທີ່ຕິດເທັດດອກນີ້ມີຄວາມເກຍດຊັງຫຼືບໍ່.
    ທິດຊອບທີ່ຕິດເທັດດອກນີ້ແມ່ນສັນຍານທີ່ເອີ້ນເກຍດຊັງ, ສຽງອື່ນຫຼືຄຳບອກທີ່ມີໃນຄວາມປວດໂທດຕໍ່ຜູ້ຫຍິງ.
    ຖ້າເປັນເຊັ່ນນັ້ນ, ຕອບວ່າ 'yes'. ບໍ່ຊັ້ນ, ຕອບວ່າ 'no'.
    """,

    "Burmese": """
    ထင်ရှားတဲ့ အမျိုးသမီးလှုပ်ရှားသူ/သတင်းထောက်တို့ရဲ့ အချိန်ဇယားကနေ Twitter မှာရှိတဲ့ တစ်ခုခုကို ပြန်ပြောကြားထားတာဖြစ်ပါတယ်: {}
    ဒီပြန်ပြောကြားထားတာဟာ အမျိုးသမီးမုန်းတီးမှုဖြစ်မဖြစ် သတ်မှတ်ပါ။
    အမျိုးသမီးမုန်းတီးမှုအဖြစ်တွေဟာ အမျိုးသမီးများကို မုန်းတီးခြင်း၊ အထင်ကရပြောဆိုခြင်း သို့မဟုတ် ဘာသာရေးအပြစ်တင်မှုကို ဖော်ပြတဲ့ အကြောင်းအရာတွေဖြစ်ပါတယ်။
    အကယ်၍ ဒီအဖွဲ့က ဟုတ်တယ်ဆိုရင် 'yes' လို့ပြန်ဆိုပါ။ ဟုတ်မဟုတ် 'no' လို့ ပြန်ဆိုပါ။
    """,

    "Cebuano": """
    Ania ang usa ka tubag sa tweet gikan sa mga timeline sa mga sikat nga babaye nga mga mamamahayag/reporter: {}
    Tukma kung ang tubag sa tweet misogynistic ba o dili.
    Ang mga misogynistic nga tweet mao kadtong mga halimbawa nga nagpakita og bisan unsang kasuko, pag-insulto o verbal nga pag-abuso batok sa mga babaye.
    Kung mao, tubag 'yes'. Kung dili, tubag 'no'.
    """,

    "Khmer": """
    នេះជាចម្លើយបញ្ចូលពីគណនី Twitter របស់អ្នកនិពន្ធ/អ្នករាយការណ៍ស្រីដែលពេញនិយម: {}
    សូមកំណត់ថាចម្លើយនេះមានភាពរើសអើងស្ត្រី ឬមិនមាន។
    Tweet រើសអើងស្ត្រី គឺជាអ្វីៗដែលបង្ហាញពីភាពស្អប់ខ្ពើម ការប្រមាថ ឬបំផ្លាញជាសរសេរ ឬមេដឹកនាំនៃការបកស្រាយមួយណាមួយដែលមិនសមរម្យ។
    ប្រសិនបើចម្លើយនោះមាន សូមឆ្លើយ 'yes' ។ ប្រសិនបើមិនមាន សូមឆ្លើយ 'no' ។
    """,

    "Tagalog": """
    Narito ang isang tugon sa tweet mula sa mga timeline ng mga sikat na babaeng mamamahayag/tagapag-ulat: {}
    Tukuyin kung ang tugon sa tweet ay misogynistic o hindi.
    Ang mga misogynistic na tweet ay yaong mga nagpapakita ng anumang galit, panlalait, o pang-aabusong salita laban sa mga kababaihan.
    Kung oo, sumagot ng 'yes'. Kung hindi, sumagot ng 'no'.
    """,

    "Hindi": """
    यहाँ एक प्रसिद्ध महिला पत्रकार/रिपोर्टर के टाइमलाइन से एक ट्वीट उत्तर है: {}
    यह निर्धारित करें कि ट्वीट उत्तर महिला विरोधी है या नहीं।
    महिला विरोधी ट्वीट वे होते हैं जो महिलाओं के प्रति किसी भी प्रकार की नफरत, अपमान या मौखिक गाली-गलौज व्यक्त करते हैं।
    यदि हाँ, तो 'yes' का उत्तर दें। अन्यथा, 'no' का उत्तर दें।
    """,

    "Bengali": """
    এখানে একজন জনপ্রিয় মহিলা সাংবাদিক/রিপোর্টারের টাইমলাইনের একটি টুইট উত্তর রয়েছে: {}
    নির্ধারণ করুন টুইটের উত্তরটি নারীবিদ্বেষী কিনা।
    নারীবিদ্বেষী টুইটগুলি সেই ঘটনাগুলি যা নারীদের প্রতি যে কোনও ঘৃণা, অপমান বা মৌখিক গালাগাল প্রকাশ করে।
    যদি হ্যাঁ, তাহলে 'yes' উত্তর দিন। অন্যথায়, 'no' উত্তর দিন।
    """,

    "Urdu": """
    یہاں ایک مشہور خاتون صحافی/رپورٹر کی ٹائم لائن سے ایک ٹویٹ جواب ہے: {}
    طے کریں کہ ٹویٹ کا جواب خواتین مخالف ہے یا نہیں۔
    خواتین مخالف ٹویٹس وہ ہیں جو خواتین کے خلاف کسی بھی نفرت، توہین یا زبانی زیادتی کا اظہار کرتی ہیں۔
    اگر ایسا ہے تو، 'yes' کا جواب دیں۔ ورنہ، 'no' کا جواب دیں۔
    """
}


# Create a DataFrame to store instructions
instructions_df = pd.DataFrame(columns=['instruction', 'output'])

# Generate instructions using randomly selected templates
instructions_df['instruction'] = df['text'].apply(
    lambda x: random.choice(list(instruction_templates.values())).format(x)
)

# Map the output to 'yes' or 'no' based on the 'misogyny' column
instructions_df['output'] = df['misogyny'].apply(lambda x: 'yes' if x == "misogyny" else 'no')

# collect a sample of n=5150
instructions_df = instructions_df.sample(n=5150, random_state=42)

# Display the first few rows of the DataFrame
instructions_df.to_parquet("../../data/LetMi/LetMi-part1.parquet", index = False)
# Define the instruction templates in different languages
instruction_templates = {
    "Chinese": """
    请将来自知名女性记者/主持人时间线的推文回复分类为以下七种厌女类别之一：
    "discredit", "derailing", "dominance", "stereotyping & objectification", "sexual harassment", "threat of violence", "damning"
    discredit: 指没有其他更大意图，仅是侮辱女性的推文。
    derailing: 推文表明了对女性的虐待的正当性，同时拒绝男性的责任，并试图中断对话以重新集中注意力。
    dominance: 推文表达了男性优越性或维持男性对女性的控制。
    stereotyping & objectification: 推文推广了对女性的普遍但固定且过于简化的形象/想法。这一标签还包括描述女性身体吸引力和/或提供狭隘标准比较的推文实例。
    sexual harassment: 推文描述了诸如性进攻、性要求和性骚扰等行为。
    threat of violence: 推文通过暴力威胁恐吓女性以使她们沉默，从而达到掌控女性的目的。
    damning: 推文包含祈祷以伤害女性；大部分祈祷涉及死亡/疾病的愿望，或请求神诅咒女性。

    推文回复：
    {}
    现在推文结束了。
    哪一种类别最适合对这条厌女推文回复进行分类？
    """,
    "English": """
    Please categorize the tweet reply from the timelines of popular female journalists/reporters into ONE of the seven misogynistic categories:
    "discredit", "derailing", "dominance", "stereotyping & objectification", "sexual harassment", "threat of violence", "damning"
    discredit: refers to tweets that combine slurring over women with no other larger intention.
    derailing: tweets that indicate a justification of women abuse while rejecting male responsibility with an attempt to disrupt the conversation in order to refocus it.
    dominance: tweets are those that express male superiority or preserve male control over women.
    stereotyping & objectification: tweets that promote a widely held but fixed and oversimplified image/idea of women. This label also refers to tweet instances that describe women’s physical appeal and/or provide comparisons to narrow standards.
    sexual harassment: tweets that describe actions such as sexual advances, requests for sexual favors, and sexual nature harassment.
    threat of violence: tweets that intimidate women to silence them with an intent to assert power over women through threats of violence physically.
    damning: tweets that contain prayers to hurt women; most of the prayers are death/illness wishes besides praying God to curse women.

    Tweet reply:
    {}
    Now the tweet ends.
    Which ONE category best categorizes the misogynistic tweet reply?
    """,
    "German": """
    Bitte kategorisieren Sie die Antwort auf den Tweet aus den Timelines beliebter Journalistinnen/Reporterinnen in EINE der sieben misogynistischen Kategorien:
    "discredit", "derailing", "dominance", "stereotyping & objectification", "sexual harassment", "threat of violence", "damning"
    discredit: bezieht sich auf Tweets, die Frauen beleidigen, ohne eine größere Absicht zu haben.
    derailing: Tweets, die Missbrauch von Frauen rechtfertigen, während die Verantwortung der Männer abgelehnt und der Versuch unternommen wird, das Gespräch zu unterbrechen, um es neu zu fokussieren.
    dominance: Tweets, die männliche Überlegenheit ausdrücken oder männliche Kontrolle über Frauen bewahren wollen.
    stereotyping & objectification: Tweets, die ein weit verbreitetes, aber festgefahrenes und vereinfachtes Bild/Idee von Frauen fördern. Dieses Label bezieht sich auch auf Tweet-Instanzen, die das körperliche Erscheinungsbild von Frauen beschreiben und/oder Vergleiche mit engen Standards ziehen.
    sexual harassment: Tweets, die Handlungen wie sexuelle Annäherungen, Aufforderungen zu sexuellen Gefälligkeiten und Belästigungen sexueller Natur beschreiben.
    threat of violence: Tweets, die Frauen einschüchtern, um sie zum Schweigen zu bringen, mit der Absicht, Macht über Frauen durch Gewaltandrohungen auszuüben.
    damning: Tweets, die Gebete enthalten, um Frauen zu verletzen; die meisten Gebete sind Todes- oder Krankheitswünsche, außerdem wird Gott gebeten, Frauen zu verfluchen.

    Tweet-Antwort:
    {}
    Jetzt ist der Tweet zu Ende.
    Welche EINER Kategorie ordnen Sie die misogynistische Tweet-Antwort am besten zu?
    """,
    "French": """
    Veuillez classer la réponse au tweet provenant des timelines de journalistes/reporters populaires dans UNE des sept catégories misogynes suivantes :
    "discredit", "derailing", "dominance", "stereotyping & objectification", "sexual harassment", "threat of violence", "damning"
    discredit: se réfère aux tweets qui insultent les femmes sans autre intention majeure.
    derailing: tweets qui indiquent une justification des abus envers les femmes tout en rejetant la responsabilité masculine avec une tentative de perturber la conversation pour la recentrer.
    dominance: tweets qui expriment la supériorité masculine ou préservent le contrôle masculin sur les femmes.
    stereotyping & objectification: tweets qui promeuvent une image/idée largement répandue mais fixe et trop simplifiée des femmes. Cette étiquette fait également référence aux instances de tweets qui décrivent l'attrait physique des femmes et/ou fournissent des comparaisons avec des normes restrictives.
    sexual harassment: tweets qui décrivent des actions telles que des avances sexuelles, des demandes de faveurs sexuelles, et du harcèlement à caractère sexuel.
    threat of violence: tweets qui intimident les femmes pour les faire taire avec l'intention d'affirmer une domination sur elles par des menaces de violence physique.
    damning: tweets qui contiennent des prières pour nuire aux femmes ; la plupart des prières sont des souhaits de mort/maladie en plus de demander à Dieu de maudire les femmes.

    Réponse au tweet :
    {}
    Maintenant, le tweet se termine.
    Quelle catégorie UNE catégorise le mieux la réponse au tweet misogyne ?
    """,
    "Spanish": """
    Por favor, clasifique la respuesta al tweet de las líneas de tiempo de periodistas/reporteras populares en UNA de las siete categorías misóginas:
    "discredit", "derailing", "dominance", "stereotyping & objectification", "sexual harassment", "threat of violence", "damning"
    discredit: se refiere a tweets que combinan insultos hacia las mujeres sin otra intención mayor.
    derailing: tweets que indican una justificación del abuso hacia las mujeres mientras se rechaza la responsabilidad masculina con un intento de interrumpir la conversación para volver a enfocarla.
    dominance: tweets que expresan la superioridad masculina o preservan el control masculino sobre las mujeres.
    stereotyping & objectification: tweets que promueven una imagen/idea ampliamente difundida pero fija y demasiado simplificada de las mujeres. Esta etiqueta también se refiere a instancias de tweets que describen el atractivo físico de las mujeres y/o proporcionan comparaciones con estándares restrictivos.
    sexual harassment: tweets que describen acciones como avances sexuales, solicitudes de favores sexuales y acoso de naturaleza sexual.
    threat of violence: tweets que intimidan a las mujeres para silenciarlas con la intención de afirmar el poder sobre ellas mediante amenazas de violencia física.
    damning: tweets que contienen oraciones para lastimar a las mujeres; la mayoría de las oraciones son deseos de muerte/enfermedad, además de pedirle a Dios que maldiga a las mujeres.

    Respuesta al tweet:
    {}
    Ahora el tweet termina.
    ¿Qué categoría UNA clasifica mejor la respuesta misógina al tweet?
    """,
    "Portuguese": """
    Por favor, categorize a resposta ao tweet das timelines de jornalistas/repórteres populares em UMA das sete categorias misóginas:
    "discredit", "derailing", "dominance", "stereotyping & objectification", "sexual harassment", "threat of violence", "damning"
    discredit: refere-se a tweets que combinam insultos às mulheres sem outra intenção maior.
    derailing: tweets que indicam uma justificação do abuso de mulheres enquanto rejeitam a responsabilidade masculina com uma tentativa de interromper a conversa para reorientá-la.
    dominance: tweets que expressam a superioridade masculina ou preservam o controle masculino sobre as mulheres.
    stereotyping & objectification: tweets que promovem uma imagem/idéia amplamente difundida, mas fixa e simplificada das mulheres. Este rótulo também se refere a instâncias de tweets que descrevem o apelo físico das mulheres e/ou fornecem comparações com padrões restritos.
    sexual harassment: tweets que descrevem ações como avanços sexuais, pedidos de favores sexuais e assédio de natureza sexual.
    threat of violence: tweets que intimidam as mulheres para silenciá-las com a intenção de afirmar poder sobre elas por meio de ameaças de violência física.
    damning: tweets que contêm orações para prejudicar as mulheres; a maioria das orações são desejos de morte/doença, além de pedir a Deus que amaldiçoe as mulheres.

    Resposta ao tweet:
    {}
    Agora o tweet termina.
    Qual categoria UMA categoriza melhor a resposta misógina ao tweet?
    """,
    "Italian": """
    Si prega di categorizzare la risposta al tweet dalle timeline di giornaliste/reporter popolari in UNA delle sette categorie misogine:
    "discredit", "derailing", "dominance", "stereotyping & objectification", "sexual harassment", "threat of violence", "damning"
    discredit: si riferisce a tweet che combinano insulti verso le donne senza altra intenzione maggiore.
    derailing: tweet che indicano una giustificazione dell'abuso sulle donne mentre si rifiuta la responsabilità maschile con un tentativo di interrompere la conversazione per rifocalizzarla.
    dominance: tweet che esprimono superiorità maschile o preservano il controllo maschile sulle donne.
    stereotyping & objectification: tweet che promuovono un'immagine/idea ampiamente diffusa ma fissa e semplificata delle donne. Questa etichetta si riferisce anche a casi di tweet che descrivono l'appeal fisico delle donne e/o forniscono confronti con standard ristretti.
    sexual harassment: tweet che descrivono azioni come avances sessuali, richieste di favori sessuali e molestie di natura sessuale.
    threat of violence: tweet che intimidiscono le donne per farle tacere con l'intento di affermare potere su di esse attraverso minacce di violenza fisica.
    damning: tweet che contengono preghiere per ferire le donne; la maggior parte delle preghiere sono desideri di morte/malattia, oltre a chiedere a Dio di maledire le donne.

    Risposta al tweet:
    {}
    Ora il tweet termina.
    Quale categoria UNA categorizza meglio la risposta misogina al tweet?
    """,
    "Dutch": """
    Categoriseer alsjeblieft de tweet-antwoord uit de tijdlijnen van populaire vrouwelijke journalisten/reporters in EEN van de zeven misogynistische categorieën:
    "discredit", "derailing", "dominance", "stereotyping & objectification", "sexual harassment", "threat of violence", "damning"
    discredit: verwijst naar tweets die vrouwen beledigen zonder andere grotere bedoeling.
    derailing: tweets die een rechtvaardiging van vrouwenmisbruik aangeven terwijl ze de verantwoordelijkheid van mannen afwijzen met een poging om het gesprek te verstoren om het opnieuw te richten.
    dominance: tweets die de mannelijke superioriteit uitdrukken of de mannelijke controle over vrouwen behouden.
    stereotyping & objectification: tweets die een wijdverbreid, maar vast en vereenvoudigd beeld/idee van vrouwen promoten. Dit label verwijst ook naar gevallen van tweets die de fysieke aantrekkingskracht van vrouwen beschrijven en/of vergelijkingen maken met beperkte normen.
    sexual harassment: tweets die acties beschrijven zoals seksuele avances, verzoeken om seksuele gunsten en seksuele intimidatie.
    threat of violence: tweets die vrouwen intimideren om ze het zwijgen op te leggen met de bedoeling om macht over vrouwen te bevestigen door middel van dreigingen van fysiek geweld.
    damning: tweets die gebeden bevatten om vrouwen te schaden; de meeste gebeden zijn wensen voor de dood/ziekte naast het bidden tot God om vrouwen te vervloeken.

    Tweet-antwoord:
    {}
    Nu eindigt de tweet.
    Welke EEN categorie categoriseert het misogynistische tweet-antwoord het beste?
    """,
    "Russian": """
    Пожалуйста, классифицируйте ответ на твит из лент популярных журналисток/репортеров в одну из семи женоненавистнических категорий:
    "discredit", "derailing", "dominance", "stereotyping & objectification", "sexual harassment", "threat of violence", "damning"
    discredit: относится к твитам, которые очерняют женщин без какой-либо другой, более значимой цели.
    derailing: твиты, которые оправдывают насилие над женщинами, отвергая при этом ответственность мужчин, и пытаются прервать разговор, чтобы перенаправить его.
    dominance: твиты, которые выражают превосходство мужчин или сохраняют контроль мужчин над женщинами.
    stereotyping & objectification: твиты, которые продвигают широко распространенный, но фиксированный и чрезмерно упрощенный образ/идею женщин. Этот ярлык также относится к случаям твитов, описывающих физическую привлекательность женщин и/или сравнивающих их с узкими стандартами.
    sexual harassment: твиты, которые описывают действия, такие как сексуальные домогательства, просьбы о сексуальных услугах и сексуальные домогательства.
    threat of violence: твиты, которые запугивают женщин, чтобы заставить их замолчать, с целью утвердить власть над ними посредством угрозы физического насилия.
    damning: твиты, которые содержат молитвы, чтобы причинить вред женщинам; большинство молитв включают пожелания смерти/болезни, а также просьбы к Богу проклясть женщин.

    Ответ на твит:
    {}
    Теперь твит заканчивается.
    Какая ОДНА категория лучше всего классифицирует женоненавистнический ответ на твит?
    """,
    "Czech": """
    Prosím, zařaďte odpověď na tweet z časové osy populárních novinářek/reportérek do JEDNÉ ze sedmi misogynních kategorií:
    "discredit", "derailing", "dominance", "stereotyping & objectification", "sexual harassment", "threat of violence", "damning"
    discredit: odkazuje na tweety, které očerňují ženy bez jiného většího záměru.
    derailing: tweety, které naznačují ospravedlnění zneužívání žen při odmítnutí odpovědnosti mužů a pokus o narušení rozhovoru s cílem přesměrovat jej.
    dominance: tweety, které vyjadřují mužskou nadřazenost nebo udržují mužskou kontrolu nad ženami.
    stereotyping & objectification: tweety, které podporují široce rozšířený, ale pevně zakořeněný a zjednodušený obraz/ideu žen. Tento štítek také odkazuje na případy tweetů, které popisují fyzickou přitažlivost žen a/nebo poskytují srovnání s úzkými standardy.
    sexual harassment: tweety, které popisují činnosti, jako jsou sexuální návrhy, žádosti o sexuální laskavosti a sexuální obtěžování.
    threat of violence: tweety, které zastrašují ženy s cílem umlčet je s úmyslem uplatnit moc nad nimi prostřednictvím výhrůžek fyzickým násilím.
    damning: tweety, které obsahují modlitby, aby ublížily ženám; většina modliteb zahrnuje přání smrti/nemoci a také žádost k Bohu, aby proklel ženy.

    Odpověď na tweet:
    {}
    Nyní tweet končí.
    Která JEDNA kategorie nejlépe kategorizuje misogynní odpověď na tweet?
    """,
    "Polish": """
    Proszę zaklasyfikować odpowiedź na tweet z linii czasu popularnych dziennikarek/reporterek do JEDNEJ z siedmiu mizoginicznych kategorii:
    "discredit", "derailing", "dominance", "stereotyping & objectification", "sexual harassment", "threat of violence", "damning"
    discredit: odnosi się do tweetów, które obrażają kobiety bez innego większego zamiaru.
    derailing: tweety, które wskazują na usprawiedliwienie nadużyć wobec kobiet, jednocześnie odrzucając odpowiedzialność mężczyzn, próbując przerwać rozmowę, aby ponownie skupić na czymś uwagę.
    dominance: tweety, które wyrażają wyższość mężczyzn lub zachowują kontrolę mężczyzn nad kobietami.
    stereotyping & objectification: tweety, które promują powszechny, ale utrwalony i nadmiernie uproszczony obraz/ideę kobiet. Ta etykieta odnosi się również do przypadków tweetów, które opisują atrakcyjność fizyczną kobiet i/lub porównują je do wąskich standardów.
    sexual harassment: tweety, które opisują działania takie jak seksualne zaloty, prośby o przysługi seksualne i molestowanie seksualne.
    threat of violence: tweety, które zastraszają kobiety w celu ich uciszenia, aby potwierdzić nad nimi władzę poprzez groźby fizycznej przemocy.
    damning: tweety, które zawierają modlitwy o zranienie kobiet; większość modlitw to życzenia śmierci/choroby, a także prośby do Boga, aby przeklął kobiety.

    Odpowiedź na tweet:
    {}
    Teraz tweet się kończy.
    Która JEDNA kategoria najlepiej klasyfikuje mizoginistyczną odpowiedź na tweet?
    """,
    "Arabic": """
    يرجى تصنيف الرد على التغريدة من جداول زمنية للصحفيات/المراسلات المشهورات في فئة واحدة من سبع فئات كارهة للنساء:
    "discredit", "derailing", "dominance", "stereotyping & objectification", "sexual harassment", "threat of violence", "damning"
    discredit: يشير إلى التغريدات التي تجمع بين الإساءة إلى النساء دون أي نية أكبر.
    derailing: تغريدات تشير إلى تبرير إساءة معاملة النساء مع رفض مسؤولية الرجال ومحاولة تعطيل المحادثة بهدف إعادة توجيهها.
    dominance: التغريدات التي تعبر عن تفوق الذكور أو الحفاظ على سيطرة الذكور على النساء.
    stereotyping & objectification: تغريدات تروج لصورة/فكرة واسعة الانتشار ولكنها ثابتة ومبسطة للغاية عن النساء. يشير هذا التصنيف أيضًا إلى التغريدات التي تصف جاذبية النساء الجسدية و/أو تقدم مقارنات مع معايير ضيقة.
    sexual harassment: التغريدات التي تصف أفعالًا مثل التقدمات الجنسية، الطلبات للحصول على خدمات جنسية، والتحرش ذو الطبيعة الجنسية.
    threat of violence: تغريدات تخيف النساء لإسكاتهن بنية تأكيد القوة عليهن من خلال تهديدات بالعنف الجسدي.
    damning: تغريدات تحتوي على صلوات لإيذاء النساء؛ معظم الصلوات تتعلق بتمني الموت/المرض بالإضافة إلى الدعاء بأن يلعن الله النساء.

    رد التغريدة:
    {}
    الآن ينتهي التغريدة.
    ما هي فئة واحدة التي تصنف رد التغريدة الكارهة للنساء بأفضل شكل؟
    """,
    "Persian": """
    لطفاً پاسخ توییت را از جدول زمانی خبرنگاران/گزارشگران زن محبوب در یکی از هفت دسته زن ستیزانه طبقه‌بندی کنید:
    "discredit", "derailing", "dominance", "stereotyping & objectification", "sexual harassment", "threat of violence", "damning"
    discredit: به توییت‌هایی اشاره دارد که با ترکیبی از توهین به زنان بدون هیچ نیت بزرگتری همراه است.
    derailing: توییت‌هایی که دلالت بر توجیه سوء استفاده از زنان دارند در حالی که مسئولیت مردان را رد می‌کنند و سعی می‌کنند مکالمه را برای تغییر مسیر آن قطع کنند.
    dominance: توییت‌هایی که برتری مردان را بیان می‌کنند یا کنترل مردان بر زنان را حفظ می‌کنند.
    stereotyping & objectification: توییت‌هایی که تصویری ثابت و ساده از زنان ترویج می‌کنند. این برچسب همچنین به مواردی از توییت‌ها اشاره دارد که به جذابیت فیزیکی زنان اشاره می‌کند و/یا مقایسه‌هایی با استانداردهای محدود ارائه می‌دهد.
    sexual harassment: توییت‌هایی که اقداماتی مانند پیشرفت‌های جنسی، درخواست‌های امتیازات جنسی و آزار و اذیت جنسی را توصیف می‌کنند.
    threat of violence: توییت‌هایی که زنان را برای خاموش کردنشان با هدف اثبات قدرت بر آنان با تهدید به خشونت فیزیکی تهدید می‌کنند.
    damning: توییت‌هایی که حاوی دعاهایی برای آسیب رساندن به زنان هستند؛ بیشتر دعاها شامل آرزوی مرگ/بیماری و همچنین دعا به خدا برای نفرین کردن زنان است.

    پاسخ توییت:
    {}
    اکنون توییت پایان می‌یابد.
    کدام یک دسته بهتر است پاسخ توییت زن ستیزانه را دسته‌بندی کند؟
    """,
    "Hebrew": """
    אנא סווגו את התגובה לציוץ מציוץ בטוויטר מהציר הזמן של עיתונאיות/כתבות פופולריות לאחת משבע קטגוריות מיזוגיניות:
    "discredit", "derailing", "dominance", "stereotyping & objectification", "sexual harassment", "threat of violence", "damning"
    discredit: מתייחס לציוצים שמשמיצים נשים ללא כוונה גדולה נוספת.
    derailing: ציוצים שמצביעים על הצדקה של התעללות בנשים תוך דחיית האחריות הגברית עם ניסיון להפריע לשיחה על מנת למקד אותה מחדש.
    dominance: ציוצים שמבטאים עליונות גברית או שומרים על שליטה גברית בנשים.
    stereotyping & objectification: ציוצים שמקדמים דימוי/רעיון נפוץ אך קבוע ומפושט מדי של נשים. התווית הזו מתייחסת גם למקרים של ציוצים שמתארים את המשיכה הפיזית של נשים ו/או מספקים השוואות לסטנדרטים מוגבלים.
    sexual harassment: ציוצים שמתארים פעולות כגון התקדמות מינית, בקשות לטובות מיניות, והטרדה מינית.
    threat of violence: ציוצים שמפחידים נשים כדי להשתיקן מתוך כוונה לאשר את הכוח עליהן באמצעות איומים של אלימות פיזית.
    damning: ציוצים שמכילים תפילות לפגוע בנשים; רוב התפילות כוללות משאלות מוות/מחלה לצד תפילה לאלוהים שיקלל נשים.

    תגובת הציוץ:
    {}
    כעת הציוץ מסתיים.
    איזו קטגוריה אחת מסווגת בצורה הטובה ביותר את תגובת הציוץ המיזוגינית?
    """,
    "Turkish": """
    Lütfen popüler kadın gazetecilerin/raportörlerin zaman çizelgelerinden gelen tweet yanıtını yedi kadın düşmanı kategoriden BİRİNE göre sınıflandırın:
    "discredit", "derailing", "dominance", "stereotyping & objectification", "sexual harassment", "threat of violence", "damning"
    discredit: kadınlara yönelik hakaret içeren, başka bir amacı olmayan tweetlere atıfta bulunur.
    derailing: kadın istismarını meşrulaştırırken erkek sorumluluğunu reddeden ve konuşmayı keserek yeniden odaklamaya çalışan tweetlerdir.
    dominance: erkek üstünlüğünü ifade eden veya erkeklerin kadınlar üzerindeki kontrolünü koruyan tweetlerdir.
    stereotyping & objectification: kadınların yaygın olarak kabul gören ancak sabit ve aşırı basitleştirilmiş bir imajını/fikrini teşvik eden tweetlerdir. Bu etiket ayrıca, kadınların fiziksel çekiciliğini tanımlayan ve/veya dar standartlarla karşılaştırmalar sunan tweet durumlarını ifade eder.
    sexual harassment: cinsel taciz, cinsel talepler ve cinsel içerikli taciz gibi davranışları tanımlayan tweetlerdir.
    threat of violence: kadınları tehditlerle susturarak üzerlerinde güç uygulamayı amaçlayan tweetlerdir.
    damning: kadınlara zarar vermek amacıyla dua içeren tweetlerdir; duaların çoğu ölüm/hastalık dilekleri içerir ve ayrıca Tanrı'ya kadınları lanetlemesi için dua eder.

    Tweet yanıtı:
    {}
    Şimdi tweet sona eriyor.
    Kadın düşmanı tweet yanıtını en iyi hangi BİRİ kategorize eder?
    """,
    "Japanese": """
    人気のある女性ジャーナリスト/レポーターのタイムラインからのツイート返信を、以下の7つの女性嫌悪カテゴリーのうちの1つに分類してください：
    "discredit", "derailing", "dominance", "stereotyping & objectification", "sexual harassment", "threat of violence", "damning"
    discredit: 女性を中傷するだけで他に意図がないツイートを指します。
    derailing: 女性への虐待を正当化し、男性の責任を否定し、会話を中断させて再集中させようとするツイートです。
    dominance: 男性の優位性を表現する、または女性に対する男性の支配を維持するツイートです。
    stereotyping & objectification: 女性の広く持たれているが固定的で過度に単純化されたイメージ/アイデアを促進するツイートです。このラベルは、女性の身体的魅力を描写し、狭い基準に基づいた比較を提供するツイートにも適用されます。
    sexual harassment: 性的な進展、性的な好意の要求、および性的な性質の嫌がらせを説明するツイートです。
    threat of violence: 女性を沈黙させるために暴力の脅威を通じて女性に対する力を主張する意図で脅迫するツイートです。
    damning: 女性を傷つけるための祈りを含むツイートです。祈りの大部分は死/病気の願いであり、さらに神に女性を呪うように祈ります。

    ツイート返信：
    {}
    これでツイートは終了です。
    女性嫌悪のツイート返信を最もよく分類するカテゴリはどれですか？
    """,
    "Korean": """
    인기 있는 여성 기자/리포터의 타임라인에서 트윗 답변을 다음의 일곱 가지 여성혐오적 범주 중 하나로 분류하십시오:
    "discredit", "derailing", "dominance", "stereotyping & objectification", "sexual harassment", "threat of violence", "damning"
    discredit: 여성에 대한 모욕을 포함한 다른 큰 의도가 없는 트윗을 가리킵니다.
    derailing: 여성 학대를 정당화하는 반면 남성의 책임을 거부하며 대화를 방해하여 다시 집중시키려는 시도를 나타내는 트윗입니다.
    dominance: 남성 우월성을 표현하거나 여성을 통제하려는 트윗입니다.
    stereotyping & objectification: 여성의 널리 퍼진 고정된 단순화된 이미지/아이디어를 홍보하는 트윗입니다. 이 라벨은 여성의 신체적 매력을 설명하거나 좁은 기준과의 비교를 제공하는 트윗 사례에도 해당됩니다.
    sexual harassment: 성적 접근, 성적 호의 요청 및 성적 성격의 괴롭힘과 같은 행동을 설명하는 트윗입니다.
    threat of violence: 여성을 위협하여 침묵하게 만들고 폭력의 위협을 통해 여성에 대한 힘을 주장하려는 의도로 여성을 위협하는 트윗입니다.
    damning: 여성을 해치기 위한 기도를 포함하는 트윗입니다; 대부분의 기도는 죽음/질병의 소망을 포함하며 하나님께 여성을 저주하도록 기도합니다.

    트윗 답변:
    {}
    이제 트윗이 끝납니다.
    여성혐오적인 트윗 답변을 가장 잘 분류하는 범주는 무엇입니까?
    """,
    "Vietnamese": """
    Vui lòng phân loại phản hồi tweet từ dòng thời gian của các nhà báo/phóng viên nữ nổi tiếng thành MỘT trong bảy loại kì thị phụ nữ:
    "discredit", "derailing", "dominance", "stereotyping & objectification", "sexual harassment", "threat of violence", "damning"
    discredit: đề cập đến những tweet kết hợp xúc phạm phụ nữ mà không có ý định lớn hơn.
    derailing: tweet thể hiện sự biện minh cho việc lạm dụng phụ nữ trong khi từ chối trách nhiệm của nam giới với một nỗ lực để làm gián đoạn cuộc trò chuyện nhằm thay đổi trọng tâm.
    dominance: tweet thể hiện sự vượt trội của nam giới hoặc duy trì sự kiểm soát của nam giới đối với phụ nữ.
    stereotyping & objectification: tweet thúc đẩy một hình ảnh/ý tưởng phổ biến nhưng cố định và đơn giản hóa quá mức về phụ nữ. Nhãn này cũng đề cập đến các trường hợp tweet mô tả sự hấp dẫn về thể chất của phụ nữ và/hoặc cung cấp các so sánh với các tiêu chuẩn hạn chế.
    sexual harassment: tweet mô tả các hành động như tấn công tình dục, yêu cầu tình dục, và quấy rối có tính chất tình dục.
    threat of violence: tweet đe dọa phụ nữ để buộc họ im lặng với ý định khẳng định quyền lực đối với phụ nữ thông qua các mối đe dọa về bạo lực thể chất.
    damning: tweet chứa những lời cầu nguyện để làm tổn thương phụ nữ; hầu hết các lời cầu nguyện là mong muốn về cái chết/bệnh tật bên cạnh việc cầu nguyện với Chúa để nguyền rủa phụ nữ.

    Phản hồi tweet:
    {}
    Bây giờ tweet kết thúc.
    Loại MỘT nào phân loại tốt nhất phản hồi tweet kì thị phụ nữ?
    """,
    "Thai": """
    โปรดจัดประเภทการตอบกลับทวีตจากไทม์ไลน์ของนักข่าว/ผู้สื่อข่าวหญิงยอดนิยมให้เป็นหนึ่งในเจ็ดหมวดหมู่ที่มีลักษณะเกลียดผู้หญิง:
    "discredit", "derailing", "dominance", "stereotyping & objectification", "sexual harassment", "threat of violence", "damning"
    discredit: หมายถึงทวีตที่แสดงความดูถูกเหยียดหยามต่อผู้หญิงโดยไม่มีเจตนาอื่นๆ ที่ใหญ่กว่า.
    derailing: ทวีตที่แสดงการแสดงเหตุผลในการทำร้ายผู้หญิง ในขณะเดียวกันก็ปฏิเสธความรับผิดชอบของผู้ชายพร้อมกับความพยายามที่จะขัดขวางการสนทนาเพื่อเปลี่ยนจุดสนใจ.
    dominance: ทวีตที่แสดงความเหนือกว่าของผู้ชายหรือรักษาการควบคุมของผู้ชายเหนือผู้หญิง.
    stereotyping & objectification: ทวีตที่ส่งเสริมภาพลักษณ์/แนวคิดที่ถือกันอย่างกว้างขวาง แต่คงที่และเรียบง่ายเกินไปเกี่ยวกับผู้หญิง. ป้ายนี้ยังหมายถึงกรณีทวีตที่อธิบายถึงความน่าสนใจทางร่างกายของผู้หญิงและ/หรือให้การเปรียบเทียบกับมาตรฐานแคบๆ.
    sexual harassment: ทวีตที่อธิบายการกระทำต่างๆ เช่น การล่วงละเมิดทางเพศ, การขอความช่วยเหลือทางเพศ, และการล่วงละเมิดทางเพศ.
    threat of violence: ทวีตที่ข่มขู่ผู้หญิงให้เงียบโดยมีเจตนาเพื่อยืนยันอำนาจเหนือผู้หญิงผ่านการข่มขู่ด้วยความรุนแรงทางกาย.
    damning: ทวีตที่มีคำอธิษฐานเพื่อทำร้ายผู้หญิง; คำอธิษฐานส่วนใหญ่เกี่ยวข้องกับความปรารถนาให้เกิดความตาย/เจ็บป่วยนอกจากการขอให้พระเจ้าแช่งผู้หญิง.

    ตอบทวีต:
    {}
    ตอนนี้ทวีตจบลงแล้ว.
    หมวดหมู่ใดที่ดีที่สุดในการจำแนกการตอบกลับทวีตที่มีลักษณะเกลียดผู้หญิง?
    """,
    "Indonesian": """
    Silakan kategorikan balasan tweet dari timeline jurnalis/reporter wanita populer ke dalam SATU dari tujuh kategori misoginis berikut:
    "discredit", "derailing", "dominance", "stereotyping & objectification", "sexual harassment", "threat of violence", "damning"
    discredit: mengacu pada tweet yang menggabungkan penghinaan terhadap wanita tanpa niat besar lainnya.
    derailing: tweet yang menunjukkan pembenaran atas penyalahgunaan terhadap wanita sambil menolak tanggung jawab laki-laki dengan upaya untuk mengganggu percakapan untuk mengalihkannya.
    dominance: tweet yang menyatakan superioritas laki-laki atau mempertahankan kendali laki-laki atas wanita.
    stereotyping & objectification: tweet yang mempromosikan citra/ide yang luas namun tetap dan terlalu menyederhanakan wanita. Label ini juga mengacu pada contoh tweet yang menggambarkan daya tarik fisik wanita dan/atau memberikan perbandingan dengan standar sempit.
    sexual harassment: tweet yang menggambarkan tindakan seperti rayuan seksual, permintaan layanan seksual, dan pelecehan seksual.
    threat of violence: tweet yang mengintimidasi wanita untuk membungkam mereka dengan maksud untuk menegaskan kekuasaan atas wanita melalui ancaman kekerasan fisik.
    damning: tweet yang berisi doa untuk menyakiti wanita; sebagian besar doa berisi keinginan untuk kematian/penyakit selain berdoa kepada Tuhan untuk mengutuk wanita.

    Balasan tweet:
    {}
    Sekarang tweet berakhir.
    Kategori SATU mana yang paling baik mengkategorikan balasan tweet yang misoginis?
    """,
    "Malay": """
    Sila kategorikan balasan tweet dari garis masa wartawan/penulis laporan wanita popular ke dalam SATU daripada tujuh kategori misoginis berikut:
    "discredit", "derailing", "dominance", "stereotyping & objectification", "sexual harassment", "threat of violence", "damning"
    discredit: merujuk kepada tweet yang menggabungkan penghinaan terhadap wanita tanpa niat yang lebih besar.
    derailing: tweet yang menunjukkan pembenaran penyalahgunaan wanita sambil menolak tanggungjawab lelaki dengan usaha untuk mengganggu perbualan untuk mengalihkannya.
    dominance: tweet yang menyatakan keunggulan lelaki atau mengekalkan kawalan lelaki ke atas wanita.
    stereotyping & objectification: tweet yang mempromosikan imej/idea yang tersebar luas tetapi tetap dan terlalu menyederhanakan mengenai wanita. Label ini juga merujuk kepada contoh tweet yang menggambarkan daya tarikan fizikal wanita dan/atau menyediakan perbandingan dengan standard yang sempit.
    sexual harassment: tweet yang menggambarkan tindakan seperti kemajuan seksual, permintaan untuk perkhidmatan seksual, dan gangguan seksual.
    threat of violence: tweet yang mengugut wanita untuk membungkam mereka dengan niat untuk menegaskan kuasa ke atas wanita melalui ancaman keganasan fizikal.
    damning: tweet yang mengandungi doa untuk menyakiti wanita; kebanyakan doa adalah harapan kematian/penyakit selain berdoa kepada Tuhan untuk mengutuk wanita.

    Balasan tweet:
    {}
    Sekarang tweet berakhir.
    Kategori SATU mana yang paling baik mengkategorikan balasan tweet yang misoginis?
    """,

    "Lao": """
    ກະລຸນາຈັດປະເພດຄຳຕອບກັບທີ່ຕິດເຕັດຈາກນັກຂ່າວ/ຜູ້ລາວທີ່ມີຊື່ສຽງ ຢູ່ໃນໜຶ່ງໃນຫົກປະເພດຕໍ່ໄປນີ້:
    "discredit", "derailing", "dominance", "stereotyping & objectification", "sexual harassment", "threat of violence", "damning"

    discredit: ໝາຍເຖິງການຫຍາບຫຼາຍຕໍ່ຜູ້ຍິງແມ່ນພວກຊາຍໄດ້ຫຍາບແມ່ນບໍ່ມີຈຸດປະສົງທີ່ຈະສະເໜັ່ນໄປເທິງອື່ນ.
    derailing: ການຕອບກັບທີ່ພະຍາຍາມບໍ່ຍຸດຊັງເພື່ອຄວາມຄອງໃຈຂອງຜູ້ຍິງໃນບົດຄວາມນີ້.
    dominance: ສະແດງຄວາມຄິດເຫັນເຫັນເຖິງພຣະສົມຫາສິດທິຂອງຜູ້ຍິງເພື່ອຜູ້ຊາຍ.
    stereotyping & objectification: ໝາຍເຖິງການຕິດເຕັດທີ່ເຫົາເຫັນວ່າຜູ້ຍິງແມ່ນຄືແບບຢ່າງໜຶ່ງເທົ່ານັ້ນ.
    sexual harassment: ການຂູ່ຄື່ນຊາຍເພື່ອການໃຫ້ຜູ້ຍິງຫຍຸ້ມໃຈເຂັ້ນ.
    threat of violence: ຄວາມຄິດຂູ່ທີ່ບໍ່ຈະມີສະຖານະການເປັນໄປໄດ້.
    damning: ຄວາມຄິດເປັນຄວາມເບື່ອຮຽນທີ່ໄດ້ເຫັນເຖິງການປ້າຍປາດ.

    ຕອບກັບທີ່ຕິດເຕັດ:
    {}
    ປະເພດໃດທີ່ຕອບກັບຄວາມຍິງທີ່ບໍ່ມີສະເໜັ່ນທີ່ຈະມີສິດໃດໆເຕືອບເວົ້າເຕັມບໍ່ໄດ້?
    """,

    "Burmese": """
    ကျေးဇူးပြု၍ နာမည်ကြီး သမားမ/သတင်းထောက် များ၏ အချိန်ဇယားမှ Twitter အကြောင်းပြန်ဆိုချက်အား အမျိုးသမီးများအပေါ် မုန်းတီးမှု အမျိုးမျိုးဖြစ်မဖြစ်ကို အောက်ပါကဏ္ဍများအရ သတ်မှတ်ပါ:
    "discredit", "derailing", "dominance", "stereotyping & objectification", "sexual harassment", "threat of violence", "damning"

    discredit: မည်သည့် ရည်ရွယ်ချက်မပါသော အသုံးအနှုန်းများနှင့် ပြောဆိုခြင်း။
    derailing: အမျိုးသမီးများအပေါ် အရှက်ကြီးစွာသော ယုံကြည်မှုများ။
    dominance: အမျိုးသမီးများအပေါ် သို့မဟုတ် ထိန်းချုပ်ထားခြင်း။
    stereotyping & objectification: အသုံးအနှုန်းများတွင် အမျိုးသမီးများကို တစ်ဖက်သားစွာဖြင့် ရှုမြင်ခြင်း။
    sexual harassment: လိင်ဆက်ဆံရေးနှင့် ပတ်သက်သော အထိအရောက်များ။
    threat of violence: အမျိုးသမီးများကို လိင်ဖော်ပြခြင်း။
    damning: အမျိုးသမီးများအပေါ် ပြင်းပြသော ထိခိုက်မှု။

    tweet:
    {}
    အမျိုးသမီးများအပေါ် အပျက်ပျက်သော ကဏ္ဍကို လျောက်ပတ်ပါ။
    """,

    "Cebuano": """
    Palihug i-categorize ang tubag sa tweet gikan sa mga timeline sa mga sikat nga babaye nga mga mamamahayag/reporters ngadto sa USA KA sa pito ka misogynistic nga mga kategorya:
    "discredit", "derailing", "dominance", "stereotyping & objectification", "sexual harassment", "threat of violence", "damning"

    discredit: mga tweet nga nagsama sa pagpasipala sa mga babaye.
    derailing: mga tweet nga naglihok isip pagtamay nga nag-ilis sa pagtutok sa mga lalaki.
    dominance: mga tweet nga nagpasiugda sa pagdumala sa mga lalaki ibabaw sa mga babaye.
    stereotyping & objectification: mga tweet nga nagpapakita og husto nga sumbanan alang sa mga babaye.
    sexual harassment: mga tweet nga nagpakita sa seksual nga pagpang-abuso.
    threat of violence: mga tweet nga nagpasidaan sa mga hulga sa pisikal nga kapintasan.
    damning: mga tweet nga adunay mga buot nga dautan nga makadaot sa mga babaye.

    Tweet:
    {}
    Asa ka nga usa ka kategorya ang mahimong sakto sa misogynistic nga tubag?
    """,

    "Khmer": """
    សូមចាត់ថ្នាក់ចម្លើយចូលក្នុងប្រភេទមួយចំនួននៃការរើសអើងស្ត្រី:
    "discredit", "derailing", "dominance", "stereotyping & objectification", "sexual harassment", "threat of violence", "damning"

    discredit: ការលើកឡើងដែលបង្ហាញពីការរើសអើង។
    derailing: ការបង្ខូចប្រធានបទក្នុងអត្ថបទដើម្បីរំលោភជនបរទេស។
    dominance: ការបង្ហាញពីកិច្ចការរបស់បុរស។
    stereotyping & objectification: ការបង្ហាញពីការប្រព្រឹត្តិដែលចេញពីកិច្ចការរបស់ស្រ្តី។
    sexual harassment: ការលើកឡើងពីឧក្រិដ្ឋកម្មស្របព្រះសាសនារៀងតែខុស។
    threat of violence: ការបង្ហាញពីគ្រោះថ្នាក់អំពីការគំរាមខ្លាច។
    damning: ការលើកឡើងដែលពោរពេញដោយឧក្រិដ្ឋ។

    ចម្លើយ:
    {}
    តើប្រភេទណាខ្ញុំអាចធ្វើបានតាមសំណើរបស់ខ្ញុំ?
    """,

    "Tagalog": """
    Mangyaring i-categorize ang tugon sa tweet mula sa timeline ng sikat na mamamahayag/reporters batay sa pito ka misogynistic kategorya:
    "discredit", "derailing", "dominance", "stereotyping & objectification", "sexual harassment", "threat of violence", "damning"

    discredit: tumutukoy sa pagsasama ng pananakit na walang intensyong mas malaki.
    derailing: mga tweet na nagpapakita ng pagpapaubaya sa mga babae sa usapang kalalakihan.
    dominance: pagpapahayag ng kalamangan ng mga lalaki.
    stereotyping & objectification: mga tweet na nagpapakita ng karaniwang sumbanan para sa mga babae.
    sexual harassment: pagpapahayag ng pang-aabuso sa sekswalidad.
    threat of violence: banta ng pisikal na karahasan laban sa kababaihan.
    damning: mga tweet na may intensyong masama laban sa mga babae.

    Tugon sa tweet:
    {}
    Ano sa palagay mo ang pinakamahusay na naglalarawan ng misogynistic tweet na ito?
    """,

    "Hindi": """
    कृपया महिला पत्रकारों/रिपोर्टरों की टाइमलाइन से प्राप्त ट्वीट उत्तर को सात मिज़ोजिनिस्टिक श्रेणियों में से किसी एक में वर्गीकृत करें:
    "discredit", "derailing", "dominance", "stereotyping & objectification", "sexual harassment", "threat of violence", "damning"

    discredit: उन ट्वीट्स का जिक्र जो बिना किसी बड़े मकसद के महिलाओं को बदनाम करते हैं।
    derailing: महिलाओं के खिलाफ दुर्व्यवहार को सही ठहराने वाले ट्वीट।
    dominance: ट्वीट्स जो पुरुषों की श्रेष्ठता या नियंत्रण को दर्शाते हैं।
    stereotyping & objectification: महिलाओं की छवि को अत्यधिक साधारण दृष्टिकोण से पेश करने वाले ट्वीट।
    sexual harassment: यौन उत्पीड़न या यौन सेवाओं की मांग वाले ट्वीट।
    threat of violence: शारीरिक हिंसा की धमकी देने वाले ट्वीट।
    damning: महिलाओं को नुकसान पहुँचाने के इरादे से प्रार्थना करने वाले ट्वीट।

    ट्वीट उत्तर:
    {}
    किस श्रेणी में यह ट्वीट सबसे अच्छा आता है?
    """,

    "Bengali": """
    দয়া করে সাংবাদিকদের টাইমলাইন থেকে টুইটের প্রতিক্রিয়া সাতটি নারীবিদ্বেষী শ্রেণিতে শ্রেণীবদ্ধ করুন:
    "discredit", "derailing", "dominance", "stereotyping & objectification", "sexual harassment", "threat of violence", "damning"

    discredit: মহিলাদের প্রতি অপমানজনক টুইট।
    derailing: মহিলাদের প্রতি নির্যাতনের ন্যায্যতা প্রদানকারী টুইট।
    dominance: পুরুষদের প্রাধান্য প্রদর্শনকারী টুইট।
    stereotyping & objectification: মহিলাদের সম্পর্কে সাধারণ ধারনাসহ টুইট।
    sexual harassment: যৌন নির্যাতনের বিবরণবর্ণনা।
    threat of violence: শারীরিক সহিংসতার হুমকি দেওয়া টুইট।
    damning: মহিলাদের ক্ষতি করার উদ্দেশ্যে টুইট।

    টুইটের প্রতিক্রিয়া:
    {}
    এই টুইটের জন্য সবচেয়ে উপযুক্ত কোন শ্রেণিটি হবে?
    """,

    "Urdu": """
    براہ کرم مشہور خواتین صحافیوں کی ٹائم لائن سے حاصل شدہ ٹویٹ کے جواب کو سات خواتین مخالف زمروں میں سے کسی ایک میں درجہ بندی کریں:
    "discredit", "derailing", "dominance", "stereotyping & objectification", "sexual harassment", "threat of violence", "damning"

    discredit: خواتین کی تذلیل کرنے والے ٹویٹس۔
    derailing: وہ ٹویٹس جو خواتین کے خلاف زیادتی کو جائز قرار دیتے ہیں۔
    dominance: مردوں کی برتری یا کنٹرول کی عکاسی کرنے والے ٹویٹس۔
    stereotyping & objectification: خواتین کو تنگ نقطہ نظر سے دیکھنے والے ٹویٹس۔
    sexual harassment: جنسی زیادتی کا اظہار کرنے والے ٹویٹس۔
    threat of violence: خواتین کو جسمانی تشدد کی دھمکی دینے والے ٹویٹس۔
    damning: خواتین کے نقصان کے لئے دعا کرنے والے ٹویٹس۔

    ٹویٹ کا جواب:
    {}
    اس ٹویٹ کے جواب میں کون سی ایک کیٹیگری بہترین ہے؟
    """
}


# Create a DataFrame to store instructions
instructions_df2 = pd.DataFrame(columns=['instruction', 'output'])

# Filter out rows where the category is 'none'
filtered_df = df[df['category'] != 'none']
filtered_df = balance_classes(filtered_df, 50, "category")



# Generate instructions using randomly selected templates
instructions_df2['instruction'] = filtered_df["text"].apply(
    lambda x: random.choice(list(instruction_templates.values())).format(x)
)

# Map the output to the category
instructions_df2['output'] = filtered_df['category']

# Display the first few rows of the DataFrame
instructions_df2.to_parquet("../../data/LetMi/LetMi-part2.parquet", index = False)