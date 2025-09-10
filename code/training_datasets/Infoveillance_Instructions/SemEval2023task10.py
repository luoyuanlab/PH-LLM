# -*- coding: utf-8 -*-
# SemEval-2023 Task 10
# - Paper: https://aclanthology.org/2023.semeval-1.305/
# - Data: https://github.com/rewire-online/edos/tree/main/data

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import pandas as pd
from utils.data_utils import balance_classes
import random

file_path = "../../data/SemEval2023Task10/edos_labelled_aggregated.csv"
data = pd.read_csv(file_path)

# Sub A
# # Function to create instruction in the target language
def get_instruction(row, templates):
    # Randomly select one of the multilingual templates
    instruction_template = random.choice(templates)
    instruction = instruction_template.format(text=row["text"])
    return instruction

# Function to create output based on the label
def get_output(row):
    return "no" if row["label_sexist"] == "not sexist" else "yes"

# Define multilingual templates for the instruction
multilingual_templates = [
    # Chinese
    """请判断以下提供的推文的主要标签是“性别歧视”还是“非性别歧视”。如果是性别歧视，请回答“yes”。如果不是，请回答“no”。
在本研究中，性别歧视被定义为基于性别，或基于性别和一个或多个其他身份属性（例如黑人女性，穆斯林女性，跨性别女性）的组合，对女性的任何虐待或负面情感。
条目本身被标记，而不是发言者。引用应按字面意思理解。即使是轻松的笑话，也应仔细评估是否含有性别歧视内容。
任务强调标记内容，而不是假设发言者的意图。
推文: "{text}"。现在推文结束。
请用'yes'或'no'作答。""",

    # English
    """Please determine if the provided tweet below its primary label is Sexist or Not Sexist. If Sexist, respond "yes". If not, respond "no".
In this study, sexism is defined as any abuse or negative sentiment that is directed towards women based on their gender, or based on their gender combined with one or more other identity attributes (e.g. Black women, Muslim women, Trans women).
The entry itself is labeled, not the speaker. Quotes should be taken at face value. Jokes should be carefully evaluated for sexist content, even if lighthearted.
The task emphasizes labeling the content rather than assuming the speaker's intent.
Tweet: "{text}". Now the tweet ends.
Please respond with 'yes' or 'no'.""",

    # German
    """Bitte bestimmen Sie, ob der unten angegebene Tweet als sexistisch oder nicht sexistisch gekennzeichnet ist. Wenn sexistisch, antworten Sie mit 'yes'. Wenn nicht, antworten Sie mit 'no'.
In dieser Studie wird Sexismus definiert als jeder Missbrauch oder negative Stimmung, die gegen Frauen aufgrund ihres Geschlechts gerichtet ist, oder basierend auf ihrem Geschlecht in Kombination mit einem oder mehreren anderen Identitätsmerkmalen (z.B. Schwarze Frauen, Muslimische Frauen, Transfrauen).
Das Eintrag selbst wird gekennzeichnet, nicht der Sprecher. Zitate sollten wörtlich genommen werden. Witze sollten sorgfältig auf sexistischen Inhalt überprüft werden, auch wenn sie unbeschwert sind.
Die Aufgabe betont die Kennzeichnung des Inhalts und nicht die Annahme der Absicht des Sprechers.
Tweet: "{text}". Jetzt endet der Tweet.
Bitte antworten Sie mit 'yes' oder 'no'.""",

    # French
    """Veuillez déterminer si l'étiquette principale du tweet fourni ci-dessous est sexiste ou non sexiste. Si sexiste, répondez 'yes'. Sinon, répondez 'no'.
Dans cette étude, le sexisme est défini comme tout abus ou sentiment négatif dirigé contre les femmes en raison de leur sexe, ou en raison de leur sexe combiné à un ou plusieurs autres attributs d'identité (par exemple, femmes noires, femmes musulmanes, femmes transgenres).
L'entrée elle-même est étiquetée, pas le locuteur. Les citations doivent être prises au pied de la lettre. Les blagues doivent être soigneusement évaluées pour leur contenu sexiste, même si elles sont légères.
La tâche met l'accent sur l'étiquetage du contenu plutôt que sur la présomption de l'intention du locuteur.
Tweet : "{text}". Maintenant, le tweet se termine.
Veuillez répondre par 'yes' ou 'no'.""",

    # Spanish
    """Por favor, determine si la etiqueta principal del tweet proporcionado a continuación es sexista o no sexista. Si es sexista, responda 'yes'. Si no, responda 'no'.
En este estudio, el sexismo se define como cualquier abuso o sentimiento negativo dirigido hacia las mujeres por su género, o basado en su género combinado con uno o más atributos de identidad adicionales (por ejemplo, mujeres negras, mujeres musulmanas, mujeres trans).
La entrada en sí está etiquetada, no el hablante. Las citas deben tomarse al pie de la letra. Las bromas deben evaluarse cuidadosamente por su contenido sexista, incluso si son ligeras.
La tarea enfatiza etiquetar el contenido en lugar de asumir la intención del hablante.
Tweet: "{text}". Ahora el tweet termina.
Por favor, responda con 'yes' o 'no'.""",

    # Portuguese
    """Por favor, determine se o rótulo principal do tweet fornecido abaixo é sexista ou não sexista. Se sexista, responda 'yes'. Se não, responda 'no'.
Neste estudo, o sexismo é definido como qualquer abuso ou sentimento negativo dirigido às mulheres com base em seu gênero, ou com base em seu gênero combinado com um ou mais outros atributos de identidade (por exemplo, mulheres negras, mulheres muçulmanas, mulheres trans).
A entrada em si é rotulada, não o orador. As citações devem ser interpretadas literalmente. As piadas devem ser cuidadosamente avaliadas quanto ao conteúdo sexista, mesmo que sejam leves.
A tarefa enfatiza rotular o conteúdo em vez de assumir a intenção do orador.
Tweet: "{text}". Agora o tweet termina.
Por favor, responda com 'yes' ou 'no'.""",

    # Italian
    """Si prega di determinare se l'etichetta principale del tweet fornito di seguito è sessista o non sessista. Se sessista, rispondere 'yes'. Se non lo è, rispondere 'no'.
In questo studio, il sessismo è definito come qualsiasi abuso o sentimento negativo rivolto alle donne in base al loro genere, o basato sul loro genere combinato con uno o più altri attributi di identità (ad esempio donne nere, donne musulmane, donne trans).
L'ingresso stesso è etichettato, non l'oratore. Le citazioni devono essere prese alla lettera. Gli scherzi devono essere attentamente valutati per il contenuto sessista, anche se leggeri.
Il compito sottolinea l'etichettatura del contenuto piuttosto che presumere l'intenzione dell'oratore.
Tweet: "{text}". Ora il tweet finisce.
Si prega di rispondere con 'yes' o 'no'.""",

    # Dutch
    """Bepaal of het primaire label van de onderstaande tweet seksistisch of niet seksistisch is. Als het seksistisch is, antwoord dan met 'yes'. Als dat niet zo is, antwoord dan met 'no'.
In dit onderzoek wordt seksisme gedefinieerd als elke vorm van misbruik of negatieve gevoelens die gericht zijn op vrouwen op basis van hun geslacht, of op basis van hun geslacht in combinatie met een of meer andere identiteitskenmerken (bijv. Zwarte vrouwen, Moslimvrouwen, Transvrouwen).
De vermelding zelf wordt gelabeld, niet de spreker. Citaten moeten letterlijk worden genomen. Grappen moeten zorgvuldig worden geëvalueerd op seksistische inhoud, zelfs als ze luchtig zijn.
De taak legt de nadruk op het labelen van de inhoud in plaats van het aannemen van de intentie van de spreker.
Tweet: "{text}". Nu eindigt de tweet.
Antwoord alstublieft met 'yes' of 'no'.""",

    # Russian
    """Пожалуйста, определите, является ли основной меткой приведенного ниже твита сексистская или несексистская. Если сексистская, ответьте 'yes'. Если нет, ответьте 'no'.
В этом исследовании сексизм определяется как любое оскорбление или негативное отношение, направленное на женщин на основе их пола, либо на основе их пола в сочетании с одним или несколькими другими атрибутами идентичности (например, чернокожие женщины, мусульманки, трансгендерные женщины).
Помечается само содержание, а не оратор. Цитаты следует воспринимать буквально. Шутки следует тщательно оценивать на предмет сексистского содержания, даже если они легкомысленны.
Задача подчеркивает маркировку содержания, а не предположение о намерениях оратора.
Твит: "{text}". Теперь твит заканчивается.
Пожалуйста, ответьте 'yes' или 'no'.""",

    # Czech
    """Prosím, určete, zda je hlavní štítek níže uvedeného tweetu sexistický nebo nesexistický. Pokud sexistický, odpovězte 'yes'. Pokud ne, odpovězte 'no'.
V této studii je sexismus definován jako jakékoli zneužívání nebo negativní postoj namířený proti ženám na základě jejich pohlaví nebo na základě jejich pohlaví v kombinaci s jedním nebo více dalšími atributy identity (např. černošky, muslimky, trans ženy).
Samotný vstup je označen, ne mluvčí. Citace by měly být brány doslova. Vtipy by měly být pečlivě hodnoceny na sexistický obsah, i když jsou lehkomyslné.
Úkol zdůrazňuje označování obsahu namísto předpokladu záměru mluvčího.
Tweet: "{text}". Nyní tweet končí.
Odpovězte prosím 'yes' nebo 'no'.""",

    # Polish
    """Proszę określić, czy główna etykieta poniższego tweeta jest seksistowska, czy nieseksistowska. Jeśli seksistowska, odpowiedz 'yes'. Jeśli nie, odpowiedz 'no'.
W tym badaniu seksizm definiowany jest jako jakiekolwiek nadużycie lub negatywne nastawienie skierowane wobec kobiet ze względu na ich płeć, lub na podstawie ich płci w połączeniu z jednym lub kilkoma innymi atrybutami tożsamości (np. czarne kobiety, muzułmanki, kobiety transpłciowe).
Etykieta dotyczy samej treści, a nie mówiącego. Cytaty powinny być traktowane dosłownie. Żarty należy dokładnie ocenić pod kątem treści seksistowskich, nawet jeśli są lekkie.
Zadanie podkreśla etykietowanie treści, a nie zakładanie intencji mówcy.
Tweet: "{text}". Teraz tweet się kończy.
Proszę odpowiedzieć 'yes' lub 'no'.""",

    # Arabic
    """يرجى تحديد ما إذا كانت التسمية الأساسية للتغريدة المقدمة أدناه هي تمييز على أساس الجنس أو ليست تمييزًا على أساس الجنس. إذا كان تمييزًا، فأجب بـ 'yes'. إذا لم يكن كذلك، فأجب بـ 'no'.
في هذه الدراسة، يُعرف التمييز على أساس الجنس بأنه أي إساءة أو شعور سلبي موجه ضد النساء بناءً على جنسهن، أو بناءً على جنسهن مع صفة هوية أخرى أو أكثر (مثل النساء السود، النساء المسلمات، النساء المتحولات جنسيًا).
يتم تصنيف المحتوى نفسه، وليس المتحدث. يجب أخذ الاقتباسات على حقيقتها. يجب تقييم النكات بعناية من حيث المحتوى التمييزي، حتى لو كانت غير ضارة.
تؤكد المهمة على تصنيف المحتوى بدلاً من افتراض نية المتحدث.
تغريدة: "{text}". الآن انتهت التغريدة.
يرجى الرد بـ 'yes' أو 'no'.""",

    # Persian
    """لطفاً تعیین کنید که آیا برچسب اصلی توییت ارائه شده در زیر جنسیت‌گرایانه است یا خیر. اگر جنسیت‌گرایانه است، پاسخ 'yes' دهید. اگر نه، پاسخ 'no' دهید.
در این مطالعه، جنسیت‌گرایی به عنوان هر گونه سوء استفاده یا احساس منفی تعریف می‌شود که به دلیل جنسیت زن یا ترکیب جنسیت زن با یک یا چند ویژگی هویتی دیگر (مانند زنان سیاه، زنان مسلمان، زنان تراجنسیتی) به سمت زنان هدایت می‌شود.
برچسب‌گذاری به خود محتوا اختصاص داده می‌شود، نه گوینده. نقل قول‌ها باید به معنای واقعی گرفته شوند. شوخی‌ها باید با دقت از نظر محتوای جنسیت‌گرایانه ارزیابی شوند، حتی اگر سبک و بدون قصد بد باشند.
این وظیفه برچسب‌گذاری محتوا را به جای فرض نیت گوینده تأکید می‌کند.
توییت: "{text}". اکنون توییت به پایان می‌رسد.
لطفاً با 'yes' یا 'no' پاسخ دهید.""",

    # Hebrew
    """אנא קבע אם התווית העיקרית של הציוץ המסופק למטה היא סקסיסטית או לא סקסיסטית. אם סקסיסטית, השב 'yes'. אם לא, השב 'no'.
במחקר זה, סקסיזם מוגדר כהתעללות או רגש שלילי שמכוונים כלפי נשים על בסיס מינן, או על בסיס מינן בשילוב עם מאפיין זהות נוסף או יותר (למשל נשים שחורות, נשים מוסלמיות, נשים טרנסיות).
הכיתוב הוא של התוכן עצמו, ולא של הדובר. יש לקחת את הציטוטים בערכם המילולי. יש להעריך בדיחות בקפידה לגבי תוכן סקסיסטי, גם אם הן קלות דעת.
המשימה מדגישה את תיוג התוכן ולא את הנחת הכוונה של הדובר.
ציוץ: "{text}". הציוץ נגמר כעת.
אנא השב 'yes' או 'no'.""",

    # Turkish
    """Lütfen aşağıda sağlanan tweet'in ana etiketinin cinsiyetçi mi yoksa cinsiyetçi olmadığını belirleyin. Eğer cinsiyetçi ise, 'yes' yanıtını verin. Eğer değilse, 'no' yanıtını verin.
Bu çalışmada cinsiyetçilik, kadınlara cinsiyetlerine dayalı veya cinsiyetleriyle bir veya daha fazla başka kimlik özelliğinin (örneğin Siyah kadınlar, Müslüman kadınlar, Trans kadınlar) birleşimine dayalı olarak yöneltilen herhangi bir kötüye kullanım veya olumsuz duygu olarak tanımlanır.
Etiketleme konuşmacıya değil, içeriğe yapılır. Alıntılar kelime anlamıyla alınmalıdır. Şakalar, hafif de olsa, cinsiyetçi içerik açısından dikkatlice değerlendirilmelidir.
Görev, konuşmacının niyetini varsaymak yerine içeriğin etiketlenmesine vurgu yapar.
Tweet: "{text}". Tweet şimdi sona erdi.
Lütfen 'yes' veya 'no' yanıtını verin.""",

    # Japanese
    """以下の提供されたツイートの主要なラベルが性差別的かどうかを判断してください。性差別的であれば「yes」と答えてください。そうでなければ「no」と答えてください。
この研究では、性差別は、性別に基づいて、または性別と1つ以上の他のアイデンティティ属性（例：黒人女性、ムスリム女性、トランス女性）を組み合わせた形で女性に向けられる虐待や否定的な感情と定義されます。
エントリ自体がラベル付けされており、発言者ではありません。引用は文字通りに受け取る必要があります。冗談であっても、性差別的な内容が含まれていないか慎重に評価する必要があります。
このタスクでは、話者の意図を推測するのではなく、コンテンツのラベル付けに重点を置いています。
ツイート：「{text}」。これでツイートは終了です。
「yes」または「no」で答えてください。""",

    # Korean
    """아래에 제공된 트윗의 주요 라벨이 성차별적인지 여부를 판단하세요. 성차별적이면 'yes'라고 답변하세요. 그렇지 않으면 'no'라고 답변하세요.
이 연구에서 성차별은 성별에 근거한 여성에 대한 모든 학대 또는 부정적인 감정을 의미하며, 성별과 하나 이상의 다른 정체성 속성(예: 흑인 여성, 무슬림 여성, 트랜스 여성)이 결합된 경우도 포함합니다.
입력된 내용 자체가 라벨링되며, 화자가 아닙니다. 인용은 그대로 받아들여야 합니다. 농담도 가볍게 평가되지 않고 성차별적인 내용이 있는지 신중하게 평가해야 합니다.
이 작업은 화자의 의도를 가정하는 대신 콘텐츠 라벨링에 중점을 둡니다.
트윗: "{text}". 이제 트윗이 끝났습니다.
'yes' 또는 'no'로 답변해 주세요.""",

    # Vietnamese
    """Vui lòng xác định xem nhãn chính của tweet được cung cấp bên dưới là phân biệt giới tính hay không phân biệt giới tính. Nếu là phân biệt giới tính, hãy trả lời 'yes'. Nếu không, hãy trả lời 'no'.
Trong nghiên cứu này, phân biệt giới tính được định nghĩa là bất kỳ lạm dụng hoặc tình cảm tiêu cực nào nhắm vào phụ nữ dựa trên giới tính của họ, hoặc dựa trên giới tính của họ kết hợp với một hoặc nhiều thuộc tính nhận dạng khác (ví dụ: phụ nữ da đen, phụ nữ Hồi giáo, phụ nữ chuyển giới).
Bản thân mục nhập được gắn nhãn, không phải người nói. Trích dẫn nên được hiểu theo nghĩa đen. Những trò đùa nên được đánh giá cẩn thận về nội dung phân biệt giới tính, ngay cả khi nhẹ nhàng.
Nhiệm vụ nhấn mạnh việc gắn nhãn nội dung hơn là giả định ý định của người nói.
Tweet: "{text}". Bây giờ tweet kết thúc.
Vui lòng trả lời bằng 'yes' hoặc 'no'.""",

    # Thai
    """โปรดตรวจสอบว่าป้ายหลักของทวีตที่ให้ไว้ด้านล่างคือการเหยียดเพศหรือไม่ใช่การเหยียดเพศ หากเป็นการเหยียดเพศ ให้ตอบ 'yes' หากไม่ใช่ ให้ตอบ 'no'
ในการศึกษานี้ การเหยียดเพศถูกกำหนดให้เป็นการละเมิดหรือความรู้สึกเชิงลบใด ๆ ที่มุ่งเป้าไปที่ผู้หญิงตามเพศของพวกเขา หรือจากเพศของพวกเธอที่รวมกับคุณลักษณะเอกลักษณ์อื่น ๆ หนึ่งหรือมากกว่า (เช่น ผู้หญิงผิวดำ ผู้หญิงมุสลิม ผู้หญิงข้ามเพศ)
รายการเองมีการติดป้ายกำกับ ไม่ใช่ลำโพง คำพูดควรถูกตีความตามมูลค่าที่กำหนดไว้ เรื่องตลกควรถูกประเมินอย่างระมัดระวังเกี่ยวกับเนื้อหาที่เกี่ยวกับเพศ แม้ว่าจะเป็นเรื่องที่ไม่ถือเอาจริงจังก็ตาม
งานเน้นการติดป้ายกำกับเนื้อหาแทนที่จะสมมติความตั้งใจของผู้พูด
ทวีต: "{text}" ตอนนี้ทวีตสิ้นสุดแล้ว
โปรดตอบ 'yes' หรือ 'no'""",

    # Indonesian
    """Silakan tentukan apakah label utama dari tweet yang disediakan di bawah ini adalah Seksis atau Tidak Seksis. Jika Seksis, jawab 'yes'. Jika tidak, jawab 'no'.
Dalam studi ini, seksisme didefinisikan sebagai segala bentuk penyalahgunaan atau sentimen negatif yang ditujukan kepada wanita berdasarkan jenis kelamin mereka, atau berdasarkan jenis kelamin mereka yang dikombinasikan dengan satu atau lebih atribut identitas lainnya (misalnya wanita kulit hitam, wanita Muslim, wanita Trans).
Entri itu sendiri diberi label, bukan pembicara. Kutipan harus diambil secara harfiah. Lelucon harus dievaluasi dengan hati-hati untuk konten seksis, bahkan jika dianggap ringan.
Tugas ini menekankan pemberian label pada konten daripada mengasumsikan niat pembicara.
Tweet: "{text}". Sekarang tweet berakhir.
Silakan jawab dengan 'yes' atau 'no'.""",

    # Malay
    """Sila tentukan sama ada label utama tweet yang diberikan di bawah adalah Seksis atau Tidak Seksis. Jika Seksis, jawab 'yes'. Jika tidak, jawab 'no'.
Dalam kajian ini, seksisme ditakrifkan sebagai sebarang penyalahgunaan atau sentimen negatif yang ditujukan kepada wanita berdasarkan jantina mereka, atau berdasarkan jantina mereka yang digabungkan dengan satu atau lebih atribut identiti lain (contohnya wanita kulit hitam, wanita Muslim, wanita Trans).
Entri itu sendiri dilabel, bukan penceramah. Petikan harus diambil secara literal. Jenaka harus dinilai dengan teliti untuk kandungan seksis, walaupun ringan.
Tugas ini menekankan pelabelan kandungan daripada mengandaikan niat penceramah.
Tweet: "{text}". Sekarang tweet tamat.
Sila jawab dengan 'yes' atau 'no'.""",

    # Lao
    """ກະລຸນາກຳນົດວ່າປ້າຍຫຼັກຂອງ tweet ທີ່ໃຫ້ມາດ້ານລຸ່ມນີ້ແມ່ນເປັນລົມຫວາຍທາງເພດ ຫຼືບໍ່ແມ່ນລົມຫວາຍທາງເພດ. ຖ້າແມ່ນລົມຫວາຍທາງເພດ, ຕອບ 'yes'. ຖ້າບໍ່ແມ່ນ, ຕອບ 'no'.
ໃນການສຶກສານີ້, ການລົມຫວາຍທາງເພດຖືກກຳນົດເປັນການລະເມີດຫຼືຄວາມຮູ້ສຶກຊື່ງບໍ່ດີທີ່ມີຕໍາຫນິຕໍ່ຫຼືຕໍາຫນິຫວາຍໃສ່ເພດຫວາຍທາງຂອງເພດຫວາຍທາງພວກນີ້ (ຍ້ອງໃຫ້ສຽງເອົາຕໍາຫນິຕໍາຫນິຫວາຍໃສ່ຫວາຍທາງຂອງເພດຫວາຍທາງຫຼືຫວາຍທາງຂອງເພດຫວາຍທາງ).
Tweet: "{text}". ຕອບ 'yes' ຫຼື 'no'.""",

    # Burmese
    """ကျေးဇူးပြု၍ အောက်တွင်ပေးထားသော တူဿ်၏အဓိကတံဆိပ်သည် လိင်ခွဲခြားမှု ဖြစ်သည်မဟုတ်ပါဘူးဆိုသည်ကို သတ်မှတ်ပါ။ ကျေးဇူးပြု၍ 'yes' ဖြင့် ဖြေကြားပါ။ လိင်ခွဲခြားမှုမဟုတ်ပါက 'no' ဖြင့် ဖြေကြားပါ။
ဤလေ့လာမှုတွင် လိင်ခွဲခြားမှုသည် အမျိုးသမီးများအား၎င်းတို့၏ လိင်အရိပ်အမြှင်၊ သို့မဟုတ် လိင်အရိပ်အမြှင်နှင့် အခြား မည်သည့် မိမိခွဲခြားမှုများကို မည်သည့် ရောနှောသော အင်္ဂါရပ်တစ်ခုနှင့် ဖြစ်နိုင်သည်။
Tweet: "{text}". 'yes' သို့မဟုတ် 'no' ဖြင့် ပြန်ကြားပါ။""",

    # Cebuano
    """Palihug pag-determinar kung ang gihatag nga tweet sa ubos ang pangunang label nga Seksyista o Dili Seksyista. Kung Seksyista, tubaga ang 'yes'. Kung dili, tubaga ang 'no'.
Sa kini nga pagtuon, ang seksismo gipatin-aw isip bisan unsang pag-abuso o negatibong sentimyento nga gipunting sa mga babaye base sa ilang sekso, o base sa ilang sekso nga gipares sa usa o daghang mga identidad nga katangian (pananglitan, Mga Itom nga babaye, Mga Muslim nga babaye, Mga Trans nga babaye).
Tweet: "{text}". Karon ang tweet mohuman na.
Palihug motubag og 'yes' o 'no'.""",

    # Khmer
    """សូមកំណត់ថាតើទំនាក់ទំនងសំខាន់នៃការបញ្ចូលនៅខាងក្រោមនេះមានអក្សរសម្បទាដែលជាទស្សនីយភាព ឬអក្សរសម្បទាដែលមិនជាទស្សនីយភាព។ ប្រសិនបើជាទស្សនីយភាព សូមឆ្លើយថា 'yes' ។ ប្រសិនបើមិនជាទស្សនីយភាព សូមឆ្លើយថា 'no' ។
Tweet: "{text}". សូមឆ្លើយថា 'yes' ឬ 'no' ។""",

    # Tagalog
    """Pakitukoy kung ang pangunahing label ng ibinigay na tweet sa ibaba ay Sexist o Hindi Sexist. Kung Sexist, sagot ng 'yes'. Kung hindi, sagot ng 'no'.
Tweet: "{text}". Pakisagot ng 'yes' o 'no'.""",

    # Hindi
    """कृपया निर्धारित करें कि नीचे दिए गए ट्वीट का प्राथमिक लेबल सेक्सिस्ट है या नॉन-सेक्सिस्ट। यदि सेक्सिस्ट है, तो 'yes' के साथ उत्तर दें। यदि नहीं, तो 'no' के साथ उत्तर दें।
ट्वीट: "{text}"। कृपया 'yes' या 'no' के साथ उत्तर दें।""",

    # Bengali
    """দয়া করে নির্ধারণ করুন যে নীচে প্রদত্ত টুইটটির প্রাথমিক লেবেলটি যৌনতাবাদী না অযৌনতাবাদী। যদি যৌনতাবাদী হয়, তাহলে 'yes' দিয়ে উত্তর দিন। যদি না হয়, তাহলে 'no' দিয়ে উত্তর দিন।
টুইট: "{text}"। 'yes' বা 'no' দিয়ে উত্তর দিন।""",

    # Urdu
    """براہ کرم تعین کریں کہ نیچے فراہم کردہ ٹویٹ کا بنیادی لیبل جنس پرستی یا غیر جنس پرستی ہے۔ اگر جنس پرستی ہے، تو 'yes' کے ساتھ جواب دیں۔ اگر نہیں، تو 'no' کے ساتھ جواب دیں۔
ٹویٹ: "{text}"۔ براہ کرم 'yes' یا 'no' کے ساتھ جواب دیں۔"""
]


# Create a blank dataframe
inst_dataA = pd.DataFrame(columns=['instruction', 'output'])

# Iterate over each row in the original DataFrame
for i, row in data.iterrows():
    output = get_output(row)
    instruction = get_instruction(row, multilingual_templates)
    inst_dataA = pd.concat([inst_dataA, pd.DataFrame({'instruction': [instruction], 'output': [output]})], ignore_index=True)

inst_dataA = balance_classes(inst_dataA, 5000, "output")

# Save the dataframe to a parquet file
inst_dataA.to_parquet("../../data/SemEval2023Task10/SemEval23_task10_subA_multilingual.parquet", index=False)

# Sub B
# Define multilingual templates for the instruction
multilingual_templates = [
    # Chinese
    """请确定以下提供的性别歧视推文属于哪个类别。
1. 威胁、计划伤害和煽动
2. 贬低，例如描述性的攻击、激进和情绪化的攻击、非人化的攻击和明显的性客体化
3. 敌意，例如随意使用性别化的侮辱、粗言秽语、侮辱、不可改变的性别差异和性别刻板印象、带有性别歧视的背后恭维、居高临下的解释或不受欢迎的建议
4. 偏见性讨论，例如支持对个别女性的虐待或对女性作为一个群体的系统性歧视

该任务强调标记内容而不是假设发言者的意图。

推文: "{text}"。现在推文结束。
请回复每个类别的索引。""",

    # English
    """Please determine what category the provided sexist tweet below belongs to.
1. Threats, plans to harm and incitement
2. Derogation, like descriptive attacks, aggressive and emotive attacks, and dehumanising attacks & overt sexual objectification
3. Animosity, like casual use of gendered slurs, profanities, insults, immutable gender differences and gender stereotypes, backhanded gendered compliments, condescending explanations or unwelcome advice
4. Prejudicial Discussions, like supporting mistreatment of individual women or systemic discrimination against women as a group

The task emphasizes labeling the content rather than assuming the speaker's intent.

Tweet: "{text}". Now the tweet ends.
Please respond with the index of each category.""",

    # German
    """Bitte bestimmen Sie, zu welcher Kategorie der unten angegebene sexistische Tweet gehört.
1. Drohungen, Pläne zu schaden und Aufstachelung
2. Herabsetzung, wie beschreibende Angriffe, aggressive und emotionale Angriffe, entmenschlichende Angriffe und offene sexuelle Objektifizierung
3. Feindseligkeit, wie beiläufiger Gebrauch von geschlechtsspezifischen Beleidigungen, Profanitäten, Beleidigungen, unveränderlichen Geschlechtsunterschieden und Geschlechterstereotypen, hinterhältigen geschlechtsspezifischen Komplimenten, herablassenden Erklärungen oder unerwünschtem Rat
4. Vorurteilige Diskussionen, wie Unterstützung von Misshandlungen einzelner Frauen oder systemische Diskriminierung von Frauen als Gruppe

Die Aufgabe betont die Kennzeichnung des Inhalts anstatt die Annahme der Absicht des Sprechers.

Tweet: "{text}". Jetzt endet der Tweet.
Bitte antworten Sie mit dem Index jeder Kategorie.""",

    # French
    """Veuillez déterminer à quelle catégorie appartient le tweet sexiste fourni ci-dessous.
1. Menaces, projets de nuire et incitation
2. Dérogation, comme les attaques descriptives, les attaques agressives et émotives, les attaques déshumanisantes et l'objectification sexuelle manifeste
3. Animosité, comme l'utilisation occasionnelle d'insultes sexistes, de vulgarités, d'insultes, de différences de genre immuables et de stéréotypes de genre, de compliments sexistes voilés, d'explications condescendantes ou de conseils non sollicités
4. Discussions préjudiciables, comme le soutien au mauvais traitement des femmes individuelles ou à la discrimination systémique contre les femmes en tant que groupe

La tâche met l'accent sur l'étiquetage du contenu plutôt que sur la présomption de l'intention du locuteur.

Tweet : "{text}". Maintenant, le tweet se termine.
Veuillez répondre avec l'index de chaque catégorie.""",

    # Spanish
    """Por favor, determine a qué categoría pertenece el tweet sexista proporcionado a continuación.
1. Amenazas, planes para dañar e incitación
2. Desprecio, como ataques descriptivos, ataques agresivos y emotivos, y ataques deshumanizantes y cosificación sexual abierta
3. Animosidad, como el uso casual de insultos de género, profanidades, insultos, diferencias de género inmutables y estereotipos de género, cumplidos condescendientes de género, explicaciones condescendientes o consejos no deseados
4. Discusiones perjudiciales, como apoyar el maltrato de mujeres individuales o la discriminación sistémica contra las mujeres como grupo

La tarea enfatiza etiquetar el contenido en lugar de asumir la intención del hablante.

Tweet: "{text}". Ahora el tweet termina.
Por favor, responda con el índice de cada categoría.""",

    # Portuguese
    """Por favor, determine a que categoria pertence o tweet sexista fornecido abaixo.
1. Ameaças, planos para prejudicar e incitação
2. Derrogação, como ataques descritivos, ataques agressivos e emotivos, e ataques desumanizadores e objetificação sexual explícita
3. Animosidade, como uso casual de insultos de gênero, profanidades, insultos, diferenças de gênero imutáveis e estereótipos de gênero, elogios de gênero condescendentes, explicações condescendentes ou conselhos indesejados
4. Discussões prejudiciais, como apoiar o maltrato de mulheres individuais ou a discriminação sistêmica contra as mulheres como grupo

A tarefa enfatiza rotular o conteúdo em vez de assumir a intenção do falante.

Tweet: "{text}". Agora o tweet termina.
Por favor, responda com o índice de cada categoria.""",

    # Italian
    """Si prega di determinare a quale categoria appartiene il tweet sessista fornito di seguito.
1. Minacce, piani di danno e incitamento
2. Squalifica, come attacchi descrittivi, attacchi aggressivi ed emotivi, attacchi disumanizzanti e oggettivazione sessuale palese
3. Animosità, come l'uso casuale di insulti di genere, volgarità, insulti, differenze di genere immutabili e stereotipi di genere, complimenti di genere condiscendenti, spiegazioni condiscendenti o consigli indesiderati
4. Discussioni pregiudizievoli, come sostenere il maltrattamento di singole donne o la discriminazione sistemica contro le donne come gruppo

Il compito sottolinea l'etichettatura del contenuto piuttosto che l'assunzione dell'intento del relatore.

Tweet: "{text}". Ora il tweet finisce.
Rispondi con l'indice di ciascuna categoria.""",

    # Dutch
    """Bepaal tot welke categorie de onderstaande seksistische tweet behoort.
1. Bedreigingen, plannen om te schaden en opruiing
2. Neerbuiging, zoals beschrijvende aanvallen, agressieve en emotionele aanvallen, en ontmenselijkende aanvallen en openlijke seksuele objectivering
3. Vijandigheid, zoals het terloops gebruik van geslachtsspecifieke scheldwoorden, vulgariteiten, beledigingen, onveranderlijke genderverschillen en genderstereotypen, neerbuigende gendercomplimenten, neerbuigende uitleg of ongewenst advies
4. Vooringenomen discussies, zoals het ondersteunen van mishandeling van individuele vrouwen of systemische discriminatie van vrouwen als groep

De taak benadrukt het labelen van de inhoud in plaats van de intentie van de spreker aan te nemen.

Tweet: "{text}". Nu eindigt de tweet.
Antwoord alstublieft met de index van elke categorie.""",

    # Russian
    """Пожалуйста, определите, к какой категории относится приведенный ниже сексистский твит.
1. Угрозы, планы причинить вред и подстрекательство
2. Уничижение, такие как описательные атаки, агрессивные и эмоциональные атаки, и обезличивание и открытая сексуальная объективация
3. Враждебность, например, случайное использование гендерных оскорблений, нецензурных выражений, оскорблений, неизменных гендерных различий и гендерных стереотипов, оскорбительные комплименты, уничижительные объяснения или нежелательные советы
4. Предвзятые обсуждения, такие как поддержка жестокого обращения с отдельными женщинами или системная дискриминация женщин как группы

Задача подчеркивает маркировку содержания, а не предположение о намерениях говорящего.

Твит: "{text}". Теперь твит заканчивается.
Пожалуйста, ответьте с индексом каждой категории.""",

    # Czech
    """Prosím, určete, do které kategorie patří níže uvedený sexistický tweet.
1. Hrozby, plány na ublížení a podněcování
2. Hanobení, jako jsou popisné útoky, agresivní a emocionální útoky a odlidšťující útoky a zjevné sexuální objektivizace
3. Animozita, jako je běžné používání genderově specifických nadávek, vulgarit, urážek, nezměnitelných genderových rozdílů a genderových stereotypů, posměšné genderové komplimenty, povýšené vysvětlení nebo nechtěné rady
4. Předsudkové diskuse, jako je podpora špatného zacházení s jednotlivými ženami nebo systematické diskriminace žen jako skupiny

Úkol zdůrazňuje označování obsahu namísto předpokladu záměru mluvčího.

Tweet: "{text}". Nyní tweet končí.
Odpovězte prosím s indexem každé kategorie.""",

    # Polish
    """Proszę określić, do której kategorii należy poniższy seksistowski tweet.
1. Groźby, plany skrzywdzenia i podżeganie
2. Derogacja, takie jak ataki opisowe, agresywne i emocjonalne ataki oraz odczłowieczające ataki i jawne uprzedmiotowienie seksualne
3. Wrogość, jak przypadkowe użycie obelg związanych z płcią, przekleństw, zniewag, niezmiennych różnic płciowych i stereotypów płciowych, złośliwych komplementów płciowych, protekcjonalnych wyjaśnień lub niechcianych porad
4. Dyskusje uprzedzeń, takie jak wspieranie złego traktowania pojedynczych kobiet lub systemowej dyskryminacji kobiet jako grupy

Zadanie kładzie nacisk na etykietowanie treści, a nie na przypuszczenie o intencjach mówcy.

Tweet: "{text}". Teraz tweet się kończy.
Proszę odpowiedzieć z indeksem każdej kategorii.""",

    # Arabic
    """يرجى تحديد الفئة التي ينتمي إليها التغريدة الجنسية المقدمة أدناه.
1. التهديدات، خطط الضرر والتحريض
2. الانتقاص، مثل الهجمات الوصفية، الهجمات العدوانية والعاطفية، الهجمات اللاإنسانية والجنس الصريح
3. العداوة، مثل الاستخدام العارض للإهانات الجندرية، الألفاظ النابية، الإهانات، الفروق الجنسية الثابتة والأنماط الجندرية، المجاملات الجندرية الخلفية، التفسيرات المتعجرفة أو النصائح غير المرحب بها
4. المناقشات المتحيزة، مثل دعم سوء معاملة النساء الفرديات أو التمييز النظامي ضد النساء كمجموعة

تؤكد المهمة على وضع العلامات على المحتوى بدلاً من افتراض نية المتحدث.

التغريدة: "{text}". الآن انتهت التغريدة.
يرجى الرد بفهرس كل فئة.""",

    # Persian
    """لطفاً تعیین کنید که توییت جنسی ارائه شده در زیر به کدام دسته تعلق دارد.
1. تهدیدها، برنامه‌های آسیب و تحریک
2. تحقیر، مانند حملات توصیفی، حملات تهاجمی و عاطفی، حملات غیرانسانی و شیء‌انگاری جنسی آشکار
3. خصومت، مانند استفاده عادی از توهین‌های جنسیتی، فحش‌ها، توهین‌ها، تفاوت‌های جنسی غیرقابل تغییر و کلیشه‌های جنسیتی، تعارفات توهین‌آمیز جنسیتی، توضیحات تحقیرآمیز یا توصیه‌های ناخوشایند
4. بحث‌های تعصبی، مانند حمایت از بدرفتاری با زنان فردی یا تبعیض سیستمی علیه زنان به عنوان یک گروه

این وظیفه برچسب‌گذاری محتوا را به جای فرض نیت گوینده تأکید می‌کند.

توییت: "{text}". اکنون توییت به پایان می‌رسد.
لطفاً با شاخص هر دسته پاسخ دهید.""",

    # Hebrew
    """אנא קבע לאיזו קטגוריה משתייך הציוץ המיני המסופק למטה.
1. איומים, תכניות לפגוע והסתה
2. השפלה, כגון התקפות תיאוריות, התקפות אגרסיביות ורגשיות, התקפות בלתי אנושיות והחפצה מינית גלויה
3. עוינות, כגון שימוש מזדמן בקללות מגדריות, גידופים, עלבונות, הבדלים מגדריים קבועים וסטריאוטיפים מגדריים, מחמאות מגדריות מזלזלות, הסברים מתנשאים או עצות לא רצויות
4. דיונים מוטים, כגון תמיכה בהתעללות בנשים בודדות או אפליה שיטתית נגד נשים כקבוצה

המשימה מדגישה את תיוג התוכן במקום להניח את כוונת הדובר.

ציוץ: "{text}". הציוץ נגמר כעת.
אנא השב עם אינדקס של כל קטגוריה.""",

    # Turkish
    """Lütfen aşağıdaki sağlanan cinsiyetçi tweet'in hangi kategoriye ait olduğunu belirleyin.
1. Tehditler, zarar verme planları ve tahrik
2. Aşağılama, tanımlayıcı saldırılar, saldırgan ve duygusal saldırılar ve insanlıktan çıkarma saldırıları ve açık cinsel nesneleştirme
3. Düşmanlık, cinsiyetli küfürlerin, küfürlerin, hakaretlerin, değişmez cinsiyet farklılıklarının ve cinsiyet stereotiplerinin, arka el cinsiyetli iltifatların, küçümseyici açıklamaların veya istenmeyen tavsiyelerin sıradan kullanımı gibi
4. Önyargılı Tartışmalar, bireysel kadınların kötü muamelesine veya kadınlara karşı sistemik ayrımcılığa destek gibi

Görev, konuşmacının niyetini varsaymak yerine içeriği etiketlemeye vurgu yapar.

Tweet: "{text}". Tweet şimdi sona erdi.
Lütfen her kategori için dizinle yanıt verin.""",

    # Japanese
    """以下の提供された性差別的なツイートがどのカテゴリーに属するかを判断してください。
1. 脅迫、害を与える計画と扇動
2. 侮辱、説明的な攻撃、攻撃的で感情的な攻撃、非人間化する攻撃と露骨な性的物化
3. 敵意、性別に基づくスラングのカジュアルな使用、暴言、侮辱、変えられない性別の違いと性別のステレオタイプ、皮肉な性別の褒め言葉、上から目線の説明や歓迎されないアドバイス
4. 偏見に基づく議論、個々の女性への虐待の支持やグループとしての女性に対する体系的な差別の支持など

このタスクは、話者の意図を推測するのではなく、コンテンツのラベル付けに重点を置いています。

ツイート：「{text}」。これでツイートは終了です。
各カテゴリーのインデックスでお答えください。""",

    # Korean
    """아래 제공된 성차별적인 트윗이 어느 범주에 속하는지 결정해 주세요.
1. 위협, 해를 끼칠 계획 및 선동
2. 비난, 설명적 공격, 공격적 및 감정적 공격, 비인간화 공격 및 노골적인 성적 대상화
3. 적대감, 성별 비방, 욕설, 모욕, 불변의 성 차이 및 성 고정관념, 뒤틀린 성별 칭찬, 경멸적인 설명 또는 원하지 않는 조언
4. 편견에 찬 토론, 개별 여성의 학대 또는 여성 그룹에 대한 체계적인 차별을 지지하는 것

이 작업은 화자의 의도를 추정하기보다는 콘텐츠 라벨링에 중점을 둡니다.

트윗: "{text}". 이제 트윗이 끝났습니다.
각 범주의 인덱스로 응답해 주세요.""",

    # Vietnamese
    """Vui lòng xác định xem tweet phân biệt giới tính được cung cấp dưới đây thuộc danh mục nào.
1. Đe dọa, kế hoạch gây hại và kích động
2. Hạ thấp, như các cuộc tấn công mô tả, các cuộc tấn công hung hăng và cảm xúc, và các cuộc tấn công phi nhân tính hóa và sự đối xử tình dục rõ ràng
3. Thù địch, như việc sử dụng thông thường các lời lăng mạ về giới tính, tục tĩu, lời xúc phạm, sự khác biệt giới tính không thể thay đổi và định kiến giới, những lời khen kiểu châm chọc về giới tính, giải thích condescending hoặc lời khuyên không mong muốn
4. Các cuộc thảo luận định kiến, như việc ủng hộ việc đối xử tệ bạc với những phụ nữ cá nhân hoặc sự phân biệt hệ thống đối với phụ nữ như một nhóm

Nhiệm vụ này nhấn mạnh việc gắn nhãn nội dung thay vì giả định ý định của người nói.

Tweet: "{text}". Bây giờ tweet kết thúc.
Vui lòng trả lời bằng chỉ mục của mỗi danh mục.""",

    # Thai
    """โปรดระบุว่าทวีตการแบ่งแยกเพศที่ให้ไว้ด้านล่างนี้จัดอยู่ในประเภทใด
1. ขู่, แผนทำร้ายและการยั่วยุ
2. ดูหมิ่น เช่น การโจมตีแบบบรรยาย การโจมตีเชิงก้าวร้าวและอารมณ์ และการโจมตีลดทอนความเป็นมนุษย์และการแยกเพศอย่างเปิดเผย
3. ความเป็นปรปักษ์ เช่น การใช้คำหยาบเพศ การสบถ การดูหมิ่น ความแตกต่างทางเพศที่เปลี่ยนแปลงไม่ได้และแบบแผนทางเพศ การเยาะเย้ย การอธิบายอย่างเห็นอกเห็นใจ หรือคำแนะนำที่ไม่พึงประสงค์
4. การอภิปรายที่มีอคติ เช่น การสนับสนุนการปฏิบัติที่ไม่ดีต่อผู้หญิงแต่ละคน หรือการเลือกปฏิบัติอย่างเป็นระบบต่อผู้หญิงในฐานะกลุ่ม

งานนี้เน้นการติดฉลากเนื้อหาแทนที่จะสันนิษฐานถึงเจตนาของผู้พูด

ทวีต: "{text}" ตอนนี้ทวีตสิ้นสุดแล้ว
กรุณาตอบด้วยดัชนีของแต่ละประเภท""",

    # Indonesian
    """Silakan tentukan kategori apa yang dimiliki oleh tweet seksis yang disediakan di bawah ini.
1. Ancaman, rencana untuk membahayakan dan hasutan
2. Penghinaan, seperti serangan deskriptif, serangan agresif dan emosional, dan serangan dehumanisasi & objektifikasi seksual terbuka
3. Permusuhan, seperti penggunaan biasa kata-kata kotor gender, sumpah serapah, hinaan, perbedaan gender yang tidak dapat diubah dan stereotip gender, pujian gender tersembunyi, penjelasan yang merendahkan atau saran yang tidak diinginkan
4. Diskusi Prasangka, seperti mendukung perlakuan buruk terhadap wanita individu atau diskriminasi sistemik terhadap wanita sebagai kelompok

Tugas ini menekankan pelabelan konten daripada mengasumsikan maksud pembicara.

Tweet: "{text}". Sekarang tweet berakhir.
Silakan jawab dengan indeks setiap kategori.""",

    # Malay
    """Sila tentukan kategori tweet seksis yang diberikan di bawah.
1. Ancaman, rancangan untuk mencederakan dan hasutan
2. Penghinaan, seperti serangan deskriptif, serangan agresif dan emosional, dan serangan yang mengurangkan martabat & objektifikasi seksual yang jelas
3. Permusuhan, seperti penggunaan biasa kata-kata kotor berasaskan gender, profaniti, penghinaan, perbezaan gender yang tidak boleh diubah dan stereotaip gender, pujian belakang gender, penjelasan merendah diri atau nasihat yang tidak diingini
4. Perbincangan Prejudis, seperti menyokong perlakuan buruk terhadap wanita individu atau diskriminasi sistematik terhadap wanita sebagai satu kumpulan

Tugas ini menekankan pelabelan kandungan berbanding dengan andaian niat penceramah.

Tweet: "{text}". Sekarang tweet tamat.
Sila jawab dengan indeks setiap kategori.""",

    # Lao
    """ກະລຸນາກຳນົດວ່າທວີດເພດຊາຍທີ່ໄດ້ຮັບດ້ານລຸ່ມນີ້ຢູ່ໃນຫມວດໃດ
1. ການຂົ່ມຂູ່, ແຜນທີ່ຈະທໍາລາຍແລະການຍຸດົນ
2. ການປົດອອກ, ເຊັ່ນການໂຈມຕີທີ່ໄດ້ຮັບການລາຍງານ, ເຊັ່ນການໂຈມຕີທີ່ໄດ້ຮັບການຍັງຢັ້ງໃຈເພື່ອທໍາລາຍຄວາມຄິດຫວັງ, ການໂຈມຕີເຊັ່ນການເຮັດເລື່ອງເຄົາະຮືບມາເພື່ອເປີດເຜີຍວ່າບໍ່ເອົາໃຈຕໍານິບໍ່ດີທາງເພດຊາຍ.
3. ຄວາມຄິດເຫັນເຊັ່ນເຊັ່ນການນໍາໃຊ້ຄຳດ່າເພດຊາຍທີ່ພົວພັນດ້ວຍວິທີທີ່ຄິດເຫັນ, ຄໍາຫມັ້ນເຫົາວິທີທີ່ບໍ່ສຸພາບ, ຄໍາໄດ້ຮັບການຫມັ້ນເຫົາວິທີທີ່ຄິດເຫັນ, ແຕກຕ່າງຂອງເພດຊາຍບໍ່ແຕກຕ່າງກັບວິທີທີ່ຄິດເຫັນ, ຄໍາຫມັ້ນເຫົາການຄິດເຫັນເພື່ອບົດບົງເລັກຫນັກຄວາມສຳເລັດອະທິບາຍພື່ນຖານ.
4. ການຂົ່ມຂູ່ຂົ່ມຄາມສຳລັບຜູ້ຍິງທີ່ບຸກຄົນຫນຶ່ງຫລືການຂົ່ມຂູ່ການເລືອກປະໂຫຍດພິເສດໄດ້ປັບໃຫ້ການເລືອກປະໂຫຍດພິເສດເປັນກຸ່ມຂອງຜູ້ຍິງທີ່ບຸກຄົນຫນຶ່ງຂົ່ມຄາມ.
ງານນີ້ມີຄວາມສຳຄັນຕໍ່ການຕິດຕາມຄວາມສຳເລັດຂອງວຽກການຂົ່ມຂູ່ແລະການເລືອກປະໂຫຍດພິເສດຄອບຄົວ.
Tweet: "{text}". ຂໍ້ຄວາມເສົາລົມນີ້ແມ່ນບໍ່ແມ່ນພຽງແຕ່ບົດບົງເລັກຫນັກ.
ກະລຸນາຕອບດ້ວຍເລກອີນເດັກຂອງແຕ່ລະຫມວດ.""",

    # Burmese
    """အောက်တွင်ပေးထားသော လိင်ခွဲခြားမှုရှိသော တူဿ်ကို ဘယ်အမျိုးအစားမှဖြစ်မည်ဟု သတ်မှတ်ပါ။
1. အန္တရာယ်များ၊ ဆိုးရွားစေခြင်းနှင့် အလှည့်ကျနိုင်ခြင်း
2. ဖျက်ဆီးခြင်း၊ ဖော်ပြမှုအပေါ်မှန်းကန်တိုက်ခိုက်မှုများ၊ ထိုးနှက်မှုနှင့် စိတ်အားထက်သန်မှုများ၊ လူသားမဟုတ်မှုများနှင့် ဖောက်ပြန်ပျက်စီးမှုများ
3. စိတ်ချော်မှု၊ ကျောတန်ဆာရောင်မှုများ၊ ထိုင်းကောက်လက်မှတ်များ၊ ပြိုင်ဆိုင်မှုများနှင့် မလိုလားအပ်သော အကြံဥာဏ်များ
4. မျိုးတုဆိုးဝါးပြောဆိုမှုများ၊ တစ်ခုခုမှ လူမဆီလေခြင်းဖြင့် နောက်ကပ်ခြင်းကို ထောက်ခံခြင်း၊ နှင့် မိမိထက်ကျော်လွန်ရန် အကြံပြုချက်များအပေါ်မှ ခြိမ်းခြောက်ခြင်းများ

ဤတာဝန်သည် ဆောင့ျခမ်းမားသည့်ဝါကျများကို တိကျစွာဖော်ပြရန်အခြားသူ၏ကြွေးမြီးထားခြင်းကို မူကြောင်းခြင်းထက် ဂရုစိုက်စစ်ဆေးခြင်းကို အလေးပေးသည်။

တူဿ်: "{text}"။ ယခုတွင် တူဿ် ပြီးဆုံးပါပြီ။
တစ်ခုခု၏အမျိုးအစားနှုန်းညွှန်ချက်ကို ရေးသားပါ။""",

    # Cebuano
    """Palihug pag-determinar kung unsang kategoriya ang gi-provide nga seksistang tweet sa ubos nab belong.
1. Mga hulga, plano sa pagdaot ug pag-aghat
2. Pagtamay, sama sa mga deskriptibong atake, agresibo ug emosyonal nga mga atake, ug dehumanisasyon nga mga atake ug hayag nga sekswal nga pag-objekta
3. Pagkapungot, sama sa kaswal nga paggamit sa gendered slurs, profanities, mga insulto, immutable nga gender differences ug gender stereotypes, backhanded gendered compliments, condescending nga mga explanations o dili gustong mga tambag
4. Mga Prejudicial nga Diskusyon, sama sa pag-suporta sa pag-abuso sa mga individual nga mga babaye o sistematikong diskriminasyon batok sa mga babaye isip grupo

Ang buluhaton nag-emphasize sa pag-label sa content imbes nga magtuo sa intensiyon sa nagsulti.

Tweet: "{text}". Karon, human ang tweet.
Palihug tubaga ang index sa matag kategoriya.""",

    # Khmer
    """សូមកំណត់ថាតើនិទ្ទេសខាងក្រោមត្រូវបានបែងចែកទៅជាចំណាត់ថ្នាក់ណាមួយ
1. ការគំរាមកំហែង ការធ្វើផែនការរក្សាតំណាង និងការជំរុញ
2. ការចោលកាដំណាក់ដូចជាការធ្វើការរិះគន់ ដឹកនាំបញ្ចៀសបង្រៀន និងការចោលការជ្រៀតចូល ការរីករាលដាលរបស់គេ និងការប្រដាប់ប្រដារ
3. ការធ្វើប្រតិបត្តិការរយះម៉ឺនដំណាក់ដូចជាការប្រើប្រាស់ពាក្យសម្ព័ន្ធដាក់បញ្ឈន់ ដាក់ពាក្យកែងដួល ដាក់ទាស់ដាក់បញ្ឈន់ ប្រជាប្រសាថាឆ្កិះបន្ទា នៃការរុំរបរ និងប្រជាប្រសាថាឆ្កិះបន្ទា នៃការរុំរបរ ការដាក់បញ្ឈន់ដាក់បញ្ឈន់ដាក់បញ្ឈន់ និងការជ្រាបកណត់រយៈ
4. ការចោលការអកាសដូចជាការគាំទ្រទៅការបំបិទអ្នកចំណាត់ទុក ការរុំរបរ និងការរក្សាបំផុសការប្រកាន់ពីការទស្សន៍ទ្រនឹមរួមបញ្ឈន់

ការងារបង្កប់ការចង់ស្លឹកមិនមែនគ្រប់គ្រងទាំងអស់ចំពោះតម្លៃឬកំណត់បញ្ឈន់នៃការសន្និដ្ឋាន។

សូមពិនិត្យមើល: "{text}"។ ឥឡូវនេះការបញ្ឈន់បានបញ្ចប់ហើយ។
សូមឆ្លើយនឹងរាងកាយនៃចំណាត់ថ្នាក់។""",

    # Tagalog
    """Pakitukoy kung aling kategorya ang ibinigay na seksistang tweet sa ibaba.
1. Mga banta, mga plano na manakit at pang-uudyok
2. Pang-iinsulto, tulad ng mga mapanglait na pag-atake, agresibo at emosyonal na pag-atake, at dehumanizing na pag-atake at hayagang sekswal na pag-objectify
3. Pagkapoot, tulad ng karaniwang paggamit ng gendered slurs, mga kabastusan, mga insulto, hindi mababago na pagkakaiba ng kasarian at mga stereotype ng kasarian, mga backhanded na puri sa kasarian, mga mapang-mataas na paliwanag o hindi nais na payo
4. Mga Prejudicial na Diskusyon, tulad ng pagsuporta sa hindi tamang pagtrato sa mga indibidwal na kababaihan o sistematikong diskriminasyon laban sa mga kababaihan bilang isang grupo

Ang gawain ay naglalayong lagyan ng label ang nilalaman sa halip na ipalagay ang intensiyon ng nagsasalita.

Tweet: "{text}". Ngayon natapos na ang tweet.
Pakisagot ang index ng bawat kategorya.""",

    # Hindi
    """कृपया निर्धारित करें कि नीचे दी गई सेक्सिस्ट ट्वीट किस श्रेणी में आती है।
1. धमकियाँ, नुकसान की योजना और उकसाना
2. अपमान, जैसे वर्णनात्मक हमले, आक्रामक और भावनात्मक हमले, और अमानवीय हमले और स्पष्ट यौन वस्तुकरण
3. शत्रुता, जैसे कि लिंग आधारित गालियों, गालियों, अपशब्दों, अपरिवर्तनीय लिंग अंतर और लिंग रूढ़ियों का आकस्मिक उपयोग, गुप्त लिंग आधारित तारीफें, अपमानजनक स्पष्टीकरण या अवांछित सलाह
4. पक्षपाती चर्चाएँ, जैसे व्यक्तिगत महिलाओं के साथ दुर्व्यवहार का समर्थन करना या महिलाओं के समूह के रूप में प्रणालीगत भेदभाव का समर्थन करना

यह कार्य वक्ता की मंशा मानने के बजाय सामग्री को लेबल करने पर जोर देता है।

ट्वीट: "{text}"। अब ट्वीट समाप्त होता है।
कृपया प्रत्येक श्रेणी का सूचकांक उत्तर दें।""",

    # Bengali
    """দয়া করে নির্ধারণ করুন যে নীচে দেওয়া লিঙ্গ বৈষম্যমূলক টুইটটি কোন বিভাগের অন্তর্গত।
1. হুমকি, ক্ষতি করার পরিকল্পনা এবং উস্কানি
2. অবমাননা, যেমন বর্ণনামূলক আক্রমণ, আগ্রাসী এবং আবেগপ্রবণ আক্রমণ, এবং অমানবিক আক্রমণ এবং স্পষ্ট যৌন বস্তুবাচকরণ
3. বিদ্বেষ, যেমন লিঙ্গভিত্তিক গালিগালাজ, অশ্লীলতা, অপমান, অপরিবর্তনীয় লিঙ্গের পার্থক্য এবং লিঙ্গের প্রচলিত ধারণাগুলি, লুকানো লিঙ্গভিত্তিক প্রশংসা, অবমাননাকর ব্যাখ্যা বা অপ্রত্যাশিত পরামর্শ
4. পক্ষপাতমূলক আলোচনা, যেমন ব্যক্তিগত মহিলাদের প্রতি সহিংস আচরণের সমর্থন করা বা মহিলাদের একটি গোষ্ঠী হিসাবে প্রাতিষ্ঠানিক বৈষম্যের সমর্থন করা

এই কাজটি বক্তার উদ্দেশ্য অনুমান করার চেয়ে বিষয়বস্তু লেবেলিংয়ের উপর জোর দেয়।

টুইট: "{text}"। এখন টুইট শেষ হয়।
অনুগ্রহ করে প্রতিটি বিভাগের সূচক উত্তর দিন।""",

    # Urdu
    """براہ کرم یہ طے کریں کہ نیچے دیے گئے جنسیت زدہ ٹویٹ کس زمرے میں آتی ہے۔
1. دھمکیاں، نقصان پہنچانے کے منصوبے اور اشتعال انگیزی
2. توہین، جیسے وضاحتی حملے، جارحانہ اور جذباتی حملے، اور انسانیت سے محروم کرنے والے حملے اور واضح جنسی اشیاء
3. دشمنی، جیسے کہ صنفی گالیوں، گالیوں، توہین، ناقابل تغیر صنفی فرق اور صنفی دقیانوسیاں، پیچھے کے صنفی تعریفی الفاظ، تحقیر آمیز وضاحتیں یا ناپسندیدہ مشورے کا معمولی استعمال
4. تعصباتی مباحثے، جیسے فردی خواتین کے ساتھ بدسلوکی کی حمایت یا گروپ کے طور پر خواتین کے خلاف نظامی امتیاز کی حمایت کرنا

یہ کام مواد کو لیبل کرنے پر زور دیتا ہے نہ کہ بولنے والے کی نیت فرض کرنے پر۔

ٹویٹ: "{text}"۔ اب ٹویٹ ختم ہو گیا ہے۔
براہ کرم ہر زمرے کا انڈیکس جواب دیں۔"""
]



# Create a blank dataframe
inst_dataB = pd.DataFrame(columns=['instruction', 'output'])

# Iterate over each row in the original DataFrame
for i, row in data.iterrows():
    if row["label_sexist"] == "sexist":
        output = row["label_category"][0]
        instruction = get_instruction(row, multilingual_templates)
        inst_dataB = pd.concat([inst_dataB, pd.DataFrame({'instruction': [instruction], 'output': [output]})], ignore_index=True)

inst_dataB = balance_classes(inst_dataB, 500, "output")

# Save the dataframe to a parquet file
inst_dataB.to_parquet("../../data/SemEval2023Task10/SemEval23_task10_subB_multilingual.parquet", index=False)