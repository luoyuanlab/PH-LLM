# -*- coding: utf-8 -*-
# RP (Rheinische Post Comments) 
# - Paper: https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/file/c9e1074f5b3f9fc8ea15d152add07294-Paper-round2.pdf
# - Data: https://zenodo.org/records/5291339#.Ybr_9VkxkUE

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import pandas as pd
import random
from utils.data_utils import balance_classes

# Load your data
df = pd.read_csv('../../data/RP/RP-Mod-Crowd.csv')


# Define the instruction templates in different languages
instruction_templates = {
    "Chinese": """
    请想象你是一个审核员（例如，报纸编辑）。根据以下评论是否可以在新闻文章下发布而无需进一步审查（无问题）进行评估。
    问题性评论是可能属于以下类别的辱骂性语言：侮辱，威胁，性别歧视，种族主义，粗俗语言。
    侮辱涉及贬低，无礼或蔑视的言论（未作进一步说明）。
    威胁涉及宣布侵犯受害者身体完整性的内容。
    性别歧视涉及基于性别（身份）对人的攻击，通常针对女性。
    种族主义涉及基于人的起源、种族、国籍的攻击（通常旨在煽动仇恨）。
    粗俗语言涉及使用露骨的性语言和不适当的语言。

    这是评论：{}

    如果上述评论存在问题（辱骂性），请回答 'yes'。否则，请回答 'no'。
    """,

    "English": """
    Please imagine you are taking the role of a moderator (e.g. a newspaper editor). Evaluate the below comment based on whether they could be published under a news article without further review (unproblematic).
    Problematic comments are abusive language that may fall into some of these categories: Insult, Threat, Sexism, Racism, Profane Language.
    Insult involves denigrating, insolvent or contemptuous statements (left without further specification).
    Threat involves announcements of the violation of the physical integrity of the victim.
    Sexism involves attacks on people based on their gender (identity), often with a focus on women.
    Racism involves attacks on people based on their origin, ethnicity, nation (typically meant to incite hatred).
    Profane language involves usage of sexually explicit and inappropriate language.

    Here is the comment: {}

    Respond 'yes' if the above comment is problematic (abusive). Otherwise, respond 'no'.
    """,

    "German": """
    Stellen Sie sich bitte vor, Sie übernehmen die Rolle eines Moderators (z.B. eines Zeitungsredakteurs). Bewerten Sie den unten stehenden Kommentar, ob er ohne weitere Überprüfung unter einem Nachrichtenartikel veröffentlicht werden könnte (unproblematisch).
    Problematische Kommentare sind missbräuchliche Sprache, die in einige dieser Kategorien fallen können: Beleidigung, Bedrohung, Sexismus, Rassismus, vulgäre Sprache.
    Beleidigung beinhaltet herabwürdigende, respektlose oder verächtliche Aussagen (ohne weitere Spezifikation).
    Bedrohung beinhaltet Ankündigungen der Verletzung der körperlichen Integrität des Opfers.
    Sexismus beinhaltet Angriffe auf Menschen aufgrund ihres Geschlechts (Identität), oft mit einem Fokus auf Frauen.
    Rassismus beinhaltet Angriffe auf Menschen aufgrund ihrer Herkunft, Ethnie, Nation (typischerweise, um Hass zu schüren).
    Vulgäre Sprache beinhaltet die Verwendung sexuell expliziter und unangemessener Sprache.

    Hier ist der Kommentar: {}

    Antworten Sie mit 'yes', wenn der obige Kommentar problematisch (missbräuchlich) ist. Andernfalls antworten Sie mit 'no'.
    """,

    "French": """
    Veuillez imaginer que vous jouez le rôle d'un modérateur (par exemple, un rédacteur en chef). Évaluez le commentaire ci-dessous en fonction de sa capacité à être publié sous un article de presse sans examen complémentaire (non problématique).
    Les commentaires problématiques sont des propos abusifs qui peuvent appartenir à certaines de ces catégories : Insulte, Menace, Sexisme, Racisme, Langage vulgaire.
    L'insulte implique des déclarations dénigrantes, insolentes ou méprisantes (sans spécification supplémentaire).
    La menace implique des annonces de violation de l'intégrité physique de la victime.
    Le sexisme implique des attaques contre des personnes en fonction de leur sexe (identité), souvent axées sur les femmes.
    Le racisme implique des attaques contre des personnes en fonction de leur origine, de leur ethnie, de leur nation (souvent destinées à inciter à la haine).
    Le langage vulgaire implique l'utilisation d'un langage sexuellement explicite et inapproprié.

    Voici le commentaire : {}

    Répondez 'yes' si le commentaire ci-dessus est problématique (abusif). Sinon, répondez 'no'.
    """,

    "Spanish": """
    Por favor, imagina que estás tomando el rol de un moderador (por ejemplo, un editor de un periódico). Evalúa el comentario a continuación en función de si se podría publicar bajo un artículo de noticias sin una revisión adicional (no problemático).
    Los comentarios problemáticos son lenguaje abusivo que puede caer en algunas de estas categorías: Insulto, Amenaza, Sexismo, Racismo, Lenguaje profano.
    Insulto implica declaraciones denigrantes, insolentes o despectivas (sin mayor especificación).
    Amenaza implica anuncios de violación de la integridad física de la víctima.
    Sexismo implica ataques a personas basados en su género (identidad), a menudo con un enfoque en las mujeres.
    Racismo implica ataques a personas basados en su origen, etnia, nación (típicamente destinados a incitar al odio).
    Lenguaje profano implica el uso de lenguaje sexualmente explícito e inapropiado.

    Aquí está el comentario: {}

    Responde 'yes' si el comentario anterior es problemático (abusivo). De lo contrario, responde 'no'.
    """,

    "Portuguese": """
    Por favor, imagine que você está assumindo o papel de um moderador (por exemplo, um editor de jornal). Avalie o comentário abaixo com base em saber se ele poderia ser publicado em um artigo de notícias sem revisão adicional (não problemático).
    Comentários problemáticos são linguagem abusiva que pode cair em algumas dessas categorias: Insulto, Ameaça, Sexismo, Racismo, Linguagem Profana.
    Insulto envolve declarações depreciativas, insolentes ou desdenhosas (deixadas sem mais especificações).
    Ameaça envolve anúncios de violação da integridade física da vítima.
    Sexismo envolve ataques a pessoas com base em seu gênero (identidade), geralmente com foco nas mulheres.
    Racismo envolve ataques a pessoas com base em sua origem, etnia, nação (tipicamente destinados a incitar ódio).
    Linguagem profana envolve o uso de linguagem sexualmente explícita e inapropriada.

    Aqui está o comentário: {}

    Responda 'yes' se o comentário acima for problemático (abusivo). Caso contrário, responda 'no'.
    """,

    "Italian": """
    Immagina di assumere il ruolo di moderatore (ad esempio, un redattore di giornale). Valuta il commento qui sotto in base alla possibilità di pubblicarlo sotto un articolo di notizie senza ulteriori revisioni (non problematico).
    I commenti problematici sono linguaggi offensivi che possono rientrare in alcune di queste categorie: Insulto, Minaccia, Sessismo, Razzismo, Linguaggio volgare.
    Insulto implica dichiarazioni denigratorie, insolenti o sprezzanti (lasciate senza ulteriori specificazioni).
    Minaccia implica annunci di violazione dell'integrità fisica della vittima.
    Sessismo implica attacchi alle persone basati sul loro genere (identità), spesso con un focus sulle donne.
    Razzismo implica attacchi alle persone basati sulla loro origine, etnia, nazione (solitamente destinati a incitare all'odio).
    Il linguaggio volgare implica l'uso di un linguaggio sessualmente esplicito e inappropriato.

    Ecco il commento: {}

    Rispondi 'yes' se il commento sopra è problematico (abusivo). Altrimenti, rispondi 'no'.
    """,

    "Dutch": """
    Stel je voor dat je de rol van een moderator op je neemt (bijvoorbeeld een krantenredacteur). Beoordeel de onderstaande opmerking op basis van de vraag of deze zonder verdere beoordeling onder een nieuwsartikel kan worden gepubliceerd (onproblematisch).
    Problematische opmerkingen zijn beledigend taalgebruik dat in een van deze categorieën kan vallen: Belediging, Bedreiging, Seksisme, Racisme, Vulgair Taalgebruik.
    Belediging omvat denigrerende, onbeschofte of minachtende uitspraken (zonder verdere specificatie).
    Bedreiging omvat aankondigingen van de schending van de fysieke integriteit van het slachtoffer.
    Seksisme omvat aanvallen op mensen op basis van hun geslacht (identiteit), vaak met een focus op vrouwen.
    Racisme omvat aanvallen op mensen op basis van hun afkomst, etniciteit, natie (meestal bedoeld om haat aan te wakkeren).
    Vulgair taalgebruik omvat het gebruik van seksueel expliciete en ongepaste taal.

    Hier is de opmerking: {}

    Antwoord 'yes' als de bovenstaande opmerking problematisch (beledigend) is. Anders antwoord 'no'.
    """,
    "Russian": """
    Представьте, что вы берете на себя роль модератора (например, редактора газеты). Оцените приведенный ниже комментарий, основываясь на том, может ли он быть опубликован под новостной статьей без дальнейшего рассмотрения (непроблематично).
    Проблемные комментарии — это оскорбительный язык, который может подпадать под некоторые из этих категорий: Оскорбление, Угроза, Сексизм, Расизм, Непристойная лексика.
    Оскорбление включает уничижительные, наглые или презрительные высказывания (оставленные без дальнейших уточнений).
    Угроза включает объявления о нарушении физической неприкосновенности жертвы.
    Сексизм включает нападки на людей по признаку пола (идентичности), часто с акцентом на женщин.
    Расизм включает нападки на людей по признаку их происхождения, этнической принадлежности, нации (обычно с целью разжигания ненависти).
    Непристойная лексика включает использование сексуально откровенной и неподобающей лексики.

    Вот комментарий: {}

    Ответьте 'yes', если вышеуказанный комментарий является проблематичным (оскорбительным). В противном случае ответьте 'no'.
    """,

    "Czech": """
    Představte si, že hrajete roli moderátora (např. redaktora novin). Zhodnoťte níže uvedený komentář na základě toho, zda by mohl být publikován pod článkem bez dalšího přezkoumání (bezproblémový).
    Problematické komentáře jsou urážlivé výrazy, které mohou spadat do některé z těchto kategorií: Urážka, Hrozba, Sexismus, Rasismus, Vulgarismus.
    Urážka zahrnuje hanlivé, neslušné nebo pohrdavé výroky (nejsou dále specifikovány).
    Hrozba zahrnuje oznámení o porušení fyzické integrity oběti.
    Sexismus zahrnuje útoky na lidi na základě jejich pohlaví (identity), často se zaměřením na ženy.
    Rasismus zahrnuje útoky na lidi na základě jejich původu, etnika, národa (obvykle zamýšlené k vyvolání nenávisti).
    Vulgarismus zahrnuje používání sexuálně explicitních a nevhodných výrazů.

    Zde je komentář: {}

    Odpovězte 'yes', pokud je výše uvedený komentář problematický (urážlivý). Jinak odpovězte 'no'.
    """,

    "Polish": """
    Wyobraź sobie, że pełnisz rolę moderatora (np. redaktora gazety). Oceń poniższy komentarz, czy może zostać opublikowany pod artykułem bez dalszej weryfikacji (bezproblemowy).
    Problematyczne komentarze to obraźliwy język, który może należeć do jednej z tych kategorii: Obelga, Groźba, Seksizm, Rasizm, Wulgarny Język.
    Obelga obejmuje uwłaczające, bezczelne lub pogardliwe wypowiedzi (pozostawione bez dalszych wyjaśnień).
    Groźba obejmuje zapowiedzi naruszenia integralności fizycznej ofiary.
    Seksizm obejmuje ataki na osoby ze względu na ich płeć (tożsamość), często z naciskiem na kobiety.
    Rasizm obejmuje ataki na osoby ze względu na ich pochodzenie, etniczność, narodowość (zazwyczaj mające na celu wzbudzenie nienawiści).
    Wulgarny język obejmuje użycie wyrażeń o charakterze seksualnym i nieodpowiednim języku.

    Oto komentarz: {}

    Odpowiedz 'yes', jeśli powyższy komentarz jest problematyczny (obraźliwy). W przeciwnym razie odpowiedz 'no'.
    """,

    "Arabic": """
    تخيل أنك تتولى دور المشرف (على سبيل المثال، محرر الصحيفة). قم بتقييم التعليق أدناه بناءً على ما إذا كان يمكن نشره تحت مقال إخباري دون مراجعة إضافية (غير إشكالي).
    التعليقات الإشكالية هي لغة مسيئة قد تندرج في بعض هذه الفئات: الإهانة، التهديد، التمييز الجنسي، العنصرية، اللغة البذيئة.
    تتضمن الإهانة تصريحات متهكمة، عدوانية أو مزدرية (دون مزيد من التحديد).
    التهديد يتضمن إعلانات عن انتهاك السلامة الجسدية للضحية.
    يتضمن التمييز الجنسي هجمات على الأشخاص بناءً على جنسهم (الهوية)، غالبًا مع التركيز على النساء.
    يتضمن العنصرية هجمات على الأشخاص بناءً على أصلهم، عرقهم، أمتهم (عادةً بهدف إثارة الكراهية).
    تتضمن اللغة البذيئة استخدام لغة جنسية صريحة وغير مناسبة.

    ها هو التعليق: {}

    أجب بـ 'yes' إذا كان التعليق أعلاه إشكاليًا (مسيئًا). خلاف ذلك، أجب بـ 'no'.
    """,

    "Persian": """
    لطفاً تصور کنید که نقش یک ناظر (مثلاً یک ویراستار روزنامه) را دارید. نظر زیر را بر اساس اینکه آیا می‌تواند بدون بررسی بیشتر در زیر یک مقاله خبری منتشر شود (بدون مشکل) ارزیابی کنید.
    نظرات مشکل‌ساز زبان توهین‌آمیزی هستند که ممکن است در برخی از این دسته‌ها قرار گیرند: توهین، تهدید، تبعیض جنسیتی، نژادپرستی، زبان زشت.
    توهین شامل اظهارات تحقیرآمیز، بی‌ادبانه یا تحقیرآمیز است (بدون مشخصات بیشتر).
    تهدید شامل اعلاماتی درباره نقض تمامیت جسمی قربانی است.
    تبعیض جنسیتی شامل حملاتی علیه افراد بر اساس جنسیت (هویت) آنهاست، اغلب با تمرکز بر زنان.
    نژادپرستی شامل حملاتی علیه افراد بر اساس منشاء، قومیت، ملیت آنهاست (معمولاً به منظور تحریک به نفرت).
    زبان زشت شامل استفاده از زبان جنسی صریح و نامناسب است.

    در اینجا نظر: {}

    اگر نظر فوق مشکل‌ساز (توهین‌آمیز) است، پاسخ دهید 'yes'. در غیر این صورت، پاسخ دهید 'no'.
    """,

    "Hebrew": """
    אנא דמיינו שאתם בתפקיד של מתווך (למשל עורך עיתון). העריכו את התגובה שלמטה על פי השאלה האם היא יכולה להתפרסם מתחת למאמר חדשותי ללא בדיקה נוספת (לא בעייתית).
    תגובות בעייתיות הן שפה פוגענית שעלולה להשתייך לכמה מהקטגוריות הללו: עלבון, איום, סקסיזם, גזענות, שפה מגונה.
    עלבון כולל הצהרות מבזות, גסות או מבזות (ללא פרטים נוספים).
    איום כולל הכרזות על פגיעה בשלמות הפיזית של הקורבן.
    סקסיזם כולל התקפות על אנשים על בסיס מינם (זהותם), לעיתים קרובות עם דגש על נשים.
    גזענות כוללת התקפות על אנשים על בסיס מוצאם, אתניותם, לאומיותם (בדרך כלל מכוונות לעורר שנאה).
    שפה מגונה כוללת שימוש בשפה בוטה ובוטה מינית.

    הנה התגובה: {}

    ענה 'yes' אם התגובה שלמעלה היא בעייתית (פוגענית). אחרת, ענה 'no'.
    """,

    "Turkish": """
    Lütfen bir moderatör rolünü üstlendiğinizi hayal edin (örneğin, bir gazete editörü). Aşağıdaki yorumu, bir haber makalesi altında ek bir inceleme olmadan yayımlanabilecek şekilde değerlendirin (sorunsuz).
    Sorunlu yorumlar, şu kategorilere girebilecek aşağılayıcı dil içeren yorumlardır: Hakaret, Tehdit, Cinsiyetçilik, Irkçılık, Küfürlü Dil.
    Hakaret, aşağılayıcı, küstah veya küçümseyici ifadeleri içerir (daha fazla ayrıntı verilmeden bırakılır).
    Tehdit, mağdurun fiziksel bütünlüğüne yönelik ihlal bildirimlerini içerir.
    Cinsiyetçilik, cinsiyetlerine (kimliklerine) dayalı insanlara yönelik saldırıları içerir, genellikle kadınlara odaklanır.
    Irkçılık, insanların kökenlerine, etnik kökenlerine, uluslarına dayalı saldırıları içerir (genellikle nefreti körüklemek amacıyla).
    Küfürlü dil, cinsel açıdan açık ve uygunsuz dil kullanımını içerir.

    İşte yorum: {}

    Yukarıdaki yorum sorunlu (aşağılayıcı) ise 'yes' yanıtını verin. Aksi takdirde, 'no' yanıtını verin.
    """,

    "Japanese": """
    モデレーター（例：新聞編集者）の役割を担っていると想像してください。以下のコメントが、さらなるレビューなしでニュース記事の下に公開できるかどうか（問題ないか）を評価してください。
    問題のあるコメントは、次のカテゴリーに該当する可能性のある虐待的な言葉です：侮辱、脅威、性差別、人種差別、卑猥な言葉。
    侮辱には、軽蔑的、無礼または軽蔑的な発言が含まれます（詳細な説明はありません）。
    脅威には、被害者の身体の完全性を侵害するという発表が含まれます。
    性差別には、主に女性に焦点を当てた、性別（アイデンティティ）に基づく人々への攻撃が含まれます。
    人種差別には、人々の出身、民族、国籍に基づく攻撃が含まれます（通常は憎悪を煽ることを目的としています）。
    卑猥な言葉には、性的に露骨で不適切な言葉の使用が含まれます。

    こちらがコメントです：{}

    上記のコメントが問題（虐待的）である場合は 'yes' と回答してください。そうでなければ、'no' と回答してください。
    """,

    "Korean": """
    당신이 중재자(예: 신문 편집자) 역할을 하고 있다고 상상해 보십시오. 아래 댓글이 추가 검토 없이 뉴스 기사 하단에 게시될 수 있는지(문제 없는지) 평가하십시오.
    문제 있는 댓글은 다음 범주에 해당할 수 있는 공격적인 언어입니다: 모욕, 위협, 성차별, 인종차별, 저속한 언어.
    모욕에는 경멸적이고 무례하거나 경멸적인 진술이 포함됩니다(더 이상 구체적으로 설명하지 않음).
    위협에는 피해자의 신체적 무결성을 침해하는 공표가 포함됩니다.
    성차별에는 성별(정체성)에 기반한 사람들에 대한 공격이 포함되며, 종종 여성에게 초점을 맞춥니다.
    인종차별에는 출신, 민족, 국가를 기반으로 한 사람들에 대한 공격이 포함됩니다(일반적으로 증오를 선동하는 것을 목표로 함).
    저속한 언어에는 성적으로 노골적이고 부적절한 언어 사용이 포함됩니다.

    여기에 댓글이 있습니다: {}

    위의 댓글이 문제(모욕적)인 경우 'yes'라고 응답하십시오. 그렇지 않으면 'no'라고 응답하십시오.
    """,

    "Vietnamese": """
    Hãy tưởng tượng bạn đang đóng vai trò của một người điều hành (ví dụ: biên tập viên báo chí). Đánh giá bình luận dưới đây dựa trên việc liệu chúng có thể được đăng dưới một bài báo mà không cần xem xét thêm hay không (không có vấn đề).
    Những bình luận có vấn đề là ngôn ngữ lăng mạ có thể rơi vào một số danh mục sau: Lăng mạ, Đe dọa, Phân biệt giới tính, Phân biệt chủng tộc, Ngôn ngữ thô tục.
    Lăng mạ liên quan đến các phát biểu làm nhục, không tôn trọng hoặc khinh miệt (để lại mà không có thêm mô tả cụ thể).
    Đe dọa liên quan đến các tuyên bố về việc vi phạm sự toàn vẹn cơ thể của nạn nhân.
    Phân biệt giới tính liên quan đến các cuộc tấn công vào người khác dựa trên giới tính (bản sắc) của họ, thường tập trung vào phụ nữ.
    Phân biệt chủng tộc liên quan đến các cuộc tấn công vào người khác dựa trên nguồn gốc, sắc tộc, quốc gia của họ (thường nhằm kích động thù hận).
    Ngôn ngữ thô tục liên quan đến việc sử dụng ngôn ngữ khiêu dâm và không phù hợp.

    Đây là bình luận: {}

    Hãy trả lời 'yes' nếu bình luận trên có vấn đề (lăng mạ). Nếu không, hãy trả lời 'no'.
    """,

    "Thai": """
    โปรดจินตนาการว่าคุณกำลังรับบทบาทเป็นผู้ดูแล (เช่น บรรณาธิการหนังสือพิมพ์) ประเมินความคิดเห็นด้านล่างนี้โดยพิจารณาจากว่าพวกเขาสามารถเผยแพร่ภายใต้บทความข่าวโดยไม่ต้องตรวจสอบเพิ่มเติม (ไม่มีปัญหา)
    ความคิดเห็นที่เป็นปัญหาคือภาษาที่ดูถูกซึ่งอาจอยู่ในบางประเภทเหล่านี้: การดูถูก การข่มขู่ การเหยียดเพศ การเหยียดเชื้อชาติ ภาษาหยาบคาย
    การดูถูกเกี่ยวข้องกับการกล่าวหา การดูหมิ่น หรือการดูถูก (โดยไม่ระบุเพิ่มเติม)
    การข่มขู่เกี่ยวข้องกับการประกาศการละเมิดความสมบูรณ์ทางกายภาพของเหยื่อ
    การเหยียดเพศเกี่ยวข้องกับการโจมตีผู้คนตามเพศ (อัตลักษณ์) โดยมักเน้นที่ผู้หญิง
    การเหยียดเชื้อชาติเกี่ยวข้องกับการโจมตีผู้คนตามเชื้อสาย ชาติพันธุ์ ประเทศ (มักจะมีเจตนาในการยุยงให้เกิดความเกลียดชัง)
    ภาษาหยาบคายเกี่ยวข้องกับการใช้ภาษาที่มีความหมายทางเพศและไม่เหมาะสม

    นี่คือความคิดเห็น: {}

    ตอบ 'yes' หากความคิดเห็นข้างต้นมีปัญหา (ดูถูก) มิฉะนั้น ให้ตอบ 'no'
    """,

    "Indonesian": """
    Silakan bayangkan Anda sedang mengambil peran sebagai moderator (misalnya editor surat kabar). Evaluasi komentar di bawah ini berdasarkan apakah mereka dapat dipublikasikan di bawah artikel berita tanpa tinjauan lebih lanjut (tidak bermasalah).
    Komentar yang bermasalah adalah bahasa kasar yang dapat jatuh ke dalam beberapa kategori berikut: Penghinaan, Ancaman, Seksisme, Rasisme, Bahasa Kasar.
    Penghinaan melibatkan pernyataan yang merendahkan, tidak sopan, atau menghina (dibiarkan tanpa penjelasan lebih lanjut).
    Ancaman melibatkan pengumuman pelanggaran integritas fisik korban.
    Seksisme melibatkan serangan terhadap orang-orang berdasarkan jenis kelamin mereka (identitas), sering kali dengan fokus pada perempuan.
    Rasisme melibatkan serangan terhadap orang-orang berdasarkan asal, etnis, negara mereka (biasanya dimaksudkan untuk menghasut kebencian).
    Bahasa kasar melibatkan penggunaan bahasa yang eksplisit secara seksual dan tidak pantas.

    Berikut adalah komentarnya: {}

    Jawab 'yes' jika komentar di atas bermasalah (kasar). Jika tidak, jawab 'no'.
    """,

    "Malay": """
    Sila bayangkan anda mengambil peranan sebagai moderator (cth. editor akhbar). Nilaikan komen di bawah berdasarkan sama ada mereka boleh diterbitkan di bawah artikel berita tanpa semakan lanjut (tidak bermasalah).
    Komen yang bermasalah adalah bahasa kasar yang mungkin tergolong dalam beberapa kategori berikut: Penghinaan, Ancaman, Seksisme, Perkauman, Bahasa Kasar.
    Penghinaan melibatkan kenyataan yang merendahkan, tidak sopan atau menghina (ditinggalkan tanpa penjelasan lanjut).
    Ancaman melibatkan pengumuman pelanggaran integriti fizikal mangsa.
    Seksisme melibatkan serangan terhadap orang berdasarkan jantina mereka (identiti), selalunya dengan fokus pada wanita.
    Perkauman melibatkan serangan terhadap orang berdasarkan asal-usul, etnik, negara mereka (biasanya bertujuan untuk menghasut kebencian).
    Bahasa kasar melibatkan penggunaan bahasa yang jelas dan tidak sesuai secara seksual.

    Berikut adalah komen: {}

    Jawab 'yes' jika komen di atas bermasalah (kasar). Jika tidak, jawab 'no'.
    """,

    "Lao": """
    ກະລຸນາຈິນຕະນາການວ່າທ່ານກຳລັງຮັບບົດບາດເປັນຜູ້ຄຸ້ມຄອງ (ຕົວຢ່າງເຊັ່ນ ບັນນາທິການເອກະສານ). ປະເມີນຄຳເຫັນດ້ານລຸ່ມນີ້ຂື້ນຢູ່ກັບວ່າພວກເຂົາສາມາດຖືກຕີພິມຢູ່ໃຕ້ບົດບາດຂ່າວໂດຍບໍ່ມີການທົດສອບຕໍ່ໄປ (ບໍ່ມີບັນຫາ).
    ຄຳເຫັນທີ່ມີບັນຫາແມ່ນພາສາທີ່ອາດຈະຕົກໄປໃນບາງຫົວຂໍ້ດັ່ງຕໍ່ໄປນີ້: ການໃຫ້ຄຳເຫັນກ່ຽວກັບການຂ່າດຄວາມຍອດຍ້ອງ, ການຂົ່ມຂູ່, ການຄຸນນະສົມບັດ, ການເຫັນແກ່ຕົນເອງ, ການເຫັນແກ່ຕົນເອງ, ຄຳເຫັນຕ່າງໆກ່ຽວກັບຄວາມຜິດພາບຂອງຄົນອື່ນ, ການເຫັນກ່ຽວກັບຄວາມຜິດພາບຂອງຄົນອື່ນ.
    ຄຳເຫັນທີ່ບໍ່ມີຄວາມຜິດພາບຈະກວດຕົວຢ່າງໄປທົ່ວໄປເພື່ອຊື່ນຊົມການຂ່າດຄວາມຍອດຍ້ອງຂອງຄົນອື່ນ.
    ນີ້ແມ່ນຄຳເຫັນ: {}

    ຕອບ 'yes' ຖ້າຄຳເຫັນຂ້າງເທິງນີ້ມີບັນຫາ (ການລົງລາຍຊື່). ມິດຢ່າງອື່ນ, ຕອບ 'no'.
    """,

    "Burmese": """
    ကျေးဇူးပြု၍ သင့်ကိုယ်ကို အခြားသူတစ်ဦး (ဥပမာ: သတင်းစာအယ်ဒီတာ) အဖြစ် ထင်ယောင်ပါ။ အောက်ပါမှတ်ချက်ကို သတင်းဆောင်းပါးအောက်တွင် ထပ်မံစစ်ဆေးမှုမရှိဘဲ စာမူပေးနိုင်မည်မျှအခြေခံ၍ အကဲဖြတ်ပါ (ပြဿနာမရှိ)။
    ပြဿနာရှိသောမှတ်ချက်များသည် ဤအမျိုးအစားအချို့တွင် ကုသနိုင်သော ဆန့်ကျင်သည့်စကားလုံးများဖြစ်သည်- ပျက်စီးမှု၊ ခြိမ်းခြောက်မှု၊ ကျားမမြားနှင့် ကွဲပြားမှု၊ ဘာသာရေး၊ ဘာသာရေး အလကားဖန့်ဖော်မှု။
    ပျက်စီးမှုတွင် အထူးပြောကြားချက်မရှိသော ဖျက်ဆီးမှု၊ ရှင်ပျောက်ဆုံးမှု သို့မဟုတ် ဂုဏ်ထူးမြင့်မှုများပါဝင်သည်။
    ခြိမ်းခြောက်မှုတွင် သက်သေရှိသည်အတွက် ဖြိုဖျက်မှုအား အလားအလာရှိသည်ကို ကြေညာမှုများပါဝင်သည်။
    ကျားမမြားနှင့် ကွဲပြားမှုတွင် အမျိုးသားများ အပေါ် ဂုဏ်ထူးပြုစွာ ဖြစ်စဉ်များအပေါ် ချိန်မိသော ကျားဘက်စွန်းစွန်းသော ထိခိုက်မှုများပါဝင်သည်။
    ဘာသာရေးတွင် လူထုအား ကြောက်ရွံ့မိသော ဆန့်ကျင်ရေးကို ဖြန့်ဖျက်ခြင်းအတွက် အစီအမံများပါဝင်သည်။
    ဘာသာရေး အလကားဖန့်ဖော်မှုတွင် လူမှုအသိုင်းအဝိုင်းနှင့် သက်ဆိုင်သော ဘာသာရေး ဖန့်ဖော်မှုများပါဝင်သည်။

    မှတ်ချက်မှာ: {}

    အထက်ပါမှတ်ချက်ကို ပြဿနာရှိ (ဆန့်ကျင်ရေး) သို့မဟုတ် 'yes' ဟု ဖြေကြားပါ။ မဟုတ်လျှင် 'no' ဟု ဖြေကြားပါ။
    """,

    "Cebuano": """
    Palihug hunahuna nga ikaw ang moderator (pananglitan usa ka editor sa mantalaan). Tantiya ang mosunod nga komento base sa kung mahimo ba kini ipatik sa ilawom sa usa ka artikulo sa balita nga walay dugang nga pagrepaso (wala’y problema).
    Ang problematic nga mga komento mao ang abusive nga sinultihan nga mahimong mahulog sa pipila niini nga mga kategorya: Insult, Threat, Sexism, Racism, Profane Language.
    Ang insulto naglakip sa pagdaugdaug, insolvent o contemptuous nga mga pahayag (gibiyaan nga walay dugang nga espesipikasyon).
    Ang hulga naglakip sa mga pahibalo sa paglapas sa pisikal nga integridad sa biktima.
    Ang seksismo naglakip sa mga pag-atake sa mga tawo base sa ilang gender (identity), kasagaran nga nakatutok sa mga babaye.
    Ang rasismo naglakip sa mga pag-atake sa mga tawo base sa ilang sinugdanan, etnisidad, nasud (pananglitan gipasabut nga pag-aghat sa kasuko).
    Ang Profane nga sinultihan naglakip sa paggamit sa sekswal nga explicit ug dili angay nga sinultihan.

    Ania ang komento: {}

    Tubaga ang 'yes' kung ang nahisgutan nga komento problematic (abusive). Kung dili, tubaga ang 'no'.
    """,

    "Khmer": """
    សូមគិតថាអ្នកកំពុងគ្រប់គ្រងនាទីរបស់អ្នកសម្របសម្រួល (ឧទាហរណ៍អ្នកនិពន្ធសារព័ត៌មាន)។ បា្រស់មើលសារដែលនៅខាងក្រោមដោយផ្អែកលើភាពអាចនឹងត្រូវបានផ្សព្វផ្សាយនៅក្រោមអត្ថបទព័ត៌មានដោយគ្មានការពិនិត្យឡើងវិញបន្ថែមទៀត (គ្មានបញ្ហា)។
    យោបល់ដែលជាបញ្ហាគឺជាភាសាដែលចាត់ទុកថាអាចទម្លាក់ទៅក្នុងក្រុមដូចខាងក្រោមនេះ៖ ការរិះគន់ ការគំរាម ការរើសអើងភេទ ការរើសអើងជាតិសាសន៍ និងភាសាត្រាស់កំហឹង។
    ការរិះគន់ពាក់ព័ន្ធនឹងការបណ្តេញ បំបែកចិត្ត ឬពាក្យគ្មានការគោរពណាមួយ (ទុកវាទុកតែឥតមានការពន្យល់បន្ថែម)។
    ការគំរាមពាក់ព័ន្ធនឹងការផ្តល់ដំណឹងអំពីការបំពានលើភាពគោរពភាពឯកតាផ្ទាល់ខ្លួននៃជនរងគ្រោះ។
    ការរើសអើងភេទពាក់ព័ន្ធនឹងការវាយប្រហារលើមនុស្សផ្អែកលើភេទរបស់ពួកគេ (អត្តសញ្ញាណ) ជាញឹកញាប់ដោយផ្តោតលើស្ត្រី។
    ការរើសអើងជាតិសាសន៍ពាក់ព័ន្ធនឹងការវាយប្រហារលើមនុស្សផ្អែកលើប្រភពដើម សាសនាធាតុជាតិ ប្រជាជាតិរបស់ពួកគេ (ជាទូទៅចង់បង្កើតការស្អប់ខ្ពើម)។
    ភាសាត្រាស់កំហឹងពាក់ព័ន្ធនឹងការប្រើប្រាស់ភាសាដែលបង្ហាញពីភេទសកម្ម និងមិនសមរម្យ។

    នេះគឺជាយោបល់: {}

    សូមឆ្លើយថា 'yes' ប្រសិនបើយោបល់ខាងលើមានបញ្ហា (បញ្ជារ)។ បើមិនដូច្នោះទេសូមឆ្លើយថា 'no'។
    """,

    "Tagalog": """
    Mangyaring isipin na ikaw ay gumaganap bilang isang moderator (hal. isang editor ng pahayagan). Suriin ang komento sa ibaba batay sa kung maaari itong mai-publish sa ilalim ng isang artikulo ng balita nang hindi na kailangang muling suriin (hindi problema).
    Ang mga problemadong komento ay mapang-abusong wika na maaaring mahulog sa ilan sa mga kategoryang ito: Insulto, Banta, Seksismo, Rasismo, Masamang Wika.
    Ang Insulto ay kinabibilangan ng nakakasirang-puri, bastos, o mapanghamak na mga pahayag (iniwan nang walang karagdagang detalye).
    Ang Banta ay nagsasangkot ng mga pahayag tungkol sa paglabag sa pisikal na integridad ng biktima.
    Ang Seksismo ay nagsasangkot ng mga pag-atake sa mga tao batay sa kanilang kasarian (pagkakakilanlan), kadalasang nakatuon sa mga kababaihan.
    Ang Rasismo ay nagsasangkot ng mga pag-atake sa mga tao batay sa kanilang pinagmulan, etnisidad, bansa (karaniwang nilalayon na pukawin ang galit).
    Ang Masamang Wika ay nagsasangkot ng paggamit ng tahasang sekswal at hindi naaangkop na wika.

    Narito ang komento: {}

    Sagutin ng 'yes' kung ang komento sa itaas ay may problema (mapang-abuso). Kung hindi, sagutin ng 'no'.
    """,

    "Hindi": """
    कृपया कल्पना करें कि आप एक मॉडरेटर की भूमिका निभा रहे हैं (उदा. एक समाचार पत्र के संपादक)। नीचे दिए गए टिप्पणी का मूल्यांकन करें कि क्या उन्हें किसी समाचार लेख के तहत बिना किसी समीक्षा के प्रकाशित किया जा सकता है (कोई समस्या नहीं)।
    समस्याग्रस्त टिप्पणियां अपमानजनक भाषा होती हैं जो इनमें से कुछ श्रेणियों में आ सकती हैं: अपमान, धमकी, लिंगभेद, नस्लवाद, अश्लील भाषा।
    अपमान में अपमानजनक, असभ्य या तिरस्कारपूर्ण बयान शामिल हैं (बिना किसी और विवरण के छोड़ दिया गया)।
    धमकी में पीड़ित की शारीरिक अखंडता का उल्लंघन करने की घोषणा शामिल है।
    लिंगभेद में लिंग (पहचान) के आधार पर लोगों पर हमला किया जाता है, जिसमें अक्सर महिलाओं पर ध्यान केंद्रित किया जाता है।
    नस्लवाद में उनके मूल, जातीयता, राष्ट्र के आधार पर लोगों पर हमला किया जाता है (आमतौर पर घृणा भड़काने के लिए)।
    अश्लील भाषा में यौन रूप से स्पष्ट और अनुचित भाषा का उपयोग शामिल है।

    यहाँ टिप्पणी है: {}

    यदि उपरोक्त टिप्पणी समस्याग्रस्त (अपमानजनक) है तो 'yes' का उत्तर दें। अन्यथा, 'no' का उत्तर दें।
    """,

    "Bengali": """
    অনুগ্রহ করে কল্পনা করুন যে আপনি একজন মডারেটরের ভূমিকা পালন করছেন (উদা. একটি সংবাদপত্রের সম্পাদক)। নীচের মন্তব্যটি মূল্যায়ন করুন যে তারা কোনও সংবাদ নিবন্ধের অধীনে আরও পর্যালোচনা ছাড়াই প্রকাশিত হতে পারে কিনা (সমস্যা ছাড়াই)।
    সমস্যাযুক্ত মন্তব্যগুলি অপমানজনক ভাষা যা এই বিভাগগুলির মধ্যে কিছুতে পড়তে পারে: অপমান, হুমকি, লিঙ্গবাদ, বর্ণবাদ, অশ্লীল ভাষা।
    অপমানের মধ্যে অবমাননাকর, অসৌজন্য বা অবজ্ঞাপূর্ণ বক্তব্য রয়েছে (অতিরিক্ত নির্দিষ্টকরণ ছাড়াই রেখে দেওয়া হয়েছে)।
    হুমকির মধ্যে ভুক্তভোগীর শারীরিক অখণ্ডতার লঙ্ঘনের ঘোষণা রয়েছে।
    লিঙ্গবাদে লিঙ্গ (পরিচয়) এর উপর ভিত্তি করে মানুষের উপর আক্রমণ জড়িত, প্রায়ই মহিলাদের উপর ফোকাস করে।
    বর্ণবাদে মানুষের উত্স, জাতিসত্তা, জাতির উপর ভিত্তি করে আক্রমণ জড়িত (সাধারণত ঘৃণা উস্কে দেওয়ার জন্য বোঝানো হয়)।
    অশ্লীল ভাষার মধ্যে যৌনতার খোলামেলা এবং অনুপযুক্ত ভাষার ব্যবহার জড়িত।

    এখানে মন্তব্য: {}

    উপরের মন্তব্যটি সমস্যাযুক্ত (অপমানজনক) হলে 'yes' উত্তর দিন। অন্যথায়, 'no' উত্তর দিন।
    """,

    "Urdu": """
    براہ کرم تصور کریں کہ آپ ایک معتدل کردار ادا کر رہے ہیں (مثال کے طور پر، ایک اخبار کا مدیر)۔ نیچے دیے گئے تبصرے کا جائزہ لیں کہ آیا انہیں کسی خبر کے مضمون کے تحت بغیر کسی مزید جائزہ کے شائع کیا جا سکتا ہے (بغیر کسی مسئلے کے)۔
    مسئلہ والے تبصرے گستاخانہ زبان ہیں جو ان میں سے کچھ زمروں میں آ سکتے ہیں: توہین، دھمکی، جنس پرستی، نسل پرستی، فحش زبان۔
    توہین میں توہین آمیز، ناپاک یا توہین آمیز بیانات شامل ہیں (بغیر کسی اور وضاحت کے چھوڑ دیا گیا)۔
    دھمکی میں متاثرہ کی جسمانی سالمیت کی خلاف ورزی کا اعلان شامل ہے۔
    جنس پرستی میں صنف (شناخت) کی بنیاد پر لوگوں پر حملے شامل ہیں، اکثر خواتین پر توجہ مرکوز ہوتی ہے۔
    نسل پرستی میں لوگوں پر ان کی اصل، نسل، قوم کی بنیاد پر حملے شامل ہیں (عام طور پر نفرت کو بھڑکانے کے ارادے سے)۔
    فحش زبان میں جنسی طور پر واضح اور نامناسب زبان کا استعمال شامل ہے۔

    یہاں تبصرہ ہے: {}

    اگر اوپر والا تبصرہ مسئلہ (گستاخانہ) ہے تو 'yes' کا جواب دیں۔ ورنہ، 'no' کا جواب دیں۔
    """
}


# Create a DataFrame to store instructions
instructions_df = pd.DataFrame(columns=['instruction', 'output'])

# Generate instructions using randomly selected templates
instructions_df['instruction'] = df["Text"].apply(
    lambda x: random.choice(list(instruction_templates.values())).format(x)
)

# Define a function to determine if both the newspaper and crowd rejected the comment
def both_reject(row):
    if row['Reject Newspaper'] == 1 and row['Reject Crowd'] == 1:
        return 'yes'
    else:
        return 'no'

# Apply the function to determine the output
instructions_df['output'] = df.apply(both_reject, axis=1)

instructions_df = balance_classes(instructions_df, 1500, "output")
instructions_df.to_parquet("../../data/RP/RP.parquet", index=False)