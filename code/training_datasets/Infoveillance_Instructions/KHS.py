# -*- coding: utf-8 -*-
# KHS (Korean Hate Speech)
# - Paper: https://www.aclweb.org/anthology/2020.socialnlp-1.4
# - Data: https://github.com/kocohub/korean-hate-speech

import pandas as pd
import random
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from utils.data_utils import downsample_to_minority_class

train = pd.read_csv('../../data/KHS/train.tsv', sep='\t')
dev = pd.read_csv('../../data/KHS/dev.tsv', sep='\t')

df = pd.concat([train, dev])

# Define the instruction templates in different languages
instruction_templates = {
    # Chinese
    "Chinese": """
    请分类以下新闻评论是否包含性别偏见。
    如果评论包括对性别角色、性取向、性别认同的偏见，以及任何关于性别相关行为的看法，请回答 "True"。否则，请回答 "False"。

    评论: {}
    请回答 "True" 或 "False"。
    """,

    # English
    "English": """
    Please classify if the news comment shown below contains gender bias.
    If the comment includes bias for gender role, sexual orientation, sexual identity, and any thoughts on gender-related acts, respond with "True". Otherwise, respond with "False".

    Comment: {}
    Please respond with "True" or "False".
    """,

    # German
    "German": """
    Bitte klassifizieren Sie, ob der unten gezeigte Nachrichtenkommentar geschlechtsspezifische Voreingenommenheit enthält.
    Wenn der Kommentar eine Voreingenommenheit gegenüber Geschlechterrollen, sexueller Orientierung, sexueller Identität und jeglichen Gedanken zu geschlechtsbezogenen Handlungen enthält, antworten Sie mit "True". Andernfalls antworten Sie mit "False".

    Kommentar: {}
    Bitte antworten Sie mit "True" oder "False".
    """,

    # French
    "French": """
    Veuillez classer si le commentaire de l'actualité ci-dessous contient un biais de genre.
    Si le commentaire inclut un biais pour le rôle de genre, l'orientation sexuelle, l'identité sexuelle et toute pensée sur les actes liés au genre, répondez par "True". Sinon, répondez par "False".

    Commentaire : {}
    Veuillez répondre par "True" ou "False".
    """,

    # Spanish
    "Spanish": """
    Por favor, clasifique si el comentario de la noticia mostrado a continuación contiene sesgo de género.
    Si el comentario incluye sesgo para el rol de género, la orientación sexual, la identidad sexual y cualquier pensamiento sobre actos relacionados con el género, responda con "True". De lo contrario, responda con "False".

    Comentario: {}
    Por favor, responda con "True" o "False".
    """,

    # Portuguese
    "Portuguese": """
    Por favor, classifique se o comentário da notícia mostrado abaixo contém preconceito de gênero.
    Se o comentário inclui preconceito em relação ao papel de gênero, orientação sexual, identidade sexual e qualquer pensamento sobre atos relacionados ao gênero, responda com "True". Caso contrário, responda com "False".

    Comentário: {}
    Por favor, responda com "True" ou "False".
    """,

    # Italian
    "Italian": """
    Si prega di classificare se il commento alla notizia mostrato di seguito contiene pregiudizi di genere.
    Se il commento include pregiudizi per il ruolo di genere, l'orientamento sessuale, l'identità sessuale e qualsiasi pensiero sugli atti relativi al genere, rispondere con "True". Altrimenti, rispondere con "False".

    Commento: {}
    Si prega di rispondere con "True" o "False".
    """,

    # Dutch
    "Dutch": """
    Gelieve te classificeren of de onderstaande nieuwscommentaar genderbias bevat.
    Als de opmerking een vooroordeel bevat voor de rol van gender, seksuele geaardheid, seksuele identiteit en eventuele gedachten over gendergerelateerde handelingen, reageer dan met "True". Anders reageer met "False".

    Opmerking: {}
    Reageer alstublieft met "True" of "False".
    """,

    # Russian
    "Russian": """
    Пожалуйста, классифицируйте, содержит ли приведенный ниже комментарий к новостям гендерные предвзятости.
    Если комментарий включает предвзятость по отношению к гендерным ролям, сексуальной ориентации, гендерной идентичности и любым мыслям о гендерных актах, ответьте "True". В противном случае ответьте "False".

    Комментарий: {}
    Пожалуйста, ответьте "True" или "False".
    """,

    # Czech
    "Czech": """
    Prosím, klasifikujte, zda níže uvedený komentář k zprávě obsahuje genderové předsudky.
    Pokud komentář obsahuje předsudky vůči genderovým rolím, sexuální orientaci, sexuální identitě a jakýmkoli myšlenkám na genderové činy, odpovězte "True". V opačném případě odpovězte "False".

    Komentář: {}
    Prosím odpovězte "True" nebo "False".
    """,

    # Polish
    "Polish": """
    Proszę sklasyfikować, czy poniższy komentarz do wiadomości zawiera uprzedzenia dotyczące płci.
    Jeśli komentarz zawiera uprzedzenia dotyczące ról płciowych, orientacji seksualnej, tożsamości płciowej i jakichkolwiek myśli na temat działań związanych z płcią, odpowiedz "True". W przeciwnym razie odpowiedz "False".

    Komentarz: {}
    Proszę odpowiedzieć "True" lub "False".
    """,

    # Arabic
    "Arabic": """
    يرجى تصنيف ما إذا كان التعليق على الأخبار الموضح أدناه يحتوي على تحيز جنساني.
    إذا كان التعليق يتضمن تحيزًا لدور الجنس، أو التوجه الجنسي، أو الهوية الجنسية، أو أي أفكار حول الأفعال المتعلقة بالجنس، يرجى الرد بـ "True". خلاف ذلك، يرجى الرد بـ "False".

    تعليق: {}
    يرجى الرد بـ "True" أو "False".
    """,

    # Persian
    "Persian": """
    لطفاً تعیین کنید که آیا نظر خبری زیر دارای تعصب جنسیتی است یا خیر.
    اگر نظر شامل تعصب برای نقش جنسیتی، گرایش جنسی، هویت جنسی و هرگونه تفکرات در مورد اعمال مرتبط با جنسیت است، با "True" پاسخ دهید. در غیر این صورت، با "False" پاسخ دهید.

    نظر: {}
    لطفاً با "True" یا "False" پاسخ دهید.
    """,

    # Hebrew
    "Hebrew": """
    אנא סווג אם התגובה לחדשות המוצגת להלן מכילה הטיה מגדרית.
    אם התגובה כוללת הטיה כלפי תפקיד מגדרי, נטייה מינית, זהות מינית וכל מחשבות על מעשים הקשורים למגדר, השב עם "True". אחרת, השב עם "False".

    תגובה: {}
    אנא השב עם "True" או "False".
    """,

    # Turkish
    "Turkish": """
    Lütfen aşağıda gösterilen haber yorumunun cinsiyet yanlılığı içerip içermediğini sınıflandırın.
    Yorum, cinsiyet rolü, cinsel yönelim, cinsel kimlik ve cinsiyetle ilgili eylemlerle ilgili herhangi bir önyargı içeriyorsa, "True" olarak yanıtlayın. Aksi takdirde "False" olarak yanıtlayın.

    Yorum: {}
    Lütfen "True" veya "False" ile yanıtlayın.
    """,

    # Japanese
    "Japanese": """
    以下のニュースコメントに性別に関する偏見が含まれているかどうかを分類してください。
    コメントに性別役割、性的指向、性的アイデンティティ、および性別関連行為に関する偏見が含まれている場合、「True」と回答してください。それ以外の場合は、「False」と回答してください。

    コメント: {}
    「True」または「False」で回答してください。
    """,

    # Korean
    "Korean": """
    아래에 표시된 뉴스 댓글에 성별 편향이 포함되어 있는지 분류하세요.
    댓글에 성 역할, 성적 지향, 성 정체성 및 성 관련 행위에 대한 편향이 포함된 경우 "True"로 응답하십시오. 그렇지 않은 경우 "False"로 응답하십시오.

    댓글: {}
    "True" 또는 "False"로 응답하십시오.
    """,

    # Vietnamese
    "Vietnamese": """
    Vui lòng phân loại liệu bình luận tin tức dưới đây có chứa sự thiên vị về giới hay không.
    Nếu bình luận bao gồm sự thiên vị về vai trò giới, xu hướng tình dục, bản sắc giới và bất kỳ suy nghĩ nào về các hành động liên quan đến giới, hãy trả lời bằng "True". Nếu không, hãy trả lời bằng "False".

    Bình luận: {}
    Vui lòng trả lời bằng "True" hoặc "False".
    """,

    # Thai
    "Thai": """
    โปรดจำแนกว่าความคิดเห็นข่าวด้านล่างนี้มีอคติเกี่ยวกับเพศหรือไม่
    หากความคิดเห็นรวมถึงอคติในบทบาททางเพศ รสนิยมทางเพศ อัตลักษณ์ทางเพศ และความคิดเกี่ยวกับการกระทำที่เกี่ยวข้องกับเพศใดๆ ให้ตอบว่า "True" มิฉะนั้น ให้ตอบว่า "False"

    ความคิดเห็น: {}
    โปรดตอบกลับด้วย "True" หรือ "False"
    """,

    # Indonesian
    "Indonesian": """
    Silakan klasifikasikan apakah komentar berita yang ditampilkan di bawah ini mengandung bias gender.
    Jika komentar tersebut mencakup bias untuk peran gender, orientasi seksual, identitas seksual, dan pemikiran terkait tindakan terkait gender, balas dengan "True". Jika tidak, balas dengan "False".

    Komentar: {}
    Silakan balas dengan "True" atau "False".
    """,

    # Malay
    "Malay": """
    Sila klasifikasikan jika komen berita yang ditunjukkan di bawah mengandungi bias jantina.
    Jika komen termasuk bias untuk peranan jantina, orientasi seksual, identiti seksual, dan sebarang pemikiran mengenai perbuatan berkaitan jantina, jawab dengan "True". Jika tidak, jawab dengan "False".

    Komen: {}
    Sila jawab dengan "True" atau "False".
    """,

    # Lao
    "Lao": """
    ກະລຸນາຈັດປະເພດວ່າຄຳຄິດເຫັນຂ່າວທີ່ສະແດງຢູ່ດ້ານລຸ່ມນີ້ມີຄວາມເອກະລາດທາງເພດຫຼືບໍ່.
    ຖ້າຄຳຄິດເຫັນມີຄວາມເອກະລາດສຳລັບບົດບາດທາງເພດ, ເພດຄົງຫຼືຄວາມຄິດເຫັນໃດກໍຕາມກ່ຽວກັບການປະພືດທາງເພດທີ່ກ່ຽວຂ້ອງ, ກະລຸນາຕອບ "True". ຖ້າບໍ່ຕອງຕອບ "False".

    ຄວາມຄິດເຫັນ: {}
    ກະລຸນາຕອບກັບ "True" ຫຼື "False".
    """,

    # Burmese
    "Burmese": """
    ကျေးဇူးပြု၍ အောက်တွင် ဖော်ပြထားသော သတင်းမှတ်ချက်တွင် ကျားမအား မတူကွဲပြားမှုပါရှိပါက ခွဲခြားဖော်ပြပါ။
    မှတ်ချက်တွင် ကျားမဝတ်ပြုမှု၊ လိင်ဆက်ဆံမှု၊ လိင်စိတ်ခံစားမှုနှင့် လိင်နှင့်ပတ်သက်သော စိတ်ကူးယဉ်မှုများပါရှိပါက "True" ဟု ပြန်ဆိုပါ။ မပါရှိပါက "False" ဟု ပြန်ဆိုပါ။

    မှတ်ချက်: {}
    ကျေးဇူးပြု၍ "True" သို့မဟုတ် "False" ဖြင့် ဖြေပါ။
    """,

    # Cebuano
    "Cebuano": """
    Palihug i-klassify kon ang komentaryo sa balita nga gipakita sa ubos naglakip og gender bias.
    Kon ang komentaryo naglakip og bias para sa papel sa gender, sekswal nga oryentasyon, sekswal nga identidad, ug bisan unsang mga hunahuna mahitungod sa mga lihok nga may kalabutan sa gender, tubaga og "True". Kung dili, tubaga og "False".

    Komentaryo: {}
    Palihug pagtubag og "True" o "False".
    """,

    # Khmer
    "Khmer": """
    សូមចាត់ថ្នាក់ថាតើការវាយតម្លៃនៃព័ត៌មាននៅខាងក្រោមនេះមានការរើសអើងយេនឌ័រ (Gender Bias) ឬទេ។
    ប្រសិនបើការវាយតម្លៃរួមបញ្ចូលនូវការរើសអើងតាមតួនាទីយេនឌ័រ, ភេទ, អត្តសញ្ញាណភេទ, និងគំនិតដែលទាក់ទងនឹងអំពើយេនឌ័រ, សូមឆ្លើយតបដោយប្រើពាក្យ "True"។ ប្រសិនបើមិនដូច្នោះទេ សូមឆ្លើយតបដោយប្រើពាក្យ "False"។

    មតិយោបល់: {}
    សូមឆ្លើយតបដោយប្រើពាក្យ "True" ឬ "False"។
    """,

    # Tagalog
    "Tagalog": """
    Pakisuri kung ang komentaryo ng balita sa ibaba ay naglalaman ng bias ng kasarian.
    Kung ang komentaryo ay may kinikilingan sa papel ng kasarian, oryentasyong sekswal, pagkakakilanlang sekswal, at anumang mga saloobin tungkol sa mga kilos na may kaugnayan sa kasarian, sumagot ng "True". Kung hindi, sumagot ng "False".

    Komento: {}
    Pakisagot ng "True" o "False".
    """,

    # Hindi
    "Hindi": """
    कृपया यह वर्गीकृत करें कि नीचे दिखाई गई समाचार टिप्पणी में लिंग पूर्वाग्रह है या नहीं।
    यदि टिप्पणी में लिंग भूमिका, यौन अभिविन्यास, यौन पहचान, और लिंग-संबंधित कृत्यों पर कोई विचार शामिल है, तो "True" के साथ प्रतिक्रिया दें। अन्यथा, "False" के साथ प्रतिक्रिया दें।

    टिप्पणी: {}
    कृपया "True" या "False" के साथ प्रतिक्रिया दें।
    """,

    # Bengali
    "Bengali": """
    অনুগ্রহ করে নীচের দেখানো সংবাদ মন্তব্যটিতে লিঙ্গ পক্ষপাতিত্ব রয়েছে কিনা তা শ্রেণীবদ্ধ করুন।
    যদি মন্তব্যে লিঙ্গ ভূমিকা, যৌন অভিমুখতা, যৌন পরিচয় এবং লিঙ্গ-সম্পর্কিত কাজের যেকোনো চিন্তা অন্তর্ভুক্ত থাকে, তবে "True" দিয়ে প্রতিক্রিয়া জানান। অন্যথায়, "False" দিয়ে প্রতিক্রিয়া জানান।

    মন্তব্য: {}
    অনুগ্রহ করে "True" বা "False" দিয়ে প্রতিক্রিয়া জানান।
    """,

    # Urdu
    "Urdu": """
    براہ کرم درجہ بندی کریں کہ آیا ذیل میں دکھائی گئی خبروں کے تبصرے میں صنفی تعصب شامل ہے۔
    اگر تبصرہ صنفی کردار، جنسی رجحان، جنسی شناخت، اور صنف سے متعلق اعمال کے بارے میں کسی تعصب کو شامل کرتا ہے تو "True" کے ساتھ جواب دیں۔ بصورت دیگر، "False" کے ساتھ جواب دیں۔

    تبصرہ: {}
    براہ کرم "True" یا "False" کے ساتھ جواب دیں۔
    """
}


# Create a new DataFrame
instruction_df1 = df[['comments', 'contain_gender_bias']].copy()

# Generate the instructions using randomly selected templates
instruction_df1['instruction'] = instruction_df1['comments'].apply(
    lambda x: random.choice(list(instruction_templates.values())).format(x)
)

# Generate the outputs without modification since "True" and "False" are already in English
instruction_df1['output'] = instruction_df1['contain_gender_bias']

# Drop the original columns
instruction_df1 = instruction_df1.drop(columns=['comments', 'contain_gender_bias'])

# Shuffle the DataFrame and save to Parquet
instruction_df1 = instruction_df1.sample(frac=1).reset_index(drop=True)
instruction_df1.to_parquet("../../data/KHS/KHS-gender.parquet", index=False)

# Display the first instruction in the DataFrame
instruction_df1_balanced = downsample_to_minority_class(instruction_df1, 'output')
print(instruction_df1_balanced.output.value_counts())

# sample n=2500 and save to the same parquet file
instruction_df1_balanced.sample(n=2500).to_parquet("../../data/KHS/KHS-gender.parquet", index=False)
# Define the instruction templates in different languages
instruction_templates = {
    "Chinese": """
    此任务是将仇恨、冒犯和无标签分配给Naver新闻评论。

    这是注释标准。请注意，在本研究中，每个类别的定义可能在某些方面与其他研究相似，但并不完全相同。请仔细遵循以下定义。

    # hate: 在评论中是否对文章的目标或相关人物、文章或评论的作者等表现出强烈的仇恨或侮辱？如果是这样，它应该被归类为hate。
    - 在侮辱的情况下，它涵盖了可能严重损害接受者社会地位的表达。
    - 在仇恨的情况下，它被定义为对具有某些特征（性别角色、性取向、性别认同、与性别相关的行为、种族、背景、国籍、民族、政治立场、肤色、宗教、残疾、年龄、外貌、财富、职业、缺乏兵役经验等）的个人/群体表达的攻击性立场的表达。
    - 此外，它还可以包括性骚扰、传播冒犯性谣言或事实、以及用于恶意目的或错误使用的造词等。
    - 文档中仅存在脏话并不总是属于这一类。

    # offensive: 虽然评论不像上述那样充满仇恨或侮辱，但它是否让目标或读者感到冒犯？如果是这样，它应该被归类为offensive。
    - 它可能包含粗鲁或攻击性的内容，例如脏话，但不至于达到仇恨或侮辱的程度。
    - 它可以通过反问或讽刺表达出讽刺的意味。
    - 它可以包括不道德的表达（例如，关于已故人物的笑话或无关的问题）。
    - 传播不明谣言的评论可能属于这一类。

    # none: 不包含任何仇恨或侮辱的评论。对于这种评论，它应该被归类为none。

    评论: {}. 现在评论结束。
    请回答 "hate", "offensive" 或 "none"。
    """,

    "English": """
    This task is to assign hate, offensive, and none labels to a comment on a news on Naver.

    This is the annotation standard. Note that in this study, the definition of each category could be similar to other studies in some aspects, but it is not exactly the same. Please follow the definition below carefully.

    # hate: Is strong hate or insulting towards the article's target or related figures, writers of the article or comments, etc. displayed in a comment? If so, it should be classified as hate.
    - In the case of insulting, it encompasses an expression that can severely harm the social status of the recipient.
    - In the case of hate, it is defined as an expression that displays aggressive stances towards individuals/groups with certain characteristics (gender role, sexual orientation, sexual identity, any thoughts on gender-related acts, race, background, nationality, ethnic group, political stance, skin color, religion, handicaps, age, appearance, richness, occupations, the absence of military service experience, etc.).
    - Additionally, it can include sexual harassment, notification of offensive rumors or facts, and coined terms for bad purposes or in bad use, etc.
    - Just the existence of bad words in the document does not always fall into this category.

    # offensive: Although a comment is not as much hateful or insulting as the above, does it make the target or the reader feel offended? If so, it should be classified as offensive.
    - It may contain rude or aggressive content, such as bad words, though not to the extent of hate or insult.
    - It can emit sarcasm through rhetorical question or irony.
    - It may encompass an unethical expression (e.g., jokes or irrelevant questions regarding the figures who passed away).
    - A comment conveying unidentified rumors can belong to this category.

    # none: A comment that does not incorporate any hate or insulting. For a comment like this, it should be classified as none.

    Comment: {}. Now the comment ends.
    Please respond with "hate", "offensive", or "none".
    """,

    "German": """
    Diese Aufgabe besteht darin, Kommentare zu einer Nachricht auf Naver mit den Labels hate, offensive und none zu versehen.

    Dies ist der Annotationsstandard. Beachten Sie, dass die Definition jeder Kategorie in dieser Studie in einigen Aspekten ähnlich wie in anderen Studien sein könnte, jedoch nicht genau gleich. Bitte befolgen Sie die folgende Definition sorgfältig.

    # hate: Wird in einem Kommentar starke Hass oder Beleidigung gegenüber dem Ziel des Artikels oder verwandten Figuren, den Autoren des Artikels oder Kommentaren usw. gezeigt? Wenn ja, sollte es als hate klassifiziert werden.
    - Im Falle einer Beleidigung umfasst es eine Ausdrucksweise, die den sozialen Status des Empfängers erheblich schädigen kann.
    - Im Falle von hate wird es als Ausdruck definiert, der aggressive Haltungen gegenüber Einzelpersonen/Gruppen mit bestimmten Merkmalen (Geschlechterrolle, sexuelle Orientierung, sexuelle Identität, Gedanken zu geschlechtsbezogenen Handlungen, Rasse, Hintergrund, Nationalität, ethnische Gruppe, politische Haltung, Hautfarbe, Religion, Behinderungen, Alter, Aussehen, Reichtum, Berufe, das Fehlen von Militärdienst) darstellt.
    - Darüber hinaus kann es sexuelle Belästigung, die Verbreitung von anstößigen Gerüchten oder Fakten und für schlechte Zwecke oder in schlechter Verwendung geprägte Begriffe umfassen.
    - Das bloße Vorhandensein von Schimpfwörtern im Dokument fällt nicht immer in diese Kategorie.

    # offensive: Obwohl ein Kommentar nicht so hasserfüllt oder beleidigend ist wie oben, fühlt sich das Ziel oder der Leser beleidigt? Wenn ja, sollte es als offensive klassifiziert werden.
    - Es kann unhöfliche oder aggressive Inhalte enthalten, wie z. B. Schimpfwörter, jedoch nicht im Ausmaß von Hass oder Beleidigung.
    - Es kann Sarkasmus durch rhetorische Frage oder Ironie ausstrahlen.
    - Es kann eine unethische Ausdrucksweise umfassen (z. B. Witze oder irrelevante Fragen zu verstorbenen Persönlichkeiten).
    - Ein Kommentar, der nicht identifizierte Gerüchte vermittelt, kann in diese Kategorie fallen.

    # none: Ein Kommentar, der keine Hass oder Beleidigungen enthält. Ein solcher Kommentar sollte als none klassifiziert werden.

    Kommentar: {}. Jetzt endet der Kommentar.
    Bitte antworten Sie mit "hate", "offensive" oder "none".
    """,

    "French": """
    Cette tâche consiste à attribuer des labels hate, offensive et none à un commentaire sur une nouvelle sur Naver.

    Voici la norme d'annotation. Notez que dans cette étude, la définition de chaque catégorie peut être similaire à d'autres études dans certains aspects, mais elle n'est pas exactement la même. Veuillez suivre attentivement la définition ci-dessous.

    # hate: Est-ce qu'une forte haine ou une insulte envers la cible de l'article ou les personnages liés, les auteurs de l'article ou les commentaires, etc. est affichée dans un commentaire? Si oui, il doit être classé comme hate.
    - En cas d'insulte, cela englobe une expression qui peut gravement nuire au statut social du destinataire.
    - En cas de hate, il est défini comme une expression affichant des positions agressives envers des individus/groupes ayant certaines caractéristiques (rôle de genre, orientation sexuelle, identité sexuelle, pensées sur les actes liés au genre, race, origine, nationalité, groupe ethnique, position politique, couleur de peau, religion, handicaps, âge, apparence, richesse, professions, absence d'expérience de service militaire, etc.).
    - De plus, il peut inclure le harcèlement sexuel, la notification de rumeurs ou de faits offensants, et les termes inventés à des fins malveillantes ou utilisés de manière abusive, etc.
    - La simple existence de mauvais mots dans le document ne relève pas toujours de cette catégorie.

    # offensive: Bien qu'un commentaire ne soit pas aussi haineux ou insultant que ci-dessus, rend-il la cible ou le lecteur offensé? Si oui, il doit être classé comme offensive.
    - Il peut contenir un contenu grossier ou agressif, tel que des gros mots, bien que pas au point de haine ou d'insulte.
    - Il peut émettre du sarcasme à travers une question rhétorique ou de l'ironie.
    - Il peut englober une expression non éthique (par exemple, des blagues ou des questions non pertinentes concernant les personnes décédées).
    - Un commentaire véhiculant des rumeurs non identifiées peut appartenir à cette catégorie.

    # none: Un commentaire qui n'incorpore aucune haine ou insulte. Pour un commentaire de ce type, il doit être classé comme none.

    Commentaire : {}. Maintenant, le commentaire se termine.
    Veuillez répondre avec "hate", "offensive" ou "none".
    """,

    "Spanish": """
    Esta tarea consiste en asignar etiquetas de hate, offensive y none a un comentario en una noticia en Naver.

    Este es el estándar de anotación. Tenga en cuenta que en este estudio, la definición de cada categoría podría ser similar a otros estudios en algunos aspectos, pero no es exactamente la misma. Siga cuidadosamente la definición a continuación.

    # hate: ¿Hay un fuerte odio o insulto hacia el objetivo del artículo o figuras relacionadas, escritores del artículo o comentarios, etc. mostrado en un comentario? Si es así, debe clasificarse como hate.
    - En el caso de insulto, abarca una expresión que puede dañar gravemente el estatus social del destinatario.
    - En el caso de hate, se define como una expresión que muestra posturas agresivas hacia individuos/grupos con ciertas características (rol de género, orientación sexual, identidad sexual, cualquier pensamiento sobre actos relacionados con el género, raza, origen, nacionalidad, grupo étnico, postura política, color de piel, religión, discapacidades, edad, apariencia, riqueza, ocupaciones, ausencia de experiencia en el servicio militar, etc.).
    - Además, puede incluir acoso sexual, notificación de rumores o hechos ofensivos y términos acuñados para malos propósitos o en mal uso, etc.
    - La mera existencia de malas palabras en el documento no siempre cae en esta categoría.

    # offensive: Aunque un comentario no es tan odioso o insultante como lo anterior, ¿hace que el objetivo o el lector se sienta ofendido? Si es así, debe clasificarse como offensive.
    - Puede contener contenido grosero o agresivo, como malas palabras, aunque no hasta el punto de odio o insulto.
    - Puede emitir sarcasmo a través de una pregunta retórica o ironía.
    - Puede abarcar una expresión poco ética (por ejemplo, bromas o preguntas irrelevantes sobre figuras que han fallecido).
    - Un comentario que transmita rumores no identificados puede pertenecer a esta categoría.

    # none: Un comentario que no incorpore ningún odio o insulto. Para un comentario como este, debe clasificarse como none.

    Comentario: {}. Ahora el comentario termina.
    Responda con "hate", "offensive" o "none".
    """,

    "Portuguese": """
    Esta tarefa é atribuir etiquetas de hate, offensive e none a um comentário sobre uma notícia no Naver.

    Este é o padrão de anotação. Note que neste estudo, a definição de cada categoria pode ser semelhante a outros estudos em alguns aspectos, mas não é exatamente a mesma. Siga cuidadosamente a definição abaixo.

    # hate: Existe ódio forte ou insulto em relação ao alvo do artigo ou figuras relacionadas, escritores do artigo ou comentários, etc. exibido em um comentário? Se sim, deve ser classificado como hate.
    - No caso de insulto, abrange uma expressão que pode prejudicar gravemente o status social do destinatário.
    - No caso de hate, é definido como uma expressão que exibe posturas agressivas em relação a indivíduos/grupos com certas características (papel de gênero, orientação sexual, identidade sexual, quaisquer pensamentos sobre atos relacionados ao gênero, raça, origem, nacionalidade, grupo étnico, posição política, cor da pele, religião, deficiências, idade, aparência, riqueza, ocupações, ausência de experiência no serviço militar, etc.).
    - Além disso, pode incluir assédio sexual, notificação de rumores ou fatos ofensivos e termos cunhados para maus propósitos ou em uso indevido, etc.
    - A mera existência de palavrões no documento não se enquadra sempre nesta categoria.

    # offensive: Embora um comentário não seja tão odioso ou insultuoso como o acima, ele faz com que o alvo ou o leitor se sinta ofendido? Se sim, deve ser classificado como offensive.
    - Pode conter conteúdo rude ou agressivo, como palavrões, embora não ao ponto de ódio ou insulto.
    - Pode emitir sarcasmo através de pergunta retórica ou ironia.
    - Pode abranger uma expressão antiética (por exemplo, piadas ou perguntas irrelevantes sobre figuras que faleceram).
    - Um comentário transmitindo rumores não identificados pode pertencer a esta categoria.

    # none: Um comentário que não incorpore nenhum ódio ou insulto. Para um comentário como este, deve ser classificado como none.

    Comentário: {}. Agora o comentário termina.
    Responda com "hate", "offensive" ou "none".
    """,

    "Italian": """
    Questo compito consiste nell'assegnare etichette di hate, offensive e none a un commento su una notizia su Naver.

    Questo è lo standard di annotazione. Si noti che in questo studio, la definizione di ciascuna categoria potrebbe essere simile ad altri studi in alcuni aspetti, ma non è esattamente la stessa. Si prega di seguire attentamente la definizione seguente.

    # hate: C'è un forte odio o insulti verso l'obiettivo dell'articolo o figure correlate, scrittori dell'articolo o commenti, ecc. mostrati in un commento? In tal caso, dovrebbe essere classificato come hate.
    - Nel caso degli insulti, comprende un'espressione che può danneggiare gravemente lo status sociale del destinatario.
    - Nel caso di hate, è definito come un'espressione che mostra posizioni aggressive nei confronti di individui/gruppi con determinate caratteristiche (ruolo di genere, orientamento sessuale, identità sessuale, qualsiasi pensiero sugli atti correlati al genere, razza, origine, nazionalità, gruppo etnico, posizione politica, colore della pelle, religione, disabilità, età, aspetto, ricchezza, occupazioni, assenza di esperienza di servizio militare, ecc.).
    - Inoltre, può includere molestie sessuali, notifica di voci o fatti offensivi e termini coniati per scopi cattivi o in cattivo uso, ecc.
    - La semplice esistenza di parolacce nel documento non rientra sempre in questa categoria.

    # offensive: Sebbene un commento non sia tanto odioso o offensivo quanto quanto sopra, fa sentire offeso il destinatario o il lettore? In tal caso, dovrebbe essere classificato come offensive.
    - Può contenere contenuti maleducati o aggressivi, come parolacce, anche se non al punto di odio o insulto.
    - Può emettere sarcasmo attraverso una domanda retorica o ironia.
    - Può comprendere un'espressione non etica (ad esempio, battute o domande irrilevanti relative a figure decedute).
    - Un commento che trasmette voci non identificate può appartenere a questa categoria.

    # none: Un commento che non incorpora alcun odio o insulto. Per un commento come questo, dovrebbe essere classificato come none.

    Commento: {}. Ora il commento finisce.
    Si prega di rispondere con "hate", "offensive" o "none".
    """,

    "Dutch": """
    Deze taak is om haat, beledigend en geen labels toe te wijzen aan een reactie op een nieuwsbericht op Naver.

    Dit is de annotatiestandaard. Merk op dat in deze studie de definitie van elke categorie in sommige opzichten vergelijkbaar kan zijn met andere studies, maar het is niet precies hetzelfde. Volg de onderstaande definitie zorgvuldig.

    # hate: Is er sterke haat of belediging gericht op het doelwit van het artikel of gerelateerde figuren, schrijvers van het artikel of opmerkingen, enz. weergegeven in een reactie? Zo ja, dan moet het worden geclassificeerd als hate.
    - In het geval van belediging omvat het een uitdrukking die de sociale status van de ontvanger ernstig kan schaden.
    - In het geval van haat wordt het gedefinieerd als een uitdrukking die agressieve houdingen weergeeft ten opzichte van individuen/groepen met bepaalde kenmerken (genderrol, seksuele geaardheid, seksuele identiteit, gedachten over gendergerelateerde handelingen, ras, achtergrond, nationaliteit, etnische groep, politieke houding, huidskleur, religie, handicaps, leeftijd, uiterlijk, rijkdom, beroepen, het ontbreken van militaire dienstervaring, enz.).
    - Daarnaast kan het seksuele intimidatie omvatten, de melding van beledigende geruchten of feiten, en verzonnen termen voor slechte doeleinden of in slecht gebruik, enz.
    - Het loutere bestaan van slechte woorden in het document valt niet altijd in deze categorie.

    # offensive: Hoewel een reactie niet zo haatdragend of beledigend is als het bovenstaande, maakt het dan dat het doelwit of de lezer zich beledigd voelt? Zo ja, dan moet het worden geclassificeerd als offensive.
    - Het kan onbeleefde of agressieve inhoud bevatten, zoals scheldwoorden, hoewel niet in de mate van haat of belediging.
    - Het kan sarcasme uitstralen door middel van een retorische vraag of ironie.
    - Het kan een onethische uitdrukking omvatten (bijvoorbeeld grappen of irrelevante vragen over overleden personen).
    - Een reactie die niet-geïdentificeerde geruchten verspreidt, kan in deze categorie vallen.

    # none: Een reactie die geen haat of belediging bevat. Voor een dergelijke reactie moet het worden geclassificeerd als none.

    Reactie: {}. Nu eindigt de reactie.
    Reageer alstublieft met "hate", "offensive" of "none".
    """,

    "Russian": """
    Эта задача заключается в присвоении меток hate, offensive и none комментариям к новости на Naver.

    Это стандарт аннотации. Обратите внимание, что в этом исследовании определение каждой категории может быть похоже на другие исследования в некоторых аспектах, но не совсем совпадает. Пожалуйста, внимательно следуйте приведенному ниже определению.

    # hate: Проявляется ли в комментарии сильная ненависть или оскорбления по отношению к цели статьи или связанным фигурам, авторам статьи или комментариям и т.д.? Если да, это следует классифицировать как hate.
    - В случае оскорбления это охватывает выражение, которое может серьезно повредить социальному статусу получателя.
    - В случае ненависти это определяется как выражение агрессивной позиции по отношению к лицам/группам с определенными характеристиками (гендерная роль, сексуальная ориентация, сексуальная идентичность, любые мысли о действиях, связанных с гендером, раса, происхождение, национальность, этническая принадлежность, политическая позиция, цвет кожи, религия, инвалидность, возраст, внешность, богатство, профессии, отсутствие опыта военной службы и т.д.).
    - Кроме того, это может включать сексуальные домогательства, уведомление об оскорбительных слухах или фактах, а также термины, придуманные для плохих целей или в плохом использовании и т.д.
    - Простое наличие плохих слов в документе не всегда попадает в эту категорию.

    # offensive: Хотя комментарий не настолько ненавистный или оскорбительный, как выше, заставляет ли он цель или читателя чувствовать себя оскорбленным? Если да, это следует классифицировать как offensive.
    - Он может содержать грубое или агрессивное содержание, такое как плохие слова, хотя и не в такой степени, как ненависть или оскорбление.
    - Он может выражать сарказм через риторический вопрос или иронию.
    - Он может охватывать неэтичное выражение (например, шутки или неуместные вопросы относительно умерших личностей).
    - Комментарий, передающий неидентифицированные слухи, может принадлежать к этой категории.

    # none: Комментарий, который не включает в себя ненависть или оскорбления. Такой комментарий следует классифицировать как none.

    Комментарий: {}. Теперь комментарий заканчивается.
    Пожалуйста, ответьте "hate", "offensive" или "none".
    """,

    "Czech": """
    Tento úkol spočívá v přiřazení štítků hate, offensive a none ke komentáři k zprávě na Naver.

    Toto je standard anotace. Všimněte si, že v této studii může být definice každé kategorie podobná jiným studiím v některých aspektech, ale není přesně stejná. Postupujte prosím pečlivě podle níže uvedené definice.

    # hate: Projevuje se v komentáři silná nenávist nebo urážky vůči cíli článku nebo souvisejícím postavám, autorům článku nebo komentářům atd.? Pokud ano, mělo by to být klasifikováno jako hate.
    - V případě urážky zahrnuje výrazy, které mohou vážně poškodit společenské postavení příjemce.
    - V případě nenávisti se to definuje jako výraz, který vyjadřuje agresivní postoje vůči jednotlivcům/skupinám s určitými charakteristikami (genderová role, sexuální orientace, sexuální identita, jakékoli myšlenky na činy související s genderem, rasa, původ, národnost, etnická skupina, politický postoj, barva pleti, náboženství, postižení, věk, vzhled, bohatství, povolání, absence vojenské služby atd.).
    - Kromě toho může zahrnovat sexuální obtěžování, oznámení urážlivých pověstí nebo faktů a vytvoření termínů pro špatné účely nebo ve špatném použití atd.
    - Pouhá existence špatných slov v dokumentu ne vždy spadá do této kategorie.

    # offensive: Ačkoli není komentář tak nenávistný nebo urážlivý jako výše, způsobuje, že se cíl nebo čtenář cítí uražen? Pokud ano, mělo by to být klasifikováno jako offensive.
    - Může obsahovat hrubý nebo agresivní obsah, jako jsou špatná slova, i když ne do té míry, jakou je nenávist nebo urážka.
    - Může vyjadřovat sarkasmus prostřednictvím rétorické otázky nebo ironie.
    - Může zahrnovat neetický výraz (např. vtipy nebo irelevantní otázky týkající se zemřelých osobností).
    - Komentář obsahující neidentifikované pověsti může spadat do této kategorie.

    # none: Komentář, který neobsahuje žádnou nenávist nebo urážky. Takový komentář by měl být klasifikován jako none.

    Komentář: {}. Nyní komentář končí.
    Prosím odpovězte "hate", "offensive" nebo "none".
    """,

    "Polish": """
    To zadanie polega na przypisaniu etykiet hate, offensive i none do komentarza do wiadomości na Naver.

    To jest standard adnotacji. Zauważ, że w tym badaniu definicja każdej kategorii może być podobna do innych badań w niektórych aspektach, ale nie jest dokładnie taka sama. Proszę dokładnie przestrzegać poniższej definicji.

    # hate: Czy w komentarzu wyrażana jest silna nienawiść lub obraźliwe komentarze wobec celu artykułu lub związanych z nim postaci, autorów artykułu lub komentarzy itp.? Jeśli tak, należy go sklasyfikować jako hate.
    - W przypadku zniewagi obejmuje to wyrażenie, które może poważnie zaszkodzić statusowi społecznemu odbiorcy.
    - W przypadku hate definiuje się to jako wyraz agresywnych postaw wobec osób/grup z pewnymi cechami (rola płci, orientacja seksualna, tożsamość płciowa, jakiekolwiek myśli dotyczące działań związanych z płcią, rasa, pochodzenie, narodowość, grupa etniczna, postawa polityczna, kolor skóry, religia, niepełnosprawności, wiek, wygląd, bogactwo, zawody, brak doświadczenia wojskowego itp.).
    - Dodatkowo może to obejmować molestowanie seksualne, powiadamianie o obraźliwych plotkach lub faktach oraz tworzenie terminów do złych celów lub w złym użyciu itp.
    - Samo istnienie wulgaryzmów w dokumencie nie zawsze należy do tej kategorii.

    # offensive: Chociaż komentarz nie jest tak nienawistny lub obraźliwy, jak powyżej, czy sprawia, że cel lub czytelnik czuje się urażony? Jeśli tak, należy go sklasyfikować jako offensive.
    - Może zawierać nieuprzejme lub agresywne treści, takie jak wulgaryzmy, chociaż nie w takim stopniu, jak nienawiść lub obraza.
    - Może emitować sarkazm poprzez pytanie retoryczne lub ironię.
    - Może obejmować nieetyczne wyrażenie (np. dowcipy lub nieistotne pytania dotyczące zmarłych postaci).
    - Komentarz przekazujący nieokreślone plotki może należeć do tej kategorii.

    # none: Komentarz, który nie zawiera żadnej nienawiści ani obrazy. Taki komentarz powinien być sklasyfikowany jako none.

    Komentarz: {}. Teraz komentarz się kończy.
    Proszę odpowiedzieć "hate", "offensive" lub "none".
    """,

    "Arabic": """
    هذه المهمة هي تخصيص تسميات hate و offensive و none لتعليق على الأخبار على Naver.

    هذا هو معيار التعليق. لاحظ أنه في هذه الدراسة، يمكن أن تكون تعريفات كل فئة مشابهة للدراسات الأخرى في بعض الجوانب، لكنها ليست متطابقة تمامًا. يرجى اتباع التعريف أدناه بعناية.

    # hate: هل يُظهر التعليق كراهية قوية أو إهانة تجاه هدف المقال أو الشخصيات ذات الصلة أو كتّاب المقال أو التعليقات وما إلى ذلك؟ إذا كان الأمر كذلك، فيجب تصنيفه على أنه hate.
    - في حالة الإهانة، يشمل التعبير الذي يمكن أن يلحق ضررًا شديدًا بالمكانة الاجتماعية للمتلقي.
    - في حالة hate، يتم تعريفه على أنه تعبير يظهر مواقف عدوانية تجاه الأفراد / المجموعات ذات الخصائص المعينة (دور الجنس، التوجه الجنسي، الهوية الجنسية، أي أفكار حول الأفعال المتعلقة بالجنس، العرق، الخلفية، الجنسية، المجموعة العرقية، الموقف السياسي، لون البشرة، الدين، الإعاقة، العمر، المظهر، الثروة، المهن، عدم وجود خبرة في الخدمة العسكرية، وما إلى ذلك).
    - بالإضافة إلى ذلك، يمكن أن يشمل التحرش الجنسي، الإبلاغ عن الشائعات المسيئة أو الحقائق، والمصطلحات التي صيغت لأغراض سيئة أو للاستخدام السيئ، وما إلى ذلك.
    - مجرد وجود كلمات سيئة في الوثيقة لا يندرج دائمًا ضمن هذه الفئة.

    # offensive: على الرغم من أن التعليق ليس مسيئًا أو مهينًا كما هو مذكور أعلاه، إلا أنه يجعل الهدف أو القارئ يشعر بالإهانة؟ إذا كان الأمر كذلك، فيجب تصنيفه على أنه offensive.
    - قد يحتوي على محتوى وقح أو عدواني، مثل الكلمات السيئة، على الرغم من أنها ليست بقدر الكراهية أو الإهانة.
    - يمكن أن ينبعث منه سخرية من خلال السؤال البلاغي أو السخرية.
    - قد يشمل تعبيرًا غير أخلاقي (على سبيل المثال، النكات أو الأسئلة غير ذات الصلة المتعلقة بالشخصيات المتوفاة).
    - التعليق الذي ينقل شائعات غير معروفة يمكن أن ينتمي إلى هذه الفئة.

    # none: تعليق لا يحتوي على أي كراهية أو إهانة. بالنسبة لتعليق من هذا النوع، يجب تصنيفه على أنه none.

    التعليق: {}. الآن انتهى التعليق.
    يرجى الرد بـ "hate" أو "offensive" أو "none".
    """,

    "Persian": """
    این کار وظیفه دارد که برچسب‌های hate، offensive و none را به یک نظر در مورد خبری در Naver اختصاص دهد.

    این استاندارد حاشیه نویسی است. توجه داشته باشید که در این مطالعه، تعریف هر دسته ممکن است در برخی جنبه‌ها مشابه مطالعات دیگر باشد، اما دقیقاً یکسان نیست. لطفاً به دقت تعریف زیر را دنبال کنید.

    # hate: آیا نفرت شدید یا توهین نسبت به هدف مقاله یا شخصیت‌های مرتبط، نویسندگان مقاله یا نظرات و غیره در نظر نشان داده می‌شود؟ اگر چنین است، باید به عنوان hate طبقه بندی شود.
    - در صورت توهین، این شامل بیانی است که می‌تواند به شدت به جایگاه اجتماعی گیرنده آسیب برساند.
    - در مورد hate، به عنوان بیانی تعریف می‌شود که مواضع تهاجمی را نسبت به افراد / گروه‌های با ویژگی‌های خاص (نقش جنسیتی، گرایش جنسی، هویت جنسی، هر گونه افکار مربوط به اعمال مرتبط با جنسیت، نژاد، پس زمینه، ملیت، گروه قومی، موضع سیاسی، رنگ پوست، دین، معلولیت، سن، ظاهر، ثروت، مشاغل، عدم تجربه خدمت نظامی و غیره) نشان می‌دهد.
    - علاوه بر این، می‌تواند شامل آزار جنسی، اطلاعیه شایعات یا حقایق توهین‌آمیز، و اصطلاحاتی که برای اهداف بد یا استفاده بد ساخته شده‌اند و غیره باشد.
    - صرفاً وجود کلمات بد در سند همیشه در این دسته قرار نمی‌گیرد.

    # offensive: اگرچه یک نظر به اندازه‌ی بالا نفرت‌انگیز یا توهین‌آمیز نیست، آیا باعث می‌شود هدف یا خواننده احساس توهین کنند؟ اگر چنین است، باید به عنوان offensive طبقه بندی شود.
    - ممکن است حاوی محتوای بی‌ادبانه یا تهاجمی، مانند کلمات بد، باشد، هرچند نه به اندازه‌ی نفرت یا توهین.
    - ممکن است از طریق پرسش بلاغی یا طعنه، کنایه ارسال کند.
    - ممکن است شامل یک بیان غیراخلاقی باشد (به عنوان مثال، شوخی‌ها یا سوالات نامربوط مربوط به شخصیت‌هایی که درگذشته‌اند).
    - نظری که شایعات نامشخصی را منتقل می‌کند، می‌تواند در این دسته قرار گیرد.

    # none: نظری که شامل هیچ نفرت یا توهینی نمی‌شود. برای نظری مانند این، باید به عنوان none طبقه بندی شود.

    نظر: {}. اکنون نظر به پایان می‌رسد.
    لطفاً با "hate"، "offensive" یا "none" پاسخ دهید.
    """,

    "Hebrew": """
    משימה זו היא להקצות תוויות hate, offensive ו-none לתגובה לחדשות ב-Naver.

    זהו תקן ההערה. שים לב שבמחקר זה, ההגדרה של כל קטגוריה עשויה להיות דומה למחקרים אחרים בחלק מהיבטים, אך היא אינה זהה בדיוק. אנא עקוב בזהירות אחר ההגדרה שלהלן.

    # hate: האם בתגובה יש שנאה עזה או עלבון כלפי מושא הכתבה או דמויות קשורות, כותבי הכתבה או תגובות וכדומה? אם כן, יש לסווג אותה כ-hate.
    - במקרה של עלבון, זה מקיף ביטוי שיכול לפגוע קשה במעמד החברתי של הנמען.
    - במקרה של hate, זה מוגדר כביטוי המפגין עמדות תוקפניות כלפי יחידים/קבוצות עם מאפיינים מסוימים (תפקיד מגדרי, נטייה מינית, זהות מינית, מחשבות על מעשים הקשורים למגדר, גזע, רקע, לאום, קבוצה אתנית, עמדה פוליטית, צבע עור, דת, מוגבלות, גיל, מראה, עושר, מקצועות, היעדר ניסיון בשירות צבאי וכו').
    - בנוסף, זה יכול לכלול הטרדה מינית, הודעה על שמועות או עובדות פוגעניות, ומונחים שטבעו למטרות רעות או בשימוש לרעה, וכו'.
    - עצם קיומם של מילים רעות במסמך אינו תמיד נופל לקטגוריה זו.

    # offensive: למרות שתגובה אינה מכילה שנאה או עלבון כמו המפורט לעיל, האם היא גורמת למטרה או לקורא להרגיש פגוע? אם כן, יש לסווג אותה כ-offensive.
    - זה עשוי להכיל תוכן גס או תוקפני, כגון מילים רעות, אם כי לא באותה מידה של שנאה או עלבון.
    - זה יכול לשדר סרקזם באמצעות שאלה רטורית או אירוניה.
    - זה עשוי לכלול ביטוי לא מוסרי (למשל, בדיחות או שאלות לא רלוונטיות לגבי דמויות שנפטרו).
    - תגובה המעבירה שמועות לא מזוהות יכולה להשתייך לקטגוריה זו.

    # none: תגובה שאינה כוללת שנאה או עלבון. לתגובה כזו יש לסווג אותה כ-none.

    תגובה: {}. עכשיו התגובה מסתיימת.
    אנא השב עם "hate", "offensive" או "none".
    """,

    "Turkish": """
    Bu görev, bir Naver haberine yapılan yoruma hate, offensive ve none etiketlerini atamaktır.

    Bu, ek açıklama standardıdır. Bu çalışmada her kategorinin tanımının bazı açılardan diğer çalışmalara benzer olabileceğini, ancak tam olarak aynı olmadığını unutmayın. Lütfen aşağıdaki tanımı dikkatlice takip edin.

    # hate: Bir yorumda, makalenin hedefi veya ilgili figürler, makalenin veya yorumların yazarları vb. hakkında güçlü bir nefret veya hakaret var mı? Öyleyse, hate olarak sınıflandırılmalıdır.
    - Hakaret durumunda, alıcının sosyal statüsüne ciddi şekilde zarar verebilecek bir ifadeyi kapsar.
    - Nefret durumunda, belirli özelliklere sahip bireyler/gruplar karşısında agresif duruşlar sergileyen bir ifade olarak tanımlanır (cinsiyet rolü, cinsel yönelim, cinsel kimlik, cinsiyetle ilgili eylemlerle ilgili herhangi bir düşünce, ırk, arka plan, milliyet, etnik grup, politik duruş, ten rengi, din, engeller, yaş, görünüm, zenginlik, meslekler, askerlik hizmeti deneyiminin olmaması vb.).
    - Ayrıca, cinsel taciz, saldırgan söylentilerin veya gerçeklerin bildirilmesi ve kötü amaçlar için uydurulmuş veya kötü kullanımda olan terimler gibi durumları içerebilir.
    - Belgede kötü kelimelerin varlığı her zaman bu kategoriye girmez.

    # offensive: Bir yorum yukarıdaki kadar nefret dolu veya hakaret içermese de hedefi veya okuyucuyu rahatsız ediyor mu? Öyleyse, offensive olarak sınıflandırılmalıdır.
    - Hakaret veya nefret düzeyinde olmasa da kaba veya saldırgan içerik içerebilir, örneğin kötü kelimeler.
    - Retorik bir soru veya ironi yoluyla alaycı bir ton yansıtabilir.
    - Ahlaki olmayan bir ifadeyi içerebilir (örneğin, ölen kişilerle ilgili şakalar veya alakasız sorular).
    - Tanımlanamayan söylentileri aktaran bir yorum bu kategoriye girebilir.

    # none: Herhangi bir nefret veya hakaret içermeyen bir yorum. Bu tür bir yorum none olarak sınıflandırılmalıdır.

    Yorum: {}. Şimdi yorum sona eriyor.
    Lütfen "hate", "offensive" veya "none" ile yanıtlayın.
    """,

    "Japanese": """
    このタスクは、Naverのニュースに対するコメントに対して「hate」「offensive」「none」のラベルを割り当てることです。

    これは注釈標準です。この研究では、各カテゴリの定義は他の研究といくつかの点で似ている可能性がありますが、正確に同じではないことに注意してください。以下の定義に注意深く従ってください。

    # hate: コメントで記事のターゲットや関連人物、記事やコメントの執筆者などに対して強い憎しみや侮辱が示されていますか？ もしそうなら、それはhateとして分類されるべきです。
    - 侮辱の場合、それは受取人の社会的地位に深刻な損害を与える可能性のある表現を含みます。
    - hateの場合、それは特定の特徴を持つ個人/グループに対する攻撃的な立場を示す表現として定義されます（ジェンダーロール、性的指向、性的アイデンティティ、ジェンダー関連の行為に関する考え、人種、背景、国籍、民族、政治的立場、肌の色、宗教、障害、年齢、外見、富、職業、軍務経験の欠如など）。
    - さらに、それはセクシャルハラスメント、攻撃的な噂や事実の通知、悪意のある目的または誤った使用のための造語などを含むことができます。
    - ドキュメント内に悪い言葉が存在するだけでは、常にこのカテゴリに該当するとは限りません。

    # offensive: コメントが上記ほど憎しみや侮辱に満ちていなくても、ターゲットや読者を不快にさせますか？ もしそうなら、それはoffensiveとして分類されるべきです。
    - 憎しみや侮辱の程度に至らなくても、悪い言葉など、無礼で攻撃的な内容を含む可能性があります。
    - それは修辞疑問や皮肉を通じて皮肉を発することができます。
    - それは倫理に反する表現を含む可能性があります（例：死亡した人物に関するジョークや無関係な質問）。
    - 未確認の噂を伝えるコメントは、このカテゴリに該当する可能性があります。

    # none: 憎しみや侮辱を含まないコメント。 このようなコメントはnoneとして分類されるべきです。

    コメント: {}。 今コメントが終了します。
    "hate"、"offensive"、または "none"で回答してください。
    """,

    "Korean": """
    이 작업은 네이버 뉴스의 댓글에 대해 hate, offensive, none 레이블을 할당하는 것입니다.

    이것은 주석 표준입니다. 이 연구에서 각 범주의 정의는 다른 연구와 몇 가지 측면에서 유사할 수 있지만 정확히 동일하지는 않습니다. 아래 정의를 주의 깊게 따르세요.

    # hate: 댓글에서 기사 대상 또는 관련 인물, 기사나 댓글 작성자 등에 대해 강한 혐오감이나 모욕감을 표현하고 있습니까? 그렇다면 hate로 분류해야 합니다.
    - 모욕의 경우, 수신자의 사회적 지위를 심각하게 손상시킬 수 있는 표현을 포함합니다.
    - hate의 경우, 특정 특성을 가진 개인/그룹에 대한 공격적인 태도를 보여주는 표현으로 정의됩니다(성 역할, 성적 지향, 성 정체성, 성 관련 행위에 대한 생각, 인종, 배경, 국적, 민족, 정치적 입장, 피부색, 종교, 장애, 나이, 외모, 부유함, 직업, 군 복무 경험의 부족 등).
    - 또한 성희롱, 모욕적인 소문이나 사실의 알림, 나쁜 목적이나 잘못된 사용을 위한 조어 등을 포함할 수 있습니다.
    - 문서에 나쁜 단어가 존재한다고 해서 항상 이 범주에 속하는 것은 아닙니다.

    # offensive: 댓글이 위의 것처럼 혐오스럽거나 모욕적이지 않더라도 대상이나 독자가 불쾌감을 느끼게 합니까? 그렇다면 offensive로 분류해야 합니다.
    - 혐오나 모욕의 정도에 이르지 않더라도 나쁜 단어와 같은 무례하거나 공격적인 내용을 포함할 수 있습니다.
    - 수사적 질문이나 아이러니를 통해 풍자를 발할 수 있습니다.
    - 비윤리적 표현을 포함할 수 있습니다(예: 사망한 인물에 대한 농담이나 관련 없는 질문).
    - 확인되지 않은 소문을 전달하는 댓글은 이 범주에 속할 수 있습니다.

    # none: 혐오나 모욕을 포함하지 않은 댓글. 이러한 댓글은 none으로 분류해야 합니다.

    댓글: {}. 이제 댓글이 끝납니다.
    "hate", "offensive" 또는 "none"으로 답변해 주세요.
    """,

    "Vietnamese": """
    Nhiệm vụ này là gán các nhãn hate, offensive và none cho một bình luận về một tin tức trên Naver.

    Đây là tiêu chuẩn chú thích. Lưu ý rằng trong nghiên cứu này, định nghĩa của từng danh mục có thể giống với các nghiên cứu khác ở một số khía cạnh, nhưng nó không hoàn toàn giống nhau. Vui lòng cẩn thận theo dõi định nghĩa dưới đây.

    # hate: Có sự thù hận mạnh mẽ hoặc xúc phạm đến mục tiêu của bài viết hoặc các nhân vật liên quan, các tác giả của bài viết hoặc bình luận, v.v. được hiển thị trong một bình luận không? Nếu vậy, nó nên được phân loại là hate.
    - Trong trường hợp xúc phạm, nó bao gồm một biểu hiện có thể gây hại nghiêm trọng đến địa vị xã hội của người nhận.
    - Trong trường hợp thù hận, nó được định nghĩa là một biểu hiện thể hiện lập trường hung hăng đối với các cá nhân / nhóm có những đặc điểm nhất định (vai trò giới, xu hướng tình dục, bản sắc giới, bất kỳ suy nghĩ nào về các hành vi liên quan đến giới tính, chủng tộc, nền tảng, quốc tịch, nhóm sắc tộc, lập trường chính trị, màu da, tôn giáo, khuyết tật, tuổi tác, ngoại hình, sự giàu có, nghề nghiệp, thiếu kinh nghiệm phục vụ quân đội, v.v.).
    - Ngoài ra, nó có thể bao gồm quấy rối tình dục, thông báo về những tin đồn hoặc sự kiện xúc phạm, và các thuật ngữ được sử dụng cho mục đích xấu hoặc sử dụng sai mục đích, v.v.
    - Chỉ riêng sự tồn tại của những lời lẽ xấu trong tài liệu không phải lúc nào cũng thuộc vào loại này.

    # offensive: Mặc dù một bình luận không đến mức hận thù hoặc xúc phạm như trên, nhưng liệu nó có khiến mục tiêu hoặc người đọc cảm thấy bị xúc phạm không? Nếu vậy, nó nên được phân loại là offensive.
    - Nó có thể chứa nội dung thô lỗ hoặc hung hăng, chẳng hạn như những lời lẽ không tốt, mặc dù không đến mức hận thù hoặc xúc phạm.
    - Nó có thể phát ra sự mỉa mai thông qua câu hỏi tu từ hoặc châm biếm.
    - Nó có thể bao gồm một biểu hiện phi đạo đức (ví dụ, những câu chuyện cười hoặc câu hỏi không liên quan đến các nhân vật đã qua đời).
    - Một bình luận truyền tải những tin đồn không xác định có thể thuộc vào loại này.

    # none: Một bình luận không bao gồm bất kỳ sự hận thù hoặc xúc phạm nào. Đối với một bình luận như thế này, nó nên được phân loại là none.

    Bình luận: {}. Bây giờ bình luận kết thúc.
    Vui lòng phản hồi với "hate", "offensive" hoặc "none".
    """,

    "Thai": """
    งานนี้คือการกำหนดป้ายกำกับ hate, offensive และ none ให้กับความคิดเห็นเกี่ยวกับข่าวใน Naver

    นี่คือมาตรฐานการใส่คำอธิบาย โปรดทราบว่าในงานศึกษานี้ การกำหนดของแต่ละหมวดหมู่อาจคล้ายคลึงกับงานศึกษาอื่นๆ ในบางประการ แต่ไม่เหมือนกันทั้งหมด โปรดปฏิบัติตามคำนิยามด้านล่างอย่างระมัดระวัง

    # hate: ความเกลียดชังอย่างรุนแรงหรือการดูถูกเป้าหมายของบทความหรือบุคคลที่เกี่ยวข้อง ผู้เขียนบทความหรือความคิดเห็น ฯลฯ แสดงออกในความคิดเห็นหรือไม่? หากเป็นเช่นนั้น ควรจัดประเภทเป็น hate
    - ในกรณีที่มีการดูถูก มันครอบคลุมการแสดงออกที่อาจเป็นอันตรายต่อสถานะทางสังคมของผู้รับอย่างรุนแรง
    - ในกรณีของ hate กำหนดเป็นการแสดงออกที่แสดงท่าทีเชิงรุกต่อบุคคล/กลุ่มที่มีลักษณะบางอย่าง (บทบาททางเพศ รสนิยมทางเพศ อัตลักษณ์ทางเพศ ความคิดเกี่ยวกับการกระทำที่เกี่ยวข้องกับเพศ เชื้อชาติ ภูมิหลัง สัญชาติ กลุ่มชาติพันธุ์ จุดยืนทางการเมือง สีผิว ศาสนา ความพิการ อายุ ลักษณะภายนอก ความร่ำรวย อาชีพ การไม่มีประสบการณ์รับราชการทหาร ฯลฯ)
    - นอกจากนี้อาจรวมถึงการล่วงละเมิดทางเพศ การแจ้งข่าวลือหรือข้อเท็จจริงที่เป็นการล่วงละเมิด และคำศัพท์ที่ประกาศใช้เพื่อจุดประสงค์ที่ไม่ดีหรือใช้งานในทางที่ผิด ฯลฯ
    - เพียงแค่การมีอยู่ของคำหยาบคายในเอกสารไม่ได้หมายความว่าเป็นประเภทนี้เสมอไป

    # offensive: แม้ว่าความคิดเห็นจะไม่เต็มไปด้วยความเกลียดชังหรือการดูถูกเหมือนข้างต้น แต่ทำให้เป้าหมายหรือผู้อ่านรู้สึกไม่พอใจหรือไม่? ถ้าใช่ ควรจัดประเภทเป็น offensive
    - อาจมีเนื้อหาที่หยาบคายหรือก้าวร้าว เช่น คำหยาบ แม้ว่าจะไม่ถึงขนาดเกลียดชังหรือดูหมิ่นก็ตาม
    - สามารถปล่อยการเสียดสีผ่านคำถามเชิงวาทศิลป์หรือการประชดประชันได้
    - อาจรวมถึงการแสดงออกที่ผิดจรรยาบรรณ (เช่น เรื่องตลกหรือคำถามที่ไม่เกี่ยวข้องเกี่ยวกับบุคคลที่เสียชีวิต)
    - ความคิดเห็นที่ถ่ายทอดข่าวลือที่ไม่ปรากฏชื่ออาจอยู่ในประเภทนี้

    # none: ความคิดเห็นที่ไม่มีความเกลียดชังหรือดูถูก สำหรับความคิดเห็นแบบนี้ ควรจัดประเภทเป็น none

    ความคิดเห็น: {}. ตอนนี้ความคิดเห็นจบลงแล้ว
    โปรดตอบกลับด้วย "hate", "offensive" หรือ "none"
    """,

    "Indonesian": """
    Tugas ini adalah untuk menetapkan label hate, offensive, dan none pada komentar berita di Naver.

    Ini adalah standar anotasi. Perhatikan bahwa dalam penelitian ini, definisi setiap kategori mungkin mirip dengan penelitian lain dalam beberapa aspek, tetapi tidak persis sama. Harap ikuti definisi di bawah ini dengan cermat.

    # hate: Apakah ada kebencian atau penghinaan yang kuat terhadap target artikel atau tokoh terkait, penulis artikel atau komentar, dll. yang ditampilkan dalam komentar? Jika demikian, itu harus diklasifikasikan sebagai hate.
    - Dalam kasus penghinaan, ini mencakup ekspresi yang dapat sangat merusak status sosial penerima.
    - Dalam kasus hate, itu didefinisikan sebagai ekspresi yang menunjukkan sikap agresif terhadap individu/kelompok dengan karakteristik tertentu (peran gender, orientasi seksual, identitas seksual, pemikiran tentang tindakan terkait gender, ras, latar belakang, kebangsaan, kelompok etnis, sikap politik, warna kulit, agama, cacat, usia, penampilan, kekayaan, pekerjaan, kurangnya pengalaman dinas militer, dll.).
    - Selain itu, dapat mencakup pelecehan seksual, pemberitahuan rumor atau fakta yang menyinggung, dan istilah yang diciptakan untuk tujuan buruk atau dalam penggunaan yang buruk, dll.
    - Keberadaan kata-kata buruk dalam dokumen tidak selalu masuk dalam kategori ini.

    # offensive: Meskipun komentar tidak seburuk kebencian atau penghinaan di atas, apakah itu membuat target atau pembaca merasa tersinggung? Jika demikian, itu harus diklasifikasikan sebagai offensive.
    - Ini mungkin berisi konten kasar atau agresif, seperti kata-kata kasar, meskipun tidak sampai pada kebencian atau penghinaan.
    - Itu dapat mengeluarkan sarkasme melalui pertanyaan retoris atau ironi.
    - Ini dapat mencakup ekspresi tidak etis (misalnya, lelucon atau pertanyaan yang tidak relevan tentang tokoh yang telah meninggal).
    - Komentar yang menyampaikan rumor yang tidak teridentifikasi dapat termasuk dalam kategori ini.

    # none: Komentar yang tidak mengandung kebencian atau penghinaan. Untuk komentar seperti ini, harus diklasifikasikan sebagai none.

    Komentar: {}. Sekarang komentar berakhir.
    Silakan jawab dengan "hate", "offensive", atau "none".
    """,

    "Malay": """
    Tugas ini adalah untuk memberikan label hate, offensive, dan none kepada komen mengenai berita di Naver.

    Ini adalah standard anotasi. Perhatikan bahawa dalam kajian ini, definisi setiap kategori mungkin serupa dengan kajian lain dalam beberapa aspek, tetapi ia tidak sama sepenuhnya. Sila ikuti definisi di bawah ini dengan teliti.

    # hate: Adakah kebencian yang kuat atau penghinaan terhadap sasaran artikel atau tokoh-tokoh yang berkaitan, penulis artikel atau komen, dll. yang dipaparkan dalam komen? Jika ya, ia harus diklasifikasikan sebagai hate.
    - Dalam kes penghinaan, ia merangkumi ekspresi yang boleh menjejaskan status sosial penerima dengan teruk.
    - Dalam kes hate, ia ditakrifkan sebagai ekspresi yang mempamerkan pendirian agresif terhadap individu/kumpulan dengan ciri-ciri tertentu (peranan gender, orientasi seksual, identiti seksual, sebarang pemikiran mengenai tindakan berkaitan gender, kaum, latar belakang, kewarganegaraan, kumpulan etnik, pendirian politik, warna kulit, agama, cacat, umur, penampilan, kekayaan, pekerjaan, kekurangan pengalaman perkhidmatan tentera, dll.).
    - Selain itu, ia boleh termasuk gangguan seksual, pemberitahuan khabar angin atau fakta yang menghina, dan istilah yang dicipta untuk tujuan buruk atau dalam penggunaan yang salah, dll.
    - Hanya kewujudan kata-kata buruk dalam dokumen tidak selalu termasuk dalam kategori ini.

    # offensive: Walaupun komen tidak sebenci atau menghina seperti di atas, adakah ia membuatkan sasaran atau pembaca berasa tersinggung? Jika ya, ia harus diklasifikasikan sebagai offensive.
    - Ia mungkin mengandungi kandungan kasar atau agresif, seperti kata-kata kasar, walaupun tidak sehingga kebencian atau penghinaan.
    - Ia boleh mengeluarkan sindiran melalui soalan retorik atau ironi.
    - Ia boleh merangkumi ekspresi tidak beretika (contohnya, jenaka atau soalan yang tidak relevan mengenai tokoh-tokoh yang telah meninggal dunia).
    - Komen yang menyampaikan khabar angin yang tidak dikenali boleh tergolong dalam kategori ini.

    # none: Komen yang tidak menggabungkan apa-apa kebencian atau penghinaan. Untuk komen seperti ini, ia harus diklasifikasikan sebagai none.

    Komen: {}. Sekarang komen berakhir.
    Sila jawab dengan "hate", "offensive", atau "none".
    """,

    "Lao": """
    ວຽກນີ້ແມ່ນເພື່ອກຳນົດປ້າຍຊື່ hate, offensive, ແລະ none ໃຫ້ກັບຄວາມຄິດເຫັນກ່ຽວກັບຂ່າວບໍລິການ Naver.

    ນີ້ແມ່ນມາດຕະຖານການກຳນົດ. ຂໍສັງເກດວ່າໃນການສຶກສານີ້, ການນິຍາມຂອງແຕ່ລະໝວດສາມາດມີຄວາມຄືກັນກັບການສຶກສາອື່ນໆໃນບາງດ້ານ, ແຕ່ບໍ່ແມ່ນທີ່ຈະຄືກັນທັງໝົດ. ກະລຸນາຕິດຕາມນິຍາມຂ້າງລຸ່ມຢ່າງລະມັດລະວັງ.

    # hate: ຄຳວ່າຄວາມຄິດເຫັນທີ່ມີຄວາມເກລີຍດຊັງ ຫຼືດູຖູກເປົ້າໝາຍຂອງບົດຄວາມ ຫຼືບຸກຄົນທີ່ມີຄວາມກ່ຽວຂ້ອງ, ຜູ້ແຕ່ງບົດຄວາມ ຫຼືຄົນທີ່ຂຽນຄວາມຄິດເຫັນ, ແລ້ວໃນຄວາມຄິດເຫັນນີ້ເປັນແບບນີ້ຫຼືບໍ່? ຖ້າເປັນແບບນັ້ນ ກໍຄວນຈະຈັດປະເພດເປັນ hate.
    - ໃນກໍລະນີຂອງຄວາມດູຖູກ, ມັນຄືການແສດສະແດງຄວາມເຫັນທີ່ສາມາດທຳລາຍສະຖານະສັງຄົມຂອງຜູ້ຮັບໄດ້ຢ່າງຮ້າຍແຮງ.
    - ໃນກໍລະນີຂອງ hate, ມັນຄືການແສດສະແດງຄວາມເຫັນທີ່ມີການໂຈມຕີຢ່າງຮຸນແຮງຕໍ່ບຸກຄົນ/ກຸ່ມຄົນທີ່ມີຄຸນສົມບັດບາງຢ່າງ (ພາວະເພດ, ລັກສະນະທາງເພດ, ລັກສະນະທາງເພດ, ການຄິດເຖິງການກະທຳກ່ຽວກັບເພດ, ຊົນເຜົ່າ, ຄວາມເປັນມາ, ສັນຊາດ, ກຸ່ມເຜົ່າ, ຈຸດຍືນທາງການເມືອງ, ສີຜິວ, ສາສະໜາ, ຄວາມພິການ, ອາຍຸ, ໜ້າຕາ, ຄວາມຮັ່ງມີ, ອາຊີບ, ການບໍ່ມີປະສົບການກັບການຮັບໃຊ້ເທັບທະຫານ ແລະອື່ນໆ).
    - ຍິ່ງໄປກວ່ານັ້ນ, ອາດລວມເອົາການລວງລະເມີດທາງເພດ, ການແຈ້ງຂ່າວລືທີ່ລົງຂ່າວຫຼືຄວາມເປັນຄວາມຈິງທີ່ລວງລະເມີດ, ຄໍາສັບທີ່ຖືກສ້າງຂຶ້ນເພື່ອຈຸດປະສົງບໍ່ດີຫຼືການນຳໃຊ້ທີ່ບໍ່ຖືກຕ້ອງ.
    - ພຽງແຕ່ການມີຢູ່ຂອງຄໍາຫຍາບໃນເອກະສານບໍ່ໄດ້ໝາຍຄວາມວ່າຄວນຈັດຢູ່ໝວດນີ້ເສມໍ່ໄປ.

    # offensive: ແມ່ນຫວ່າງຄວາມຄິດເຫັນນີ້ບໍ່ມີຄວາມເກລີຍດຊັງເຫມືອນເທິງໆນີ້, ແຕ່ມັນເຮັດໃຫ້ເປົ້າຫມາຍ ຫຼືຜູ້ອ່ານຮູ້ສຶກບໍ່ສະບາຍຫຼືບໍ່? ຖ້າເປັນແບບນັ້ນກໍຄວນຈະຈັດປະເພດເປັນ offensive.
    - ມັນອາດມີເນື້ອຫາທີ່ບໍ່ດີ ຫຼືຮຸນແຮງ, ເຊັ່ນຄໍາຫຍາບໆ, ແມ່ນວ່າມັນຈະບໍ່ໄດ້ຄົງຢູ່ໃນຄວາມເກລີຍດຊັງ ຫຼືການດູຖູກເທົ່ານັ້ນກໍຕາມ.
    - ມັນສາມາດເຫັນເປັນການເສຍດີເດັ່ນຂຶ້ນມາດ້ວຍຄຳຖາມຊັກເລກ ຫຼືການຫົວເສົາ.
    - ມັນສາມາດລວມເອົາການແສດສະແດງທີ່ບໍ່ມີສິນທຳ (ເຊັ່ນຕະລາດ ຫຼືຄຳຖາມບໍ່ທີ່ສຳຄັນກ່ຽວກັບບຸກຄົນທີ່ເສຍຊີວິດ).
    - ຄວາມຄິດເຫັນທີ່ແຈ້ງຂ່າວລືທີ່ບໍ່ຊັດເຈນອາດຈະຈັດປະເພດນີ້ໄດ້.

    # none: ຄວາມຄິດເຫັນທີ່ບໍ່ມີຄວາມເກລີຍດຊັງ ຫຼືຄວາມດູຖູກ. ສຳລັບຄວາມຄິດເຫັນທີ່ຄືກັນນີ້, ມັນຄວນຈະຖືກຈັດເປັນ none.

    ຄຳເຫັນ: {}. ຄຳເຫັນດັ່ງກ່າວສິ້ນສຸດແລ້ວ.
    ກະລຸນາຕອບກັບ "hate", "offensive", ຫຼື "none".
    """,

    "Burmese": """
    ဒီတာဝန်မှာ Naver ရဲ့ သတင်းတစ်ပုဒ်မှာ ပါဝင်တဲ့ မတူကွဲပြားမှုတွေကို hate, offensive, none ဆိုတဲ့ label တွေ သတ်မှတ်ပေးရမှာ ဖြစ်ပါတယ်။

    ဒါက annotation standard ဖြစ်ပါတယ်။ ဒီလေ့လာမှုမှာ အမျိုးအစားတိုင်းရဲ့ အဓိပ္ပါယ်က အခြားလေ့လာမှုတွေနဲ့ တချို့ aspect တွေအရ တူညီတဲ့အချက်တွေလည်း ရှိနိုင်ပါတယ်။ ဒါပေမယ့် အတိအကျတော့ မတူပါဘူး။ အောက်မှာ ဖော်ပြထားတဲ့ အဓိပ္ပါယ်တွေကို ဂရုတစိုက်နာပါ။

    # hate: မုန်းတီးမှုတစ်ခုဟာ target article အားဖြင့်ဖြစ်မိပါသလား။ ဒါမှမဟုတ် ကွန်မင့်တစ်ခုမှာ မတူကွဲပြားမှုကိုပါ ပဋိပက္ခဖြစ်အောင် ဆောင်ရွက်မှုတွေပါသလား။ အဲဒီအခါ hate လို့ မှတ်သားရပါမယ်။
    - ဒီပြသနာတွေဟာ လူ့အသိုင်းအဝိုင်းတည်နေရာတွေကို ထိခိုက်နိုင်တဲ့အမှုတွေလည်း ပါပါသည်။
    - ဥပမာ: လူမှုဆက်ဆံရေးစနစ်နဲ့ မနီးမစပ်တဲ့ အထူးအချက်အလက်တွေပါပါသည်။
    - ဒါ့အပြင် ဒီလိုအမှုတွေမှာ ဖော်ပြချက်တွေထဲမှာ စစ်မှန်တဲ့အသုံးအနှုန်းတွေကိုလည်း အသုံးပြုပါတယ်။

    # offensive: offensive လို့ မှတ်သားဖို့လိုတဲ့အခါ target comment ထဲမှာ သတိထားဖို့လိုတဲ့မူများလည်း ပါလာနိုင်သည်။ offensive မှာ လှုံ့ဆော်စရာ အကြောင်းအရာတွေ ပါဝင်နိုင်တယ်။
    - သတိထားစရာရောက်တဲ့ offensive စကားလုံးတွေကို ပြောဆိုလာရတဲ့ အခါမှာ offensive လို့ မှတ်တမ်းတင်ရပါမယ်။
    - offensive သည် ဖော်ပြချက်များပြည့်စုံပါသည်။

    # none: သဘောထားအဖြစ် ထိခိုက်စရာမပါဝင်ပါက none လို့ မှတ်တမ်းတင်ပါမယ်။

    ကောက်ချက်: {}. အခါအကျေနပ်စွာ ပြင်ဆင်ပြီးစီးသည်။
    ကျေးဇူးပြု၍ "hate", "offensive", ဒါမှမဟုတ် "none" ဖြင့် ပြန်လည်တုံ့ပြန်ပါ။
    """,

    "Cebuano": """
    Kini nga buluhaton mao ang paghatag ug hate, offensive, ug none nga mga label sa usa ka komento sa usa ka balita sa Naver.

    Kini mao ang annotation standard. Timan-e nga sa niini nga pagtuon, ang kahulogan sa matag kategoriya mahimong managsama sa ubang mga pagtuon sa pipila ka mga aspeto, apan dili kini eksaktong parehas. Palihug sundon ug maayo ang kahulogan sa ubos.

    # hate: Aduna bay kusog nga kasuko o pagpasipala sa tumong sa artikulo o mga may kalabotan nga mga tawo, mga magsusulat sa artikulo o mga komento, ug uban pa nga gipakita sa usa ka komento? Kung mao, kini kinahanglan nga iklasipikar isip hate.
    - Sa kaso sa pagpasipala, kini naglakip sa usa ka ekspresyon nga mahimo’g grabe nga makadaot sa sosyal nga kahimtang sa nakadawat.
    - Sa kaso sa hate, kini gihubit nga usa ka ekspresyon nga nagpakita og agresibo nga mga pamatasan batok sa mga indibidwal/mga grupo nga adunay mga piho nga kinaiya (gender role, sexual orientation, sexual identity, bisan unsang hunahuna sa mga buhat nga may kalabutan sa gender, lahi, background, nationality, ethnic group, political stance, skin color, religion, handicaps, age, appearance, richness, occupations, ang kakulang sa kasinatian sa military service, ug uban pa).
    - Dugang pa, kini mahimo’g maglakip sa sexual harassment, abiso sa mga pasipala nga tsismis o mga kamatuoran, ug mga naugmad nga mga termino alang sa daotan nga mga katuyoan o sa sayop nga paggamit, ug uban pa.
    - Ang paglungtad lamang sa mga daotan nga mga pulong sa dokumento dili kanunay nga mahulog sa kini nga kategoriya.

    # offensive: Bisan tuod usa ka komento dili ingon ka hate o pagpasipala sama sa itaas, nagmugna ba kini sa tumong o ang magbabasa nga mobati og kasuko? Kung mao, kini kinahanglan nga iklasipikar isip offensive.
    - Kini mahimong maglakip sa bastos o agresibo nga sulud, sama sa daotan nga mga pulong, bisan tuod dili sa giladmon sa hate o pagpasipala.
    - Kini mahimong magpagawas og sarcasm pinaagi sa pangutana nga rhetorical o irony.
    - Kini mahimong maglakip sa usa ka unethical nga ekspresyon (pananglitan, mga joke o dili angay nga mga pangutana bahin sa mga tawo nga namatay).
    - Usa ka komento nga nagpadala og dili mailhan nga tsismis mahimo nga mahulog sa kini nga kategoriya.

    # none: Usa ka komento nga wala naglakip sa bisan unsang hate o pagpasipala. Alang sa usa ka komento sama niini, kini kinahanglan nga iklasipikar isip none.

    Komento: {}. Karong ang komento mohunong.
    Palihug pagtubag og "hate", "offensive", o "none".
    """,

    "Khmer": """
    លទ្ធកម្មនេះគឺដើម្បីផ្ដល់នូវស្លាក hate, offensive និង none ចំពោះមតិមួយក្នុងសារព័ត៌មាននៅលើ Naver។

    នេះជាស្តង់ដារបន្ថែមតែមួយ។ ចំណាំថាក្នុងសិក្សានេះ ការជំរុញនៃប្រភេទនីមួយៗអាចស្រដៀងទៅនឹងសិក្សាផ្សេងៗជាច្រើនតែមិនស្មើគ្នាទាំងស្រុង។ សូមបន្ថែមតាមការណែនាំខាងក្រោមយ៉ាងប្រុងប្រយ័ត្ន។

    # hate: តើមានការរិះគន់ដ៏ខ្លាំងក្លាឬការប្រមាថដ៏រឹងមាំចំពោះគោលដៅនៃអត្ថបទនេះ ឬនានាបុគ្គលដែលពាក់ព័ន្ធ មនុស្សដែលបង្កើតអត្ថបទនោះ ឬមតិរបស់នោះ មាននៅក្នុងមតិនេះទេ? ប្រសិនបើបាទ វាគួរតែត្រូវបានចាត់ទុកជាភាពគេងងប់លើ។
    - ជាការគំរាមកំហែង វាគ្របដណ្តប់នូវការដែលអាចបង្កគ្រោះថ្នាក់យ៉ាងខ្លាំងដល់អំណាចសង្គមរបស់មនុស្សដែលកំពុងទទួលនូវឆន្ទៈនិងពត៌មានបែបនោះ។
    - ជាសកម្មភាពកាចវា គឺជាការប្រែក្លាយនៃមោទនភាពនិងការអត់ពន្លឺទៅតាមគំនិតនានាដែលពាក់ព័ន្ធនិងសកម្មភាពកាចមួយចំនួន ទៅលើបុគ្គល ឬក្រុមបុគ្គលមួយចំនួនដែលមានលក្ខណៈសម្បត្តិពិសេសដែលមនុស្សគួរតែគោរពក្នុងសង្គម សម្បត្តិទាំងនេះរួមមានតួនាទីនៃស្ត្រី, និយមស្រឡាញ់ភេទដូចគ្នា, តំណាងគុណភាពស្រីបុរស, និងមាតិកាដែលគួរតែគោរពនិងរៀបចំកោតក្នុងចំណែកនៃភេទ និង សកម្មភាពសង្ឃឹមទៅជាភាពកិច្ចការមួយចំនួន។
    - លើសពីនេះ វាបន្ថែមនូវការលួចប្រកាសពត៌មានបញ្ចូលកម្លាំងនិងពិតបញ្ចូលបញ្ញាប់ទាក់ទងទៅនិងពាក្យទិន្នន័យដែលបង្កើតឡើងសម្រាប់គោលបំណងអាក្រក់ឬឆ្គាមឆ្ងល់យ៉ាងខ្លាំង។
    - ការមាននូវពាក្យដែលមានមេរ៉ាអាក្រក់ក្នុងឯកសារមិនត្រូវបានគេចាត់ទុកជាពាក្យដ៏ខ្លាំងក្រោមមាតិកានេះទេ។

    # offensive: ទោះបីជាមតិមិនខ្លាំងក្លានិងគួរអោយខ្លាចដូចបានលើកឡើងខាងលើទេ តែវានាំឱ្យគោលដៅ ឬមនុស្សដែលអានឆន្ទៈនោះ មានអារម្មណ៍ថាអ្នកកំពុងប្រមាថដែរឬទេ? ប្រសិនបើបាទ វាគួរតែត្រូវបានចាត់ទុកជា offensive។
    - វាអាចរួមមានអំពើសម្បូរនិងគុណភាពដែលចង់បានដូចបញ្ហាប្រហារអាចមានពាក្យសរសេរ អញ្ចឹងវាអាចនាំទៅនឹងពាក្យពន្យល់សាកល្បង ដែលសមស្របតាមមនុស្សដែលមានទំនុកចិត្តយ៉ាងខ្លាំង។
    - វាអាចបញ្ចេញកម្លាំងបញ្ញាដោយគួរពន្យល់ពីការសាកល្បងសាកសង្វាង ដែលមនុស្សបញ្ចេញនូវការសាកល្បងបញ្ចូលបញ្ញា ដើម្បីគួរពន្យល់ថាមានកម្លាំងបញ្ញាទៅលើមនុស្សដែលមានឈ្មោះសូមបង្វែរកន្លែងសាកល្បងបញ្ចូលបញ្ញា។
    - វាអាចនិងលោភឡំរំដែលមនុស្សបានសាកល្បងសាកសង្វាងទៅលើមនុស្សដែលបានស្លាប់ទៅហើយស្លាប់។

    # none: មតិដែលមិនបញ្ចូលភាពគេងងប់លើនិងគម្របដាច់។ សម្រាប់មតិនេះ វាគួរតែត្រូវបានចាត់ទុកជា none។

    មតិ: {}. ឥឡូវនេះមតិបានបញ្ចប់។
    សូមបញ្ចូលចម្លើយជាមួយនិង "hate", "offensive", ឬ "none"។
    """,

    "Tagalog": """
    Ang gawain na ito ay magtalaga ng mga label na hate, offensive, at none sa isang komento sa balita sa Naver.

    Ito ang pamantayan sa pag-annotate. Tandaan na sa pag-aaral na ito, ang kahulugan ng bawat kategorya ay maaaring maging katulad ng sa ibang pag-aaral sa ilang aspeto, ngunit hindi ito eksaktong pareho. Mangyaring maingat na sundin ang kahulugan sa ibaba.

    # hate: Mayroon bang matinding pagkamuhi o pang-iinsulto patungo sa paksa ng artikulo o kaugnay na mga tauhan, mga manunulat ng artikulo o mga komento, atbp. na ipinapakita sa isang komento? Kung oo, ito ay dapat iklasipika bilang hate.
    - Sa kaso ng pang-iinsulto, sakop nito ang isang pagpapahayag na maaaring lubos na makasira sa panlipunang katayuan ng tumanggap.
    - Sa kaso ng hate, ito ay tinutukoy bilang isang pagpapahayag na nagpapakita ng agresibong mga posisyon patungo sa mga indibidwal/grupo na may ilang katangian (tungkulin ng kasarian, oryentasyong sekswal, pagkakakilanlang sekswal, anumang kaisipan tungkol sa mga kilos na may kaugnayan sa kasarian, lahi, pinagmulan, nasyonalidad, pangkat etniko, paninindigan sa politika, kulay ng balat, relihiyon, kapansanan, edad, hitsura, kayamanan, trabaho, kawalan ng karanasan sa serbisyo militar, atbp.).
    - Bukod pa rito, maaari itong magsama ng panggigipit sa sekswal, pagpapahayag ng mga nakakasakit na tsismis o katotohanan, at mga salitang binuo para sa masamang layunin o sa maling paggamit, atbp.
    - Ang simpleng pag-iral ng mga masamang salita sa dokumento ay hindi palaging kasama sa kategoryang ito.

    # offensive: Bagaman ang isang komento ay hindi kasing puno ng galit o pang-iinsulto gaya ng nasa itaas, pinaparamdam ba nito sa target o sa mambabasa ang pagkasakit ng loob? Kung oo, ito ay dapat iklasipika bilang offensive.
    - Maaari itong maglaman ng magaspang o agresibong nilalaman, tulad ng mga masamang salita, bagaman hindi sa antas ng galit o pang-iinsulto.
    - Maaari itong maglabas ng sarkasmo sa pamamagitan ng retorikal na tanong o pang-uuyam.
    - Maaari itong magsama ng hindi etikal na pagpapahayag (hal., mga biro o hindi kaugnay na mga tanong tungkol sa mga pumanaw na tauhan).
    - Ang isang komento na nagpapahayag ng hindi kilalang mga tsismis ay maaaring mapabilang sa kategoryang ito.

    # none: Isang komento na hindi naglalaman ng anumang pagkamuhi o pang-iinsulto. Para sa isang komentong tulad nito, ito ay dapat iklasipika bilang none.

    Komento: {}. Ngayon, nagtatapos na ang komento.
    Mangyaring tumugon gamit ang "hate", "offensive", o "none".
    """,

    "Hindi": """
    यह कार्य Naver पर एक समाचार टिप्पणी को hate, offensive और none लेबल असाइन करने के लिए है।

    यह एनोटेशन मानक है। ध्यान दें कि इस अध्ययन में, प्रत्येक श्रेणी की परिभाषा कुछ पहलुओं में अन्य अध्ययनों के समान हो सकती है, लेकिन यह बिल्कुल समान नहीं है। कृपया नीचे दी गई परिभाषा का सावधानीपूर्वक पालन करें।

    # hate: क्या किसी टिप्पणी में लेख के लक्ष्य या संबंधित व्यक्तियों, लेख या टिप्पणियों के लेखकों आदि के प्रति तीव्र घृणा या अपमान प्रदर्शित होता है? यदि हाँ, तो इसे hate के रूप में वर्गीकृत किया जाना चाहिए।
    - अपमान के मामले में, इसमें एक अभिव्यक्ति शामिल है जो प्राप्तकर्ता की सामाजिक स्थिति को गंभीर रूप से नुकसान पहुंचा सकती है।
    - hate के मामले में, इसे व्यक्तियों/समूहों के प्रति आक्रामक रुख प्रदर्शित करने वाली अभिव्यक्ति के रूप में परिभाषित किया गया है जिनमें कुछ विशेषताएँ (लिंग भूमिका, यौन अभिविन्यास, यौन पहचान, लिंग-संबंधी कृत्यों के बारे में विचार, जाति, पृष्ठभूमि, राष्ट्रीयता, जातीय समूह, राजनीतिक रुख, त्वचा का रंग, धर्म, विकलांगता, आयु, रूप, धन, व्यवसाय, सैन्य सेवा अनुभव की अनुपस्थिति, आदि) होती हैं।
    - इसके अतिरिक्त, इसमें यौन उत्पीड़न, अपमानजनक अफवाहों या तथ्यों की अधिसूचना और बुरे उद्देश्यों के लिए या बुरे उपयोग में गढ़े गए शब्द आदि शामिल हो सकते हैं।
    - दस्तावेज़ में केवल बुरे शब्दों की उपस्थिति हमेशा इस श्रेणी में नहीं आती।

    # offensive: हालांकि कोई टिप्पणी उपरोक्त की तुलना में इतनी घृणास्पद या अपमानजनक नहीं है, लेकिन क्या यह लक्ष्य या पाठक को अपमानित महसूस कराती है? यदि हाँ, तो इसे offensive के रूप में वर्गीकृत किया जाना चाहिए।
    - इसमें अशिष्ट या आक्रामक सामग्री हो सकती है, जैसे कि बुरे शब्द, हालांकि यह घृणा या अपमान की सीमा तक नहीं है।
    - यह व्यंग्यात्मक प्रश्न या व्यंग्य के माध्यम से व्यंग्य व्यक्त कर सकता है।
    - इसमें एक अनैतिक अभिव्यक्ति शामिल हो सकती है (उदाहरण के लिए, मृत व्यक्तियों के बारे में मजाक या अप्रासंगिक प्रश्न)।
    - एक टिप्पणी जो अज्ञात अफवाहों को व्यक्त करती है, वह इस श्रेणी में आ सकती है।

    # none: एक टिप्पणी जिसमें कोई भी घृणा या अपमान शामिल नहीं है। ऐसी टिप्पणी के लिए इसे none के रूप में वर्गीकृत किया जाना चाहिए।

    टिप्पणी: {}. अब टिप्पणी समाप्त होती है।
    कृपया "hate", "offensive" या "none" के साथ प्रतिक्रिया दें।
    """,

    "Bengali": """
    এই কাজটি হল নেভারে একটি সংবাদে মন্তব্যে hate, offensive এবং none লেবেল বরাদ্দ করা।

    এটি ব্যাখ্যার মানদণ্ড। লক্ষ্য করুন যে এই গবেষণায়, প্রতিটি বিভাগের সংজ্ঞা কিছু ক্ষেত্রে অন্যান্য গবেষণার অনুরূপ হতে পারে, তবে এটি একেবারে একই নয়। দয়া করে নীচের সংজ্ঞাটি সাবধানে অনুসরণ করুন।

    # hate: একটি মন্তব্যে কি প্রবন্ধের লক্ষ্য বা সম্পর্কিত ব্যক্তিবর্গ, প্রবন্ধ বা মন্তব্যের লেখক ইত্যাদির প্রতি শক্তিশালী ঘৃণা বা অপমান প্রদর্শিত হয়েছে? যদি তাই হয়, তবে এটি hate হিসাবে শ্রেণীবদ্ধ করা উচিত।
    - অপমানের ক্ষেত্রে, এতে এমন একটি অভিব্যক্তি অন্তর্ভুক্ত রয়েছে যা প্রাপকের সামাজিক মর্যাদাকে মারাত্মকভাবে ক্ষতিগ্রস্থ করতে পারে।
    - hate এর ক্ষেত্রে, এটি ব্যক্তিদের/গোষ্ঠীর প্রতি আক্রমণাত্মক মনোভাব প্রদর্শনকারী একটি অভিব্যক্তি হিসাবে সংজ্ঞায়িত করা হয়েছে যার নির্দিষ্ট বৈশিষ্ট্য রয়েছে (লিঙ্গ ভূমিকা, যৌন অভিমুখিতা, যৌন পরিচয়, লিঙ্গ-সম্পর্কিত কাজের উপর যেকোন চিন্তা, জাতি, পটভূমি, জাতীয়তা, জাতিগত গোষ্ঠী, রাজনৈতিক অবস্থান, ত্বকের রঙ, ধর্ম, প্রতিবন্ধকতা, বয়স, চেহারা, ধন, পেশা, সামরিক সেবার অভিজ্ঞতার অনুপস্থিতি, ইত্যাদি)।
    - অতিরিক্তভাবে, এটি যৌন হয়রানি, আক্রমণাত্মক গুজব বা তথ্যের বিজ্ঞপ্তি এবং খারাপ উদ্দেশ্য বা খারাপ ব্যবহারের জন্য তৈরি করা শব্দগুলি অন্তর্ভুক্ত করতে পারে ইত্যাদি।
    - নথিতে খারাপ শব্দের উপস্থিতি সবসময় এই বিভাগে পড়ে না।

    # offensive: যদিও একটি মন্তব্য উপরের মতো ঘৃণ্য বা অপমানজনক নয়, তবে এটি কি লক্ষ্য বা পাঠককে অপমানিত বোধ করে? যদি তাই হয়, তবে এটি offensive হিসাবে শ্রেণীবদ্ধ করা উচিত।
    - এতে অশ্লীল বা আক্রমণাত্মক সামগ্রী থাকতে পারে, যেমন খারাপ শব্দ, যদিও ঘৃণা বা অপমানের মাত্রা পর্যন্ত নয়।
    - এটি রূপক প্রশ্ন বা বিদ্রূপের মাধ্যমে বিদ্রূপ প্রকাশ করতে পারে।
    - এতে একটি অনৈতিক অভিব্যক্তি অন্তর্ভুক্ত থাকতে পারে (যেমন, মৃত ব্যক্তিদের সম্পর্কে কৌতুক বা অসঙ্গতিপূর্ণ প্রশ্ন)।
    - একটি মন্তব্য যা অজানা গুজবগুলি প্রকাশ করে তা এই বিভাগে অন্তর্ভুক্ত হতে পারে।

    # none: একটি মন্তব্য যাতে কোনো ঘৃণা বা অপমান অন্তর্ভুক্ত নেই। এরকম মন্তব্যের জন্য এটি none হিসাবে শ্রেণীবদ্ধ করা উচিত।

    মন্তব্য: {}। এখন মন্তব্য শেষ হয়।
    অনুগ্রহ করে "hate", "offensive" বা "none" এর সাথে প্রতিক্রিয়া জানান।
    """,

    "Urdu": """
    یہ کام نیور پر ایک خبر کے تبصرے کو hate، offensive، اور none کے لیبل دینے کے لیے ہے۔

    یہ تشریحی معیار ہے۔ نوٹ کریں کہ اس مطالعے میں، ہر زمرے کی تعریف کچھ پہلوؤں میں دوسرے مطالعات کے ساتھ ملتی جلتی ہو سکتی ہے، لیکن یہ بالکل ایک جیسی نہیں ہے۔ براہ کرم نیچے دی گئی تعریف کو احتیاط سے فالو کریں۔

    # hate: کیا کسی تبصرے میں مضمون کے ہدف یا متعلقہ شخصیات، مضمون یا تبصروں کے مصنفین وغیرہ کی طرف شدید نفرت یا توہین ظاہر ہوتی ہے؟ اگر ایسا ہے تو اسے hate کے طور پر درجہ بند کیا جانا چاہئے۔
    - توہین کی صورت میں، اس میں ایک اظہار شامل ہے جو وصول کنندہ کی سماجی حیثیت کو شدید نقصان پہنچا سکتا ہے۔
    - hate کے معاملے میں، اسے انفرادی افراد/گروہوں کے خلاف جارحانہ موقف ظاہر کرنے والے اظہار کے طور پر بیان کیا گیا ہے جن میں کچھ خصوصیات (صنفی کردار، جنسی رجحان، جنسی شناخت، جنس سے متعلقہ اعمال کے بارے میں کوئی بھی خیالات، نسل، پس منظر، قومیت، نسلی گروہ، سیاسی موقف، جلد کا رنگ، مذہب، معذوری، عمر، ظاہری شکل، دولت، پیشے، فوجی خدمت کے تجربے کی کمی، وغیرہ) ہوتے ہیں۔
    - اضافی طور پر، اس میں جنسی ہراسیت، توہین آمیز افواہوں یا حقائق کی اطلاع، اور بری مقاصد یا غلط استعمال کے لیے بنائی گئی اصطلاحات وغیرہ شامل ہو سکتی ہیں۔
    - دستاویز میں خراب الفاظ کی موجودگی ہمیشہ اس زمرے میں نہیں آتی۔

    # offensive: اگرچہ کوئی تبصرہ اوپر کے مقابلے میں اتنا زیادہ نفرت انگیز یا توہین آمیز نہیں ہے، کیا یہ ہدف یا قاری کو توہین آمیز محسوس کرتا ہے؟ اگر ہاں، تو اسے offensive کے طور پر درجہ بند کیا جانا چاہئے۔
    - اس میں بے ہودہ یا جارحانہ مواد ہو سکتا ہے، جیسے کہ خراب الفاظ، حالانکہ نفرت یا توہین کی حد تک نہیں۔
    - یہ طنزیہ سوال یا طنز کے ذریعے طنز کا اظہار کر سکتا ہے۔
    - اس میں ایک غیر اخلاقی اظہار شامل ہو سکتا ہے (مثلاً، مزاحیہ یا غیر متعلقہ سوالات جو مرنے والوں کے بارے میں ہیں)۔
    - ایک تبصرہ جو غیر شناخت شدہ افواہیں ظاہر کرتا ہے وہ اس زمرے میں آ سکتا ہے۔

    # none: ایک تبصرہ جس میں کوئی بھی نفرت یا توہین شامل نہیں ہے۔ اس طرح کے تبصرے کے لیے اسے none کے طور پر درجہ بند کیا جانا چاہئے۔

    تبصرہ: {}۔ اب تبصرہ ختم ہوتا ہے۔
    براہ کرم "hate"، "offensive" یا "none" کے ساتھ جواب دیں۔
    """
}


# Create a new DataFrame
instruction_df2 = df[['comments', 'hate']].copy()

# Generate the instructions using randomly selected templates
instruction_df2['instruction'] = instruction_df2['comments'].apply(
    lambda x: random.choice(list(instruction_templates.values())).format(x)
)

# Generate the outputs without modification since "hate", "offensive", and "none" are already in English
instruction_df2['output'] = instruction_df2['hate']

# Drop the original columns
instruction_df2 = instruction_df2.drop(columns=['comments', 'hate'])

# Shuffle the DataFrame and save to Parquet
instruction_df2 = instruction_df2.sample(frac=1).reset_index(drop=True)
instruction_df2.to_parquet("../../data/KHS/KHS-hatespeech.parquet", index=False)
instruction_df2_balanced = downsample_to_minority_class(instruction_df2, 'output')

# sample n=800 and save to the same parquet file
instruction_df2_balanced.sample(n=6000).to_parquet("../../data/KHS/KHS-hatespeech.parquet", index=False)


