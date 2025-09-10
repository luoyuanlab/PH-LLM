# -*- coding: utf-8 -*-
# DR (Depression – Reddit)
# - Paper: https://doi.org/10.1145/3485447.3512128
# - Data: https://github.com/usmaann/Depression_Severity_Dataset
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import pandas as pd
import random
from utils.data_utils import downsample_to_minority_class

df = pd.read_csv("../../data/DR/Reddit_depression_dataset.csv")


# List of prompts for all languages
prompts = [
    # Chinese
    "我想确定Reddit帖子中表达的抑郁程度。在四个抑郁程度等级中：最低、轻度、中度和重度，您需要选择一个最能描述Reddit帖子的等级。根据下面提供的Reddit帖子内容，请选择最适合的抑郁程度等级。我们的抑郁程度定义基于Beck's Depression Inventory。Reddit帖子内容：{text}。Reddit帖子到此结束。请回答：哪个抑郁程度等级（最低、轻度、中度或重度）最能描述上述Reddit帖子？请用minimum, mild, moderate, 或severe (用英文) 回答。",
    "请帮助确定Reddit帖子中表达的抑郁程度。在四个抑郁程度等级中：最低、轻度、中度和重度，选择一个最能描述Reddit帖子的等级。我们的抑郁程度定义基于Beck's Depression Inventory。Reddit帖子内容：{text}。Reddit帖子到此结束。请回答：哪个抑郁程度等级（最低、轻度、中度或重度）最能描述上述Reddit帖子？请用minimum, mild, moderate, 或severe (用英文) 回答。",

    # English
    "I would like to identify the severity of depression expressed in a Reddit post. Among four levels of depression severity: minimum, mild, moderate, and severe, you need to choose ONE level that best describes the Reddit post. Based on the Reddit post provided below, please select the most appropriate level of depression severity that describes the post. Our definition of depression severity level is based on Beck's Depression Inventory. Reddit post content: {text}. The Reddit post now ends. Please respond with minimum, mild, moderate, or severe (in English).",
    "Please determine the severity of depression expressed in the following Reddit post. There are four levels of depression severity: minimum, mild, moderate, and severe. Choose the level that best describes the post. Our definition of depression severity is based on Beck's Depression Inventory. Reddit post content: {text}. The Reddit post ends here. Please respond with minimum, mild, moderate, or severe (in English).",

    # German
    "Ich möchte den Schweregrad der in einem Reddit-Post ausgedrückten Depression identifizieren. Unter vier Schweregraden der Depression: minimal, mild, moderat und schwer, müssen Sie einen auswählen, der den Reddit-Post am besten beschreibt. Basierend auf dem unten stehenden Reddit-Post wählen Sie bitte den am besten passenden Schweregrad aus. Unsere Definition des Schweregrads basiert auf dem Beck's Depression Inventory. Inhalt des Reddit-Posts: {text}. Der Reddit-Post endet hier. Bitte antworten Sie mit minimum, mild, moderate, oder severe (auf Englisch).",
    "Bitte bestimmen Sie den Schweregrad der Depression, die im folgenden Reddit-Post ausgedrückt wird. Es gibt vier Stufen der Schwere der Depression: minimal, mild, moderat und schwer. Wählen Sie die Stufe, die den Post am besten beschreibt. Unsere Definition der Schweregrade basiert auf dem Beck's Depression Inventory. Inhalt des Reddit-Posts: {text}. Der Reddit-Post endet hier. Bitte antworten Sie mit minimum, mild, moderate, oder severe (auf Englisch).",

    # French
    "Je souhaite identifier la gravité de la dépression exprimée dans un post Reddit. Parmi quatre niveaux de gravité de la dépression : minimum, légère, modérée et sévère, vous devez choisir un niveau qui décrit le mieux le post Reddit. Sur la base du post Reddit ci-dessous, veuillez sélectionner le niveau de gravité le plus approprié. Notre définition de la gravité est basée sur l'Inventaire de Dépression de Beck. Contenu du post Reddit : {text}. Le post Reddit se termine ici. Veuillez répondre par minimum, mild, moderate, ou severe (en anglais).",
    "Veuillez déterminer la gravité de la dépression exprimée dans le post Reddit suivant. Il existe quatre niveaux de gravité de la dépression : minimum, légère, modérée et sévère. Choisissez celui qui décrit le mieux le post. Notre définition de la gravité de la dépression est basée sur l'Inventaire de Dépression de Beck. Contenu du post Reddit : {text}. Le post Reddit se termine ici. Veuillez répondre par minimum, mild, moderate, ou severe (en anglais).",

    # Spanish
    "Me gustaría identificar la gravedad de la depresión expresada en una publicación de Reddit. Entre cuatro niveles de gravedad de la depresión: mínimo, leve, moderado y severo, necesitas elegir uno que mejor describa la publicación. Basado en la publicación proporcionada a continuación, selecciona el nivel más apropiado. Nuestra definición de los niveles de gravedad se basa en el Inventario de Depresión de Beck. Contenido de la publicación de Reddit: {text}. La publicación termina aquí. Responde con minimum, mild, moderate, o severe (en inglés).",
    "Por favor, determine la gravedad de la depresión expresada en la siguiente publicación de Reddit. Existen cuatro niveles de gravedad de la depresión: mínimo, leve, moderado y severo. Elige el nivel que mejor describa la publicación. Nuestra definición de la gravedad de la depresión se basa en el Inventario de Depresión de Beck. Contenido de la publicación de Reddit: {text}. La publicación termina aquí. Responde con minimum, mild, moderate, o severe (en inglés).",

    # Portuguese
    "Gostaria de identificar a gravidade da depressão expressa em uma postagem do Reddit. Entre quatro níveis de gravidade da depressão: mínimo, leve, moderado e grave, você precisa escolher um que melhor descreva a postagem. Com base na postagem fornecida abaixo, selecione o nível mais apropriado. Nossa definição dos níveis de gravidade é baseada no Inventário de Depressão de Beck. Conteúdo da postagem do Reddit: {text}. A postagem termina aqui. Responda com minimum, mild, moderate, ou severe (em inglês).",
    "Por favor, determine a gravidade da depressão expressa na seguinte postagem do Reddit. Existem quatro níveis de gravidade da depressão: mínimo, leve, moderado e grave. Escolha o nível que melhor descreva a postagem. Nossa definição da gravidade da depressão é baseada no Inventário de Depressão de Beck. Conteúdo da postagem do Reddit: {text}. A postagem termina aqui. Responda com minimum, mild, moderate, ou severe (em inglês).",

    # Italian
    "Vorrei identificare la gravità della depressione espressa in un post su Reddit. Tra quattro livelli di gravità della depressione: minimo, lieve, moderato e grave, devi scegliere uno che descrive meglio il post. Basandoti sul post fornito qui sotto, seleziona il livello più appropriato. La nostra definizione dei livelli di gravità si basa sul Beck's Depression Inventory. Contenuto del post su Reddit: {text}. Il post termina qui. Rispondi con minimum, mild, moderate, o severe (in inglese).",
    "Per favore, determina la gravità della depressione espressa nel seguente post su Reddit. Ci sono quattro livelli di gravità della depressione: minimo, lieve, moderato e grave. Scegli il livello che meglio descrive il post. La nostra definizione della gravità della depressione si basa sul Beck's Depression Inventory. Contenuto del post su Reddit: {text}. Il post termina qui. Rispondi con minimum, mild, moderate, o severe (in inglese).",

    # Dutch
    "Ik wil de ernst van de depressie vaststellen die in een Reddit-post wordt uitgedrukt. Van de vier niveaus van depressie-ernst: minimaal, mild, matig en ernstig, moet je één kiezen die de post het beste beschrijft. Op basis van de onderstaande post, selecteer alstublieft het meest geschikte niveau. Onze definitie van de ernstniveaus is gebaseerd op de Beck's Depression Inventory. Inhoud van de Reddit-post: {text}. De post eindigt hier. Reageer met minimum, mild, moderate, of severe (in het Engels).",
    "Bepaal de ernst van de depressie die in de volgende Reddit-post wordt uitgedrukt. Er zijn vier niveaus van depressie-ernst: minimaal, mild, matig en ernstig. Kies het niveau dat de post het beste beschrijft. Onze definitie van de ernstniveaus is gebaseerd op de Beck's Depression Inventory. Inhoud van de Reddit-post: {text}. De post eindigt hier. Reageer met minimum, mild, moderate, of severe (in het Engels).",

    # Russian
    "Я хотел бы определить степень тяжести депрессии, выраженной в посте на Reddit. Среди четырех уровней тяжести депрессии: минимальный, легкий, умеренный и тяжелый, выберите один, который лучше всего описывает пост. На основе приведенного ниже поста, выберите наиболее подходящий уровень. Наше определение уровней тяжести основано на шкале депрессии Бека. Содержание поста на Reddit: {text}. Пост заканчивается здесь. Ответьте минимум, mild, moderate или severe (на английском).",
    "Пожалуйста, определите степень тяжести депрессии, выраженной в следующем посте на Reddit. Существует четыре уровня тяжести депрессии: минимальный, легкий, умеренный и тяжелый. Выберите уровень, который лучше всего описывает пост. Наше определение степени тяжести депрессии основано на шкале депрессии Бека. Содержание поста на Reddit: {text}. Пост заканчивается здесь. Ответьте минимум, mild, moderate или severe (на английском).",

    # Czech
    "Chtěl bych identifikovat závažnost deprese vyjádřené v příspěvku na Redditu. Mezi čtyřmi úrovněmi závažnosti deprese: minimální, mírná, střední a těžká, vyberte jednu, která nejlépe popisuje příspěvek. Na základě níže uvedeného příspěvku zvolte nejvhodnější úroveň. Naše definice úrovní závažnosti vychází z Beckova depresivního inventáře. Obsah příspěvku na Redditu: {text}. Příspěvek zde končí. Odpovězte minimum, mild, moderate, nebo severe (v angličtině).",
    "Určete závažnost deprese vyjádřené v následujícím příspěvku na Redditu. Existují čtyři úrovně závažnosti deprese: minimální, mírná, střední a těžká. Vyberte úroveň, která nejlépe popisuje příspěvek. Naše definice úrovní závažnosti vychází z Beckova depresivního inventáře. Obsah příspěvku na Redditu: {text}. Příspěvek zde končí. Odpovězte minimum, mild, moderate, nebo severe (v angličtině).",

    # Polish
    "Chciałbym zidentyfikować stopień nasilenia depresji wyrażony w poście na Reddit. Spośród czterech poziomów nasilenia depresji: minimalny, łagodny, umiarkowany i ciężki, wybierz jeden, który najlepiej opisuje post. Na podstawie poniższego postu wybierz najbardziej odpowiedni poziom. Nasza definicja poziomów nasilenia opiera się na Inwentarzu Depresji Becka. Treść postu na Reddit: {text}. Post kończy się tutaj. Odpowiedz minimum, mild, moderate lub severe (w języku angielskim).",
    "Proszę określić stopień nasilenia depresji wyrażony w następującym poście na Reddit. Istnieją cztery poziomy nasilenia depresji: minimalny, łagodny, umiarkowany i ciężki. Wybierz poziom, który najlepiej opisuje post. Nasza definicja poziomów nasilenia opiera się na Inwentarzu Depresji Becka. Treść postu na Reddit: {text}. Post kończy się tutaj. Odpowiedz minimum, mild, moderate lub severe (w języku angielskim).",

    # Arabic
    "أود تحديد شدة الاكتئاب المعبر عنها في منشور على Reddit. من بين أربعة مستويات من شدة الاكتئاب: الحد الأدنى، الخفيف، المتوسط، والشديد، اختر واحدًا يصف المنشور بشكل أفضل. بناءً على المنشور أدناه، اختر المستوى الأكثر ملاءمة. يعتمد تعريفنا لمستويات الشدة على مقياس بيك للاكتئاب. محتوى منشور Reddit: {text}. ينتهي المنشور هنا. الرجاء الرد بالحد الأدنى، الخفيف، المتوسط، أو الشديد (باللغة الإنجليزية).",
    "يرجى تحديد شدة الاكتئاب المعبر عنها في المنشور التالي على Reddit. هناك أربعة مستويات من شدة الاكتئاب: الحد الأدنى، الخفيف، المتوسط، والشديد. اختر المستوى الذي يصف المنشور بشكل أفضل. يعتمد تعريفنا لمستويات الشدة على مقياس بيك للاكتئاب. محتوى منشور Reddit: {text}. ينتهي المنشور هنا. الرجاء الرد بالحد الأدنى، الخفيف، المتوسط، أو الشديد (باللغة الإنجليزية).",

    # Persian
    "می‌خواهم شدت افسردگی بیان‌شده در یک پست Reddit را شناسایی کنم. از بین چهار سطح شدت افسردگی: حداقل، خفیف، متوسط و شدید، یکی را انتخاب کنید که پست را بهتر توصیف کند. بر اساس پست زیر، لطفاً مناسب‌ترین سطح را انتخاب کنید. تعریف ما از سطوح شدت افسردگی بر اساس پرسشنامه افسردگی بک است. محتوای پست Reddit: {text}. پست در اینجا به پایان می‌رسد. لطفاً با حداقل، خفیف، متوسط، یا شدید (به انگلیسی) پاسخ دهید.",
    "لطفاً شدت افسردگی بیان‌شده در پست زیر در Reddit را تعیین کنید. چهار سطح شدت افسردگی وجود دارد: حداقل، خفیف، متوسط و شدید. سطحی را انتخاب کنید که پست را بهتر توصیف کند. تعریف ما از سطوح شدت افسردگی بر اساس پرسشنامه افسردگی بک است. محتوای پست Reddit: {text}. پست در اینجا به پایان می‌رسد. لطفاً با حداقل، خفیف، متوسط، یا شدید (به انگلیسی) پاسخ دهید.",

    # Hebrew
    "אני רוצה לזהות את חומרת הדיכאון המתבטאת בפוסט ב-Reddit. מבין ארבע רמות חומרת הדיכאון: מינימום, קל, מתון וחמור, עליך לבחור אחת שמתארת את הפוסט בצורה הטובה ביותר. בהתבסס על הפוסט שלמטה, אנא בחר את הרמה המתאימה ביותר. ההגדרה שלנו לרמות חומרת הדיכאון מבוססת על Beck's Depression Inventory. תוכן הפוסט ב-Reddit: {text}. הפוסט מסתיים כאן. אנא השב במילים minimum, mild, moderate, או severe (באנגלית).",
    "אנא קבע את חומרת הדיכאון המתבטאת בפוסט הבא ב-Reddit. קיימות ארבע רמות חומרת דיכאון: מינימום, קל, מתון וחמור. בחר את הרמה שמתארת את הפוסט בצורה הטובה ביותר. ההגדרה שלנו לרמות חומרת הדיכאון מבוססת על Beck's Depression Inventory. תוכן הפוסט ב-Reddit: {text}. הפוסט מסתיים כאן. אנא השב במילים minimum, mild, moderate, או severe (באנגלית).",

    # Turkish
    "Bir Reddit gönderisinde ifade edilen depresyonun ciddiyetini belirlemek istiyorum. Depresyon ciddiyetinin dört seviyesi arasında: minimum, hafif, orta ve şiddetli, gönderiyi en iyi tanımlayan bir seviyeyi seçmeniz gerekiyor. Aşağıda verilen Reddit gönderisine dayanarak, en uygun seviyeyi seçin. Depresyon ciddiyeti tanımımız, Beck Depresyon Envanteri'ne dayanmaktadır. Reddit gönderisi içeriği: {text}. Reddit gönderisi burada sona eriyor. Lütfen minimum, mild, moderate veya severe (İngilizce olarak) yanıt verin.",
    "Lütfen bir sonraki Reddit gönderisinde ifade edilen depresyon ciddiyetini belirleyin. Depresyon ciddiyetinin dört seviyesi vardır: minimum, hafif, orta ve şiddetli. Gönderiyi en iyi tanımlayan seviyeyi seçin. Depresyon ciddiyetinin tanımı Beck Depresyon Envanteri'ne dayanmaktadır. Reddit gönderisi içeriği: {text}. Reddit gönderisi burada sona eriyor. Lütfen minimum, mild, moderate veya severe (İngilizce olarak) yanıt verin.",

    # Japanese
    "Redditの投稿に表現されているうつ病の重症度を特定したいと思います。うつ病の重症度には4つのレベルがあります：最低、軽度、中度、重度です。投稿を最もよく表現する1つのレベルを選んでください。以下の投稿に基づいて、最も適切なレベルを選択してください。私たちの定義はBeckのうつ病評価表に基づいています。Reddit投稿の内容：{text}。Redditの投稿はここで終わります。minimum、mild、moderate、またはsevere（英語で）で答えてください。",
    "以下のReddit投稿に表現されているうつ病の重症度を決定してください。うつ病の重症度には4つのレベルがあります：最低、軽度、中度、重度です。投稿を最もよく表現するレベルを選んでください。私たちの定義はBeckのうつ病評価表に基づいています。Reddit投稿の内容：{text}。Redditの投稿はここで終わります。minimum、mild、moderate、またはsevere（英語で）で答えてください。",

    # Korean
    "Reddit 게시물에 표현된 우울증의 심각성을 파악하고 싶습니다. 우울증 심각성의 네 가지 수준 중: 최소, 경도, 중등도, 중증 중에서 게시물을 가장 잘 설명하는 수준을 선택해야 합니다. 아래에 제공된 Reddit 게시물을 기반으로 가장 적절한 수준을 선택하십시오. 우리의 우울증 심각성 정의는 Beck 우울증 척도를 기반으로 합니다. Reddit 게시물 내용: {text}. Reddit 게시물은 여기에서 끝납니다. 최소, mild, moderate, 또는 severe (영어로)로 응답하십시오.",
    "다음 Reddit 게시물에 표현된 우울증의 심각성을 결정하십시오. 우울증 심각성의 네 가지 수준이 있습니다: 최소, 경도, 중등도, 중증. 게시물을 가장 잘 설명하는 수준을 선택하십시오. 우리의 우울증 심각성 정의는 Beck 우울증 척도를 기반으로 합니다. Reddit 게시물 내용: {text}. Reddit 게시물은 여기에서 끝납니다. 최소, mild, moderate, 또는 severe (영어로)로 응답하십시오.",

    # Vietnamese
    "Tôi muốn xác định mức độ nghiêm trọng của trầm cảm được thể hiện trong một bài đăng trên Reddit. Có bốn mức độ nghiêm trọng của trầm cảm: tối thiểu, nhẹ, vừa phải và nghiêm trọng, bạn cần chọn một mức độ mô tả tốt nhất bài đăng. Dựa trên bài đăng dưới đây, hãy chọn mức độ phù hợp nhất. Định nghĩa của chúng tôi về các mức độ nghiêm trọng dựa trên Beck's Depression Inventory. Nội dung bài đăng trên Reddit: {text}. Bài đăng kết thúc ở đây. Vui lòng trả lời với minimum, mild, moderate, hoặc severe (bằng tiếng Anh).",
    "Vui lòng xác định mức độ nghiêm trọng của trầm cảm được thể hiện trong bài đăng sau trên Reddit. Có bốn mức độ nghiêm trọng của trầm cảm: tối thiểu, nhẹ, vừa phải và nghiêm trọng. Hãy chọn mức độ mô tả tốt nhất bài đăng. Định nghĩa của chúng tôi về các mức độ nghiêm trọng dựa trên Beck's Depression Inventory. Nội dung bài đăng trên Reddit: {text}. Bài đăng kết thúc ở đây. Vui lòng trả lời với minimum, mild, moderate, hoặc severe (bằng tiếng Anh).",

    # Thai
    "ฉันต้องการระบุระดับความรุนแรงของภาวะซึมเศร้าที่แสดงออกในโพสต์ Reddit ระดับความรุนแรงของภาวะซึมเศร้ามีทั้งหมด 4 ระดับ: ขั้นต่ำ เล็กน้อย ปานกลาง และรุนแรง คุณต้องเลือกหนึ่งระดับที่อธิบายโพสต์ได้ดีที่สุด จากโพสต์ด้านล่าง โปรดเลือกระดับที่เหมาะสมที่สุด คำจำกัดความของเราขึ้นอยู่กับ Beck's Depression Inventory เนื้อหาโพสต์ใน Reddit: {text} โพสต์จบที่นี่ โปรดตอบกลับด้วย minimum, mild, moderate, หรือ severe (เป็นภาษาอังกฤษ)",
    "โปรดกำหนดระดับความรุนแรงของภาวะซึมเศร้าที่แสดงออกในโพสต์ Reddit ต่อไปนี้ ระดับความรุนแรงของภาวะซึมเศร้ามี 4 ระดับ: ขั้นต่ำ เล็กน้อย ปานกลาง และรุนแรง โปรดเลือกระดับที่อธิบายโพสต์ได้ดีที่สุด คำจำกัดความของเราขึ้นอยู่กับ Beck's Depression Inventory เนื้อหาโพสต์ใน Reddit: {text} โพสต์จบที่นี่ โปรดตอบกลับด้วย minimum, mild, moderate, หรือ severe (เป็นภาษาอังกฤษ)",

    # Indonesian
    "Saya ingin mengidentifikasi tingkat keparahan depresi yang diekspresikan dalam postingan Reddit. Di antara empat tingkat keparahan depresi: minimum, ringan, sedang, dan parah, Anda perlu memilih satu yang paling menggambarkan postingan tersebut. Berdasarkan postingan di bawah ini, pilih tingkat yang paling sesuai. Definisi kami tentang tingkat keparahan berdasarkan Beck's Depression Inventory. Konten postingan Reddit: {text}. Postingan berakhir di sini. Harap balas dengan minimum, mild, moderate, atau severe (dalam bahasa Inggris).",
    "Harap tentukan tingkat keparahan depresi yang diekspresikan dalam postingan Reddit berikut. Ada empat tingkat keparahan depresi: minimum, ringan, sedang, dan parah. Pilih tingkat yang paling menggambarkan postingan tersebut. Definisi kami tentang tingkat keparahan berdasarkan Beck's Depression Inventory. Konten postingan Reddit: {text}. Postingan berakhir di sini. Harap balas dengan minimum, mild, moderate, atau severe (dalam bahasa Inggris).",

    # Malay
    "Saya ingin mengenal pasti tahap keparahan kemurungan yang dinyatakan dalam siaran Reddit. Di antara empat tahap keparahan kemurungan: minimum, ringan, sederhana, dan teruk, anda perlu memilih satu yang paling menggambarkan siaran tersebut. Berdasarkan siaran di bawah, pilih tahap yang paling sesuai. Definisi kami tentang tahap keparahan berdasarkan Beck's Depression Inventory. Kandungan siaran Reddit: {text}. Siaran berakhir di sini. Sila balas dengan minimum, mild, moderate, atau severe (dalam bahasa Inggeris).",
    "Sila tentukan tahap keparahan kemurungan yang dinyatakan dalam siaran Reddit berikut. Terdapat empat tahap keparahan kemurungan: minimum, ringan, sederhana, dan teruk. Pilih tahap yang paling menggambarkan siaran tersebut. Definisi kami tentang tahap keparahan berdasarkan Beck's Depression Inventory. Kandungan siaran Reddit: {text}. Siaran berakhir di sini. Sila balas dengan minimum, mild, moderate, atau severe (dalam bahasa Inggeris).",

    # Lao
    "ຂ້ອຍຕ້ອງການຈະລະບຸລະດັບຄວາມຮຸນແຮງຂອງຄວາມເສົ້າໃນໂພສ Reddit. ໃນບັນດາລະດັບຄວາມຮຸນແຮງຂອງຄວາມເສົ້າ: ຕ່ຳສຸດ, ບາງໆ, ປານກາງ, ແລະຮຸນແຮງ, ເຈົ້າຈຳເປັນຕ້ອງເລືອກລະດັບທີ່ດີທີ່ສຸດ. Beck's Depression Inventory ອີງໃສ່ການນິຍາມ. ໂພສ Reddit: {text}. ຈົ່ງຕອບ minimum, mild, moderate, ຫຼື severe (ໃນພາສາອັງກິດ).",
    "ກະລຸນາກຳນົດລະດັບຄວາມຮຸນແຮງຂອງຄວາມເສົ້າໃນໂພສ Reddit ນີ້. ມີສີ່ລະດັບ: ຕ່ຳສຸດ, ບາງໆ, ປານກາງ, ແລະຮຸນແຮງ. ເລືອກລະດັບທີ່ມີຄວາມເຫມາະສົມທີ່ສຸດ. ເນື້ອຫາໂພສ Reddit: {text}. ກະລຸນາຕອບ minimum, mild, moderate, ຫຼື severe (ໃນພາສາອັງກິດ).",

    # Burmese
    "Reddit ပို့စ်တစ်ခုတွင် ဖော်ပြထားသော စိတ်ကျရောဂါ၏ ဆိုးရွားမှုကို သိရှိလိုသည်။ စိတ်ကျရောဂါဆိုးရွားမှုအဆင့် ၄ ခုရှိသည်: အနည်းဆုံး, ပျော့နှောင်း, အလယ်အလတ်နှင့် အလေးအနက်. အဆိုပါ Beck's Depression Inventory ကို အခြေခံ၍ စိတ်ကျရောဂါ၏ အဆင့်ကို သတ်မှတ်ပါ. Reddit ပို့စ်၏ အကြောင်းအရာ: {text}. အဆုံးသတ်ပါပြီ. ကျေးဇူးပြု၍ minimum, mild, moderate, သို့မဟုတ် severe (အင်္ဂလိပ်) ဖြင့် ပြန်ကြားပါ။",
    "သင့်ကို ရှေ့မှာဖော်ပြထားသော Reddit ပို့စ်တွင်ပါရှိသော စိတ်ကျရောဂါအဆင့် တစ်ခုကို ရွေးချယ်ရန်လိုအပ်သည်။ Beck's Depression Inventory အရ အနည်းဆုံး, ပျော့နှောင်း, အလယ်အလတ်, အလေးအနက်အဆင့်များစွာရှိပါသည်. Reddit ပို့စ်: {text}. ကျေးဇူးပြု၍ minimum, mild, moderate, သို့မဟုတ် severe (အင်္ဂလိပ်) ဖြင့် ပြန်ကြားပါ။",

    # Cebuano
    "Gusto nakong mahibal-an ang lebel sa depresyon sa usa ka Reddit post. Ang mga lebel sa depresyon mao ang: minimum, mild, moderate, ug severe. Palihug pilia ang labing angay nga lebel base sa Beck's Depression Inventory. Reddit post: {text}. Tubaga gamit ang minimum, mild, moderate, o severe (Ingles).",
    "Palihug tukma-a ang lebel sa depresyon nga gipadayag sa Reddit post nga mosunod. Pilia ang labing angay nga lebel. Beck's Depression Inventory gigamit isip basehan. Reddit post: {text}. Palihug tubaga gamit ang minimum, mild, moderate, o severe (Ingles).",

    # Khmer
    "ខ្ញុំចង់កំណត់កម្រិតនៃការរលួយស្មារតីនៅក្នុងការបង្ហោះ Reddit. កម្រិតនៃការរលួយស្មារតីមានបួន: ទាបបំផុត, ប្រាថ្នា, បន្ថយ, និងធ្ងន់ធ្ងរ. ពន្យល់ពីការរលួយស្មារតីតាម Beck's Depression Inventory. អត្ថបទ Reddit: {text}. សូមឆ្លើយ minimum, mild, moderate, ឬ severe (ភាសាអង់គ្លេស)។",
    "សូមកំណត់កម្រិតនៃការរលួយស្មារតីសម្រាប់ការបង្ហោះនៅលើ Reddit ខាងក្រោម. ប្រើបួនកម្រិត: ទាបបំផុត, ប្រាថ្នា, បន្ថយ, និងធ្ងន់ធ្ងរ. ពន្យល់តាម Beck's Depression Inventory. Reddit: {text}. សូមឆ្លើយ minimum, mild, moderate, ឬ severe (ភាសាអង់គ្លេស)។",

    # Tagalog
    "Nais kong tukuyin ang antas ng depresyon sa isang Reddit post. Apat na antas ng depresyon: minimal, mild, moderate, at severe. Batay sa Beck's Depression Inventory. Reddit post: {text}. Sagutin ng minimum, mild, moderate, o severe (sa Ingles).",
    "Pakisuri ang depresyon sa sumusunod na Reddit post. May apat na antas: minimal, mild, moderate, at severe. Batay sa Beck's Depression Inventory. Reddit: {text}. Sagutin ng minimum, mild, moderate, o severe (sa Ingles).",

    # Hindi
    "मैं Reddit पोस्ट में अवसाद के स्तर का पता लगाना चाहता हूँ। अवसाद के स्तरों में न्यूनतम, हल्का, मध्यम, और गंभीर होते हैं। Beck's Depression Inventory पर आधारित। Reddit पोस्ट: {text}. उत्तर दें minimum, mild, moderate, या severe (अंग्रेज़ी में)।",
    "कृपया Reddit पोस्ट में अवसाद के स्तर का निर्धारण करें। चार स्तर हैं: न्यूनतम, हल्का, मध्यम, और गंभीर। Beck's Depression Inventory के आधार पर। Reddit: {text}. उत्तर दें minimum, mild, moderate, या severe (अंग्रेज़ी में)।",

    # Bengali
    "আমি Reddit পোস্টের মাধ্যমে বিষণ্নতার মাত্রা চিহ্নিত করতে চাই। বিষণ্নতার চারটি স্তর: সর্বনিম্ন, মৃদু, মাঝারি, এবং গুরুতর। Beck's Depression Inventory ব্যবহার করে। Reddit: {text}. দয়া করে উত্তর দিন minimum, mild, moderate, বা severe (ইংরেজিতে)।",
    "দয়া করে নিম্নলিখিত Reddit পোস্টে বিষণ্নতার স্তর নির্ধারণ করুন। স্তরগুলি: সর্বনিম্ন, মৃদু, মাঝারি, এবং গুরুতর। Beck's Depression Inventory ব্যবহার। Reddit: {text}. দয়া করে উত্তর দিন minimum, mild, moderate, বা severe (ইংরেজিতে)।",

    # Urdu
    "میں Reddit پوسٹ میں ڈپریشن کی شدت کا پتہ لگانا چاہتا ہوں۔ چار سطحیں ہیں: کم سے کم، ہلکی، درمیانی، اور شدید۔ Beck's Depression Inventory پر مبنی۔ Reddit پوسٹ: {text}. براہ کرم minimum, mild, moderate, یا severe (انگریزی) میں جواب دیں۔",
    "براہ کرم اگلے Reddit پوسٹ میں ڈپریشن کی شدت کا تعین کریں۔ سطحیں: کم سے کم، ہلکی، درمیانی، اور شدید۔ Beck's Depression Inventory کے مطابق۔ Reddit: {text}. جواب دیں minimum, mild, moderate, یا severe (انگریزی میں)۔"
]

# Construct the output column
def construct_output(row):
    return row['label']

# Construct the instruction column
def construct_instruction(row):
    instruction_template = random.choice(prompts)
    instruction = instruction_template.format(text=row['text'])
    return instruction

# Apply the functions to create the dataset
df['output'] = df.apply(construct_output, axis=1)
df['instruction'] = df.apply(construct_instruction, axis=1)

# Save the resulting dataset to a parquet file
df_result = df[["instruction", "output"]]
df_result.to_parquet("../../data/DR/dr.parquet", index=False)

# Example usage:
# Assuming df_result is your DataFrame and 'output' is the target column
instruction_df_balanced = downsample_to_minority_class(df_result, 'output')
instruction_df_balanced.to_parquet("../../data/DR/dr.parquet", index=False)
# sample n=1000 and save to the same parquet file
instruction_df_balanced.sample(n=500).to_parquet("../../data/DR/dr.parquet", index=False)