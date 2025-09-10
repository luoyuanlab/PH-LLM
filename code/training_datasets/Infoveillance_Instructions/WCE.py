# -*- coding: utf-8 -*-\
# WCE (Weibo COVID emotion)  
# - Paper: https://doi.org/10.1007/978-3-030-60450-9_56
# - Data: https://github.com/COVID-19-Weibo-data/COVID-19-sentiment-analysis-dataset-Weibo/blob/master/%E6%83%85%E6%84%9F%E8%AE%AD%E7%BB%83%E9%9B%86.csv
import pandas as pd
import random

df = pd.read_csv("../../data/WCE/train.csv")

# Define the output dictionary
output_dict = {
    0: 'Fear',
    1: 'Disgust',
    2: 'Optimism',
    3: 'Surprise',
    4: 'Gratitude',
    5: 'Sadness',
    6: 'Anger',
}

# Define the list of prompts for all languages
prompts = [
    # Chinese
    "这是一个数据标注任务。你需要从恐惧、厌恶、乐观、惊讶、感激、悲伤和愤怒中选择一个最能描述我稍后发送给你的微博帖子所传达的情感类别。基于以下微博内容，请选择最适合描述该微博帖子的情感类别。微博内容：{}。问题：在恐惧、厌恶、乐观、惊讶、感激、悲伤和愤怒中，哪一个类别最能描述这条微博帖子？请用英文作答：Fear, Disgust, Optimism, Surprise, Gratitude, Sadness, Anger。",

    # English
    "This is a data labeling task. You need to select ONE category that best describes the emotion conveyed in the Weibo post I will send you later. There are a total of 7 possible categories: Fear, Disgust, Optimism, Surprise, Gratitude, Sadness, and Anger. Based on the Weibo content I provide below, please select the most appropriate category that describes the Weibo post. Weibo post: {}. Question: Which ONE of these categories (Fear, Disgust, Optimism, Surprise, Gratitude, Sadness, Anger) best describes this Weibo post?",

    # German
    "Dies ist eine Aufgabe zur Datenkennzeichnung. Sie müssen EINE Kategorie auswählen, die die in dem Weibo-Beitrag, den ich Ihnen später senden werde, vermittelte Emotion am besten beschreibt. Es gibt insgesamt 7 mögliche Kategorien: Angst, Ekel, Optimismus, Überraschung, Dankbarkeit, Traurigkeit und Wut. Basierend auf dem unten bereitgestellten Weibo-Inhalt wählen Sie bitte die am besten passende Kategorie aus, die den Weibo-Beitrag beschreibt. Weibo-Inhalt: {}. Frage: Welche EINER dieser Kategorien (Fear, Disgust, Optimism, Surprise, Gratitude, Sadness, Anger) beschreibt diesen Weibo-Beitrag am besten? Bitte antworten Sie in Englisch.",

    # French
    "Il s'agit d'une tâche de marquage de données. Vous devez sélectionner UNE catégorie qui décrit le mieux l'émotion véhiculée dans le post Weibo que je vais vous envoyer plus tard. Il y a un total de 7 catégories possibles : Peur, Dégoût, Optimisme, Surprise, Gratitude, Tristesse et Colère. En vous basant sur le contenu du Weibo ci-dessous, veuillez sélectionner la catégorie la plus appropriée qui décrit le post Weibo. Contenu Weibo : {}. Question : Laquelle de ces catégories (Fear, Disgust, Optimism, Surprise, Gratitude, Sadness, Anger) décrit le mieux ce post Weibo ? Répondez en anglais, s'il vous plaît.",

    # Spanish
    "Esta es una tarea de etiquetado de datos. Debe seleccionar UNA categoría que mejor describa la emoción transmitida en la publicación de Weibo que le enviaré más tarde. Hay un total de 7 categorías posibles: Miedo, Disgusto, Optimismo, Sorpresa, Gratitud, Tristeza y Enfado. Basado en el contenido de Weibo que se proporciona a continuación, seleccione la categoría más adecuada que describa la publicación de Weibo. Publicación de Weibo: {}. Pregunta: ¿Cuál de estas categorías (Fear, Disgust, Optimism, Surprise, Gratitude, Sadness, Anger) describe mejor esta publicación de Weibo? Responda en inglés, por favor.",

    # Portuguese
    "Esta é uma tarefa de rotulagem de dados. Você precisa selecionar UMA categoria que melhor descreva a emoção transmitida na postagem do Weibo que enviarei a seguir. Existem 7 categorias possíveis: Medo, Nojo, Otimismo, Surpresa, Gratidão, Tristeza e Raiva. Com base no conteúdo do Weibo fornecido abaixo, selecione a categoria mais apropriada que descreve a postagem do Weibo. Postagem do Weibo: {}. Pergunta: Qual destas categorias (Fear, Disgust, Optimism, Surprise, Gratitude, Sadness, Anger) descreve melhor esta postagem do Weibo? Responda em inglês, por favor.",

    # Italian
    "Questo è un compito di etichettatura dei dati. Devi selezionare UNA categoria che descriva meglio l'emozione trasmessa nel post di Weibo che ti invierò in seguito. Ci sono un totale di 7 categorie possibili: Paura, Disgusto, Ottimismo, Sorpresa, Gratitudine, Tristezza e Rabbia. In base al contenuto di Weibo che fornisco di seguito, seleziona la categoria più appropriata che descrive il post di Weibo. Post di Weibo: {}. Domanda: Quale di queste categorie (Fear, Disgust, Optimism, Surprise, Gratitude, Sadness, Anger) descrive meglio questo post di Weibo? Rispondi in inglese, per favore.",

    # Dutch
    "Dit is een gegevenslabelingstaak. U moet EEN categorie selecteren die het best de emotie beschrijft die wordt overgebracht in de Weibo-post die ik u later zal sturen. Er zijn in totaal 7 mogelijke categorieën: Angst, Walging, Optimisme, Verrassing, Dankbaarheid, Verdriet en Woede. Op basis van de Weibo-inhoud die ik hieronder verstrek, selecteert u de meest geschikte categorie die de Weibo-post beschrijft. Weibo-post: {}. Vraag: Welke van deze categorieën (Fear, Disgust, Optimism, Surprise, Gratitude, Sadness, Anger) beschrijft deze Weibo-post het beste? Antwoord in het Engels, alstublieft.",

    # Russian
    "Это задание по маркировке данных. Вам нужно выбрать ОДНУ категорию, которая лучше всего описывает эмоцию, переданную в посте Weibo, который я отправлю вам позже. Всего есть 7 возможных категорий: Страх, Отвращение, Оптимизм, Удивление, Благодарность, Печаль и Гнев. На основании содержимого Weibo, которое я предоставляю ниже, выберите наиболее подходящую категорию, описывающую пост Weibo. Содержимое Weibo: {}. Вопрос: Какая из этих категорий (Fear, Disgust, Optimism, Surprise, Gratitude, Sadness, Anger) лучше всего описывает этот пост Weibo? Пожалуйста, ответьте на английском.",

    # Czech
    "Jedná se o úkol označování dat. Musíte vybrat JEDNU kategorii, která nejlépe popisuje emoci, kterou sděluje příspěvek na Weibo, který vám později pošlu. Celkem existuje 7 možných kategorií: Strach, Znechucení, Optimismus, Překvapení, Vděčnost, Smutek a Hněv. Na základě obsahu Weibo, který uvádím níže, vyberte nejvhodnější kategorii, která popisuje příspěvek na Weibo. Příspěvek na Weibo: {}. Otázka: Která z těchto kategorií (Fear, Disgust, Optimism, Surprise, Gratitude, Sadness, Anger) nejlépe popisuje tento příspěvek na Weibo? Odpovězte prosím anglicky.",

    # Polish
    "To jest zadanie etykietowania danych. Musisz wybrać JEDNĄ kategorię, która najlepiej opisuje emocję przekazaną w poście Weibo, który wyślę Ci później. Istnieje łącznie 7 możliwych kategorii: Strach, Wstręt, Optymizm, Zaskoczenie, Wdzięczność, Smutek i Gniew. Na podstawie poniższej treści Weibo wybierz najodpowiedniejszą kategorię, która opisuje post Weibo. Post na Weibo: {}. Pytanie: Która z tych kategorii (Fear, Disgust, Optimism, Surprise, Gratitude, Sadness, Anger) najlepiej opisuje ten post na Weibo? Proszę odpowiedzieć po angielsku.",

    # Arabic
    "هذه مهمة لتصنيف البيانات. تحتاج إلى اختيار فئة واحدة تصف بشكل أفضل العاطفة التي ينقلها منشور Weibo الذي سأرسله لك لاحقًا. هناك 7 فئات ممكنة: الخوف، الاشمئزاز، التفاؤل، المفاجأة، الامتنان، الحزن، والغضب. بناءً على محتوى Weibo الذي سأقدمه لك أدناه، يرجى اختيار الفئة الأكثر ملاءمة التي تصف منشور Weibo. منشور Weibo: {}. السؤال: أي واحدة من هذه الفئات (Fear, Disgust, Optimism, Surprise, Gratitude, Sadness, Anger) تصف هذا المنشور بشكل أفضل؟ يرجى الرد بالإنجليزية.",

    # Persian
    "این یک وظیفه برچسب‌گذاری داده‌ها است. شما باید یک دسته‌بندی را انتخاب کنید که به بهترین شکل احساس منتقل شده در پست Weibo که بعداً برای شما ارسال می‌کنم را توصیف کند. در مجموع ۷ دسته ممکن وجود دارد: ترس، انزجار، خوشبینی، تعجب، قدردانی، غم و خشم. بر اساس محتوای Weibo که در زیر ارائه می‌شود، لطفاً مناسب‌ترین دسته‌بندی را که پست Weibo را توصیف می‌کند، انتخاب کنید. پست Weibo: {}. سوال: کدام یک از این دسته‌ها (Fear, Disgust, Optimism, Surprise, Gratitude, Sadness, Anger) این پست Weibo را به بهترین شکل توصیف می‌کند؟ لطفاً به انگلیسی پاسخ دهید.",

    # Hebrew
    "זוהי משימת תיוג נתונים. עליך לבחור קטגוריה אחת שמתארת בצורה הטובה ביותר את הרגש שמועבר בפוסט ה-Weibo שאשלח לך בהמשך. ישנן בסך הכל 7 קטגוריות אפשריות: פחד, גועל, אופטימיות, הפתעה, תודה, עצב וכעס. בהתבסס על תוכן ה-Weibo שאני מספק למטה, בחר את הקטגוריה המתאימה ביותר שמתארת את פוסט ה-Weibo. תוכן ה-Weibo: {}. שאלה: איזו מהקטגוריות האלה (Fear, Disgust, Optimism, Surprise, Gratitude, Sadness, Anger) מתארת בצורה הטובה ביותר את פוסט ה-Weibo הזה? אנא ענה באנגלית.",

    # Turkish
    "Bu bir veri etiketleme görevidir. Daha sonra size göndereceğim Weibo gönderisinde aktarılan duyguyu en iyi şekilde tanımlayan BİR kategori seçmelisiniz. Toplamda 7 olası kategori vardır: Korku, Tiksinme, İyimserlik, Şaşkınlık, Minnettarlık, Üzüntü ve Öfke. Aşağıda sağladığım Weibo içeriğine dayanarak, Weibo gönderisini en iyi tanımlayan en uygun kategoriyi seçin. Weibo gönderisi: {}. Soru: Bu kategorilerden hangisi (Fear, Disgust, Optimism, Surprise, Gratitude, Sadness, Anger) bu Weibo gönderisini en iyi tanımlar? Lütfen İngilizce cevap verin.",

    # Japanese
    "これはデータラベリングのタスクです。後でお送りするWeibo投稿で伝えられる感情を最もよく表すカテゴリを1つ選択してください。選択可能なカテゴリは合計で7つあります: 恐怖、嫌悪、楽観、驚き、感謝、悲しみ、怒りです (Fear, Disgust, Optimism, Surprise, Gratitude, Sadness, Anger)。以下に提供するWeiboの内容に基づいて、Weibo投稿を最もよく表す適切なカテゴリを選択してください。Weibo投稿: {}。質問: これらのカテゴリのうち、どれがこのWeibo投稿を最もよく表していますか? 英語で回答してください。",

    # Korean
    "이것은 데이터 라벨링 작업입니다. 나중에 보내드릴 Weibo 게시물에서 전달된 감정을 가장 잘 설명하는 하나의 범주를 선택해야 합니다. 가능한 범주는 총 7개입니다: 공포, 혐오, 낙관, 놀라움, 감사, 슬픔, 분노 (Fear, Disgust, Optimism, Surprise, Gratitude, Sadness, Anger). 아래 제공된 Weibo 내용을 기반으로 Weibo 게시물을 가장 잘 설명하는 범주를 선택하세요. Weibo 게시물: {}. 질문: 이 범주 중에서 어떤 것이 이 Weibo 게시물을 가장 잘 설명합니까? 영어로 답변해 주세요.",

    # Vietnamese
    "Đây là một nhiệm vụ gán nhãn dữ liệu. Bạn cần chọn MỘT danh mục mô tả tốt nhất cảm xúc được truyền tải trong bài đăng Weibo mà tôi sẽ gửi cho bạn sau. Có tổng cộng 7 danh mục có thể: Sợ hãi, Ghê tởm, Lạc quan, Ngạc nhiên, Biết ơn, Buồn bã và Tức giận. Dựa trên nội dung Weibo tôi cung cấp dưới đây, vui lòng chọn danh mục phù hợp nhất mô tả bài đăng Weibo. Bài đăng Weibo: {}. Câu hỏi: Danh mục nào trong số này (Fear, Disgust, Optimism, Surprise, Gratitude, Sadness, Anger) mô tả tốt nhất bài đăng Weibo này? Vui lòng trả lời bằng tiếng Anh.",

    # Thai
    "นี่คืองานการติดป้ายข้อมูล คุณต้องเลือกหนึ่งหมวดหมู่ที่อธิบายอารมณ์ที่ถ่ายทอดในโพสต์ Weibo ที่ฉันจะส่งให้คุณในภายหลังได้ดีที่สุด มีทั้งหมด 7 หมวดหมู่ที่เป็นไปได้: ความกลัว, ความขยะแขยง, การมองโลกในแง่ดี, ความประหลาดใจ, ความกตัญญู, ความเศร้าโศก และความโกรธ ขึ้นอยู่กับเนื้อหา Weibo ที่ฉันให้ไว้ด้านล่าง โปรดเลือกหมวดหมู่ที่เหมาะสมที่สุดที่อธิบายโพสต์ Weibo โพสต์ Weibo: {} คำถาม: หนึ่งในหมวดหมู่เหล่านี้ (Fear, Disgust, Optimism, Surprise, Gratitude, Sadness, Anger) อธิบายโพสต์ Weibo นี้ได้ดีที่สุด โปรดตอบเป็นภาษาอังกฤษ",

    # Indonesian
    "Ini adalah tugas pelabelan data. Anda perlu memilih SATU kategori yang paling tepat menggambarkan emosi yang disampaikan dalam postingan Weibo yang akan saya kirimkan kepada Anda nanti. Ada total 7 kategori yang mungkin: Ketakutan, Jijik, Optimisme, Kejutan, Rasa Terima Kasih, Kesedihan, dan Kemarahan. Berdasarkan konten Weibo yang saya berikan di bawah ini, silakan pilih kategori yang paling tepat yang menggambarkan postingan Weibo. Postingan Weibo: {} Pertanyaan: Yang mana dari kategori ini (Fear, Disgust, Optimism, Surprise, Gratitude, Sadness, Anger) yang paling tepat menggambarkan postingan Weibo ini? Silakan jawab dalam bahasa Inggris.",

    # Malay
    "Ini adalah tugas pelabelan data. Anda perlu memilih SATU kategori yang paling tepat menggambarkan emosi yang disampaikan dalam kiriman Weibo yang akan saya hantar kepada anda nanti. Terdapat 7 kategori yang mungkin: Ketakutan, Jijik, Optimisme, Kejutan, Rasa Terima Kasih, Kesedihan, dan Kemarahan. Berdasarkan kandungan Weibo yang saya berikan di bawah, sila pilih kategori yang paling sesuai yang menggambarkan kiriman Weibo. Kiriman Weibo: {} Soalan: Yang manakah daripada kategori ini (Fear, Disgust, Optimism, Surprise, Gratitude, Sadness, Anger) yang paling tepat menggambarkan kiriman Weibo ini? Sila jawab dalam bahasa Inggeris.",

    # Lao
    "นี่คือภารกิจในการติดป้ายกำกับข้อมูล คุณต้องเลือกหมวดหมู่หนึ่งที่อธิบายอารมณ์ที่ถ่ายทอดในโพสต์ Weibo ที่ฉันจะส่งให้คุณในภายหลังได้ดีที่สุด มีทั้งหมด 7 หมวดหมู่ที่เป็นไปได้: ความกลัว, ความขยะแขยง, การมองโลกในแง่ดี, ความประหลาดใจ, ความกตัญญู, ความเศร้าโศก และความโกรธ ขึ้นอยู่กับเนื้อหา Weibo ที่ฉันให้ไว้ด้านล่าง โปรดเลือกหมวดหมู่ที่เหมาะสมที่สุดที่อธิบายโพสต์ Weibo นี้ โพสต์ Weibo: {} คำถาม: หนึ่งในหมวดหมู่เหล่านี้ (Fear, Disgust, Optimism, Surprise, Gratitude, Sadness, Anger) อธิบายโพสต์ Weibo นี้ได้ดีที่สุด โปรดตอบเป็นภาษาอังกฤษ",

    # Burmese
    "ဤသည်သည်ဒေတာအမှတ်အသားလုပ်ငန်းဖြစ်သည်။ သင်သည်ကျွန်ုပ်မှနောက်ပိုင်းတွင်ပေးပို့မည့်Weiboပို့စ်တွင်ဖော်ပြသည့်ခံစားမှုကိုအကောင်းဆုံးဖော်ပြသည့်တစ်ခုသောအမျိုးအစားကိုရွေးချယ်ရန်လိုအပ်သည်။ စုစုပေါင်း 7 မျိုးအမျိုးအစားများရှိသည်: ကြောက်ရွံ့မှု၊ ရိုက်ကြိုက်မှု၊ အကောင်းမြင်မှု၊ အံ့သြစရာ၊ ကျေးဇူးတင်မှု၊ ဝမ်းနည်းမှုနှင့် ဒေါသ ဖြစ်ပါသည်။ ကျွန်ုပ်ထောက်ပံ့သည့် Weibo အကြောင်းအရာကိုအခြေခံ၍ Weibo ပို့စ်ကိုဖော်ပြသည့်အကောင်းဆုံးအမျိုးအစားကိုရွေးချယ်ပါ။ Weibo ပို့စ်: {} မေးခွန်း: အမျိုးအစားများထဲတွင် (Fear, Disgust, Optimism, Surprise, Gratitude, Sadness, Anger) သည် Weibo ပို့စ်ကိုအကောင်းဆုံးဖော်ပြသည့်အမျိုးအစားတစ်ခုကိုရွေးချယ်ပါ။ အင်္ဂလိပ်ဘာသာဖြင့်ဖြေပါ။",

    # Cebuano
    "Kini usa ka buluhaton sa pag-label sa datos. Kinahanglan nimo nga mopili og USA ka kategoriya nga labing maayo nga naghulagway sa emosyon nga gipadayag sa Weibo nga post nga akong ipadala kanimo sa ulahi. Adunay total nga 7 ka posible nga mga kategoriya: Kahadlok, Pagdumot, Optimismo, Katingalahan, Pasalamat, Kasubo, ug Kasuko. Base sa sulod sa Weibo nga akong gihatag sa ubos, palihug pilia ang labing angay nga kategoriya nga naghulagway sa Weibo nga post. Weibo nga post: {} Pangutana: Asa niini nga mga kategoriya (Fear, Disgust, Optimism, Surprise, Gratitude, Sadness, Anger) ang labing maayo nga naghulagway niini nga Weibo nga post? Palihug pagtubag sa Ingles.",

    # Khmer
    "នេះជាការងារតំឡើងស្លាកទិន្នន័យ។ អ្នកត្រូវតែជ្រើសរើសប្រភេទមួយដែលល្អបំផុតក្នុងការពិពណ៌នាអារម្មណ៍ដែលបានបញ្ជូនមកក្នុងការបង្ហោះ Weibo ដែលខ្ញុំ​នឹងផ្ញើទៅអ្នកនៅពេលក្រោយ មានប្រភេទទាំង 7 ដែលអាចធ្វើទៅបាន: ភ័យខ្លាច, ស្អប់ខ្ពើម, ល្អប្រសើរ, ភ្ញាក់ផ្អើល, អរគុណ, សោកស្តាយ និង សំណាងដ៏អាក្រក់។ ដោយផ្អែកលើខ្លឹមសារ Weibo ខ្ញុំបានផ្តល់នៅខាងក្រោម សូមជ្រើសរើសប្រភេទដែលសមរម្យបំផុតដែលពិពណ៌នាអំពីការបង្ហោះ Weibo នេះ។ ការបង្ហោះ Weibo: {} សំណួរ៖ តើប្រភេទមួយណាខាងក្រោមនេះ (ភ័យខ្លាច, ស្អប់ខ្ពើម, ល្អប្រសើរ, ភ្ញាក់ផ្អើល, អរគុណ, សោកស្តាយ និង សំណាងដ៏អាក្រក់)(Fear, Disgust, Optimism, Surprise, Gratitude, Sadness, Anger) ដែលពិពណ៌នាការបង្ហោះ Weibo នេះបានល្អបំផុត? សូមឆ្លើយជាភាសាអង់គ្លេស។",

    # Tagalog
    "Ito ay isang gawain sa pag-label ng data. Kailangan mong pumili ng ISA sa mga kategorya na pinakamainam na naglalarawan sa damdaming ipinahahayag sa post ng Weibo na ipapadala ko sa iyo mamaya. May kabuuang 7 posibleng kategorya: Takot, Pagka-disgusto, Optimismo, Pagkamangha, Pasasalamat, Kalungkutan, at Galit. Batay sa nilalaman ng Weibo na ibinibigay ko sa ibaba, mangyaring piliin ang pinakaangkop na kategorya na naglalarawan sa Weibo post. Weibo post: {} Tanong: Alin sa mga kategoryang ito (Fear, Disgust, Optimism, Surprise, Gratitude, Sadness, Anger) ang pinakamahusay na naglalarawan sa post na ito ng Weibo? Pakisagot sa Ingles.",

    # Hindi
    "यह एक डेटा लेबलिंग कार्य है। आपको एक श्रेणी का चयन करना है जो सबसे अच्छा वर्णन करती है कि मैं आपको बाद में भेजे जाने वाले Weibo पोस्ट में कौन सी भावना व्यक्त की गई है। कुल 7 संभावित श्रेणियाँ हैं: भय, घृणा, आशावाद, आश्चर्य, कृतज्ञता, उदासी और क्रोध। नीचे दिए गए Weibo सामग्री के आधार पर, कृपया सबसे उपयुक्त श्रेणी चुनें जो Weibo पोस्ट का वर्णन करती है। Weibo पोस्ट: {} प्रश्न: इन श्रेणियों में से कौन सा (Fear, Disgust, Optimism, Surprise, Gratitude, Sadness, Anger) इस Weibo पोस्ट का सबसे अच्छा वर्णन करता है? कृपया अंग्रेजी में उत्तर दें।",

    # Bengali
    "এটি একটি ডেটা লেবেলিং কাজ। আপনাকে একটি ক্যাটেগরি নির্বাচন করতে হবে যা সবচেয়ে ভালভাবে বর্ণনা করে যে আমি আপনাকে পরে পাঠাবো Weibo পোস্টে কোন আবেগ প্রকাশিত হয়েছে। মোট ৭টি সম্ভাব্য ক্যাটেগরি রয়েছে: ভয়, ঘৃণা, আশাবাদ, বিস্ময়, কৃতজ্ঞতা, বিষণ্ণতা এবং ক্রোধ। নিচে দেওয়া Weibo বিষয়বস্তুর উপর ভিত্তি করে, দয়া করে সবচেয়ে উপযুক্ত ক্যাটেগরি নির্বাচন করুন যা Weibo পোস্টটিকে বর্ণনা করে। Weibo পোস্ট: {} প্রশ্ন: এই ক্যাটেগরিগুলির মধ্যে কোনটি (Fear, Disgust, Optimism, Surprise, Gratitude, Sadness, Anger) এই Weibo পোস্টটিকে সবচেয়ে ভালভাবে বর্ণনা করে? দয়া করে ইংরেজিতে উত্তর দিন।",

    # Urdu
    "یہ ایک ڈیٹا لیبلنگ کا کام ہے۔ آپ کو ایک زمرہ منتخب کرنا ہوگا جو میں آپ کو بعد میں بھیجوں گا اس ویبو پوسٹ میں کون سا جذبہ بہترین طریقے سے بیان کرتا ہے۔ کل 7 ممکنہ زمروں میں سے ایک ہیں: خوف، نفرت، امید پرستی، حیرت، شکرگزاری، غم اور غصہ۔ نیچے دیے گئے ویبو مواد کی بنیاد پر، براہ کرم وہ زمرہ منتخب کریں جو ویبو پوسٹ کی بہترین وضاحت کرتا ہے۔ ویبو پوسٹ: {} سوال: ان زمروں میں سے کون سا (Fear, Disgust, Optimism, Surprise, Gratitude, Sadness, Anger) اس ویبو پوسٹ کو بہترین بیان کرتا ہے؟ براہ کرم انگریزی میں جواب دیں۔"
]


# Function to construct the output column
def construct_output(row):
    return output_dict[row['label']]

# Function to construct the instruction column
def construct_instruction(row):
    instruction_template = random.choice(prompts)
    instruction = instruction_template.format(row['review'])
    return instruction

# Apply the functions to create the dataset
df['output'] = df.apply(construct_output, axis=1)
df['instruction'] = df.apply(construct_instruction, axis=1)
print(df.output.value_counts())
print(df.shape)
# 2200 per class
# df = balance_classes(df, 2000, 'output')

# Save the resulting DataFrame to a parquet file
df[['instruction', 'output']].sample(n=10500, random_state=42).to_parquet("../../data/WCE/WCE.parquet", index=False)
# df.output.value_counts()