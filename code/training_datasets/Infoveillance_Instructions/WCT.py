# -*- coding: utf-8 -*-
# WCT (Weibo COVID-19 Testing)
# - Paper: https://doi.org/10.2196/26895
# - Data: Not publicly available

import pandas as pd
import random

# Load data from Excel files
df1 = pd.read_excel('../../data/WCT/核酸检测_人工标记数据_北京内.xlsx')
df2 = pd.read_excel('../../data/WCT/核酸检测_人工标记数据_北京外.xlsx')
df = pd.concat([df1, df2])

# Data processing
df['text'] = df['微博内容'].fillna('') + df['原微博(如果这条微博为转发）'].fillna('')
df['topic'] = df['topic'].map(lambda x: ' ' + str(x) + ' ')
df = df[['text', 'topic']]

# Define categories and instructions
categories = ['18']

# Define the list of prompts for all languages
prompts = [ 
    # Chinese
    "微博帖子内容：{text}。请标记这条微博帖子是否属于真实的微博用户个人表达的核酸检测相关的内容，比如表达个人的态度、意见、观点，或记叙某个人的行为做法。也就是说，这个帖子肯定不是1）新闻报道；2）政府、公司等机构账户发布的公告；3）无个人评论的纯转发，或类似的无关或垃圾信息；4）来自机器人的发帖。如果是个人表达，请回答 'yes'；否则，如果是新闻、公告、纯转发、机器人发的帖等无关个人表达的微博帖子，请回答 'no'（请用英文作答）",

    # English
    "Post content: {text}. Please determine if this post is a personal expression by a real user related to COVID-19 testing, such as expressing personal attitudes, opinions, viewpoints, or describing someone's behavior. This post should definitely NOT be: 1) a news report; 2) an announcement from a government, company, or other institutional account; 3) a pure repost without personal commentary or other irrelevant or spam content; 4) a post made by a bot. If it is personal expression, answer 'yes'; otherwise, if it is a news report, announcement, pure repost, bot post, or other non-personal expression, answer 'no' (answer in English).",

    # German
    "Beitragsinhalt: {text}. Bitte geben Sie an, ob dieser Beitrag ein persönlicher Ausdruck eines echten Benutzers in Bezug auf COVID-19-Tests ist, wie z. B. das Äußern von persönlichen Einstellungen, Meinungen, Ansichten oder das Beschreiben des Verhaltens einer Person. Dieser Beitrag sollte auf keinen Fall sein: 1) ein Nachrichtenbericht; 2) eine Ankündigung von einer Regierungs-, Unternehmens- oder anderen institutionellen Seite; 3) eine reine Weiterleitung ohne persönlichen Kommentar oder anderer irrelevanter/spamartiger Inhalt; 4) ein Beitrag, der von einem Bot erstellt wurde. Wenn es sich um einen persönlichen Ausdruck handelt, antworten Sie bitte mit 'yes'; andernfalls, wenn es ein Nachrichtenbericht, eine Ankündigung, eine reine Weiterleitung, ein Bot-Beitrag oder ein anderer nicht-persönlicher Ausdruck ist, antworten Sie bitte mit 'no' (Antwort auf Englisch).",

    # French
    "Contenu du post : {text}. Veuillez indiquer si ce post est une expression personnelle d'un utilisateur réel en lien avec les tests de COVID-19, telle qu'une attitude, une opinion, un point de vue personnel ou la description d'un comportement. Ce post ne doit absolument pas être : 1) un rapport d'actualité ; 2) une annonce provenant d'un compte gouvernemental, d'une entreprise ou d'une autre institution ; 3) un simple partage sans commentaire personnel, ou tout autre contenu non pertinent ou spam ; 4) un post fait par un bot. Si c'est une expression personnelle, répondez 'yes' ; sinon, s'il s'agit d'un rapport d'actualité, d'une annonce, d'un simple partage, d'un post de bot ou d'une autre expression non personnelle, répondez 'no' (répondez en anglais).",

    # Spanish
    "Contenido de la publicación: {text}. Por favor, indique si esta publicación es una expresión personal de un usuario real relacionada con las pruebas de COVID-19, como expresar actitudes, opiniones, puntos de vista personales o describir el comportamiento de alguien. Esta publicación definitivamente NO debe ser: 1) un informe de noticias; 2) un anuncio de una cuenta gubernamental, empresarial o de otra institución; 3) una simple republicación sin comentario personal u otro contenido irrelevante o spam; 4) una publicación hecha por un bot. Si es una expresión personal, responda 'yes'; de lo contrario, si es un informe de noticias, un anuncio, una simple republicación, una publicación de bot u otra expresión no personal, responda 'no' (responda en inglés).",

    # Portuguese
    "Conteúdo da publicação: {text}. Por favor, indique se esta publicação é uma expressão pessoal de um usuário real relacionada com testes de COVID-19, como a expressão de atitudes, opiniões, pontos de vista pessoais ou a descrição do comportamento de alguém. Esta publicação definitivamente NÃO deve ser: 1) um relatório de notícias; 2) um anúncio de uma conta governamental, empresarial ou de outra instituição; 3) uma repostagem pura sem comentário pessoal ou outro conteúdo irrelevante ou de spam; 4) uma publicação feita por um bot. Se for uma expressão pessoal, responda 'yes'; caso contrário, se for um relatório de notícias, anúncio, repostagem pura, publicação de bot ou outra expressão não pessoal, responda 'no' (responda em inglês).",

    # Italian
    "Contenuto del post: {text}. Si prega di indicare se questo post è un'espressione personale di un utente reale relativa ai test COVID-19, come esprimere atteggiamenti, opinioni, punti di vista personali o descrivere il comportamento di qualcuno. Questo post non dovrebbe assolutamente essere: 1) un rapporto di notizie; 2) un annuncio da parte di un account governativo, aziendale o di un'altra istituzione; 3) una ripubblicazione pura senza commenti personali o altro contenuto irrilevante o spam; 4) un post fatto da un bot. Se si tratta di un'espressione personale, rispondi 'yes'; altrimenti, se è un rapporto di notizie, un annuncio, una ripubblicazione pura, un post di bot o un'altra espressione non personale, rispondi 'no' (rispondi in inglese).",

    # Dutch
    "Inhoud van de post: {text}. Gelieve aan te geven of deze post een persoonlijke uiting is van een echte gebruiker met betrekking tot COVID-19-testen, zoals het uiten van persoonlijke houdingen, meningen, standpunten of het beschrijven van iemands gedrag. Deze post mag zeker GEEN van het volgende zijn: 1) een nieuwsrapport; 2) een aankondiging van een overheid, bedrijf of andere institutionele account; 3) een pure repost zonder persoonlijk commentaar of andere irrelevante of spamachtige inhoud; 4) een post gemaakt door een bot. Als het een persoonlijke uiting is, antwoord 'yes'; anders, als het een nieuwsrapport, aankondiging, pure repost, botpost of andere niet-persoonlijke uiting is, antwoord 'no' (antwoord in het Engels).",

    # Russian
    "Содержание поста: {text}. Пожалуйста, укажите, является ли этот пост личным выражением реального пользователя, связанным с тестированием на COVID-19, например, выражением личных настроений, мнений, точек зрения или описанием чьего-то поведения. Этот пост точно НЕ должен быть: 1) новостным сообщением; 2) объявлением от правительственного, корпоративного или другого институционального аккаунта; 3) чистым репостом без личных комментариев или другим нерелевантным или спам-контентом; 4) постом, созданным ботом. Если это личное выражение, ответьте 'yes'; если это новостное сообщение, объявление, чистый репост, пост бота или другое не личное выражение, ответьте 'no' (ответ на английском).",

    # Czech
    "Obsah příspěvku: {text}. Prosím určete, zda se jedná o osobní vyjádření skutečného uživatele týkající se testování na COVID-19, jako například vyjádření osobních postojů, názorů, pohledů nebo popisu něčího chování. Tento příspěvek by rozhodně NEMĚL být: 1) zprávou; 2) oznámením od vládního, firemního nebo jiného institucionálního účtu; 3) čistým repostem bez osobního komentáře nebo jiný irelevantní nebo spamový obsah; 4) příspěvkem vytvořeným botem. Pokud jde o osobní vyjádření, odpovězte 'yes'; jinak, pokud se jedná o zprávu, oznámení, čistý repost, příspěvek bota nebo jiné neosobní vyjádření, odpovězte 'no' (odpovězte anglicky).",

    # Polish
    "Treść posta: {text}. Proszę określić, czy ten post jest osobistą wypowiedzią prawdziwego użytkownika dotyczącą testów na COVID-19, np. wyrażaniem osobistych postaw, opinii, punktów widzenia lub opisywaniem czyjegoś zachowania. Ten post na pewno NIE powinien być: 1) raportem informacyjnym; 2) ogłoszeniem z konta rządowego, firmowego lub innego instytucjonalnego; 3) czystym repostem bez osobistego komentarza lub inną nieistotną lub spamową treścią; 4) postem stworzonym przez bota. Jeśli jest to osobista wypowiedź, odpowiedz 'yes'; w przeciwnym razie, jeśli jest to raport informacyjny, ogłoszenie, czysty repost, post bota lub inna nieosobista wypowiedź, odpowiedz 'no' (odpowiedz po angielsku).",

    # Arabic
    "محتوى المنشور: {text}. يرجى تحديد ما إذا كان هذا المنشور تعبيرًا شخصيًا من مستخدم حقيقي متعلقًا بفحوصات COVID-19، مثل التعبير عن مواقف شخصية أو آراء أو وجهات نظر، أو وصف سلوك شخص ما. يجب أن لا يكون هذا المنشور بالتأكيد: 1) تقريرًا إخباريًا؛ 2) إعلانًا من حساب حكومي أو شركة أو حساب مؤسسي آخر؛ 3) إعادة نشر بدون تعليق شخصي أو أي محتوى غير ذي صلة أو مزعج؛ 4) منشورًا من صنع روبوت. إذا كان تعبيرًا شخصيًا، أجب 'yes'؛ وإلا، إذا كان تقريرًا إخباريًا أو إعلانًا أو إعادة نشر أو منشور روبوت أو أي تعبير غير شخصي آخر، أجب 'no' (أجب باللغة الإنجليزية).",

    # Persian
    "محتوای پست: {text}. لطفاً مشخص کنید که آیا این پست یک بیان شخصی از کاربر واقعی مرتبط با تست‌های COVID-19 است، مانند بیان نگرش‌ها، نظرات یا دیدگاه‌های شخصی یا توصیف رفتار کسی. این پست قطعاً نباید باشد: 1) گزارش خبری؛ 2) اعلان از طرف حساب دولتی، شرکتی یا حساب‌های نهادی دیگر؛ 3) بازنشر خالص بدون تفسیر شخصی یا محتوای نامربوط یا اسپم؛ 4) پستی که توسط ربات ساخته شده است. اگر بیان شخصی است، پاسخ 'yes' بدهید؛ در غیر این صورت، اگر گزارش خبری، اعلان، بازنشر خالص، پست ربات یا بیان غیر شخصی است، پاسخ 'no' بدهید (به انگلیسی پاسخ دهید).",

    # Hebrew
    "תוכן הפוסט: {text}. נא לציין אם הפוסט הזה הוא ביטוי אישי של משתמש אמיתי שקשור לבדיקות COVID-19, כמו הבעת עמדות אישיות, דעות, נקודות מבט או תיאור התנהגות של מישהו. הפוסט הזה בוודאות לא צריך להיות: 1) דיווח חדשותי; 2) הודעה מחשבון ממשלתי, חברה או גוף מוסדי אחר; 3) שיתוף מחדש ללא תגובה אישית או תוכן לא רלוונטי או ספאם; 4) פוסט שנעשה על ידי בוט. אם זה ביטוי אישי, אנא השב 'yes'; אחרת, אם זה דיווח חדשותי, הודעה, שיתוף מחדש, פוסט של בוט או ביטוי לא אישי אחר, השב 'no' (השב באנגלית).",

    # Turkish
    "Gönderi içeriği: {text}. Lütfen bu gönderinin bir kullanıcının COVID-19 testi ile ilgili kişisel ifadesi olup olmadığını belirtin. Örneğin, kişisel tutumlar, görüşler veya birinin davranışını anlatma gibi. Bu gönderi kesinlikle şunlar olmamalıdır: 1) bir haber raporu; 2) bir hükümet, şirket veya başka bir kurumsal hesaptan yapılan bir duyuru; 3) kişisel yorum içermeyen salt bir paylaşım veya başka bir ilgisiz veya spam içerik; 4) bir bot tarafından yapılan bir gönderi. Kişisel bir ifade ise 'yes' yanıtını verin; aksi takdirde, eğer haber raporu, duyuru, salt paylaşım, bot gönderisi veya başka bir kişisel olmayan ifade ise 'no' yanıtını verin (İngilizce cevaplayın).",

    # Japanese
    "投稿内容: {text}。この投稿が、COVID-19の検査に関連する実在のユーザーによる個人的な表現、たとえば個人的な態度、意見、見解の表明や、誰かの行動を記述したものかどうかを判断してください。この投稿は、次のいずれかであってはなりません：1）ニュース報道；2）政府、企業、その他の機関のアカウントによる発表；3）個人的なコメントのない単なるリポスト、その他無関係またはスパム的なコンテンツ；4）ボットによる投稿。個人的な表現である場合は 'yes' と答えてください。それ以外、ニュース報道、発表、単なるリポスト、ボット投稿、またはその他の個人的でない表現である場合は 'no' と答えてください（英語で回答してください）。",

    # Korean
    "게시물 내용: {text}. 이 게시물이 COVID-19 테스트와 관련된 실제 사용자의 개인적인 표현인지 여부를 판단해 주세요. 예를 들어, 개인적인 태도, 의견, 관점을 표현하거나 누군가의 행동을 설명하는지 여부입니다. 이 게시물은 다음과 같아서는 안 됩니다: 1) 뉴스 보도; 2) 정부, 회사 또는 기타 기관 계정에서 게시한 공지; 3) 개인적인 의견 없이 단순히 공유한 게시물이나 기타 관련 없는 스팸성 콘텐츠; 4) 봇이 작성한 게시물. 만약 개인적인 표현이라면 'yes'라고 답해 주세요. 그렇지 않으면 뉴스 보도, 공지, 단순 공유, 봇 게시물 또는 기타 개인적이지 않은 표현이라면 'no'라고 답해 주세요 (영어로 답변해 주세요).",

    # Vietnamese
    "Nội dung bài đăng: {text}. Vui lòng xác định xem bài đăng này có phải là sự bày tỏ cá nhân của một người dùng thực tế liên quan đến việc xét nghiệm COVID-19 hay không, chẳng hạn như bày tỏ thái độ, ý kiến, quan điểm cá nhân hoặc mô tả hành vi của ai đó. Bài đăng này chắc chắn không được là: 1) một báo cáo tin tức; 2) một thông báo từ tài khoản của chính phủ, công ty hoặc tổ chức khác; 3) một bài chia sẻ lại không có bình luận cá nhân hoặc nội dung không liên quan hoặc spam; 4) một bài đăng do bot tạo ra. Nếu đó là bày tỏ cá nhân, hãy trả lời 'yes'; nếu không, nếu đó là báo cáo tin tức, thông báo, chia sẻ lại, bài đăng của bot hoặc biểu đạt không mang tính cá nhân khác, hãy trả lời 'no' (trả lời bằng tiếng Anh).",

    # Thai
    "เนื้อหาโพสต์: {text}. โปรดระบุว่าโพสต์นี้เป็นการแสดงออกส่วนตัวของผู้ใช้จริงที่เกี่ยวข้องกับการตรวจหาเชื้อ COVID-19 หรือไม่ เช่น การแสดงทัศนคติส่วนตัว ความคิดเห็น มุมมอง หรือการบรรยายพฤติกรรมของบุคคลใด ๆ โพสต์นี้ไม่ควรเป็น: 1) รายงานข่าว; 2) ประกาศจากบัญชีของรัฐบาล บริษัท หรือบัญชีของสถาบันอื่น ๆ; 3) การโพสต์ซ้ำที่ไม่มีความคิดเห็นส่วนตัว หรือเนื้อหาที่ไม่เกี่ยวข้องหรือเป็นสแปม; 4) โพสต์ที่สร้างโดยบอท หากเป็นการแสดงออกส่วนตัว โปรดตอบว่า 'yes'; หากไม่ใช่ ไม่ว่าจะเป็นรายงานข่าว ประกาศ การโพสต์ซ้ำ โพสต์ของบอท หรือการแสดงออกที่ไม่ใช่ส่วนตัว โปรดตอบว่า 'no' (โปรดตอบเป็นภาษาอังกฤษ)",

    # Indonesian
    "Isi postingan: {text}. Tolong tentukan apakah postingan ini adalah ungkapan pribadi oleh pengguna nyata terkait dengan pengujian COVID-19, seperti mengungkapkan sikap pribadi, opini, pandangan, atau mendeskripsikan perilaku seseorang. Postingan ini jelas tidak boleh berupa: 1) laporan berita; 2) pengumuman dari akun pemerintah, perusahaan, atau institusi lainnya; 3) repost murni tanpa komentar pribadi atau konten tidak relevan atau spam lainnya; 4) postingan yang dibuat oleh bot. Jika itu adalah ungkapan pribadi, jawab 'yes'; jika tidak, jika itu adalah laporan berita, pengumuman, repost murni, postingan bot, atau ekspresi non-pribadi lainnya, jawab 'no' (jawab dalam bahasa Inggris).",

    # Malay
    "Kandungan kiriman: {text}. Sila tentukan sama ada kiriman ini adalah ekspresi peribadi oleh pengguna sebenar berkaitan dengan ujian COVID-19, seperti menyatakan sikap peribadi, pendapat, pandangan, atau menggambarkan tingkah laku seseorang. Kiriman ini semestinya TIDAK boleh: 1) laporan berita; 2) pengumuman daripada akaun kerajaan, syarikat, atau institusi lain; 3) kiriman semula tanpa komen peribadi atau kandungan tidak berkaitan atau spam; 4) kiriman yang dibuat oleh bot. Jika ia adalah ekspresi peribadi, jawab 'yes'; jika tidak, jika ia adalah laporan berita, pengumuman, kiriman semula, kiriman bot, atau ekspresi bukan peribadi, jawab 'no' (jawab dalam bahasa Inggeris).",

    # Lao
    "ເນື້ອໃນໂພສ: {text}. ກະລຸນາກຳນົດວ່າໂພສນີ້ເປັນການແດງອອກສ່ວນຕົວໂດຍຜູ້ໃຊ້ຈິງທີ່ກ່ຽວຂ້ອງກັບການກວດ COVID-19 ຫຼືບໍ່, ຍ້ຽງເຊັ່ນການແດງອອກຄວາມຄິດເຫັນສ່ວນຕົວ, ທັດສະນະສ່ວນຕົວ, ຫຼືການພະນາຍຄວາມປະພຶດຂອງບາງຄົນ. ໂພສນີ້ຕ້ອງບໍ່ແມ່ນ: 1) ການລາຍງານຂ່າວ; 2) ການປະກາດຈາກບັນຊີລັດຖະບານ, ບໍລິສັດ, ຫຼືບັນຊີອົງການອື່ນ; 3) ໂພສທີ່ເປັນການແຊຣ໌ເທົ່ານັ້ນໂດຍບໍ່ມີຄຳຄິດເຫັນສ່ວນຕົວ, ຫຼືເນື້ອຫາທີ່ບໍ່ແມ່ນກ່ຽວຂ້ອງຫຼືສະແປມ; 4) ໂພສທີ່ຖືກສ້າງໂດຍບອດ. ຖ້າແມ່ນການແດງອອກສ່ວນຕົວ, ກະລຸນາຕອບ 'yes'; ຖ້າບໍ່, ຖ້າມັນແມ່ນການລາຍງານຂ່າວ, ປະກາດ, ແຊຣ໌ເທົ່ານັ້ນ, ໂພສຂອງບອດ, ຫຼືການແດງອອກທີ່ບໍ່ແມ່ນສ່ວນຕົວອື່ນ, ກະລຸນາຕອບ 'no' (ຕອບເປັນພາສາອັງກິດ).",

    # Burmese
    "ပို့စ်အကြောင်းအရာ: {text}။ ကျေးဇူးပြု၍ ဒီပို့စ်ဟာ COVID-19 စစ်ဆေးမှုနှင့်စပ်လျဉ်းပြီး တကယ့်အသုံးပြုသူတစ်ဦး၏ ကိုယ်ပိုင်ဖော်ပြချက်ဖြစ်မဖြစ် သေချာစွာခွဲခြားပါ၊ ဥပမာ ကိုယ်ပိုင်ရည်ရွယ်ချက်များ၊ ထင်မြင်ချက်များ၊ သဘောထားများ ဖော်ပြခြင်း သို့မဟုတ် တစ်စုံတစ်ဦး၏ အပြုအမူကို ဖော်ပြခြင်းစသည့် အကြောင်းများဖြစ်ပါသည်။ ဒီပို့စ်ဟာ အောက်ပါအချို့ဖြစ်မှာ မဟုတ်သင့်ပါဘူး- ၁) သတင်းအစီရင်ခံချက်၊ ၂) အစိုးရ၊ ကုမ္ပဏီ သို့မဟုတ် အခြားအဖွဲ့အစည်းများမှ အကောင့်ဖြင့် ထုတ်ပြန်ချက်၊ ၃) ကိုယ်ပိုင်မှတ်ချက် မပါသော repost သီးသန့် တစ်ခု သို့မဟုတ် အခြား မသင့်တော်သော သို့မဟုတ် spam အကြောင်းအရာများ၊ ၄) bot ဖြင့် ဖန်တီးထားသော ပို့စ်။ ပုဂ္ဂိုလ်ရေးဖော်ပြချက်ဖြစ်လျှင် 'yes' ဟုဖြေကြောင်းပြုပါ။ သို့မဟုတ် သတင်းအစီရင်ခံချက်၊ ထုတ်ပြန်ချက်၊ repost သီးသန့်၊ bot ဖြင့် ဖန်တီးထားသော ပို့စ် သို့မဟုတ် ပုဂ္ဂိုလ်ရေး မဟုတ်သည့် ဖော်ပြချက် ဖြစ်လျှင် 'no' ဟုဖြေပါ (အင်္ဂလိပ်လို ဖြေပါ)။",

    # Cebuano
    "Sulod sa post: {text}. Palihug tukma nga itino kung kini nga post usa ka personal nga ekspresyon gikan sa tinuod nga tiggamit bahin sa COVID-19 testing, sama sa pagpadayag sa personal nga mga pamatasan, opinyon, panan-aw, o paglarawan sa pamatasan sa usa ka tawo. Kini nga post dili gayud mahimong: 1) usa ka report sa balita; 2) usa ka anunsyo gikan sa usa ka gobyerno, kompanya, o uban pang institusyon nga account; 3) usa ka tin-aw nga repost nga walay personal nga komentaryo o uban pang walay pulos nga sulod o spam; 4) usa ka post nga gihimo sa usa ka bot. Kung kini usa ka personal nga ekspresyon, tubaga 'yes'; kon dili, kon kini usa ka balita nga report, anunsyo, repost, post nga bot o uban pang dili personal nga ekspresyon, tubaga 'no' (tubaga sa Iningles).",

    # Khmer
    "ខ្លឹមសារប្រកាស៖ {text}។ សូមកំណត់ថាតើប្រកាសនេះជាការបញ្ចេញអារម្មណ៍ផ្ទាល់ខ្លួនរបស់អ្នកប្រើប្រាស់ពិតដែលទាក់ទងនឹងការធ្វើតេស្ត COVID-19 ដូចជាការបញ្ចេញអាកប្បកិរិយាផ្ទាល់ខ្លួន មតិយោបល់ មូលមតិ ឬពិពណ៌នាពីពាក្យចចាមអារាមរបស់នរណាម្នាក់។ ប្រកាសនេះមិនគួរតែជា៖ ១) របាយការណ៍ព័ត៌មាន; ២) សេចក្តីប្រកាសពីគណនីរដ្ឋាភិបាល ក្រុមហ៊ុន ឬគណនីស្ថាប័នផ្សេងទៀត; ៣) ការផ្សាយឡើងវិញដោយមិនមានមតិយោបល់ផ្ទាល់ខ្លួន ឬខ្លឹមសារមិនពាក់ព័ន្ធឬសារឥតប្រយោជន៍ផ្សេងៗទៀត; ៤) ប្រកាសដែលត្រូវបានបង្កើតដោយបុត។ ប្រសិនបើវាជាការបញ្ចេញអារម្មណ៍ផ្ទាល់ខ្លួន សូមឆ្លើយ 'yes'; បើមិនដូច្នោះទេ ប្រសិនបើវាជារបាយការណ៍ព័ត៌មាន សេចក្តីប្រកាស ការផ្សាយឡើងវិញ ប្រកាសបុត ឬការបញ្ចេញអារម្មណ៍ផ្សេងទៀតដែលមិនមែនជាផ្ទាល់ខ្លួន សូមឆ្លើយ 'no' (សូមឆ្លើយជាភាសាអង់គ្លេស)។",

    # Tagalog
    "Nilalaman ng post: {text}. Pakisuri kung ang post na ito ay personal na pahayag ng isang totoong user na may kaugnayan sa COVID-19 testing, tulad ng pagpapahayag ng mga personal na saloobin, opinyon, pananaw, o paglalarawan ng kilos ng isang tao. Ang post na ito ay dapat na HINDI: 1) ulat ng balita; 2) isang anunsyo mula sa isang account ng gobyerno, kumpanya, o ibang institusyonal na account; 3) isang simpleng repost na walang personal na komento o ibang hindi kaugnay o spam na content; 4) isang post na ginawa ng bot. Kung ito ay personal na pahayag, sagutin 'yes'; kung hindi, kung ito ay ulat ng balita, anunsyo, simpleng repost, bot post, o iba pang hindi personal na pahayag, sagutin 'no' (sagutin sa Ingles).",

    # Hindi
    "पोस्ट की सामग्री: {text}। कृपया निर्धारित करें कि यह पोस्ट COVID-19 परीक्षण से संबंधित किसी वास्तविक उपयोगकर्ता की व्यक्तिगत अभिव्यक्ति है या नहीं, जैसे कि व्यक्तिगत दृष्टिकोण, राय, विचारों की अभिव्यक्ति या किसी के व्यवहार का वर्णन करना। यह पोस्ट निश्चित रूप से नहीं होनी चाहिए: 1) एक समाचार रिपोर्ट; 2) सरकार, कंपनी या अन्य संस्थागत खाते से एक घोषणा; 3) व्यक्तिगत टिप्पणी के बिना एक सरल पुनः पोस्ट या अन्य अप्रासंगिक या स्पैम सामग्री; 4) एक बॉट द्वारा बनाई गई पोस्ट। यदि यह व्यक्तिगत अभिव्यक्ति है, तो उत्तर दें 'yes'; अन्यथा, यदि यह समाचार रिपोर्ट, घोषणा, सरल पुनः पोस्ट, बॉट पोस्ट या अन्य गैर-व्यक्तिगत अभिव्यक्ति है, तो उत्तर दें 'no' (अंग्रेजी में उत्तर दें)।",

    # Bengali
    "পোস্টের বিষয়বস্তু: {text}। দয়া করে নির্ধারণ করুন এই পোস্টটি একজন বাস্তব ব্যবহারকারীর COVID-19 পরীক্ষা সম্পর্কিত ব্যক্তিগত মতামত, যেমন ব্যক্তিগত মনোভাব, মতামত, দৃষ্টিভঙ্গি প্রকাশ করা বা কারো আচরণ বর্ণনা করা। এই পোস্টটি অবশ্যই হওয়া উচিত নয়: 1) একটি সংবাদ প্রতিবেদন; 2) সরকার, কোম্পানি বা অন্যান্য প্রাতিষ্ঠানিক অ্যাকাউন্ট থেকে একটি ঘোষণা; 3) কোনো ব্যক্তিগত মন্তব্য ছাড়া শুধুমাত্র পুনঃপোস্ট বা অন্যান্য অপ্রাসঙ্গিক বা স্প্যাম সামগ্রী; 4) একটি বট দ্বারা তৈরি করা পোস্ট। যদি এটি ব্যক্তিগত মতামত হয়, তবে উত্তর দিন 'yes'; অন্যথায়, যদি এটি একটি সংবাদ প্রতিবেদন, ঘোষণা, শুধুমাত্র পুনঃপোস্ট, বট পোস্ট বা অন্যান্য অ-ব্যক্তিগত মতামত হয়, তবে উত্তর দিন 'no' (ইংরেজিতে উত্তর দিন)।",

    # Urdu
    "پوسٹ کا مواد: {text}۔ براہ کرم یہ تعین کریں کہ یہ پوسٹ ایک حقیقی صارف کی ذاتی رائے ہے جو COVID-19 ٹیسٹنگ سے متعلق ہے، جیسے کہ ذاتی رویے، آراء، نقطہ نظر کا اظہار یا کسی کے طرز عمل کی وضاحت کرنا۔ یہ پوسٹ یقینی طور پر نہیں ہونی چاہیے: 1) ایک نیوز رپورٹ؛ 2) حکومت، کمپنی یا دیگر ادارہ جاتی اکاؤنٹ سے اعلان؛ 3) ذاتی تبصرہ کے بغیر ایک سادہ ری پوسٹ یا دیگر غیر متعلقہ یا اسپام مواد؛ 4) ایک بوٹ کی طرف سے بنایا گیا پوسٹ۔ اگر یہ ذاتی اظہار ہے تو جواب دیں 'yes'؛ بصورت دیگر، اگر یہ نیوز رپورٹ، اعلان، سادہ ری پوسٹ، بوٹ پوسٹ یا کوئی اور غیر ذاتی اظہار ہے تو جواب دیں 'no' (جواب انگریزی میں دیں)۔",
]



# Function to create instruction for a given row and category
def create_instruction(text):
    instruction_template = random.choice(prompts)
    instruction = instruction_template.format(text=text)
    return instruction

# Create a new DataFrame that will contain the instructions for each label
instructions_df = pd.DataFrame(columns=['instruction', 'output', 'category'])

# Iterate over each row in the original DataFrame
for index, row in df.iterrows():
    text = str(row['text'])
    for category in categories:
        # Create instruction for the category
        instruction = create_instruction(text)
        if row['topic'].find(" "+category+" ") >= 0:
            output = 'no'
        else:
            output = 'yes'
        instructions_df = pd.concat([instructions_df, pd.DataFrame({'instruction': [instruction], 'output': [output], 'category': [category]})], ignore_index=True)
        
# print(instructions_df.output.value_counts())     
instructions_df[['instruction', 'output']].sample(n=11000, random_state=42).to_parquet("../../data/WCT/WCT-subtask-A.parquet", index=False)


# 然后我们进行多分类
df = df[df['topic']!= ' 18 ']


categories = ['1.2', '1.3', '2.1', '2.2', '2.3', '2.4',
              '3.1', '3.2', '3.3', '4.1', '4.2', '4.3',
              '5.1', '5.2', '6.1', '6.2', '7', '8', '9',
              '10.1', '10.2', '12.1', '12.2', '12.3',
              '13.1', '13.2', '13.3', '13.4', '14', '15']

# Define all the multilingual templates for each category
multilingual_templates = {
    '1.2': [
        # Chinese
        "请分析以下微博内容，并判断其是否提到预约将做或计划做核酸检测。指的是，帖子提到，即将开始检测、已预约、计划预约、准备做核酸检测: {}。如果是，请回答'yes'，否则回答'no'。",
        # English
        "Please analyze the following post and determine if it mentions plans or appointments for nucleic acid testing. This indicates that the post mentions starting the test soon, having made an appointment, planning to make an appointment, or preparing for the test: {}. If so, respond 'yes'; otherwise, respond 'no'.",
        # German
        "Bitte analysieren Sie den folgenden Beitrag und bestimmen Sie, ob er Pläne oder Termine für einen Nukleinsäuretest erwähnt. Dies bedeutet, dass der Beitrag das baldige Starten des Tests, das Vereinbaren eines Termins, das Planen eines Termins oder die Vorbereitung auf den Test erwähnt: {}. Wenn ja, antworten Sie mit 'yes', andernfalls mit 'no'.",
        # French
        "Veuillez analyser le post ci-dessous et déterminer s'il mentionne des plans ou des rendez-vous pour un test d'acide nucléique. Cela indique que le post mentionne le début du test bientôt, la prise d'un rendez-vous, la planification d'un rendez-vous ou la préparation pour le test: {}. Si c'est le cas, répondez 'yes', sinon répondez 'no'.",
        # Spanish
        "Por favor, analice el siguiente post y determine si menciona planes o citas para la prueba de ácido nucleico. Esto indica que la publicación menciona comenzar la prueba pronto, haber hecho una cita, planear hacer una cita o prepararse para la prueba: {}. Si es así, responda 'yes', de lo contrario, responda 'no'.",
        # Portuguese
        "Por favor, analise a postagem abaixo e determine se ela menciona planos ou compromissos para o teste de ácido nucleico. Isso indica que a postagem menciona começar o teste em breve, ter marcado uma consulta, planejar marcar uma consulta ou se preparar para o teste: {}. Se for o caso, responda 'yes', caso contrário, responda 'no'.",
        # Italian
        "Si prega di analizzare il seguente post e determinare se menziona piani o appuntamenti per il test dell'acido nucleico. Ciò indica che il post menziona l'inizio del test a breve, aver fissato un appuntamento, pianificare un appuntamento o prepararsi per il test: {}. In tal caso, rispondi 'yes', altrimenti rispondi 'no'.",
        # Dutch
        "Analyseer alstublieft de volgende post en bepaal of er plannen of afspraken voor nucleïnezuurtesten worden genoemd. Dit geeft aan dat de post vermeldt dat de test binnenkort begint, een afspraak heeft gemaakt, een afspraak plant of zich op de test voorbereidt: {}. Als dat het geval is, antwoord dan met 'yes', anders met 'no'.",
        # Russian
        "Пожалуйста, проанализируйте следующий пост и определите, упоминаются ли в нем планы или записи на тестирование на нуклеиновую кислоту. Это указывает на то, что в посте упоминается скорое начало теста, назначение встречи, планирование встречи или подготовка к тесту: {}. Если да, ответьте 'yes', в противном случае ответьте 'no'.",
        # Czech
        "Analyzujte prosím následující příspěvek a určete, zda zmiňuje plány nebo schůzky na testování nukleové kyseliny. To naznačuje, že příspěvek zmiňuje brzké zahájení testu, dohodnutí schůzky, plánování schůzky nebo přípravu na test: {}. Pokud ano, odpovězte 'yes', jinak odpovězte 'no'.",
        # Polish
        "Proszę przeanalizować poniższy post i ustalić, czy wspomina o planach lub umowach dotyczących testu kwasu nukleinowego. Wskazuje to, że post wspomina o rozpoczęciu testu wkrótce, umówieniu się na spotkanie, planowaniu spotkania lub przygotowaniu się do testu: {}. Jeśli tak, odpowiedz 'yes', w przeciwnym razie odpowiedz 'no'.",
        # Arabic
        "يرجى تحليل المنشور أدناه وتحديد ما إذا كان يشير إلى خطط أو مواعيد لاختبار الحمض النووي. يشير هذا إلى أن المنشور يشير إلى بدء الاختبار قريبًا، أو تحديد موعد، أو التخطيط لتحديد موعد، أو الاستعداد للاختبار: {}. إذا كان الأمر كذلك، أجب 'yes'، وإلا أجب 'no'.",
        # Persian
        "لطفاً پست زیر را تحلیل کنید و تعیین کنید که آیا به برنامه‌ها یا قرارهای آزمایش اسید نوکلئیک اشاره می‌شود یا خیر. این نشان می‌دهد که پست به شروع آزمایش به زودی، تعیین وقت، برنامه‌ریزی وقت یا آماده شدن برای آزمایش اشاره دارد: {}. اگر چنین است، با 'yes' پاسخ دهید، در غیر این صورت 'no' پاسخ دهید.",
        # Hebrew
        "אנא נתחו את הפוסט הבא וקבעו אם הוא מזכיר תוכניות או תורים לבדיקת חומצת גרעין. זה מציין שהפוסט מציין התחלת הבדיקה בקרוב, קביעת פגישה, תכנון פגישה או הכנה לבדיקה: {}. אם כן, השב 'yes', אחרת השב 'no'.",
        # Turkish
        "Lütfen aşağıdaki gönderiyi analiz edin ve nükleik asit testi için plan veya randevulardan bahsedilip bahsedilmediğini belirleyin. Bu, gönderinin yakında teste başlayacağını, randevu aldığını, randevu yapmayı planladığını veya teste hazırlık yaptığını belirttiğini gösterir: {}. Eğer öyleyse, 'yes' yanıtını verin, aksi takdirde 'no' yanıtını verin.",
        # Japanese
        "以下の投稿を分析し、それが核酸検査の計画や予約について言及しているかどうかを判断してください。これにより、投稿がテストをすぐに開始すること、予約をしたこと、予約を計画していること、またはテストの準備をしていることが示されます: {}. そうであれば「yes」と答え、それ以外の場合は「no」と答えてください。",
        # Korean
        "다음 게시물을 분석하고 핵산 검사에 대한 계획이나 약속이 언급되었는지 확인하십시오. 이로 인해 게시물이 곧 검사를 시작하거나, 예약을 하거나, 예약을 계획하거나, 검사를 준비하고 있다고 언급한 것으로 나타납니다: {}. 그렇다면 'yes'로 답하고, 그렇지 않다면 'no'로 답하십시오.",
        # Vietnamese
        "Vui lòng phân tích bài viết dưới đây và xác định xem nó có đề cập đến kế hoạch hoặc cuộc hẹn cho xét nghiệm axit nucleic hay không. Điều này cho thấy rằng bài viết đề cập đến việc bắt đầu thử nghiệm sớm, đã đặt lịch hẹn, lên kế hoạch hẹn hoặc chuẩn bị cho bài kiểm tra: {}. Nếu có, hãy trả lời 'yes', nếu không, hãy trả lời 'no'.",
        # Thai
        "โปรดวิเคราะห์โพสต์ด้านล่างนี้และระบุว่ามีการกล่าวถึงแผนหรือการนัดหมายสำหรับการทดสอบกรดนิวคลีอิกหรือไม่ สิ่งนี้บ่งชี้ว่าการโพสต์ระบุถึงการเริ่มทดสอบในเร็ว ๆ นี้ การนัดหมาย การวางแผนที่จะนัดหมาย หรือการเตรียมการสำหรับการทดสอบ: {}. หากเป็นเช่นนั้น ให้ตอบว่า 'yes' มิฉะนั้นให้ตอบว่า 'no'",
        # Indonesian
        "Harap analisis postingan di bawah ini dan tentukan apakah disebutkan rencana atau janji untuk pengujian asam nukleat. Ini menunjukkan bahwa postingan tersebut menyebutkan memulai pengujian dalam waktu dekat, telah membuat janji, merencanakan janji, atau mempersiapkan pengujian: {}. Jika demikian, balas dengan 'yes', jika tidak, balas dengan 'no'.",
        # Malay
        "Sila analisis siaran di bawah dan tentukan sama ada ia menyebut rancangan atau janji temu untuk ujian asid nukleik. Ini menunjukkan bahawa siaran itu menyebutkan memulakan ujian tidak lama lagi, telah membuat janji temu, merancang untuk membuat janji temu, atau membuat persiapan untuk ujian: {}. Jika ya, sila jawab 'yes', jika tidak, jawab 'no'.",
        # Lao
        "โปรดวิเคราะห์โพสต์ด้านล่างนี้และระบุว่ามีการกล่าวถึงแผนหรือการนัดหมายสำหรับการทดสอบกรดนิวคลีอิกหรือไม่ สิ่งนี้บ่งชี้ว่าการโพสต์ระบุถึงการเริ่มทดสอบในเร็ว ๆ นี้ การนัดหมาย การวางแผนที่จะนัดหมาย หรือการเตรียมการสำหรับการทดสอบ: {}. หากเป็นเช่นนั้น ให้ตอบว่า 'yes' มิฉะนั้นให้ตอบว่า 'no'",
        # Burmese
        "ကျေးဇူးပြု၍ အောက်ပါစာမျက်နှာကိုသုံးသပ်ပြီး နျဴကလီဝမ်အက်ဆစ်စစ်ဆေးမှုအတွက် အစီအစဉ်များ သို့မဟုတ် ချိန်းဆိုချက်များကို ရှာဖွေပါ။ ၎င်းသည် စစ်ဆေးမှုကိုမကြာမီစတင်ရန်၊ ချိန်းဆိုချက်တစ်ခုပြုလုပ်ပြီး၊ ချိန်းဆိုချက်တစ်ခုစီစဉ်ထားခြင်း သို့မဟုတ် စစ်ဆေးမှုအတွက်ပြင်ဆင်ထားကြောင်းကိုဖော်ပြသည်: {}. အကယ်၍အကယ်၍တစ်ခုပြုလုပ်ရန် 'yes' ဖြင့်ပြန်ကြားပါ၊ မဟုတ်ပါက 'no' ဖြင့်ပြန်ကြားပါ။",
        # Cebuano
        "Pagtudlo sa mga plano o mga appointment alang sa nucleic acid testing. Nagpasabot kini nga ang post naghisgot bahin sa pagsugod sa pagsulay sa dili madugay, nga naghimo og appointment, nagplano sa paghimo og appointment, o pagpangandam alang sa pagsulay: {}. Palihug paghimo usa ka paghukom base sa sulod sa ibabaw. Kung mao, tubaga nga 'yes', kung dili, tubaga nga 'no'.",
        # Khmer
        "សូម​វិភាគ​អត្ថបទ​សេដ្ឋកិច្ច​ខាងក្រោម​នេះ ហើយ​សម្រេចថា តើ​វា​មាន​គម្រោង​នោះទេ ឬ​ការណាត់​ជួប​សម្រាប់​ការសាកល្បង​អាស៊ីត​នុយក្លេអ៊ីក។ វា​មានន័យថា​ការ​ប្រកាស​បាន​រំឭក​ពី​ការ​ចាប់ផ្តើម​ធ្វើ​តេស្ត​នៅក្នុង​ពេលឆាប់​នេះ ការណាត់​ជួប ការធ្វើគម្រោង​នេះ សម្រាប់ការណាត់ជួប ឬ​ការ​រៀបចំ​សម្រាប់​ការសាកល្បង៖ {}. ប្រសិនបើ​បាទ/ចាស សូមឆ្លើយថា 'yes' បើ​មិនមែន ឆ្លើយថា 'no'.",
        # Tagalog
        "Pakisuri ang sumusunod na post at tukuyin kung ito ay binabanggit ang mga plano o mga appointment para sa pagsusuri ng nucleic acid. Ipinapahiwatig nito na ang post ay binanggit na magsisimula ang pagsusuri, nag-set na ng appointment, plano na mag-set ng appointment, o naghahanda na para sa pagsusuri: {}. Kung gayon, sumagot ng 'yes', kung hindi, sumagot ng 'no'.",
        # Hindi
        "कृपया निम्नलिखित पोस्ट का विश्लेषण करें और निर्धारित करें कि क्या इसमें न्यूक्लिक एसिड परीक्षण के लिए योजनाओं या अपॉइंटमेंट का उल्लेख है। इसका मतलब है कि पोस्ट में परीक्षण शुरू करने, अपॉइंटमेंट लेने, अपॉइंटमेंट की योजना बनाने या परीक्षण की तैयारी का उल्लेख किया गया है: {}. यदि हां, तो 'yes' का उत्तर दें; अन्यथा, 'no' का उत्तर दें।",
        # Bengali
        "অনুগ্রহ করে নিম্নলিখিত পোস্টটি বিশ্লেষণ করুন এবং নির্ধারণ করুন এটি নিউক্লিক অ্যাসিড পরীক্ষার জন্য পরিকল্পনা বা অ্যাপয়েন্টমেন্টের উল্লেখ করে কিনা। এর মানে হল পোস্টটি পরীক্ষাটি শুরু করতে, একটি অ্যাপয়েন্টমেন্ট সেট করতে, অ্যাপয়েন্টমেন্টের পরিকল্পনা করতে বা পরীক্ষার জন্য প্রস্তুতির উল্লেখ করেছে: {}. যদি তাই হয়, 'yes' দিয়ে উত্তর দিন; অন্যথায়, 'no' দিয়ে উত্তর দিন।",
        # Urdu
        "براہ کرم نیچے دی گئی پوسٹ کا تجزیہ کریں اور تعین کریں کہ آیا یہ نیوکلیک ایسڈ ٹیسٹنگ کے لیے منصوبوں یا اپوائنٹمنٹس کا ذکر کرتی ہے۔ اس کا مطلب یہ ہے کہ پوسٹ ٹیسٹ شروع کرنے، اپوائنٹمنٹ کرنے، اپوائنٹمنٹ کا منصوبہ بنانے یا ٹیسٹ کی تیاری کا ذکر کرتی ہے: {}. اگر ایسا ہے تو، 'yes' کے ساتھ جواب دیں؛ بصورت دیگر، 'no' کے ساتھ جواب دیں۔"
    ],
    '1.3': [
        # Chinese
        "请分析以下微博内容，并判断其是否提到做过核酸检测，且结果呈阴性: {}。如果是，请回答'yes'，否则回答'no'。",
        # English
        "Please analyze the following post and determine if it mentions having done a nucleic acid test, with a negative result: {}. If so, respond 'yes'; otherwise, respond 'no'.",
        # German
        "Bitte analysieren Sie den folgenden Beitrag und bestimmen Sie, ob er erwähnt, einen Nukleinsäuretest gemacht zu haben, mit einem negativen Ergebnis: {}. Wenn ja, antworten Sie mit 'yes', andernfalls mit 'no'.",
        # French
        "Veuillez analyser le post ci-dessous et déterminer s'il mentionne avoir fait un test d'acide nucléique avec un résultat négatif: {}. Si c'est le cas, répondez 'yes', sinon répondez 'no'.",
        # Spanish
        "Por favor, analice el siguiente post y determine si menciona haber hecho una prueba de ácido nucleico, con un resultado negativo: {}. Si es así, responda 'yes', de lo contrario, responda 'no'.",
        # Portuguese
        "Por favor, analise a postagem abaixo e determine se ela menciona ter feito um teste de ácido nucleico, com um resultado negativo: {}. Se for o caso, responda 'yes', caso contrário, responda 'no'.",
        # Italian
        "Si prega di analizzare il seguente post e determinare se menziona di aver fatto un test dell'acido nucleico, con un risultato negativo: {}. In tal caso, rispondi 'yes', altrimenti rispondi 'no'.",
        # Dutch
        "Analyseer alstublieft de volgende post en bepaal of er wordt vermeld dat een nucleïnezuurtest is uitgevoerd, met een negatief resultaat: {}. Als dat het geval is, antwoord dan met 'yes', anders met 'no'.",
        # Russian
        "Пожалуйста, проанализируйте следующий пост и определите, упоминается ли в нем, что был проведен тест на нуклеиновую кислоту с отрицательным результатом: {}. Если да, ответьте 'yes', в противном случае ответьте 'no'.",
        # Czech
        "Analyzujte prosím následující příspěvek a určete, zda zmiňuje provedení testu na nukleovou kyselinu, s negativním výsledkem: {}. Pokud ano, odpovězte 'yes', jinak odpovězte 'no'.",
        # Polish
        "Proszę przeanalizować poniższy post i ustalić, czy wspomina o przeprowadzeniu testu na kwas nukleinowy, z wynikiem negatywnym: {}. Jeśli tak, odpowiedz 'yes', w przeciwnym razie odpowiedz 'no'.",
        # Arabic
        "يرجى تحليل المنشور أدناه وتحديد ما إذا كان يذكر أنه تم إجراء اختبار الحمض النووي بنتيجة سلبية: {}. إذا كان الأمر كذلك، أجب 'yes'، وإلا أجب 'no'.",
        # Persian
        "لطفاً پست زیر را تحلیل کنید و تعیین کنید که آیا اشاره شده است که آزمایش اسید نوکلئیک با نتیجه منفی انجام شده است یا خیر: {}. اگر چنین است، با 'yes' پاسخ دهید، در غیر این صورت 'no' پاسخ دهید.",
        # Hebrew
        "אנא נתחו את הפוסט הבא וקבעו אם הוא מזכיר ביצוע בדיקת חומצת גרעין עם תוצאה שלילית: {}. אם כן, השב 'yes', אחרת השב 'no'.",
        # Turkish
        "Lütfen aşağıdaki gönderiyi analiz edin ve nükleik asit testi yapıldığını ve sonucunun negatif olduğunu belirtip belirtmediğini belirleyin: {}. Eğer öyleyse, 'yes' yanıtını verin, aksi takdirde 'no' yanıtını verin.",
        # Japanese
        "以下の投稿を分析し、核酸検査を受けたことが記載されており、その結果が陰性であるかどうかを判断してください: {}。 そうであれば「yes」と答え、それ以外の場合は「no」と答えてください。",
        # Korean
        "다음 게시물을 분석하고 핵산 검사를 받았다고 언급되었으며 결과가 음성인지 확인하십시오: {}. 그렇다면 'yes'로 답하고, 그렇지 않다면 'no'로 답하십시오.",
        # Vietnamese
        "Vui lòng phân tích bài viết dưới đây và xác định xem nó có đề cập đến việc đã thực hiện xét nghiệm axit nucleic, với kết quả âm tính hay không: {}. Nếu có, hãy trả lời 'yes', nếu không, hãy trả lời 'no'.",
        # Thai
        "โปรดวิเคราะห์โพสต์ด้านล่างนี้และระบุว่ามีการกล่าวถึงการทดสอบกรดนิวคลีอิกหรือไม่ และผลลัพธ์เป็นลบหรือไม่: {}. หากเป็นเช่นนั้น ให้ตอบว่า 'yes' มิฉะนั้นให้ตอบว่า 'no'",
        # Indonesian
        "Harap analisis postingan di bawah ini dan tentukan apakah disebutkan bahwa telah dilakukan pengujian asam nukleat, dengan hasil negatif: {}. Jika demikian, balas dengan 'yes', jika tidak, balas dengan 'no'.",
        # Malay
        "Sila analisis siaran di bawah dan tentukan sama ada ia menyebut bahawa ujian asid nukleik telah dilakukan, dengan hasil negatif: {}. Jika ya, sila jawab 'yes', jika tidak, jawab 'no'.",
        # Lao
        "โปรดวิเคราะห์โพสต์ด้านล่างนี้และระบุว่ามีการกล่าวถึงการทดสอบกรดนิวคลีอิกหรือไม่ และผลลัพธ์เป็นลบหรือไม่: {}. หากเป็นเช่นนั้น ให้ตอบว่า 'yes' มิฉะนั้นให้ตอบว่า 'no'",
        # Burmese
        "ကျေးဇူးပြု၍ အောက်ပါစာမျက်နှာကိုသုံးသပ်ပြီး နျဴကလီဝမ်အက်ဆစ်စစ်ဆေးမှုအတွက် အစီအစဉ်များ သို့မဟုတ် ချိန်းဆိုချက်များကို ရှာဖွေပါ။ ၎င်းသည် စစ်ဆေးမှုကိုမကြာမီစတင်ရန်၊ ချိန်းဆိုချက်တစ်ခုပြုလုပ်ပြီး၊ ချိန်းဆိုချက်တစ်ခုစီစဉ်ထားခြင်း သို့မဟုတ် စစ်ဆေးမှုအတွက်ပြင်ဆင်ထားကြောင်းကိုဖော်ပြသည်။ {}. အကယ်၍အကယ်၍တစ်ခုပြုလုပ်ရန် 'yes' ဖြင့်ပြန်ကြားပါ၊ မဟုတ်ပါက 'no' ဖြင့်ပြန်ကြားပါ။",
        # Cebuano
        "Pagtudlo sa mga plano o mga appointment alang sa nucleic acid testing. Nagpasabot kini nga ang post naghisgot bahin sa pagsugod sa pagsulay sa dili madugay, nga naghimo og appointment, nagplano sa paghimo og appointment, o pagpangandam alang sa pagsulay: {}. Palihug paghimo usa ka paghukom base sa sulod sa ibabaw. Kung mao, tubaga nga 'yes', kung dili, tubaga nga 'no'.",
        # Khmer
        "សូម​វិភាគ​អត្ថបទ​សេដ្ឋកិច្ច​ខាងក្រោម​នេះ ហើយ​សម្រេចថា តើ​វា​មាន​គម្រោង​នោះទេ ឬ​ការណាត់​ជួប​សម្រាប់​ការសាកល្បង​អាស៊ីត​នុយក្លេអ៊ីក។ វា​មានន័យថា​ការ​ប្រកាស​បាន​រំឭក​ពី​ការ​ចាប់ផ្តើម​ធ្វើ​តេស្ត​នៅក្នុង​ពេលឆាប់​នេះ ការណាត់​ជួប ការធ្វើគម្រោង​នេះ សម្រាប់ការណាត់ជួប ឬ​ការ​រៀបចំ​សម្រាប់​ការសាកល្បង៖ {}. ប្រសិនបើ​បាទ/ចាស សូមឆ្លើយថា 'yes' បើ​មិនមែន ឆ្លើយថា 'no'.",
        # Tagalog
        "Pakisuri ang sumusunod na post at tukuyin kung ito ay binabanggit ang mga plano o mga appointment para sa pagsusuri ng nucleic acid. Ipinapahiwatig nito na ang post ay binanggit na magsisimula ang pagsusuri, nag-set na ng appointment, plano na mag-set ng appointment, o naghahanda na para sa pagsusuri: {}. Kung gayon, sumagot ng 'yes', kung hindi, sumagot ng 'no'.",
        # Hindi
        "कृपया निम্নलिखित पोस्ट का विश्लेषण करें और निर्धारित करें कि क्या इसमें न्यूक्लिक एसिड परीक्षण के लिए योजनाओं या अपॉइंटमेंट का उल्लेख है। इसका मतलब है कि पोस्ट में परीक्षण शुरू करने, अपॉइंटमेंट लेने, अपॉइंटमेंट की योजना बनाने या परीक्षण की तैयारी का उल्लेख किया गया है: {}. यदि हां, तो 'yes' का उत्तर दें; अन्यथा, 'no' का उत्तर दें।",
        # Bengali
        "অনুগ্রহ করে নিম্নলিখিত পোস্টটি বিশ্লেষণ করুন এবং নির্ধারণ করুন এটি নিউক্লিক অ্যাসিড পরীক্ষার জন্য পরিকল্পনা বা অ্যাপয়েন্টমেন্টের উল্লেখ করে কিনা। এর মানে হল পোস্টটি পরীক্ষাটি শুরু করতে, একটি অ্যাপয়েন্টমেন্ট সেট করতে, অ্যাপয়েন্টমেন্টের পরিকল্পনা করতে বা পরীক্ষার জন্য প্রস্তুতির উল্লেখ করেছে: {}. যদি তাই হয়, 'yes' দিয়ে উত্তর দিন; অন্যথায়, 'no' দিয়ে উত্তর দিন।",
        # Urdu
        "براہ کرم نیچے دی گئی پوسٹ کا تجزیہ کریں اور تعین کریں کہ آیا یہ نیوکلیک ایسڈ ٹیسٹنگ کے لیے منصوبوں یا اپوائنٹمنٹس کا ذکر کرتی ہے۔ اس کا مطلب یہ ہے کہ پوسٹ ٹیسٹ شروع کرنے، اپوائنٹمنٹ کرنے، اپوائنٹمنٹ کا منصوبہ بنانے یا ٹیسٹ کی تیاری کا ذکر کرتی ہے: {}. اگر ایسا ہے تو، 'yes' کے ساتھ جواب دیں؛ بصورت دیگر، 'no' کے ساتھ جواب دیں۔"
    ],
    '1.4': [
        # Chinese
        "请分析以下微博内容，并判断其是否提到做过核酸检测，结果阳性或者假阳性: {}。如果是，请回答'yes'，否则回答'no'。",
        # English
        "Please analyze the following post and determine if it mentions having done a nucleic acid test, with a positive or false positive result: {}. If so, respond 'yes'; otherwise, respond 'no'.",
        # German
        "Bitte analysieren Sie den folgenden Beitrag und bestimmen Sie, ob er erwähnt, einen Nukleinsäuretest gemacht zu haben, mit einem positiven oder falsch positiven Ergebnis: {}. Wenn ja, antworten Sie mit 'yes', andernfalls mit 'no'.",
        # French
        "Veuillez analyser le post ci-dessous et déterminer s'il mentionne avoir fait un test d'acide nucléique avec un résultat positif ou un faux positif: {}. Si c'est le cas, répondez 'yes', sinon répondez 'no'.",
        # Spanish
        "Por favor, analice el siguiente post y determine si menciona haber hecho una prueba de ácido nucleico, con un resultado positivo o falso positivo: {}. Si es así, responda 'yes', de lo contrario, responda 'no'.",
        # Portuguese
        "Por favor, analise a postagem abaixo e determine se ela menciona ter feito um teste de ácido nucleico, com um resultado positivo ou falso positivo: {}. Se for o caso, responda 'yes', caso contrário, responda 'no'.",
        # Italian
        "Si prega di analizzare il seguente post e determinare se menziona di aver fatto un test dell'acido nucleico, con un risultato positivo o falso positivo: {}. In tal caso, rispondi 'yes', altrimenti rispondi 'no'.",
        # Dutch
        "Analyseer alstublieft de volgende post en bepaal of er wordt vermeld dat een nucleïnezuurtest is uitgevoerd, met een positief of vals positief resultaat: {}. Als dat het geval is, antwoord dan met 'yes', anders met 'no'.",
        # Russian
        "Пожалуйста, проанализируйте следующий пост и определите, упоминается ли в нем, что был проведен тест на нуклеиновую кислоту с положительным или ложноположительным результатом: {}. Если да, ответьте 'yes', в противном случае ответьте 'no'.",
        # Czech
        "Analyzujte prosím následující příspěvek a určete, zda zmiňuje provedení testu na nukleovou kyselinu, s pozitivním nebo falešně pozitivním výsledkem: {}. Pokud ano, odpovězte 'yes', jinak odpovězte 'no'.",
        # Polish
        "Proszę przeanalizować poniższy post i ustalić, czy wspomina o przeprowadzeniu testu na kwas nukleinowy, z wynikiem pozytywnym lub fałszywie pozytywnym: {}. Jeśli tak, odpowiedz 'yes', w przeciwnym razie odpowiedz 'no'.",
        # Arabic
        "يرجى تحليل المنشور أدناه وتحديد ما إذا كان يذكر أنه تم إجراء اختبار الحمض النووي بنتيجة إيجابية أو إيجابية كاذبة: {}. إذا كان الأمر كذلك، أجب 'yes'، وإلا أجب 'no'.",
        # Persian
        "لطفاً پست زیر را تحلیل کنید و تعیین کنید که آیا اشاره شده است که آزمایش اسید نوکلئیک با نتیجه مثبت یا مثبت کاذب انجام شده است یا خیر: {}. اگر چنین است، با 'yes' پاسخ دهید، در غیر این صورت 'no' پاسخ دهید.",
        # Hebrew
        "אנא נתחו את הפוסט הבא וקבעו אם הוא מזכיר ביצוע בדיקת חומצת גרעין עם תוצאה חיובית או חיובית כוזבת: {}. אם כן, השב 'yes', אחרת השב 'no'.",
        # Turkish
        "Lütfen aşağıdaki gönderiyi analiz edin ve nükleik asit testi yapıldığını ve sonucunun pozitif veya yanlış pozitif olduğunu belirtip belirtmediğini belirleyin: {}. Eğer öyleyse, 'yes' yanıtını verin, aksi takdirde 'no' yanıtını verin.",
        # Japanese
        "以下の投稿を分析し、核酸検査を受けたことが記載されており、その結果が陽性または偽陽性であるかどうかを判断してください: {}。 そうであれば「yes」と答え、それ以外の場合は「no」と答えてください。",
        # Korean
        "다음 게시물을 분석하고 핵산 검사를 받았다고 언급되었으며 결과가 양성 또는 위양성인지 확인하십시오: {}. 그렇다면 'yes'로 답하고, 그렇지 않다면 'no'로 답하십시오.",
        # Vietnamese
        "Vui lòng phân tích bài viết dưới đây và xác định xem nó có đề cập đến việc đã thực hiện xét nghiệm axit nucleic, với kết quả dương tính hoặc dương tính giả hay không: {}. Nếu có, hãy trả lời 'yes', nếu không, hãy trả lời 'no'.",
        # Thai
        "โปรดวิเคราะห์โพสต์ด้านล่างนี้และระบุว่ามีการกล่าวถึงการทดสอบกรดนิวคลีอิกหรือไม่ และผลลัพธ์เป็นบวกหรือบวกเท็จหรือไม่: {}. หากเป็นเช่นนั้น ให้ตอบว่า 'yes' มิฉะนั้นให้ตอบว่า 'no'",
        # Indonesian
        "Harap analisis postingan di bawah ini dan tentukan apakah disebutkan bahwa telah dilakukan pengujian asam nukleat, dengan hasil positif atau positif palsu: {}. Jika demikian, balas dengan 'yes', jika tidak, balas dengan 'no'.",
        # Malay
        "Sila analisis siaran di bawah dan tentukan sama ada ia menyebut bahawa ujian asid nukleik telah dilakukan, dengan hasil positif atau positif palsu: {}. Jika ya, sila jawab 'yes', jika tidak, jawab 'no'.",
        # Lao
        "โปรดวิเคราะห์โพสต์ด้านล่างนี้และระบุว่ามีการกล่าวถึงการทดสอบกรดนิวคลีอิกหรือไม่ และผลลัพธ์เป็นบวกหรือบวกเท็จหรือไม่: {}. หากเป็นเช่นนั้น ให้ตอบว่า 'yes' มิฉะนั้นให้ตอบว่า 'no'",
        # Burmese
        "ကျေးဇူးပြု၍ အောက်ပါစာမျက်နှာကိုသုံးသပ်ပြီး နျဴကလီဝမ်အက်ဆစ်စစ်ဆေးမှုအတွက် အစီအစဉ်များ သို့မဟုတ် ချိန်းဆိုချက်များကို ရှာဖွေပါ။ ၎င်းသည် စစ်ဆေးမှုကိုမကြာမီစတင်ရန်၊ ချိန်းဆိုချက်တစ်ခုပြုလုပ်ပြီး၊ ချိန်းဆိုချက်တစ်ခုစီစဉ်ထားခြင်း သို့မဟုတ် စစ်ဆေးမှုအတွက်ပြင်ဆင်ထားကြောင်းကိုဖော်ပြသည်။ {}. အကယ်၍အကယ်၍တစ်ခုပြုလုပ်ရန် 'yes' ဖြင့်ပြန်ကြားပါ၊ မဟုတ်ပါက 'no' ဖြင့်ပြန်ကြားပါ။",
        # Cebuano
        "Pagtudlo sa mga plano o mga appointment alang sa nucleic acid testing. Nagpasabot kini nga ang post naghisgot bahin sa pagsugod sa pagsulay sa dili madugay, nga naghimo og appointment, nagplano sa paghimo og appointment, o pagpangandam alang sa pagsulay: {}. Palihug paghimo usa ka paghukom base sa sulod sa ibabaw. Kung mao, tubaga nga 'yes', kung dili, tubaga nga 'no'.",
        # Khmer
        "សូម​វិភាគ​អត្ថបទ​សេដ្ឋកិច្ច​ខាងក្រោម​នេះ ហើយ​សម្រេចថា តើ​វា​មាន​គម្រោង​នោះទេ ឬ​ការណាត់​ជួប​សម្រាប់​ការសាកល្បង​អាស៊ីត​នុយក្លេអ៊ីក។ វា​មានន័យថា​ការ​ប្រកាស​បាន​រំឭក​ពី​ការ​ចាប់ផ្តើម​ធ្វើ​តេស្ត​នៅក្នុង​ពេលឆាប់​នេះ ការណាត់​ជួប ការធ្វើគម្រោង​នេះ សម្រាប់ការណាត់ជួប ឬ​ការ​រៀបចំ​សម្រាប់​ការសាកល្បង៖ {}. ប្រសិនបើ​បាទ/ចាស សូមឆ្លើយថា 'yes' បើ​មិនមែន ឆ្លើយថា 'no'.",
        # Tagalog
        "Pakisuri ang sumusunod na post at tukuyin kung ito ay binabanggit ang mga plano o mga appointment para sa pagsusuri ng nucleic acid. Ipinapahiwatig nito na ang post ay binanggit na magsisimula ang pagsusuri, nag-set na ng appointment, plano na mag-set ng appointment, o naghahanda na para sa pagsusuri: {}. Kung gayon, sumagot ng 'yes', kung hindi, sumagot ng 'no'.",
        # Hindi
        "कृपया निम्नलिखित पोस्ट का विश्लेषण करें और निर्धारित करें कि क्या इसमें न्यूक्लिक एसिड परीक्षण के लिए योजनाओं या अपॉइंटमेंट का उल्लेख है। इसका मतलब है कि पोस्ट में परीक्षण शुरू करने, अपॉइंटमेंट लेने, अपॉइंटमेंट की योजना बनाने या परीक्षण की तैयारी का उल्लेख किया गया है: {}. यदि हां, तो 'yes' का उत्तर दें; अन्यथा, 'no' का उत्तर दें।",
        # Bengali
        "অনুগ্রহ করে নিম্নলিখিত পোস্টটি বিশ্লেষণ করুন এবং নির্ধারণ করুন এটি নিউক্লিক অ্যাসিড পরীক্ষার জন্য পরিকল্পনা বা অ্যাপয়েন্টমেন্টের উল্লেখ করে কিনা। এর মানে হল পোস্টটি পরীক্ষাটি শুরু করতে, একটি অ্যাপয়েন্টমেন্ট সেট করতে, অ্যাপয়েন্টমেন্টের পরিকল্পনা করতে বা পরীক্ষার জন্য প্রস্তুতির উল্লেখ করেছে: {}. যদি তাই হয়, 'yes' দিয়ে উত্তর দিন; অন্যথায়, 'no' দিয়ে উত্তর দিন।",
        # Urdu
        "براہ کرم نیچے دی گئی پوسٹ کا تجزیہ کریں اور تعین کریں کہ آیا یہ نیوکلیک ایسڈ ٹیسٹنگ کے لیے منصوبوں یا اپوائنٹمنٹس کا ذکر کرتی ہے۔ اس کا مطلب یہ ہے کہ پوسٹ ٹیسٹ شروع کرنے، اپوائنٹمنٹ کرنے، اپوائنٹمنٹ کا منصوبہ بنانے یا ٹیسٹ کی تیاری کا ذکر کرتی ہے: {}. اگر ایسا ہے تو، 'yes' کے ساتھ جواب دیں؛ بصورت دیگر، 'no' کے ساتھ جواب دیں۔"
    ],
    '2.1': [
        # Chinese
        "请分析以下微博内容，并判断其是否提到个人因为自己暴露或出现症状而主动去做核酸检测: {}。如果是，请回答'yes'，否则回答'no'。",
        # English
        "Please analyze the following post and determine if it mentions that the individual took the initiative to get tested for nucleic acid due to exposure or symptoms: {}. If so, respond 'yes'; otherwise, respond 'no'.",
        # German
        "Bitte analysieren Sie den folgenden Beitrag und bestimmen Sie, ob er erwähnt, dass die Person aufgrund von Exposition oder Symptomen die Initiative ergriffen hat, sich einem Nukleinsäuretest zu unterziehen: {}. Wenn ja, antworten Sie mit 'yes', andernfalls mit 'no'.",
        # French
        "Veuillez analyser le post ci-dessous et déterminer s'il mentionne que la personne a pris l'initiative de se faire tester pour l'acide nucléique en raison d'une exposition ou de symptômes: {}. Si c'est le cas, répondez 'yes', sinon répondez 'no'.",
        # Spanish
        "Por favor, analice el siguiente post y determine si menciona que el individuo tomó la iniciativa de hacerse la prueba de ácido nucleico debido a la exposición o los síntomas: {}. Si es así, responda 'yes', de lo contrario, responda 'no'.",
        # Portuguese
        "Por favor, analise a postagem abaixo e determine se ela menciona que o indivíduo tomou a iniciativa de fazer o teste de ácido nucleico devido à exposição ou sintomas: {}. Se for o caso, responda 'yes', caso contrário, responda 'no'.",
        # Italian
        "Si prega di analizzare il seguente post e determinare se menziona che l'individuo ha preso l'iniziativa di sottoporsi a un test per l'acido nucleico a causa dell'esposizione o dei sintomi: {}. In tal caso, rispondi 'yes', altrimenti rispondi 'no'.",
        # Dutch
        "Analyseer alstublieft de volgende post en bepaal of er wordt vermeld dat het individu het initiatief heeft genomen om een nucleïnezuurtest te laten doen vanwege blootstelling of symptomen: {}. Als dat het geval is, antwoord dan met 'yes', anders met 'no'.",
        # Russian
        "Пожалуйста, проанализируйте следующий пост и определите, упоминается ли в нем, что человек сам проявил инициативу для прохождения теста на нуклеиновую кислоту из-за воздействия или симптомов: {}. Если да, ответьте 'yes', в противном случае ответьте 'no'.",
        # Czech
        "Analyzujte prosím následující příspěvek a určete, zda zmiňuje, že jedinec z vlastní iniciativy podstoupil test na nukleovou kyselinu kvůli expozici nebo symptomům: {}. Pokud ano, odpovězte 'yes', jinak odpovězte 'no'.",
        # Polish
        "Proszę przeanalizować poniższy post i ustalić, czy wspomina on, że osoba z własnej inicjatywy poddała się testowi na kwas nukleinowy z powodu narażenia lub objawów: {}. Jeśli tak, odpowiedz 'yes', w przeciwnym razie odpowiedz 'no'.",
        # Arabic
        "يرجى تحليل المنشور أدناه وتحديد ما إذا كان يذكر أن الشخص بادر بإجراء اختبار الحمض النووي بسبب التعرض أو الأعراض: {}. إذا كان الأمر كذلك، أجب 'yes'، وإلا أجب 'no'.",
        # Persian
        "لطفاً پست زیر را تحلیل کنید و تعیین کنید که آیا اشاره شده است که فرد به دلیل قرار گرفتن در معرض ویروس یا علائم، به‌صورت خودخواسته آزمایش اسید نوکلئیک انجام داده است: {}. اگر چنین است، با 'yes' پاسخ دهید، در غیر این صورت 'no' پاسخ دهید.",
        # Hebrew
        "אנא נתחו את הפוסט הבא וקבעו אם הוא מזכיר שהאדם לקח יוזמה להיבדק לחומצת גרעין עקב חשיפה או תסמינים: {}. אם כן, השב 'yes', אחרת השב 'no'.",
        # Turkish
        "Lütfen aşağıdaki gönderiyi analiz edin ve kişinin maruziyet veya semptomlar nedeniyle nükleik asit testi yaptırmak için inisiyatif alıp almadığını belirtip belirtmediğini belirleyin: {}. Eğer öyleyse, 'yes' yanıtını verin, aksi takdirde 'no' yanıtını verin.",
        # Japanese
        "以下の投稿を分析し、個人が曝露や症状のために核酸検査を自発的に受けたと記載されているかどうかを判断してください: {}。 そうであれば「yes」と答え、それ以外の場合は「no」と答えてください。",
        # Korean
        "다음 게시물을 분석하고 개인이 노출이나 증상으로 인해 자발적으로 핵산 검사를 받았다고 언급되었는지 확인하십시오: {}. 그렇다면 'yes'로 답하고, 그렇지 않다면 'no'로 답하십시오.",
        # Vietnamese
        "Vui lòng phân tích bài viết dưới đây và xác định xem nó có đề cập rằng cá nhân đã chủ động làm xét nghiệm axit nucleic do tiếp xúc hoặc có triệu chứng hay không: {}. Nếu có, hãy trả lời 'yes', nếu không, hãy trả lời 'no'.",
        # Thai
        "โปรดวิเคราะห์โพสต์ด้านล่างนี้และระบุว่ามีการกล่าวถึงว่าบุคคลนั้นริเริ่มที่จะตรวจกรดนิวคลีอิกเนื่องจากการสัมผัสหรืออาการหรือไม่: {}. หากเป็นเช่นนั้น ให้ตอบว่า 'yes' มิฉะนั้นให้ตอบว่า 'no'",
        # Indonesian
        "Harap analisis postingan di bawah ini dan tentukan apakah disebutkan bahwa individu mengambil inisiatif untuk menjalani pengujian asam nukleat karena paparan atau gejala: {}. Jika demikian, balas dengan 'yes', jika tidak, balas dengan 'no'.",
        # Malay
        "Sila analisis siaran di bawah dan tentukan sama ada ia menyebut bahawa individu tersebut mengambil inisiatif untuk menjalani ujian asid nukleik kerana pendedahan atau gejala: {}. Jika ya, sila jawab 'yes', jika tidak, jawab 'no'.",
        # Lao
        "โปรดวิเคราะห์โพสต์ด้านล่างนี้และระบุว่ามีการกล่าวถึงว่าบุคคลนั้นริเริ่มที่จะตรวจกรดนิวคลีอิกเนื่องจากการสัมผัสหรืออาการหรือไม่: {}. หากเป็นเช่นนั้น ให้ตอบว่า 'yes' มิฉะนั้นให้ตอบว่า 'no'",
        # Burmese
        "ကျေးဇူးပြု၍ အောက်ပါစာမျက်နှာကိုသုံးသပ်ပြီး နျဴကလီဝမ်အက်ဆစ်စစ်ဆေးမှုအတွက် အစီအစဉ်များ သို့မဟုတ် ချိန်းဆိုချက်များကို ရှာဖွေပါ။ ၎င်းသည် စစ်ဆေးမှုကိုမကြာမီစတင်ရန်၊ ချိန်းဆိုချက်တစ်ခုပြုလုပ်ပြီး၊ ချိန်းဆိုချက်တစ်ခုစီစဉ်ထားခြင်း သို့မဟုတ် စစ်ဆေးမှုအတွက်ပြင်ဆင်ထားကြောင်းကိုဖော်ပြသည်။ {}. အကယ်၍အကယ်၍တစ်ခုပြုလုပ်ရန် 'yes' ဖြင့်ပြန်ကြားပါ၊ မဟုတ်ပါက 'no' ဖြင့်ပြန်ကြားပါ။",
        # Cebuano
        "Pagtudlo sa mga plano o mga appointment alang sa nucleic acid testing. Nagpasabot kini nga ang post naghisgot bahin sa pagsugod sa pagsulay sa dili madugay, nga naghimo og appointment, nagplano sa paghimo og appointment, o pagpangandam alang sa pagsulay: {}. Palihug paghimo usa ka paghukom base sa sulod sa ibabaw. Kung mao, tubaga nga 'yes', kung dili, tubaga nga 'no'.",
        # Khmer
        "សូម​វិភាគ​អត្ថបទ​សេដ្ឋកិច្ច​ខាងក្រោម​នេះ ហើយ​សម្រេចថា តើ​វា​មាន​គម្រោង​នោះទេ ឬ​ការណាត់​ជួប​សម្រាប់​ការសាកល្បង​អាស៊ីត​នុយក្លេអ៊ីក។ វា​មានន័យថា​ការ​ប្រកាស​បាន​រំឭក​ពី​ការ​ចាប់ផ្តើម​ធ្វើ​តេស្ត​នៅក្នុង​ពេលឆាប់​នេះ ការណាត់​ជួប ការធ្វើគម្រោង​នេះ សម្រាប់ការណាត់ជួប ឬ​ការ​រៀបចំ​សម្រាប់​ការសាកល្បង៖ {}. ប្រសិនបើ​បាទ/ចាស សូមឆ្លើយថា 'yes' បើ​មិនមែន ឆ្លើយថា 'no'.",
        # Tagalog
        "Pakisuri ang sumusunod na post at tukuyin kung ito ay binabanggit ang mga plano o mga appointment para sa pagsusuri ng nucleic acid. Ipinapahiwatig nito na ang post ay binanggit na magsisimula ang pagsusuri, nag-set na ng appointment, plano na mag-set ng appointment, o naghahanda na para sa pagsusuri: {}. Kung gayon, sumagot ng 'yes', kung hindi, sumagot ng 'no'.",
        # Hindi
        "कृपया निम्नलिखित पोस्ट का विश्लेषण करें और निर्धारित करें कि क्या इसमें न्यूक্লिक एसिड परीक्षण के लिए योजनाओं या अपॉइंटمنٹ का उल्लेख है। इसका मतलब है कि पोस्ट में परीक्षण शुरू करने, अपॉइंटमेंट लेने, अपॉइंटमेंट की योजना बनाने या परीक्षण की तैयारी का उल्लेख किया गया है: {}. यदि हां, तो 'yes' का उत्तर दें; अन्यथा, 'no' का उत्तर दें।",
        # Bengali
        "অনুগ্রহ করে নিম্নলিখিত পোস্টটি বিশ্লেষণ করুন এবং নির্ধারণ করুন এটি নিউক্লিক অ্যাসিড পরীক্ষার জন্য পরিকল্পনা বা অ্যাপয়েন্টমেন্টের উল্লেখ করে কিনা। এর মানে হল পোস্টটি পরীক্ষাটি শুরু করতে, একটি অ্যাপয়েন্টমেন্ট সেট করতে, অ্যাপয়েন্টমেন্টের পরিকল্পনা করতে বা পরীক্ষার জন্য প্রস্তুতির উল্লেখ করেছে: {}. যদি তাই হয়, 'yes' দিয়ে উত্তর দিন; অন্যথায়, 'no' দিয়ে উত্তর দিন।",
        # Urdu
        "براہ کرم نیچے دی گئی پوسٹ کا تجزیہ کریں اور تعین کریں کہ آیا یہ نیوکلیک ایسڈ ٹیسٹنگ کے لیے منصوبوں یا اپوائنٹمنٹس کا ذکر کرتی ہے۔ اس کا مطلب یہ ہے کہ پوسٹ ٹیسٹ شروع کرنے، اپوائنٹمنٹ کرنے، اپوائنٹمنٹ کا منصوبہ بنانے یا ٹیسٹ کی تیاری کا ذکر کرتی ہے: {}. اگر ایسا ہے تو، 'yes' کے ساتھ جواب دیں؛ بصورت دیگر، 'no' کے ساتھ جواب دیں۔"
    ],

    '2.2': [
        # Chinese
        "请分析以下微博内容，并判断其是否提到虽然没有症状，但出于担心而主动去做核酸检测: {}。如果是，请回答'yes'，否则回答'no'。",
        # English
        "Please analyze the following post and determine if it mentions that the individual, despite not having symptoms, took the initiative to get tested for nucleic acid out of concern: {}. If so, respond 'yes'; otherwise, respond 'no'.",
        # German
        "Bitte analysieren Sie den folgenden Beitrag und bestimmen Sie, ob er erwähnt, dass die Person, obwohl sie keine Symptome hat, aus Sorge die Initiative ergriffen hat, sich einem Nukleinsäuretest zu unterziehen: {}. Wenn ja, antworten Sie mit 'yes', andernfalls mit 'no'.",
        # French
        "Veuillez analyser le post ci-dessous et déterminer s'il mentionne que la personne, bien qu'elle ne présente pas de symptômes, a pris l'initiative de se faire tester pour l'acide nucléique par précaution: {}. Si c'est le cas, répondez 'yes', sinon répondez 'no'.",
        # Spanish
        "Por favor, analice el siguiente post y determine si menciona que el individuo, a pesar de no tener síntomas, tomó la iniciativa de hacerse la prueba de ácido nucleico por precaución: {}. Si es así, responda 'yes', de lo contrario, responda 'no'.",
        # Portuguese
        "Por favor, analise a postagem abaixo e determine se ela menciona que o indivíduo, apesar de não ter sintomas, tomou a iniciativa de fazer o teste de ácido nucleico por precaução: {}. Se for o caso, responda 'yes', caso contrário, responda 'no'.",
        # Italian
        "Si prega di analizzare il seguente post e determinare se menziona che l'individuo, pur non avendo sintomi, ha preso l'iniziativa di sottoporsi a un test per l'acido nucleico per precauzione: {}. In tal caso, rispondi 'yes', altrimenti rispondi 'no'.",
        # Dutch
        "Analyseer alstublieft de volgende post en bepaal of er wordt vermeld dat het individu, ondanks het ontbreken van symptomen, uit bezorgdheid het initiatief heeft genomen om een nucleïnezuurtest te laten doen: {}. Als dat het geval is, antwoord dan met 'yes', anders met 'no'.",
        # Russian
        "Пожалуйста, проанализируйте следующий пост и определите, упоминается ли в нем, что человек, несмотря на отсутствие симптомов, сам проявил инициативу для прохождения теста на нуклеиновую кислоту из-за обеспокоенности: {}. Если да, ответьте 'yes', в противном случае ответьте 'no'.",
        # Czech
        "Analyzujte prosím následující příspěvek a určete, zda zmiňuje, že jedinec, přestože nemá příznaky, z vlastní iniciativy podstoupil test na nukleovou kyselinu z obavy: {}. Pokud ano, odpovězte 'yes', jinak odpovězte 'no'.",
        # Polish
        "Proszę przeanalizować poniższy post i ustalić, czy wspomina on, że osoba, mimo braku objawów, z własnej inicjatywy poddała się testowi na kwas nukleinowy z powodu obaw: {}. Jeśli tak, odpowiedz 'yes', w przeciwnym razie odpowiedz 'no'.",
        # Arabic
        "يرجى تحليل المنشور أدناه وتحديد ما إذا كان يذكر أن الشخص بادر بإجراء اختبار الحمض النووي رغم عدم ظهور أعراض عليه بدافع القلق: {}. إذا كان الأمر كذلك، أجب 'yes'، وإلا أجب 'no'.",
        # Persian
        "لطفاً پست زیر را تحلیل کنید و تعیین کنید که آیا اشاره شده است که فرد با وجود نداشتن علائم، به دلیل نگرانی به‌صورت خودخواسته آزمایش اسید نوکلئیک انجام داده است: {}. اگر چنین است، با 'yes' پاسخ دهید، در غیر این صورت 'no' پاسخ دهید.",
        # Hebrew
        "אנא נתחו את הפוסט הבא וקבעו אם הוא מזכיר שהאדם, למרות שאין לו תסמינים, לקח יוזמה להיבדק לחומצת גרעין מתוך חשש: {}. אם כן, השב 'yes', אחרת השב 'no'.",
        # Turkish
        "Lütfen aşağıdaki gönderiyi analiz edin ve kişinin semptomları olmamasına rağmen endişeden dolayı nükleik asit testi yaptırmak için inisiyatif alıp almadığını belirtip belirtmediğini belirleyin: {}. Eğer öyleyse, 'yes' yanıtını verin, aksi takdirde 'no' yanıtını verin.",
        # Japanese
        "以下の投稿を分析し、個人が症状がないにもかかわらず、懸念から核酸検査を自発的に受けたと記載されているかどうかを判断してください: {}。 そうであれば「yes」と答え、それ以外の場合は「no」と答えてください。",
        # Korean
        "다음 게시물을 분석하고 개인이 증상이 없는데도 불구하고 우려 때문에 자발적으로 핵산 검사를 받았다고 언급되었는지 확인하십시오: {}. 그렇다면 'yes'로 답하고, 그렇지 않다면 'no'로 답하십시오.",
        # Vietnamese
        "Vui lòng phân tích bài viết dưới đây và xác định xem nó có đề cập rằng cá nhân, mặc dù không có triệu chứng, đã chủ động làm xét nghiệm axit nucleic vì lo ngại hay không: {}. Nếu có, hãy trả lời 'yes', nếu không, hãy trả lời 'no'.",
        # Thai
        "โปรดวิเคราะห์โพสต์ด้านล่างนี้และระบุว่าบุคคลนั้นริเริ่มที่จะตรวจกรดนิวคลีอิกแม้ว่าจะไม่มีอาการแต่ก็มีความกังวลหรือไม่: {}. หากเป็นเช่นนั้น ให้ตอบว่า 'yes' มิฉะนั้นให้ตอบว่า 'no'",
        # Indonesian
        "Harap analisis postingan di bawah ini dan tentukan apakah disebutkan bahwa individu, meskipun tidak memiliki gejala, mengambil inisiatif untuk menjalani pengujian asam nukleat karena kekhawatiran: {}. Jika demikian, balas dengan 'yes', jika tidak, balas dengan 'no'.",
        # Malay
        "Sila analisis siaran di bawah dan tentukan sama ada ia menyebut bahawa individu tersebut mengambil inisiatif untuk menjalani ujian asid nukleik kerana kebimbangan walaupun tiada gejala: {}. Jika ya, sila jawab 'yes', jika tidak, jawab 'no'.",
        # Lao
        "โปรดวิเคราะห์โพสต์ด้านล่างนี้และระบุว่าบุคคลนั้นริเริ่มที่จะตรวจกรดนิวคลีอิกแม้ว่าจะไม่มีอาการแต่ก็มีความกังวลหรือไม่: {}. หากเป็นเช่นนั้น ให้ตอบว่า 'yes' มิฉะนั้นให้ตอบว่า 'no'",
        # Burmese
        "ကျေးဇူးပြု၍ အောက်ပါစာမျက်နှာကိုသုံးသပ်ပြီး နျဴကလီဝမ်အက်ဆစ်စစ်ဆေးမှုအတွက် အစီအစဉ်များ သို့မဟုတ် ချိန်းဆိုချက်များကို ရှာဖွေပါ။ ၎င်းသည် စစ်ဆေးမှုကိုမကြာမီစတင်ရန်၊ ချိန်းဆိုချက်တစ်ခုပြုလုပ်ပြီး၊ ချိန်းဆိုချက်တစ်ခုစီစဉ်ထားခြင်း သို့မဟုတ် စစ်ဆေးမှုအတွက်ပြင်ဆင်ထားကြောင်းကိုဖော်ပြသည်။ {}. အကယ်၍အကယ်၍တစ်ခုပြုလုပ်ရန် 'yes' ဖြင့်ပြန်ကြားပါ၊ မဟုတ်ပါက 'no' ဖြင့်ပြန်ကြားပါ။",
        # Cebuano
        "Pagtudlo sa mga plano o mga appointment alang sa nucleic acid testing. Nagpasabot kini nga ang post naghisgot bahin sa pagsugod sa pagsulay sa dili madugay, nga naghimo og appointment, nagplano sa paghimo og appointment, o pagpangandam alang sa pagsulay: {}. Palihug paghimo usa ka paghukom base sa sulod sa ibabaw. Kung mao, tubaga nga 'yes', kung dili, tubaga nga 'no'.",
        # Khmer
        "សូម​វិភាគ​អត្ថបទ​សេដ្ឋកិច្ច​ខាងក្រោម​នេះ ហើយ​សម្រេចថា តើ​វា​មាន​គម្រោង​នោះទេ ឬ​ការណាត់​ជួប​សម្រាប់​ការសាកល្បង​អាស៊ីត​នុយក្លេអ៊ីក។ វា​មានន័យថា​ការ​ប្រកាស​បាន​រំឭក​ពី​ការ​ចាប់ផ្តើម​ធ្វើ​តេស្ត​នៅក្នុង​ពេលឆាប់​នេះ ការណាត់​ជួប ការធ្វើគម្រោង​នេះ សម្រាប់ការណាត់ជួប ឬ​ការ​រៀបចំ​សម្រាប់​ការសាកល្បង៖ {}. ប្រសិនបើ​បាទ/ចាស សូមឆ្លើយថា 'yes' បើ​មិនមែន ឆ្លើយថា 'no'.",
        # Tagalog
        "Pakisuri ang sumusunod na post at tukuyin kung ito ay binabanggit ang mga plano o mga appointment para sa pagsusuri ng nucleic acid. Ipinapahiwatig nito na ang post ay binanggit na magsisimula ang pagsusuri, nag-set na ng appointment, plano na mag-set ng appointment, o naghahanda na para sa pagsusuri: {}. Kung gayon, sumagot ng 'yes', kung hindi, sumagot ng 'no'.",
        # Hindi
        "कृपया निम्नलिखित पोस्ट का विश्लेषण करें और निर्धारित करें कि क्या इसमें न्यूक्लिक ایسिड परीक्षण کے لیے منصوبوں یا اپوائنٹمنٹس کا ذکر کرتی ہے۔ اس کا مطلب یہ ہے کہ پوسٹ ٹیسٹ شروع کرنے، اپوائنٹمنٹ کرنے، اپوائنٹمنٹ کا منصوبہ بنانے یا ٹیسٹ کی تیاری کا ذکر کرتی ہے: {}. اگر ایسا ہے تو، 'yes' کے ساتھ جواب دیں؛ بصورت دیگر، 'no' کے ساتھ جواب دیں۔"
    ],
    '2.3': [
        # Chinese
        "请分析以下微博内容，并判断其是否提到做核酸检测是由于政策规定或其他外在因素的影响: {}。如果是，请回答'yes'，否则回答'no'。",
        # English
        "Please analyze the following post and determine if it mentions undergoing nucleic acid testing due to policy regulations or other external factors: {}. If so, respond 'yes'; otherwise, respond 'no'.",
        # German
        "Bitte analysieren Sie den folgenden Beitrag und bestimmen Sie, ob er erwähnt, dass aufgrund von politischen Vorschriften oder anderen externen Faktoren ein Nukleinsäuretest durchgeführt wurde: {}. Wenn ja, antworten Sie mit 'yes', andernfalls mit 'no'.",
        # French
        "Veuillez analyser le post ci-dessous et déterminer s'il mentionne qu'un test d'acide nucléique a été réalisé en raison de règlements politiques ou d'autres facteurs externes: {}. Si c'est le cas, répondez 'yes', sinon répondez 'no'.",
        # Spanish
        "Por favor, analice el siguiente post y determine si menciona que se realizó una prueba de ácido nucleico debido a regulaciones políticas u otros factores externos: {}. Si es así, responda 'yes', de lo contrario, responda 'no'.",
        # Portuguese
        "Por favor, analise a postagem abaixo e determine se ela menciona que o teste de ácido nucleico foi realizado devido a regulamentos políticos ou outros fatores externos: {}. Se for o caso, responda 'yes', caso contrário, responda 'no'.",
        # Italian
        "Si prega di analizzare il seguente post e determinare se menziona che è stato eseguito un test per l'acido nucleico a causa di regolamenti politici o altri fattori esterni: {}. In tal caso, rispondi 'yes', altrimenti rispondi 'no'.",
        # Dutch
        "Analyseer alstublieft de volgende post en bepaal of er wordt vermeld dat een nucleïnezuurtest is uitgevoerd vanwege politieke voorschriften of andere externe factoren: {}. Als dat het geval is, antwoord dan met 'yes', anders met 'no'.",
        # Russian
        "Пожалуйста, проанализируйте следующий пост и определите, упоминается ли в нем, что тест на нуклеиновую кислоту был проведен из-за политических регуляций или других внешних факторов: {}. Если да, ответьте 'yes', в противном случае ответьте 'no'.",
        # Czech
        "Analyzujte prosím následující příspěvek a určete, zda zmiňuje, že test na nukleovou kyselinu byl proveden kvůli politickým předpisům nebo jiným vnějším faktorům: {}. Pokud ano, odpovězte 'yes', jinak odpovězte 'no'.",
        # Polish
        "Proszę przeanalizować poniższy post i ustalić, czy wspomina, że test na kwas nukleinowy został przeprowadzony z powodu przepisów politycznych lub innych czynników zewnętrznych: {}. Jeśli tak, odpowiedz 'yes', w przeciwnym razie odpowiedz 'no'.",
        # Arabic
        "يرجى تحليل المنشور أدناه وتحديد ما إذا كان يذكر أن اختبار الحمض النووي تم إجراؤه بسبب اللوائح السياسية أو العوامل الخارجية الأخرى: {}. إذا كان الأمر كذلك، أجب 'yes'، وإلا أجب 'no'.",
        # Persian
        "لطفاً پست زیر را تحلیل کنید و تعیین کنید که آیا اشاره شده است که آزمایش اسید نوکلئیک به دلیل مقررات سیاسی یا سایر عوامل خارجی انجام شده است: {}. اگر چنین است، با 'yes' پاسخ دهید، در غیر این صورت 'no' پاسخ دهید.",
        # Hebrew
        "אנא נתחו את הפוסט הבא וקבעו אם הוא מזכיר שבדיקת חומצת גרעין בוצעה עקב תקנות מדיניות או גורמים חיצוניים אחרים: {}. אם כן, השב 'yes', אחרת השב 'no'.",
        # Turkish
        "Lütfen aşağıdaki gönderiyi analiz edin ve nükleik asit testinin politik düzenlemeler veya diğer dış etkenler nedeniyle yapıldığını belirtip belirtmediğini belirleyin: {}. Eğer öyleyse, 'yes' yanıtını verin, aksi takdirde 'no' yanıtını verin.",
        # Japanese
        "以下の投稿を分析し、政策規制や他の外部要因により核酸検査が行われたと記載されているかどうかを判断してください: {}。 そうであれば「yes」と答え、それ以外の場合は「no」と答えてください。",
        # Korean
        "다음 게시물을 분석하고 정책 규제 또는 기타 외부 요인으로 인해 핵산 검사가 수행되었다고 언급되었는지 확인하십시오: {}. 그렇다면 'yes'로 답하고, 그렇지 않다면 'no'로 답하십시오.",
        # Vietnamese
        "Vui lòng phân tích bài viết dưới đây và xác định xem nó có đề cập rằng xét nghiệm axit nucleic được thực hiện do các quy định chính sách hoặc các yếu tố bên ngoài khác hay không: {}. Nếu có, hãy trả lời 'yes', nếu không, hãy trả lời 'no'.",
        # Thai
        "โปรดวิเคราะห์โพสต์ด้านล่างนี้และระบุว่ามีการกล่าวถึงว่าการทดสอบกรดนิวคลีอิกนั้นดำเนินการเนื่องจากข้อบังคับนโยบายหรือปัจจัยภายนอกอื่น ๆ หรือไม่: {}. หากเป็นเช่นนั้น ให้ตอบว่า 'yes' มิฉะนั้นให้ตอบว่า 'no'",
        # Indonesian
        "Harap analisis postingan di bawah ini dan tentukan apakah disebutkan bahwa pengujian asam nukleat dilakukan karena peraturan kebijakan atau faktor eksternal lainnya: {}. Jika demikian, balas dengan 'yes', jika tidak, balas dengan 'no'.",
        # Malay
        "Sila analisis siaran di bawah dan tentukan sama ada ia menyebut bahawa ujian asid nukleik dijalankan kerana peraturan dasar atau faktor luaran yang lain: {}. Jika ya, sila jawab 'yes', jika tidak, jawab 'no'.",
        # Lao
        "โปรดวิเคราะห์โพสต์ด้านล่างนี้และระบุว่ามีการกล่าวถึงว่าการทดสอบกรดนิวคลีอิกนั้นดำเนินการเนื่องจากข้อบังคับนโยบายหรือปัจจัยภายนอกอื่น ๆ หรือไม่: {}. หากเป็นเช่นนั้น ให้ตอบว่า 'yes' มิฉะนั้นให้ตอบว่า 'no'",
        # Burmese
        "ကျေးဇူးပြု၍ အောက်ပါစာမျက်နှာကိုသုံးသပ်ပြီး နျဴကလီဝမ်အက်ဆစ်စစ်ဆေးမှုအတွက် အစီအစဉ်များ သို့မဟုတ် ချိန်းဆိုချက်များကို ရှာဖွေပါ။ ၎င်းသည် စစ်ဆေးမှုကိုမကြာမီစတင်ရန်၊ ချိန်းဆိုချက်တစ်ခုပြုလုပ်ပြီး၊ ချိန်းဆိုချက်တစ်ခုစီစဉ်ထားခြင်း သို့မဟုတ် စစ်ဆေးမှုအတွက်ပြင်ဆင်ထားကြောင်းကိုဖော်ပြသည်။ {}. အကယ်၍အကယ်၍တစ်ခုပြုလုပ်ရန် 'yes' ဖြင့်ပြန်ကြားပါ၊ မဟုတ်ပါက 'no' ဖြင့်ပြန်ကြားပါ။",
        # Cebuano
        "Pagtudlo sa mga plano o mga appointment alang sa nucleic acid testing. Nagpasabot kini nga ang post naghisgot bahin sa pagsugod sa pagsulay sa dili madugay, nga naghimo og appointment, nagplano sa paghimo og appointment, o pagpangandam alang sa pagsulay: {}. Palihug paghimo usa ka paghukom base sa sulod sa ibabaw. Kung mao, tubaga nga 'yes', kung dili, tubaga nga 'no'.",
        # Khmer
        "សូម​វិភាគ​អត្ថបទ​សេដ្ឋកិច្ច​ខាងក្រោម​នេះ ហើយ​សម្រេចថា តើ​វា​មាន​គម្រោង​នោះទេ ឬ​ការណាត់​ជួប​សម្រាប់​ការសាកល្បង​អាស៊ីត​នុយក្លេអ៊ីក។ វា​មានន័យថា​ការ​ប្រកាស​បាន​រំឭក​ពី​ការ​ចាប់ផ្តើម​ធ្វើ​តេស្ត​នៅក្នុង​ពេលឆាប់​នេះ ការណាត់​ជួប ការធ្វើគម្រោង​នេះ សម្រាប់ការណាត់ជួប ឬ​ការ​រៀបចំ​សម្រាប់​ការសាកល្បង៖ {}. ប្រសិនបើ​បាទ/ចាស សូមឆ្លើយថា 'yes' បើ​មិនមែន ឆ្លើយថា 'no'.",
        # Tagalog
        "Pakisuri ang sumusunod na post at tukuyin kung ito ay binabanggit ang mga plano o mga appointment para sa pagsusuri ng nucleic acid. Ipinapahiwatig nito na ang post ay binanggit na magsisimula ang pagsusuri, nag-set na ng appointment, plano na mag-set ng appointment, o naghahanda na para sa pagsusuri: {}. Kung gayon, sumagot ng 'yes', kung hindi, sumagot ng 'no'.",
        # Hindi
        "कृपया निम्नलिखित पोस्ट का विश्लेषण करें और निर्धारित करें कि क्या इसमें न्यूक्लिक ایسिड परीक्षण کے لیے منصوبوں یا اپوائنٹمنٹس کا ذکر کرتی ہے۔ اس کا مطلب یہ ہے کہ پوسٹ ٹیسٹ شروع کرنے، اپوائنٹمنٹ کرنے، اپوائنٹمنٹ کا منصوبہ بنانے یا ٹیسٹ کی تیاری کا ذکر کرتی ہے: {}. اگر ایسا ہے تو، 'yes' کے ساتھ جواب دیں؛ بصورت دیگر، 'no' کے ساتھ جواب دیں۔"
    ],

    '2.4': [
        # Chinese
        "请分析以下微博内容，并判断其是否提到政府组织的大规模群体核酸检测: {}。如果是，请回答'yes'，否则回答'no'。",
        # English
        "Please analyze the following post and determine if it mentions government-organized large-scale nucleic acid testing for the community: {}. If so, respond 'yes'; otherwise, respond 'no'.",
        # German
        "Bitte analysieren Sie den folgenden Beitrag und bestimmen Sie, ob er erwähnt, dass die Regierung groß angelegte Nukleinsäuretests für die Gemeinschaft organisiert hat: {}. Wenn ja, antworten Sie mit 'yes', andernfalls mit 'no'.",
        # French
        "Veuillez analyser le post ci-dessous et déterminer s'il mentionne que le gouvernement a organisé des tests d'acide nucléique à grande échelle pour la communauté: {}. Si c'est le cas, répondez 'yes', sinon répondez 'no'.",
        # Spanish
        "Por favor, analice el siguiente post y determine si menciona que el gobierno ha organizado pruebas de ácido nucleico a gran escala para la comunidad: {}. Si es así, responda 'yes', de lo contrario, responda 'no'.",
        # Portuguese
        "Por favor, analise a postagem abaixo e determine se ela menciona que o governo organizou testes de ácido nucleico em grande escala para a comunidade: {}. Se for o caso, responda 'yes', caso contrário, responda 'no'.",
        # Italian
        "Si prega di analizzare il seguente post e determinare se menziona che il governo ha organizzato test su larga scala per l'acido nucleico per la comunità: {}. In tal caso, rispondi 'yes', altrimenti rispondi 'no'.",
        # Dutch
        "Analyseer alstublieft de volgende post en bepaal of er wordt vermeld dat de regering grootschalige nucleïnezuurtesten voor de gemeenschap heeft georganiseerd: {}. Als dat het geval is, antwoord dan met 'yes', anders met 'no'.",
        # Russian
        "Пожалуйста, проанализируйте следующий пост и определите, упоминается ли в нем, что правительство организовало массовое тестирование на нуклеиновую кислоту для сообщества: {}. Если да, ответьте 'yes', в противном случае ответьте 'no'.",
        # Czech
        "Analyzujte prosím následující příspěvek a určete, zda zmiňuje, že vláda zorganizovala hromadné testování nukleové kyseliny pro komunitu: {}. Pokud ano, odpovězte 'yes', jinak odpovězte 'no'.",
        # Polish
        "Proszę przeanalizować poniższy post i ustalić, czy wspomina, że rząd zorganizował masowe testowanie na kwas nukleinowy dla społeczności: {}. Jeśli tak, odpowiedz 'yes', w przeciwnym razie odpowiedz 'no'.",
        # Arabic
        "يرجى تحليل المنشور أدناه وتحديد ما إذا كان يذكر أن الحكومة نظمت اختبارًا جماعيًا لحمض النووي للمجتمع: {}. إذا كان الأمر كذلك، أجب 'yes'، وإلا أجب 'no'.",
        # Persian
        "لطفاً پست زیر را تحلیل کنید و تعیین کنید که آیا اشاره شده است که دولت آزمایش اسید نوکلئیک گسترده‌ای را برای جامعه سازماندهی کرده است: {}. اگر چنین است، با 'yes' پاسخ دهید، در غیر این صورت 'no' پاسخ دهید.",
        # Hebrew
        "אנא נתחו את הפוסט הבא וקבעו אם הוא מזכיר שהממשלה ארגנה בדיקות חומצת גרעין רחבות היקף לקהילה: {}. אם כן, השב 'yes', אחרת השב 'no'.",
        # Turkish
        "Lütfen aşağıdaki gönderiyi analiz edin ve hükümetin topluluk için büyük ölçekli nükleik asit testi düzenlediğini belirtip belirtmediğini belirleyin: {}. Eğer öyleyse, 'yes' yanıtını verin, aksi takdirde 'no' yanıtını verin.",
        # Japanese
        "以下の投稿を分析し、政府が地域社会のために大規模な核酸検査を実施したと記載されているかどうかを判断してください: {}。 そうであれば「yes」と答え、それ以外の場合は「no」と答えてください。",
        # Korean
        "다음 게시물을 분석하고 정부가 지역사회를 위한 대규모 핵산 검사를 조직했다고 언급되었는지 확인하십시오: {}. 그렇다면 'yes'로 답하고, 그렇지 않다면 'no'로 답하십시오.",
        # Vietnamese
        "Vui lòng phân tích bài viết dưới đây và xác định xem nó có đề cập rằng chính phủ đã tổ chức xét nghiệm axit nucleic quy mô lớn cho cộng đồng hay không: {}. Nếu có, hãy trả lời 'yes', nếu không, hãy trả lời 'no'.",
        # Thai
        "โปรดวิเคราะห์โพสต์ด้านล่างนี้และระบุว่ามีการกล่าวถึงว่ารัฐบาลจัดการทดสอบกรดนิวคลีอิกในวงกว้างสำหรับชุมชนหรือไม่: {}. หากเป็นเช่นนั้น ให้ตอบว่า 'yes' มิฉะนั้นให้ตอบว่า 'no'",
        # Indonesian
        "Harap analisis postingan di bawah ini dan tentukan apakah disebutkan bahwa pemerintah menyelenggarakan pengujian asam nukleat berskala besar untuk komunitas: {}. Jika demikian, balas dengan 'yes', jika tidak, balas dengan 'no'.",
        # Malay
        "Sila analisis siaran di bawah dan tentukan sama ada ia menyebut bahawa kerajaan telah menganjurkan ujian asid nukleik berskala besar untuk komuniti: {}. Jika ya, sila jawab 'yes', jika tidak, jawab 'no'.",
        # Lao
        "โปรดวิเคราะห์โพสต์ด้านล่างนี้และระบุว่ามีการกล่าวถึงว่ารัฐบาลจัดการทดสอบกรดนิวคลีอิกในวงกว้างสำหรับชุมชนหรือไม่: {}. หากเป็นเช่นนั้น ให้ตอบว่า 'yes' มิฉะนั้นให้ตอบว่า 'no'",
        # Burmese
        "ကျေးဇူးပြု၍ အောက်ပါစာမျက်နှာကိုသုံးသပ်ပြီး နျဴကလီဝမ်အက်ဆစ်စစ်ဆေးမှုအတွက် အစီအစဉ်များ သို့မဟုတ် ချိန်းဆိုချက်များကို ရှာဖွေပါ။ ၎င်းသည် စစ်ဆေးမှုကိုမကြာမီစတင်ရန်၊ ချိန်းဆိုချက်တစ်ခုပြုလုပ်ပြီး၊ ချိန်းဆိုချက်တစ်ခုစီစဉ်ထားခြင်း သို့မဟုတ် စစ်ဆေးမှုအတွက်ပြင်ဆင်ထားကြောင်းကိုဖော်ပြသည်။ {}. အကယ်၍အကယ်၍တစ်ခုပြုလုပ်ရန် 'yes' ဖြင့်ပြန်ကြားပါ၊ မဟုတ်ပါက 'no' ဖြင့်ပြန်ကြားပါ။",
        # Cebuano
        "Pagtudlo sa mga plano o mga appointment alang sa nucleic acid testing. Nagpasabot kini nga ang post naghisgot bahin sa pagsugod sa pagsulay sa dili madugay, nga naghimo og appointment, nagplano sa paghimo og appointment, o pagpangandam alang sa pagsulay: {}. Palihug paghimo usa ka paghukom base sa sulod sa ibabaw. Kung mao, tubaga nga 'yes', kung dili, tubaga nga 'no'.",
        # Khmer
        "សូម​វិភាគ​អត្ថបទ​សេដ្ឋកិច្ច​ខាងក្រោម​នេះ ហើយ​សម្រេចថា តើ​វា​មាន​គម្រោង​នោះទេ ឬ​ការណាត់​ជួប​សម្រាប់​ការសាកល្បង​អាស៊ីត​នុយក្លេអ៊ីក។ វា​មានន័យថា​ការ​ប្រកាស​បាន​រំឭក​ពី​ការ​ចាប់ផ្តើម​ធ្វើ​តេស្ត​នៅក្នុង​ពេលឆាប់​នេះ ការណាត់ជួប ការធ្វើគម្រោង​នេះ សម្រាប់ការណាត់ជួប ឬ​ការ​រៀបចំ​សម្រាប់​ការសាកល្បង៖ {}. ប្រសិនបើ​បាទ/ចាស សូមឆ្លើយថា 'yes' បើ​មិនមែន ឆ្លើយថា 'no'.",
        # Tagalog
        "Pakisuri ang sumusunod na post at tukuyin kung ito ay binabanggit ang mga plano o mga appointment para sa pagsusuri ng nucleic acid. Ipinapahiwatig nito na ang post ay binanggit na magsisimula ang pagsusuri, nag-set na ng appointment, plano na mag-set ng appointment, o naghahanda na para sa pagsusuri: {}. Kung gayon, sumagot ng 'yes', kung hindi, sumagot ng 'no'.",
        # Hindi
        "कृपया निम्नलिखित पोस्ट का विश्लेषण करें और निर्धारित کریں कि क्या इसमें न्यूक्लिक ایسिड परीक्षण کے لیے منصوبوں یا اپوائنٹمنٹس کا ذکر کرتی ہے۔ اس کا مطلب یہ ہے کہ پوسٹ ٹیسٹ شروع کرنے، اپوائنٹمنٹ کرنے، اپوائنٹمنٹ کا منصوبہ بنانے یا ٹیسٹ کی تیاری کا ذکر کرتی ہے: {}. اگر ایسا ہے تو، 'yes' کے ساتھ جواب دیں؛ بصورت دیگر، 'no' کے ساتھ جواب دیں۔"
    ],
    '3.1': [
        # Chinese
        "请分析以下微博内容，并判断其是否提到支持核酸检测: {}。如果是，请回答'yes'，否则回答'no'。",
        # English
        "Please analyze the following post and determine if it mentions supporting nucleic acid testing: {}. If so, respond 'yes'; otherwise, respond 'no'.",
        # German
        "Bitte analysieren Sie den folgenden Beitrag und bestimmen Sie, ob er die Unterstützung von Nukleinsäuretests erwähnt: {}. Wenn ja, antworten Sie mit 'yes', andernfalls mit 'no'.",
        # French
        "Veuillez analyser le post ci-dessous et déterminer s'il mentionne soutenir les tests d'acide nucléique: {}. Si c'est le cas, répondez 'yes', sinon répondez 'no'.",
        # Spanish
        "Por favor, analice el siguiente post y determine si menciona apoyar las pruebas de ácido nucleico: {}. Si es así, responda 'yes', de lo contrario, responda 'no'.",
        # Portuguese
        "Por favor, analise a postagem abaixo e determine se ela menciona apoiar os testes de ácido nucleico: {}. Se for o caso, responda 'yes', caso contrário, responda 'no'.",
        # Italian
        "Si prega di analizzare il seguente post e determinare se menziona il supporto ai test dell'acido nucleico: {}. In tal caso, rispondi 'yes', altrimenti rispondi 'no'.",
        # Dutch
        "Analyseer alstublieft de volgende post en bepaal of er wordt vermeld dat nucleïnezuurtesten worden ondersteund: {}. Als dat het geval is, antwoord dan met 'yes', anders met 'no'.",
        # Russian
        "Пожалуйста, проанализируйте следующий пост и определите, упоминается ли в нем поддержка тестирования на нуклеиновую кислоту: {}. Если да, ответьте 'yes', в противном случае ответьте 'no'.",
        # Czech
        "Analyzujte prosím následující příspěvek a určete, zda zmiňuje podporu testování nukleové kyseliny: {}. Pokud ano, odpovězte 'yes', jinak odpovězte 'no'.",
        # Polish
        "Proszę przeanalizować poniższy post i ustalić, czy wspomina o wsparciu testów na kwas nukleinowy: {}. Jeśli tak, odpowiedz 'yes', w przeciwnym razie odpowiedz 'no'.",
        # Arabic
        "يرجى تحليل المنشور أدناه وتحديد ما إذا كان يذكر دعم اختبار الحمض النووي: {}. إذا كان الأمر كذلك، أجب 'yes'، وإلا أجب 'no'.",
        # Persian
        "لطفاً پست زیر را تحلیل کنید و تعیین کنید که آیا اشاره شده است که از آزمایش اسید نوکلئیک حمایت شده است: {}. اگر چنین است، با 'yes' پاسخ دهید، در غیر این صورت 'no' پاسخ دهید.",
        # Hebrew
        "אנא נתחו את הפוסט הבא וקבעו אם הוא מזכיר תמיכה בבדיקות חומצת גרעין: {}. אם כן, השב 'yes', אחרת השב 'no'.",
        # Turkish
        "Lütfen aşağıdaki gönderiyi analiz edin ve nükleik asit testlerini desteklediğini belirtip belirtmediğini belirleyin: {}. Eğer öyleyse, 'yes' yanıtını verin, aksi takdirde 'no' yanıtını verin.",
        # Japanese
        "以下の投稿を分析し、核酸検査を支持するかどうか記載されているかどうかを判断してください: {}。 そうであれば「yes」と答え、それ以外の場合は「no」と答えてください。",
        # Korean
        "다음 게시물을 분석하고 핵산 검사를 지지한다고 언급되었는지 확인하십시오: {}. 그렇다면 'yes'로 답하고, 그렇지 않다면 'no'로 답하십시오.",
        # Vietnamese
        "Vui lòng phân tích bài viết dưới đây và xác định xem nó có đề cập đến việc ủng hộ xét nghiệm axit nucleic hay không: {}. Nếu có, hãy trả lời 'yes', nếu không, hãy trả lời 'no'.",
        # Thai
        "โปรดวิเคราะห์โพสต์ด้านล่างนี้และระบุว่ามีการกล่าวถึงการสนับสนุนการทดสอบกรดนิวคลีอิกหรือไม่: {}. หากเป็นเช่นนั้น ให้ตอบว่า 'yes' มิฉะนั้นให้ตอบว่า 'no'",
        # Indonesian
        "Harap analisis postingan di bawah ini dan tentukan apakah disebutkan mendukung pengujian asam nukleat: {}. Jika demikian, balas dengan 'yes', jika tidak, balas dengan 'no'.",
        # Malay
        "Sila analisis siaran di bawah dan tentukan sama ada ia menyebutkan sokongan untuk ujian asid nukleik: {}. Jika ya, sila jawab 'yes', jika tidak, jawab 'no'.",
        # Lao
        "โปรดวิเคราะห์โพสต์ด้านล่างนี้และระบุว่ามีการกล่าวถึงการสนับสนุนการทดสอบกรดนิวคลีอิกหรือไม่: {}. หากเป็นเช่นนั้น ให้ตอบว่า 'yes' มิฉะนั้นให้ตอบว่า 'no'",
        # Burmese
        "ကျေးဇူးပြု၍ အောက်ပါစာမျက်နှာကိုသုံးသပ်ပြီး နျဴကလီဝမ်အက်ဆစ်စစ်ဆေးမှုအတွက် အစီအစဉ်များ သို့မဟုတ် ချိန်းဆိုချက်များကို ရှာဖွေပါ။ ၎င်းသည် စစ်ဆေးမှုကိုမကြာမီစတင်ရန်၊ ချိန်းဆိုချက်တစ်ခုပြုလုပ်ပြီး၊ ချိန်းဆိုချက်တစ်ခုစီစဉ်ထားခြင်း သို့မဟုတ် စစ်ဆေးမှုအတွက်ပြင်ဆင်ထားကြောင်းကိုဖော်ပြသည်။ {}. အကယ်၍အကယ်၍တစ်ခုပြုလုပ်ရန် 'yes' ဖြင့်ပြန်ကြားပါ၊ မဟုတ်ပါက 'no' ဖြင့်ပြန်ကြားပါ။",
        # Cebuano
        "Pagtudlo sa mga plano o mga appointment alang sa nucleic acid testing. Nagpasabot kini nga ang post naghisgot bahin sa pagsugod sa pagsulay sa dili madugay, nga naghimo og appointment, nagplano sa paghimo og appointment, o pagpangandam alang sa pagsulay: {}. Palihug paghimo usa ka paghukom base sa sulod sa ibabaw. Kung mao, tubaga nga 'yes', kung dili, tubaga nga 'no'.",
        # Khmer
        "សូម​វិភាគ​អត្ថបទ​សេដ្ឋកិច្ច​ខាងក្រោម​នេះ ហើយ​សម្រេចថា តើ​វា​មាន​គម្រោង​នោះទេ ឬ​ការណាត់​ជួប​សម្រាប់​ការសាកល្បង​អាស៊ីត​នុយក្លេអ៊ីក។ វា​មានន័យថា​ការ​ប្រកាស​បាន​រំឭក​ពី​ការ​ចាប់ផ្តើម​ធ្វើ​តេស្ត​នៅក្នុង​ពេលឆាប់​នេះ ការណាត់ជួប ការធ្វើគម្រោង​នេះ សម្រាប់ការណាត់ជួប ឬ​ការ​រៀបចំ​សម្រាប់​ការសាកល្បង៖ {}. ប្រសិនបើ​បាទ/ចាស សូមឆ្លើយថា 'yes' បើ​មិនមែន ឆ្លើយថា 'no'.",
        # Tagalog
        "Pakisuri ang sumusunod na post at tukuyin kung ito ay binabanggit ang mga plano o mga appointment para sa pagsusuri ng nucleic acid. Ipinapahiwatig nito na ang post ay binanggit na magsisimula ang pagsusuri, nag-set na ng appointment, plano na mag-set ng appointment, o naghahanda na para sa pagsusuri: {}. Kung gayon, sumagot ng 'yes', kung hindi, sumagot ng 'no'.",
        # Hindi
        "कृपया निम्नलिखित पोस्ट का विश्लेषण करें और निर्धारित करें कि क्या इसमें न्यूक्लिक ایسڈ परीक्षण کے لیے منصوبوں یا اپوائنٹمنٹس کا ذکر کرتی ہے۔ اس کا مطلب یہ ہے کہ پوسٹ ٹیسٹ شروع کرنے، اپوائنٹمنٹ کرنے، اپوائنٹمنٹ کا منصوبہ بنانے یا ٹیسٹ کی تیاری کا ذکر کرتی ہے: {}. اگر ایسا ہے تو، 'yes' کے ساتھ جواب دیں؛ بصورت دیگر، 'no' کے ساتھ جواب دیں۔"
    ],

    '3.2': [
        # Chinese
        "请分析以下微博内容，并判断其是否提到对核酸检测持犹豫态度: {}。如果是，请回答'yes'，否则回答'no'。",
        # English
        "Please analyze the following post and determine if it mentions hesitancy toward nucleic acid testing: {}. If so, respond 'yes'; otherwise, respond 'no'.",
        # German
        "Bitte analysieren Sie den folgenden Beitrag und bestimmen Sie, ob er Zögern gegenüber Nukleinsäuretests erwähnt: {}. Wenn ja, antworten Sie mit 'yes', andernfalls mit 'no'.",
        # French
        "Veuillez analyser le post ci-dessous et déterminer s'il mentionne une hésitation à l'égard des tests d'acide nucléique: {}. Si c'est le cas, répondez 'yes', sinon répondez 'no'.",
        # Spanish
        "Por favor, analice el siguiente post y determine si menciona una hesitación hacia las pruebas de ácido nucleico: {}. Si es así, responda 'yes', de lo contrario, responda 'no'.",
        # Portuguese
        "Por favor, analise a postagem abaixo e determine se ela menciona hesitação em relação aos testes de ácido nucleico: {}. Se for o caso, responda 'yes', caso contrário, responda 'no'.",
        # Italian
        "Si prega di analizzare il seguente post e determinare se menziona esitazione nei confronti dei test dell'acido nucleico: {}. In tal caso, rispondi 'yes', altrimenti rispondi 'no'.",
        # Dutch
        "Analyseer alstublieft de volgende post en bepaal of er wordt vermeld dat er aarzeling is ten opzichte van nucleïnezuurtesten: {}. Als dat het geval is, antwoord dan met 'yes', anders met 'no'.",
        # Russian
        "Пожалуйста, проанализируйте следующий пост и определите, упоминается ли в нем нерешительность по отношению к тестированию на нуклеиновую кислоту: {}. Если да, ответьте 'yes', в противном случае ответьте 'no'.",
        # Czech
        "Analyzujte prosím následující příspěvek a určete, zda zmiňuje váhavost vůči testování nukleové kyseliny: {}. Pokud ano, odpovězte 'yes', jinak odpovězte 'no'.",
        # Polish
        "Proszę przeanalizować poniższy post i ustalić, czy wspomina o wahaniu wobec testów na kwas nukleinowy: {}. Jeśli tak, odpowiedz 'yes', w przeciwnym razie odpowiedz 'no'.",
        # Arabic
        "يرجى تحليل المنشور أدناه وتحديد ما إذا كان يذكر التردد تجاه اختبار الحمض النووي: {}. إذا كان الأمر كذلك، أجب 'yes'، وإلا أجب 'no'.",
        # Persian
        "لطفاً پست زیر را تحلیل کنید و تعیین کنید که آیا اشاره شده است که نسبت به انجام آزمایش اسید نوکلئیک تردید وجود دارد: {}. اگر چنین است، با 'yes' پاسخ دهید، در غیر این صورت 'no' پاسخ دهید.",
        # Hebrew
        "אנא נתחו את הפוסט הבא וקבעו אם הוא מזכיר היסוס כלפי בדיקות חומצת גרעין: {}. אם כן, השב 'yes', אחרת השב 'no'.",
        # Turkish
        "Lütfen aşağıdaki gönderiyi analiz edin ve nükleik asit testlerine yönelik tereddütten bahsedilip bahsedilmediğini belirleyin: {}. Eğer öyleyse, 'yes' yanıtını verin, aksi takdirde 'no' yanıtını verin.",
        # Japanese
        "以下の投稿を分析し、核酸検査に対する躊躇が記載されているかどうかを判断してください: {}。 そうであれば「yes」と答え、それ以外の場合は「no」と答えてください。",
        # Korean
        "다음 게시물을 분석하고 핵산 검사에 대한 주저함이 언급되었는지 확인하십시오: {}. 그렇다면 'yes'로 답하고, 그렇지 않다면 'no'로 답하십시오.",
        # Vietnamese
        "Vui lòng phân tích bài viết dưới đây và xác định xem nó có đề cập đến sự do dự đối với việc xét nghiệm axit nucleic hay không: {}. Nếu có, hãy trả lời 'yes', nếu không, hãy trả lời 'no'.",
        # Thai
        "โปรดวิเคราะห์โพสต์ด้านล่างนี้และระบุว่ามีการกล่าวถึงความลังเลต่อการทดสอบกรดนิวคลีอิกหรือไม่: {}. หากเป็นเช่นนั้น ให้ตอบว่า 'yes' มิฉะนั้นให้ตอบว่า 'no'",
        # Indonesian
        "Harap analisis postingan di bawah ini dan tentukan apakah disebutkan keraguan terhadap pengujian asam nukleat: {}. Jika demikian, balas dengan 'yes', jika tidak, balas dengan 'no'.",
        # Malay
        "Sila analisis siaran di bawah dan tentukan sama ada ia menyebutkan keraguan terhadap ujian asid nukleik: {}. Jika ya, sila jawab 'yes', jika tidak, jawab 'no'.",
        # Lao
        "โปรดวิเคราะห์โพสต์ด้านล่างนี้และระบุว่ามีการกล่าวถึงความลังเลต่อการทดสอบกรดนิวคลีอิกหรือไม่: {}. หากเป็นเช่นนั้น ให้ตอบว่า 'yes' มิฉะนั้นให้ตอบว่า 'no'",
        # Burmese
        "ကျေးဇူးပြု၍ အောက်ပါစာမျက်နှာကိုသုံးသပ်ပြီး နျဴကလီဝမ်အက်ဆစ်စစ်ဆေးမှုအတွက် အစီအစဉ်များ သို့မဟုတ် ချိန်းဆိုချက်များကို ရှာဖွေပါ။ ၎င်းသည် စစ်ဆေးမှုကိုမကြာမီစတင်ရန်၊ ချိန်းဆိုချက်တစ်ခုပြုလုပ်ပြီး၊ ချိန်းဆိုချက်တစ်ခုစီစဉ်ထားခြင်း သို့မဟုတ် စစ်ဆေးမှုအတွက်ပြင်ဆင်ထားကြောင်းကိုဖော်ပြသည်။ {}. အကယ်၍အကယ်၍တစ်ခုပြုလုပ်ရန် 'yes' ဖြင့်ပြန်ကြားပါ၊ မဟုတ်ပါက 'no' ဖြင့်ပြန်ကြားပါ။",
        # Cebuano
        "Pagtudlo sa mga plano o mga appointment alang sa nucleic acid testing. Nagpasabot kini nga ang post naghisgot bahin sa pagsugod sa pagsulay sa dili madugay, nga naghimo og appointment, nagplano sa paghimo og appointment, o pagpangandam alang sa pagsulay: {}. Palihug paghimo usa ka paghukom base sa sulod sa ibabaw. Kung mao, tubaga nga 'yes', kung dili, tubaga nga 'no'.",
        # Khmer
        "សូម​វិភាគ​អត្ថបទ​សេដ្ឋកិច្ច​ខាងក្រោម​នេះ ហើយ​សម្រេចថា តើ​វា​មាន​គម្រោង​នោះទេ ឬ​ការណាត់​ជួប​សម្រាប់​ការសាកល្បង​អាស៊ីត​នុយក្លេអ៊ីក។ វា​មានន័យថា​ការ​ប្រកាស​បាន​រំឭក​ពី​ការ​ចាប់ផ្តើម​ធ្វើ​តេស្ត​នៅក្នុង​ពេលឆាប់​នេះ ការណាត់ជួប ការធ្វើគម្រោង​នេះ សម្រាប់ការណាត់ជួប ឬ​ការ​រៀបចំ​សម្រាប់​ការសាកល្បង៖ {}. ប្រសិនបើ​បាទ/ចាស សូមឆ្លើយថា 'yes' បើ​មិនមែន ឆ្លើយថា 'no'.",
        # Tagalog
        "Pakisuri ang sumusunod na post at tukuyin kung ito ay binabanggit ang mga plano o mga appointment para sa pagsusuri ng nucleic acid. Ipinapahiwatig nito na ang post ay binanggit na magsisimula ang pagsusuri, nag-set na ng appointment, plano na mag-set ng appointment, o naghahanda na para sa pagsusuri: {}. Kung gayon, sumagot ng 'yes', kung hindi, sumagot ng 'no'.",
        # Hindi
        "कृपया निम्नलिखित پوسٹ کا تجزیہ کریں اور یہ بتائیں کہ کیا اس میں نیوکلک ایسڈ ٹیسٹ کے بارے میں ہچکچاہٹ کا ذکر ہے: {}. اگر ہاں تو 'yes' جواب دیں؛ بصورت دیگر، 'no' جواب دیں۔"
    ],

    '3.3': [
        # Chinese
        "请分析以下微博内容，并判断其是否提到不愿意做核酸检测: {}。如果是，请回答'yes'，否则回答'no'。",
        # English
        "Please analyze the following post and determine if it mentions unwillingness to undergo nucleic acid testing: {}. If so, respond 'yes'; otherwise, respond 'no'.",
        # German
        "Bitte analysieren Sie den folgenden Beitrag und bestimmen Sie, ob er erwähnt, dass jemand nicht bereit ist, sich einem Nukleinsäuretest zu unterziehen: {}. Wenn ja, antworten Sie mit 'yes', andernfalls mit 'no'.",
        # French
        "Veuillez analyser le post ci-dessous et déterminer s'il mentionne une réticence à subir un test d'acide nucléique: {}. Si c'est le cas, répondez 'yes', sinon répondez 'no'.",
        # Spanish
        "Por favor, analice el siguiente post y determine si menciona la falta de disposición para someterse a una prueba de ácido nucleico: {}. Si es así, responda 'yes', de lo contrario, responda 'no'.",
        # Portuguese
        "Por favor, analise a postagem abaixo e determine se ela menciona a falta de vontade de se submeter a um teste de ácido nucleico: {}. Se for o caso, responda 'yes', caso contrário, responda 'no'.",
        # Italian
        "Si prega di analizzare il seguente post e determinare se menziona la riluttanza a sottoporsi a un test per l'acido nucleico: {}. In tal caso, rispondi 'yes', altrimenti rispondi 'no'.",
        # Dutch
        "Analyseer alstublieft de volgende post en bepaal of er wordt vermeld dat iemand niet bereid is een nucleïnezuurtest te ondergaan: {}. Als dat het geval is, antwoord dan met 'yes', anders met 'no'.",
        # Russian
        "Пожалуйста, проанализируйте следующий пост и определите, упоминается ли в нем нежелание пройти тест на нуклеиновую кислоту: {}. Если да, ответьте 'yes', в противном случае ответьте 'no'.",
        # Czech
        "Analyzujte prosím následující příspěvek a určete, zda zmiňuje neochotu podstoupit test na nukleovou kyselinu: {}. Pokud ano, odpovězte 'yes', jinak odpovězte 'no'.",
        # Polish
        "Proszę przeanalizować poniższy post i ustalić, czy wspomina o niechęci do poddania się testowi na kwas nukleinowy: {}. Jeśli tak, odpowiedz 'yes', w przeciwnym razie odpowiedz 'no'.",
        # Arabic
        "يرجى تحليل المنشور أدناه وتحديد ما إذا كان يذكر عدم الرغبة في إجراء اختبار الحمض النووي: {}. إذا كان الأمر كذلك، أجب 'yes'، وإلا أجب 'no'.",
        # Persian
        "لطفاً پست زیر را تحلیل کنید و تعیین کنید که آیا اشاره شده است که تمایلی به انجام آزمایش اسید نوکلئیک وجود ندارد: {}. اگر چنین است، با 'yes' پاسخ دهید، در غیر این صورت 'no' پاسخ دهید.",
        # Hebrew
        "אנא נתחו את הפוסט הבא וקבעו אם הוא מזכיר חוסר רצון לעבור בדיקת חומצת גרעין: {}. אם כן, השב 'yes', אחרת השב 'no'.",
        # Turkish
        "Lütfen aşağıdaki gönderiyi analiz edin ve nükleik asit testine girmeme isteğinden bahsedilip bahsedilmediğini belirleyin: {}. Eğer öyleyse, 'yes' yanıtını verin, aksi takdirde 'no' yanıtını verin.",
        # Japanese
        "以下の投稿を分析し、核酸検査を受けることに消極的であるかどうか記載されているかどうかを判断してください: {}。 そうであれば「yes」と答え、それ以外の場合は「no」と答えてください。",
        # Korean
        "다음 게시물을 분석하고 핵산 검사를 받고 싶지 않다는 내용이 언급되었는지 확인하십시오: {}. 그렇다면 'yes'로 답하고, 그렇지 않다면 'no'로 답하십시오.",
        # Vietnamese
        "Vui lòng phân tích bài viết dưới đây và xác định xem nó có đề cập đến sự không sẵn lòng trải qua xét nghiệm axit nucleic hay không: {}. Nếu có, hãy trả lời 'yes', nếu không, hãy trả lời 'no'.",
        # Thai
        "โปรดวิเคราะห์โพสต์ด้านล่างนี้และระบุว่ามีการกล่าวถึงความไม่เต็มใจที่จะทดสอบกรดนิวคลีอิกหรือไม่: {}. หากเป็นเช่นนั้น ให้ตอบว่า 'yes' มิฉะนั้นให้ตอบว่า 'no'",
        # Indonesian
        "Harap analisis postingan di bawah ini dan tentukan apakah disebutkan ketidakmauan untuk menjalani pengujian asam nukleat: {}. Jika demikian, balas dengan 'yes', jika tidak, balas dengan 'no'.",
        # Malay
        "Sila analisis siaran di bawah dan tentukan sama ada ia menyebutkan ketidaksediaan untuk menjalani ujian asid nukleik: {}. Jika ya, sila jawab 'yes', jika tidak, jawab 'no'.",
        # Lao
        "โปรดวิเคราะห์โพสต์ด้านล่างนี้และระบุว่ามีการกล่าวถึงความไม่เต็มใจที่จะทดสอบกรดนิวคลีอิกหรือไม่: {}. หากเป็นเช่นนั้น ให้ตอบว่า 'yes' มิฉะนั้นให้ตอบว่า 'no'",
        # Burmese
        "ကျေးဇူးပြု၍ အောက်ပါစာမျက်နှာကိုသုံးသပ်ပြီး နျဴကလီဝမ်အက်ဆစ်စစ်ဆေးမှုအတွက် အစီအစဉ်များ သို့မဟုတ် ချိန်းဆိုချက်များကို ရှာဖွေပါ။ ၎င်းသည် စစ်ဆေးမှုကိုမကြာမီစတင်ရန်၊ ချိန်းဆိုချက်တစ်ခုပြုလုပ်ပြီး၊ ချိန်းဆိုချက်တစ်ခုစီစဉ်ထားခြင်း သို့မဟုတ် စစ်ဆေးမှုအတွက်ပြင်ဆင်ထားကြောင်းကိုဖော်ပြသည်။ {}. အကယ်၍အကယ်၍တစ်ခုပြုလုပ်ရန် 'yes' ဖြင့်ပြန်ကြားပါ၊ မဟုတ်ပါက 'no' ဖြင့်ပြန်ကြားပါ။",
        # Cebuano
        "Pagtudlo sa mga plano o mga appointment alang sa nucleic acid testing. Nagpasabot kini nga ang post naghisgot bahin sa pagsugod sa pagsulay sa dili madugay, nga naghimo og appointment, nagplano sa paghimo og appointment, o pagpangandam alang sa pagsulay: {}. Palihug paghimo usa ka paghukom base sa sulod sa ibabaw. Kung mao, tubaga nga 'yes', kung dili, tubaga nga 'no'.",
        # Khmer
        "សូម​វិភាគ​អត្ថបទ​សេដ្ឋកិច្ច​ខាងក្រោម​នេះ ហើយ​សម្រេចថា តើ​វា​មាន​គម្រោង​នោះទេ ឬ​ការណាត់​ជួប​សម្រាប់​ការសាកល្បង​អាស៊ីត​នុយក្លេអ៊ីក។ វា​មានន័យថា​ការ​ប្រកាស​បាន​រំឭក​ពី​ការ​ចាប់ផ្តើម​ធ្វើ​តេស្ត​នៅក្នុង​ពេលឆាប់​នេះ ការណាត់ជួប ការធ្វើគម្រោង​នេះ សម្រាប់ការណាត់ជួប ឬ​ការ​រៀបចំ​សម្រាប់​ការសាកល្បង៖ {}. ប្រសិនបើ​បាទ/ចាស សូមឆ្លើយថា 'yes' បើ​មិនមែន ឆ្លើយថា 'no'.",
        # Tagalog
        "Pakisuri ang sumusunod na post at tukuyin kung ito ay binabanggit ang mga plano o mga appointment para sa pagsusuri ng nucleic acid. Ipinapahiwatig nito na ang post ay binanggit na magsisimula ang pagsusuri, nag-set na ng appointment, plano na mag-set ng appointment, o naghahanda na para sa pagsusuri: {}. Kung gayon, sumagot ng 'yes', kung hindi, sumagot ng 'no'.",
        # Hindi
        "कृपया निम्नलिखित پوسٹ کا تجزیہ کریں اور یہ بتائیں کہ آیا اس میں نیوکلک ایسڈ ٹیسٹ کے بارے میں ہچکچاہٹ کا ذکر ہے: {}. اگر ہاں تو 'yes' جواب دیں؛ بصورت دیگر، 'no' جواب دیں۔"
    ],
    '4.1': [
        # Chinese
        "请分析以下微博内容，并判断其是否提到支持在出行或就医等情况下的核酸检测要求: {}。如果是，请回答'yes'，否则回答'no'。",
        # English
        "Please analyze the following post and determine if it mentions supporting nucleic acid testing requirements during travel or medical visits: {}. If so, respond 'yes'; otherwise, respond 'no'.",
        # German
        "Bitte analysieren Sie den folgenden Beitrag und bestimmen Sie, ob er die Unterstützung von Nukleinsäuretestanforderungen während Reisen oder Arztbesuchen erwähnt: {}. Wenn ja, antworten Sie mit 'yes', andernfalls mit 'no'.",
        # French
        "Veuillez analyser le post ci-dessous et déterminer s'il mentionne soutenir les exigences de tests d'acide nucléique lors de voyages ou de visites médicales: {}. Si c'est le cas, répondez 'yes', sinon répondez 'no'.",
        # Spanish
        "Por favor, analice el siguiente post y determine si menciona apoyar los requisitos de pruebas de ácido nucleico durante viajes o visitas médicas: {}. Si es así, responda 'yes', de lo contrario, responda 'no'.",
        # Portuguese
        "Por favor, analise a postagem abaixo e determine se ela menciona apoiar os requisitos de testes de ácido nucleico durante viagens ou visitas médicas: {}. Se for o caso, responda 'yes', caso contrário, responda 'no'.",
        # Italian
        "Si prega di analizzare il seguente post e determinare se menziona il supporto ai requisiti di test dell'acido nucleico durante viaggi o visite mediche: {}. In tal caso, rispondi 'yes', altrimenti rispondi 'no'.",
        # Dutch
        "Analyseer alstublieft de volgende post en bepaal of er wordt vermeld dat nucleïnezuurtesten tijdens reizen of medische bezoeken worden ondersteund: {}. Als dat het geval is, antwoord dan met 'yes', anders met 'no'.",
        # Russian
        "Пожалуйста, проанализируйте следующий пост и определите, упоминается ли в нем поддержка требований к тестированию на нуклеиновую кислоту во время поездок или медицинских визитов: {}. Если да, ответьте 'yes', в противном случае ответьте 'no'.",
        # Czech
        "Analyzujte prosím následující příspěvek a určete, zda zmiňuje podporu požadavkům na testování nukleové kyseliny během cestování nebo lékařských návštěv: {}. Pokud ano, odpovězte 'yes', jinak odpovězte 'no'.",
        # Polish
        "Proszę przeanalizować poniższy post i ustalić, czy wspomina o wsparciu wymogów testów na kwas nukleinowy podczas podróży lub wizyt lekarskich: {}. Jeśli tak, odpowiedz 'yes', w przeciwnym razie odpowiedz 'no'.",
        # Arabic
        "يرجى تحليل المنشور أدناه وتحديد ما إذا كان يذكر دعم متطلبات اختبار الحمض النووي أثناء السفر أو الزيارات الطبية: {}. إذا كان الأمر كذلك، أجب 'yes'، وإلا أجب 'no'.",
        # Persian
        "لطفاً پست زیر را تحلیل کنید و تعیین کنید که آیا اشاره شده است که از الزامات انجام آزمایش اسید نوکلئیک در سفرها یا بازدیدهای پزشکی حمایت می‌شود: {}. اگر چنین است، با 'yes' پاسخ دهید، در غیر این صورت 'no' پاسخ دهید.",
        # Hebrew
        "אנא נתחו את הפוסט הבא וקבעו אם הוא מזכיר תמיכה בדרישות בדיקות חומצת גרעין במהלך נסיעות או ביקורים רפואיים: {}. אם כן, השב 'yes', אחרת השב 'no'.",
        # Turkish
        "Lütfen aşağıdaki gönderiyi analiz edin ve seyahatler veya tıbbi ziyaretler sırasında nükleik asit testi gerekliliklerini desteklediğini belirtip belirtmediğini belirleyin: {}. Eğer öyleyse, 'yes' yanıtını verin, aksi takdirde 'no' yanıtını verin.",
        # Japanese
        "以下の投稿を分析し、旅行や診療時に核酸検査の要件を支持しているかどうか記載されているかどうかを判断してください: {}。 そうであれば「yes」と答え、それ以外の場合は「no」と答えてください。",
        # Korean
        "다음 게시물을 분석하고 여행이나 의료 방문 중에 핵산 검사 요구 사항을 지지하는지 여부를 확인하십시오: {}. 그렇다면 'yes'로 답하고, 그렇지 않다면 'no'로 답하십시오.",
        # Vietnamese
        "Vui lòng phân tích bài viết dưới đây và xác định xem nó có đề cập đến việc ủng hộ các yêu cầu xét nghiệm axit nucleic trong quá trình đi lại hoặc thăm khám y tế hay không: {}. Nếu có, hãy trả lời 'yes', nếu không, hãy trả lời 'no'.",
        # Thai
        "โปรดวิเคราะห์โพสต์ด้านล่างนี้และระบุว่ามีการกล่าวถึงการสนับสนุนข้อกำหนดในการทดสอบกรดนิวคลีอิกในระหว่างการเดินทางหรือการไปพบแพทย์หรือไม่: {}. หากเป็นเช่นนั้น ให้ตอบว่า 'yes' มิฉะนั้นให้ตอบว่า 'no'",
        # Indonesian
        "Harap analisis postingan di bawah ini dan tentukan apakah disebutkan mendukung persyaratan pengujian asam nukleat selama perjalanan atau kunjungan medis: {}. Jika demikian, balas dengan 'yes', jika tidak, balas dengan 'no'.",
        # Malay
        "Sila analisis siaran di bawah dan tentukan sama ada ia menyebutkan sokongan untuk keperluan ujian asid nukleik semasa perjalanan atau lawatan perubatan: {}. Jika ya, sila jawab 'yes', jika tidak, jawab 'no'.",
        # Lao
        "โปรดวิเคราะห์โพสต์ด้านล่างนี้และระบุว่ามีการกล่าวถึงการสนับสนุนข้อกำหนดในการทดสอบกรดนิวคลีอิกในระหว่างการเดินทางหรือการไปพบแพทย์หรือไม่: {}. หากเป็นเช่นนั้น ให้ตอบว่า 'yes' มิฉะนั้นให้ตอบว่า 'no'",
        # Burmese
        "ကျေးဇူးပြု၍ အောက်ပါစာမျက်နှာကိုသုံးသပ်ပြီး နျဴကလီဝမ်အက်ဆစ်စစ်ဆေးမှုအတွက် အစီအစဉ်များ သို့မဟုတ် ချိန်းဆိုချက်များကို ရှာဖွေပါ။ ၎င်းသည် စစ်ဆေးမှုကိုမကြာမီစတင်ရန်၊ ချိန်းဆိုချက်တစ်ခုပြုလုပ်ပြီး၊ ချိန်းဆိုချက်တစ်ခုစီစဉ်ထားခြင်း သို့မဟုတ် စစ်ဆေးမှုအတွက်ပြင်ဆင်ထားကြောင်းကိုဖော်ပြသည်။ {}. အကယ်၍အကယ်၍တစ်ခုပြုလုပ်ရန် 'yes' ဖြင့်ပြန်ကြားပါ၊ မဟုတ်ပါက 'no' ဖြင့်ပြန်ကြားပါ။",
        # Cebuano
        "Pagtudlo sa mga plano o mga appointment alang sa nucleic acid testing. Nagpasabot kini nga ang post naghisgot bahin sa pagsugod sa pagsulay sa dili madugay, nga naghimo og appointment, nagplano sa paghimo og appointment, o pagpangandam alang sa pagsulay: {}. Palihug paghimo usa ka paghukom base sa sulod sa ibabaw. Kung mao, tubaga nga 'yes', kung dili, tubaga nga 'no'.",
        # Khmer
        "សូម​វិភាគ​អត្ថបទ​សេដ្ឋកិច្ច​ខាងក្រោម​នេះ ហើយ​សម្រេចថា តើ​វា​មាន​គម្រោង​នោះទេ ឬ​ការណាត់​ជួប​សម្រាប់​ការសាកល្បង​អាស៊ីត​នុយក្លេអ៊ីក។ វា​មានន័យថា​ការ​ប្រកាស​បាន​រំឭក​ពី​ការ​ចាប់ផ្តើម​ធ្វើ​តេស្ត​នៅក្នុង​ពេលឆាប់​នេះ ការណាត់ជួប ការធ្វើគម្រោង​នេះ សម្រាប់ការណាត់ជួប ឬ​ការ​រៀបចំ​សម្រាប់​ការសាកល្បង៖ {}. ប្រសិនបើ​បាទ/ចាស សូមឆ្លើយថា 'yes' បើ​មិនមែន ឆ្លើយថា 'no'.",
        # Tagalog
        "Pakisuri ang sumusunod na post at tukuyin kung ito ay binabanggit ang mga plano o mga appointment para sa pagsusuri ng nucleic acid. Ipinapahiwatig nito na ang post ay binanggit na magsisimula ang pagsusuri, nag-set na ng appointment, plano na mag-set ng appointment, o naghahanda na para sa pagsusuri: {}. Kung gayon, sumagot ng 'yes', kung hindi, sumagot ng 'no'.",
        # Hindi
        "कृपया निम्नलिखित پوسٹ کا تجزیہ کریں اور یہ بتائیں کہ کیا اس میں نیوکلک ایسڈ ٹیسٹ کے بارے میں ہچکچاہٹ کا ذکر ہے: {}. اگر ہاں تو 'yes' جواب دیں؛ بصورت دیگر، 'no' جواب دیں۔"
    ],

    '4.2': [
        # Chinese
        "请分析以下微博内容，并判断其是否提到对出行或就医情况下的核酸检测政策持中立态度: {}。如果是，请回答'yes'，否则回答'no'。",
        # English
        "Please analyze the following post and determine if it mentions a neutral stance toward nucleic acid testing policies during travel or medical visits: {}. If so, respond 'yes'; otherwise, respond 'no'.",
        # German
        "Bitte analysieren Sie den folgenden Beitrag und bestimmen Sie, ob er eine neutrale Haltung gegenüber den Nukleinsäuretestvorschriften während Reisen oder Arztbesuchen erwähnt: {}. Wenn ja, antworten Sie mit 'yes', andernfalls mit 'no'.",
        # French
        "Veuillez analyser le post ci-dessous et déterminer s'il mentionne une position neutre vis-à-vis des politiques de tests d'acide nucléique lors de voyages ou de visites médicales: {}. Si c'est le cas, répondez 'yes', sinon répondez 'no'.",
        # Spanish
        "Por favor, analice el siguiente post y determine si menciona una postura neutral hacia las políticas de pruebas de ácido nucleico durante viajes o visitas médicas: {}. Si es así, responda 'yes', de lo contrario, responda 'no'.",
        # Portuguese
        "Por favor, analise a postagem abaixo e determine se ela menciona uma postura neutra em relação às políticas de testes de ácido nucleico durante viagens ou visitas médicas: {}. Se for o caso, responda 'yes', caso contrário, responda 'no'.",
        # Italian
        "Si prega di analizzare il seguente post e determinare se menziona un atteggiamento neutro verso le politiche di test dell'acido nucleico durante viaggi o visite mediche: {}. In tal caso, rispondi 'yes', altrimenti rispondi 'no'.",
        # Dutch
        "Analyseer alstublieft de volgende post en bepaal of er een neutrale houding ten opzichte van nucleïnezuurtestbeleid tijdens reizen of medische bezoeken wordt vermeld: {}. Als dat het geval is, antwoord dan met 'yes', anders met 'no'.",
        # Russian
        "Пожалуйста, проанализируйте следующий пост и определите, упоминается ли в нем нейтральное отношение к политике тестирования на нуклеиновую кислоту во время поездок или медицинских визитов: {}. Если да, ответьте 'yes', в противном случае ответьте 'no'.",
        # Czech
        "Analyzujte prosím následující příspěvek a určete, zda zmiňuje neutrální postoj vůči politice testování nukleové kyseliny během cestování nebo lékařských návštěv: {}. Pokud ano, odpovězte 'yes', jinak odpovězte 'no'.",
        # Polish
        "Proszę przeanalizować poniższy post i ustalić, czy wspomina o neutralnym nastawieniu do polityki testów na kwas nukleinowy podczas podróży lub wizyt lekarskich: {}. Jeśli tak, odpowiedz 'yes', w przeciwnym razie odpowiedz 'no'.",
        # Arabic
        "يرجى تحليل المنشور أدناه وتحديد ما إذا كان يذكر موقفًا محايدًا تجاه سياسات اختبار الحمض النووي أثناء السفر أو الزيارات الطبية: {}. إذا كان الأمر كذلك، أجب 'yes'، وإلا أجب 'no'.",
        # Persian
        "لطفاً پست زیر را تحلیل کنید و تعیین کنید که آیا اشاره شده است که نسبت به سیاست‌های انجام آزمایش اسید نوکلئیک در سفرها یا بازدیدهای پزشکی، موضعی خنثی وجود دارد: {}. اگر چنین است، با 'yes' پاسخ دهید، در غیر این صورت 'no' پاسخ دهید.",
        # Hebrew
        "אנא נתחו את הפוסט הבא וקבעו אם הוא מזכיר עמדה ניטרלית כלפי מדיניות בדיקות חומצת גרעין במהלך נסיעות או ביקורים רפואיים: {}. אם כן, השב 'yes', אחרת השב 'no'.",
        # Turkish
        "Lütfen aşağıdaki gönderiyi analiz edin ve seyahatler veya tıbbi ziyaretler sırasında nükleik asit testi politikalarına karşı tarafsız bir duruş sergilediğini belirtip belirtmediğini belirleyin: {}. Eğer öyleyse, 'yes' yanıtını verin, aksi takdirde 'no' yanıtını verin.",
        # Japanese
        "以下の投稿を分析し、旅行や診療時の核酸検査政策に対する中立的な立場を取っているかどうか記載されているかどうかを判断してください: {}。 そうであれば「yes」と答え、それ以外の場合は「no」と答えてください。",
        # Korean
        "다음 게시물을 분석하고 여행 또는 의료 방문 중에 핵산 검사 정책에 대해 중립적인 입장을 취하고 있는지 여부를 확인하십시오: {}. 그렇다면 'yes'로 답하고, 그렇지 않다면 'no'로 답하십시오.",
        # Vietnamese
        "Vui lòng phân tích bài viết dưới đây và xác định xem nó có đề cập đến lập trường trung lập đối với các chính sách xét nghiệm axit nucleic trong quá trình đi lại hoặc thăm khám y tế hay không: {}. Nếu có, hãy trả lời 'yes', nếu không, hãy trả lời 'no'.",
        # Thai
        "โปรดวิเคราะห์โพสต์ด้านล่างนี้และระบุว่ามีการกล่าวถึงจุดยืนที่เป็นกลางต่อข้อกำหนดในการทดสอบกรดนิวคลีอิกในระหว่างการเดินทางหรือการไปพบแพทย์หรือไม่: {}. หากเป็นเช่นนั้น ให้ตอบว่า 'yes' มิฉะนั้นให้ตอบว่า 'no'",
        # Indonesian
        "Harap analisis postingan di bawah ini dan tentukan apakah disebutkan sikap netral terhadap kebijakan pengujian asam nukleat selama perjalanan atau kunjungan medis: {}. Jika demikian, balas dengan 'yes', jika tidak, balas dengan 'no'.",
        # Malay
        "Sila analisis siaran di bawah dan tentukan sama ada ia menyebutkan pendirian neutral terhadap dasar ujian asid nukleik semasa perjalanan atau lawatan perubatan: {}. Jika ya, sila jawab 'yes', jika tidak, jawab 'no'.",
        # Lao
        "โปรดวิเคราะห์โพสต์ด้านล่างนี้และระบุว่ามีการกล่าวถึงจุดยืนที่เป็นกลางต่อข้อกำหนดในการทดสอบกรดนิวคลีอิกในระหว่างการเดินทางหรือการไปพบแพทย์หรือไม่: {}. หากเป็นเช่นนั้น ให้ตอบว่า 'yes' มิฉะนั้นให้ตอบว่า 'no'",
        # Burmese
        "ကျေးဇူးပြု၍ အောက်ပါစာမျက်နှာကိုသုံးသပ်ပြီး နျဴကလီဝမ်အက်ဆစ်စစ်ဆေးမှုအတွက် အစီအစဉ်များ သို့မဟုတ် ချိန်းဆိုချက်များကို ရှာဖွေပါ။ ၎င်းသည် စစ်ဆေးမှုကိုမကြာမီစတင်ရန်၊ ချိန်းဆိုချက်တစ်ခုပြုလုပ်ပြီး၊ ချိန်းဆိုချက်တစ်ခုစီစဉ်ထားခြင်း သို့မဟုတ် စစ်ဆေးမှုအတွက်ပြင်ဆင်ထားကြောင်းကိုဖော်ပြသည်။ {}. အကယ်၍အကယ်၍တစ်ခုပြုလုပ်ရန် 'yes' ဖြင့်ပြန်ကြားပါ၊ မဟုတ်ပါက 'no' ဖြင့်ပြန်ကြားပါ။",
        # Cebuano
        "Pagtudlo sa mga plano o mga appointment alang sa nucleic acid testing. Nagpasabot kini nga ang post naghisgot bahin sa pagsugod sa pagsulay sa dili madugay, nga naghimo og appointment, nagplano sa paghimo og appointment, o pagpangandam alang sa pagsulay: {}. Palihug paghimo usa ka paghukom base sa sulod sa ibabaw. Kung mao, tubaga nga 'yes', kung dili, tubaga nga 'no'.",
        # Khmer
        "សូម​វិភាគ​អត្ថបទ​សេដ្ឋកិច្ច​ខាងក្រោម​នេះ ហើយ​សម្រេចថា តើ​វា​មាន​គម្រោង​នោះទេ ឬ​ការណាត់​ជួប​សម្រាប់​ការសាកល្បង​អាស៊ីត​នុយក្លេអ៊ីក។ វា​មានន័យថា​ការ​ប្រកាស​បាន​រំឭក​ពី​ការ​ចាប់ផ្តើម​ធ្វើ​តេស្ត​នៅក្នុង​ពេលឆាប់​នេះ ការណាត់ជួប ការធ្វើគម្រោង​នេះ សម្រាប់ការណាត់ជួប ឬ​ការ​រៀបចំ​សម្រាប់​ការសាកល្បង៖ {}. ប្រសិនបើ​បាទ/ចាស សូមឆ្លើយថា 'yes' បើ​មិនមែន ឆ្លើយថា 'no'.",
        # Tagalog
        "Pakisuri ang sumusunod na post at tukuyin kung ito ay binabanggit ang mga plano o mga appointment para sa pagsusuri ng nucleic acid. Ipinapahiwatig nito na ang post ay binanggit na magsisimula ang pagsusuri, nag-set na ng appointment, plano na mag-set ng appointment, o naghahanda na para sa pagsusuri: {}. Kung gayon, sumagot ng 'yes', kung hindi, sumagot ng 'no'.",
        # Hindi
        "कृपया निम्नलिखित پوسٹ کا تجزیہ کریں اور یہ بتائیں کہ کیا اس میں نیوکلک ایسڈ ٹیسٹ کے بارے میں ہچکچاہٹ کا ذکر ہے: {}. اگر ہاں تو 'yes' جواب دیں؛ بصورت دیگر، 'no' جواب دیں۔"
    ],

    '4.3': [
        # Chinese
        "请分析以下微博内容，并判断其是否提到反对在出行或就医情况下的核酸检测政策: {}。如果是，请回答'yes'，否则回答'no'。",
        # English
        "Please analyze the following post and determine if it mentions opposing nucleic acid testing policies during travel or medical visits: {}. If so, respond 'yes'; otherwise, respond 'no'.",
        # German
        "Bitte analysieren Sie den folgenden Beitrag und bestimmen Sie, ob er die Ablehnung der Nukleinsäuretestpolitik während Reisen oder Arztbesuchen erwähnt: {}. Wenn ja, antworten Sie mit 'yes', andernfalls mit 'no'.",
        # French
        "Veuillez analyser le post ci-dessous et déterminer s'il mentionne l'opposition aux politiques de tests d'acide nucléique lors de voyages ou de visites médicales: {}. Si c'est le cas, répondez 'yes', sinon répondez 'no'.",
        # Spanish
        "Por favor, analice el siguiente post y determine si menciona oponerse a las políticas de pruebas de ácido nucleico durante viajes o visitas médicas: {}. Si es así, responda 'yes', de lo contrario, responda 'no'.",
        # Portuguese
        "Por favor, analise a postagem abaixo e determine se ela menciona a oposição às políticas de testes de ácido nucleico durante viagens ou visitas médicas: {}. Se for o caso, responda 'yes', caso contrário, responda 'no'.",
        # Italian
        "Si prega di analizzare il seguente post e determinare se menziona l'opposizione alle politiche di test dell'acido nucleico durante viaggi o visite mediche: {}. In tal caso, rispondi 'yes', altrimenti rispondi 'no'.",
        # Dutch
        "Analyseer alstublieft de volgende post en bepaal of er wordt vermeld dat iemand tegen nucleïnezuurtestbeleid tijdens reizen of medische bezoeken is: {}. Als dat het geval is, antwoord dan met 'yes', anders met 'no'.",
        # Russian
        "Пожалуйста, проанализируйте следующий пост и определите, упоминается ли в нем противодействие политике тестирования на нуклеиновую кислоту во время поездок или медицинских визитов: {}. Если да, ответьте 'yes', в противном случае ответьте 'no'.",
        # Czech
        "Analyzujte prosím následující příspěvek a určete, zda zmiňuje odpor vůči politice testování nukleové kyseliny během cestování nebo lékařských návštěv: {}. Pokud ano, odpovězte 'yes', jinak odpovězte 'no'.",
        # Polish
        "Proszę przeanalizować poniższy post i ustalić, czy wspomina o sprzeciwie wobec polityki testów na kwas nukleinowy podczas podróży lub wizyt lekarskich: {}. Jeśli tak, odpowiedz 'yes', w przeciwnym razie odpowiedz 'no'.",
        # Arabic
        "يرجى تحليل المنشور أدناه وتحديد ما إذا كان يذكر معارضة سياسات اختبار الحمض النووي أثناء السفر أو الزيارات الطبية: {}. إذا كان الأمر كذلك، أجب 'yes'، وإلا أجب 'no'.",
        # Persian
        "لطفاً پست زیر را تحلیل کنید و تعیین کنید که آیا اشاره شده است که نسبت به سیاست‌های انجام آزمایش اسید نوکلئیک در سفرها یا بازدیدهای پزشکی مخالفتی وجود دارد: {}. اگر چنین است، با 'yes' پاسخ دهید، در غیر این صورت 'no' پاسخ دهید.",
        # Hebrew
        "אנא נתחו את הפוסט הבא וקבעו אם הוא מזכיר התנגדות למדיניות בדיקות חומצת גרעין במהלך נסיעות או ביקורים רפואיים: {}. אם כן, השב 'yes', אחרת השב 'no'.",
        # Turkish
        "Lütfen aşağıdaki gönderiyi analiz edin ve seyahatler veya tıbbi ziyaretler sırasında nükleik asit testi politikalarına karşı çıktığını belirtip belirtmediğini belirleyin: {}. Eğer öyleyse, 'yes' yanıtını verin, aksi takdirde 'no' yanıtını verin.",
        # Japanese
        "以下の投稿を分析し、旅行や診療時の核酸検査政策に反対しているかどうか記載されているかどうかを判断してください: {}。 そうであれば「yes」と答え、それ以外の場合は「no」と答えてください。",
        # Korean
        "다음 게시물을 분석하고 여행 또는 의료 방문 중에 핵산 검사 정책에 반대하는지 여부를 확인하십시오: {}. 그렇다면 'yes'로 답하고, 그렇지 않다면 'no'로 답하십시오.",
        # Vietnamese
        "Vui lòng phân tích bài viết dưới đây và xác định xem nó có đề cập đến việc phản đối các chính sách xét nghiệm axit nucleic trong quá trình đi lại hoặc thăm khám y tế hay không: {}. Nếu có, hãy trả lời 'yes', nếu không, hãy trả lời 'no'.",
        # Thai
        "โปรดวิเคราะห์โพสต์ด้านล่างนี้และระบุว่ามีการกล่าวถึงการคัดค้านข้อกำหนดในการทดสอบกรดนิวคลีอิกในระหว่างการเดินทางหรือการไปพบแพทย์หรือไม่: {}. หากเป็นเช่นนั้น ให้ตอบว่า 'yes' มิฉะนั้นให้ตอบว่า 'no'",
        # Indonesian
        "Harap analisis postingan di bawah ini dan tentukan apakah disebutkan menentang kebijakan pengujian asam nukleat selama perjalanan atau kunjungan medis: {}. Jika demikian, balas dengan 'yes', jika tidak, balas dengan 'no'.",
        # Malay
        "Sila analisis siaran di bawah dan tentukan sama ada ia menyebutkan penentangan terhadap dasar ujian asid nukleik semasa perjalanan atau lawatan perubatan: {}. Jika ya, sila jawab 'yes', jika tidak, jawab 'no'.",
        # Lao
        "โปรดวิเคราะห์โพสต์ด้านล่างนี้และระบุว่ามีการกล่าวถึงการคัดค้านข้อกำหนดในการทดสอบกรดนิวคลีอิกในระหว่างการเดินทางหรือการไปพบแพทย์หรือไม่: {}. หากเป็นเช่นนั้น ให้ตอบว่า 'yes' มิฉะนั้นให้ตอบว่า 'no'",
        # Burmese
        "ကျေးဇူးပြု၍ အောက်ပါစာမျက်နှာကိုသုံးသပ်ပြီး နျဴကလီဝမ်အက်ဆစ်စစ်ဆေးမှုအတွက် အစီအစဉ်များ သို့မဟုတ် ချိန်းဆိုချက်များကို ရှာဖွေပါ။ ၎င်းသည် စစ်ဆေးမှုကိုမကြာမီစတင်ရန်၊ ချိန်းဆိုချက်တစ်ခုပြုလုပ်ပြီး၊ ချိန်းဆိုချက်တစ်ခုစီစဉ်ထားခြင်း သို့မဟုတ် စစ်ဆေးမှုအတွက်ပြင်ဆင်ထားကြောင်းကိုဖော်ပြသည်။ {}. အကယ်၍အကယ်၍တစ်ခုပြုလုပ်ရန် 'yes' ဖြင့်ပြန်ကြားပါ၊ မဟုတ်ပါက 'no' ဖြင့်ပြန်ကြားပါ။",
        # Cebuano
        "Pagtudlo sa mga plano o mga appointment alang sa nucleic acid testing. Nagpasabot kini nga ang post naghisgot bahin sa pagsugod sa pagsulay sa dili madugay, nga naghimo og appointment, nagplano sa paghimo og appointment, o pagpangandam alang sa pagsulay: {}. Palihug paghimo usa ka paghukom base sa sulod sa ibabaw. Kung mao, tubaga nga 'yes', kung dili, tubaga nga 'no'.",
        # Khmer
        "សូម​វិភាគ​អត្ថបទ​សេដ្ឋកិច្ច​ខាងក្រោម​នេះ ហើយ​សម្រេចថា តើ​វា​មាន​គម្រោង​នោះទេ ឬ​ការណាត់ជួប​សម្រាប់​ការសាកល្បង​អាស៊ីត​នុយក្លេអ៊ីក។ វា​មានន័យថា​ការ​ប្រកាស​បាន​រំឭក​ពី​ការ​ចាប់ផ្តើម​ធ្វើ​តេស្ត​នៅក្នុង​ពេលឆាប់​នេះ ការណាត់ជួប ការធ្វើគម្រោង​នេះ សម្រាប់ការណាត់ជួប ឬ​ការ​រៀបចំ​សម្រាប់​ការសាកល្បង៖ {}. ប្រសិនបើ​បាទ/ចាស សូមឆ្លើយថា 'yes' បើ​មិនមែន ឆ្លើយថា 'no'.",
        # Tagalog
        "Pakisuri ang sumusunod na post at tukuyin kung ito ay binabanggit ang mga plano o mga appointment para sa pagsusuri ng nucleic acid. Ipinapahiwatig nito na ang post ay binanggit na magsisimula ang pagsusuri, nag-set na ng appointment, plano na mag-set ng appointment, o naghahanda na para sa pagsusuri: {}. Kung gayon, sumagot ng 'yes', kung hindi, sumagot ng 'no'.",
        # Hindi
        "कृपया निम्नलिखित پوسٹ کا تجزیہ کریں اور یہ بتائیں کہ کیا اس میں نیوکلک ایسڈ ٹیسٹ کے بارے میں ہچکچاہٹ کا ذکر ہے: {}. اگر ہاں تو 'yes' جواب دیں؛ بصورت دیگر، 'no' جواب دیں۔"
    ],

    '5.1': [
        # Chinese
        "请分析以下微博内容，并判断其是否提到支持大规模群体核酸检测政策: {}。如果是，请回答'yes'，否则回答'no'。",
        # English
        "Please analyze the following post and determine if it mentions supporting large-scale community nucleic acid testing policies: {}. If so, respond 'yes'; otherwise, respond 'no'.",
        # German
        "Bitte analysieren Sie den folgenden Beitrag und bestimmen Sie, ob er die Unterstützung groß angelegter Gemeinschaftstestvorschriften für Nukleinsäure erwähnt: {}. Wenn ja, antworten Sie mit 'yes', andernfalls mit 'no'.",
        # French
        "Veuillez analyser le post ci-dessous et déterminer s'il mentionne le soutien aux politiques de tests d'acide nucléique à grande échelle dans la communauté: {}. Si c'est le cas, répondez 'yes', sinon répondez 'no'.",
        # Spanish
        "Por favor, analice el siguiente post y determine si menciona el apoyo a las políticas de pruebas de ácido nucleico a gran escala en la comunidad: {}. Si es así, responda 'yes', de lo contrario, responda 'no'.",
        # Portuguese
        "Por favor, analise a postagem abaixo e determine se ela menciona o apoio às políticas de testes de ácido nucleico em grande escala na comunidade: {}. Se for o caso, responda 'yes', caso contrário, responda 'no'.",
        # Italian
        "Si prega di analizzare il seguente post e determinare se menziona il supporto alle politiche di test dell'acido nucleico su larga scala per la comunità: {}. In tal caso, rispondi 'yes', altrimenti rispondi 'no'.",
        # Dutch
        "Analyseer alstublieft de volgende post en bepaal of er wordt vermeld dat er steun is voor het beleid inzake grootschalige nucleïnezuurtesten in de gemeenschap: {}. Als dat het geval is, antwoord dan met 'yes', anders met 'no'.",
        # Russian
        "Пожалуйста, проанализируйте следующий пост и определите, упоминается ли в нем поддержка политики массового тестирования на нуклеиновую кислоту в сообществе: {}. Если да, ответьте 'yes', в противном случае ответьте 'no'.",
        # Czech
        "Analyzujte prosím následující příspěvek a určete, zda zmiňuje podporu politiky hromadného testování nukleové kyseliny v komunitě: {}. Pokud ano, odpovězte 'yes', jinak odpovězte 'no'.",
        # Polish
        "Proszę przeanalizować poniższy post i ustalić, czy wspomina o wsparciu dla polityki masowych testów na kwas nukleinowy w społeczności: {}. Jeśli tak, odpowiedz 'yes', w przeciwnym razie odpowiedz 'no'.",
        # Arabic
        "يرجى تحليل المنشور أدناه وتحديد ما إذا كان يذكر دعم سياسات اختبار الحمض النووي واسعة النطاق في المجتمع: {}. إذا كان الأمر كذلك، أجب 'yes'، وإلا أجب 'no'.",
        # Persian
        "لطفاً پست زیر را تحلیل کنید و تعیین کنید که آیا اشاره شده است که از سیاست‌های انجام آزمایش اسید نوکلئیک در سطح گسترده در جامعه حمایت می‌شود: {}. اگر چنین است، با 'yes' پاسخ دهید، در غیر این صورت 'no' پاسخ دهید.",
        # Hebrew
        "אנא נתחו את הפוסט הבא וקבעו אם הוא מזכיר תמיכה במדיניות בדיקות חומצת גרעין נרחבות בקהילה: {}. אם כן, השב 'yes', אחרת השב 'no'.",
        # Turkish
        "Lütfen aşağıdaki gönderiyi analiz edin ve toplulukta geniş çaplı nükleik asit test politikalarını desteklediğini belirtip belirtmediğini belirleyin: {}. Eğer öyleyse, 'yes' yanıtını verin, aksi takdirde 'no' yanıtını verin.",
        # Japanese
        "以下の投稿を分析し、コミュニティでの大規模な核酸検査政策を支持しているかどうか記載されているかどうかを判断してください: {}。 そうであれば「yes」と答え、それ以外の場合は「no」と答えてください。",
        # Korean
        "다음 게시물을 분석하고 지역 사회에서 대규모 핵산 검사 정책을 지지하는지 여부를 확인하십시오: {}. 그렇다면 'yes'로 답하고, 그렇지 않다면 'no'로 답하십시오.",
        # Vietnamese
        "Vui lòng phân tích bài viết dưới đây và xác định xem nó có đề cập đến việc ủng hộ các chính sách xét nghiệm axit nucleic quy mô lớn trong cộng đồng hay không: {}. Nếu có, hãy trả lời 'yes', nếu không, hãy trả lời 'no'.",
        # Thai
        "โปรดวิเคราะห์โพสต์ด้านล่างนี้และระบุว่ามีการกล่าวถึงการสนับสนุนข้อกำหนดในการทดสอบกรดนิวคลีอิกขนาดใหญ่สำหรับชุมชนหรือไม่: {}. หากเป็นเช่นนั้น ให้ตอบว่า 'yes' มิฉะนั้นให้ตอบว่า 'no'",
        # Indonesian
        "Harap analisis postingan di bawah ini dan tentukan apakah disebutkan mendukung kebijakan pengujian asam nukleat berskala besar untuk komunitas: {}. Jika demikian, balas dengan 'yes', jika tidak, balas dengan 'no'.",
        # Malay
        "Sila analisis siaran di bawah dan tentukan sama ada ia menyebutkan sokongan terhadap dasar ujian asid nukleik berskala besar untuk komuniti: {}. Jika ya, sila jawab 'yes', jika tidak, jawab 'no'.",
        # Lao
        "โปรดวิเคราะห์โพสต์ด้านล่างนี้และระบุว่ามีการกล่าวถึงการสนับสนุนข้อกำหนดในการทดสอบกรดนิวคลีอิกขนาดใหญ่สำหรับชุมชนหรือไม่: {}. หากเป็นเช่นนั้น ให้ตอบว่า 'yes' มิฉะนั้นให้ตอบว่า 'no'",
        # Burmese
        "ကျေးဇူးပြု၍ အောက်ပါစာမျက်နှာကိုသုံးသပ်ပြီး နျဴကလီဝမ်အက်ဆစ်စစ်ဆေးမှုအတွက် အစီအစဉ်များ သို့မဟုတ် ချိန်းဆိုချက်များကို ရှာဖွေပါ။ ၎င်းသည် စစ်ဆေးမှုကိုမကြာမီစတင်ရန်၊ ချိန်းဆိုချက်တစ်ခုပြုလုပ်ပြီး၊ ချိန်းဆိုချက်တစ်ခုစီစဉ်ထားခြင်း သို့မဟုတ် စစ်ဆေးမှုအတွက်ပြင်ဆင်ထားကြောင်းကိုဖော်ပြသည်။ {}. အကယ်၍အကယ်၍တစ်ခုပြုလုပ်ရန် 'yes' ဖြင့်ပြန်ကြားပါ၊ မဟုတ်ပါက 'no' ဖြင့်ပြန်ကြားပါ။",
        # Cebuano
        "Pagtudlo sa mga plano o mga appointment alang sa nucleic acid testing. Nagpasabot kini nga ang post naghisgot bahin sa pagsugod sa pagsulay sa dili madugay, nga naghimo og appointment, nagplano sa paghimo og appointment, o pagpangandam alang sa pagsulay: {}. Palihug paghimo usa ka paghukom base sa sulod sa ibabaw. Kung mao, tubaga nga 'yes', kung dili, tubaga nga 'no'.",
        # Khmer
        "សូម​វិភាគ​អត្ថបទ​សេដ្ឋកិច្ច​ខាងក្រោម​នេះ ហើយ​សម្រេចថា តើ​វា​មាន​គម្រោង​នោះទេ ឬ​ការណាត់ជួប​សម្រាប់​ការសាកល្បង​អាស៊ីត​នុយក្លេអ៊ីក។ វា​មានន័យថា​ការ​ប្រកាស​បាន​រំឭក​ពី​ការ​ចាប់ផ្តើម​ធ្វើ​តេស្ត​នៅក្នុង​ពេលឆាប់​នេះ ការណាត់ជួប ការធ្វើគម្រោង​នេះ សម្រាប់ការណាត់ជួប ឬ​ការ​រៀបចំ​សម្រាប់​ការសាកល្បង៖ {}. ប្រសិនបើ​បាទ/ចាស សូមឆ្លើយថា 'yes' បើ​មិនមែន ឆ្លើយថា 'no'.",
        # Tagalog
        "Pakisuri ang sumusunod na post at tukuyin kung ito ay binabanggit ang mga plano o mga appointment para sa pagsusuri ng nucleic acid. Ipinapahiwatig nito na ang post ay binanggit na magsisimula ang pagsusuri, nag-set na ng appointment, plano na mag-set ng appointment, o naghahanda na para sa pagsusuri: {}. Kung gayon, sumagot ng 'yes', kung hindi, sumagot ng 'no'.",
        # Hindi
        "कृपया निम्नलिखित پوسٹ کا تجزیہ کریں اور یہ بتائیں کہ کیا اس میں نیوکلک ایسڈ ٹیسٹ کے بارے میں ہچکچاہٹ کا ذکر ہے: {}. اگر ہاں تو 'yes' جواب دیں؛ بصورت دیگر، 'no' جواب دیں۔"
    ],

    '5.2': [
        # Chinese
        "请分析以下微博内容，并判断其是否提到质疑大规模群体核酸检测的必要性: {}。如果是，请回答'yes'，否则回答'no'。",
        # English
        "Please analyze the following post and determine if it mentions questioning the necessity of large-scale community nucleic acid testing: {}. If so, respond 'yes'; otherwise, respond 'no'.",
        # German
        "Bitte analysieren Sie den folgenden Beitrag und bestimmen Sie, ob er die Notwendigkeit groß angelegter Gemeinschaftstestvorschriften für Nukleinsäure in Frage stellt: {}. Wenn ja, antworten Sie mit 'yes', andernfalls mit 'no'.",
        # French
        "Veuillez analyser le post ci-dessous et déterminer s'il mentionne la remise en question de la nécessité des tests d'acide nucléique à grande échelle dans la communauté: {}. Si c'est le cas, répondez 'yes', sinon répondez 'no'.",
        # Spanish
        "Por favor, analice el siguiente post y determine si menciona cuestionar la necesidad de pruebas de ácido nucleico a gran escala en la comunidad: {}. Si es así, responda 'yes', de lo contrario, responda 'no'.",
        # Portuguese
        "Por favor, analise a postagem abaixo e determine se ela menciona questionar a necessidade de testes de ácido nucleico em grande escala na comunidade: {}. Se for o caso, responda 'yes', caso contrário, responda 'no'.",
        # Italian
        "Si prega di analizzare il seguente post e determinare se menziona mettere in discussione la necessità di test dell'acido nucleico su larga scala per la comunità: {}. In tal caso, rispondi 'yes', altrimenti rispondi 'no'.",
        # Dutch
        "Analyseer alstublieft de volgende post en bepaal of er wordt vermeld dat de noodzaak van grootschalige nucleïnezuurtesten in de gemeenschap in twijfel wordt getrokken: {}. Als dat het geval is, antwoord dan met 'yes', anders met 'no'.",
        # Russian
        "Пожалуйста, проанализируйте следующий пост и определите, упоминается ли в нем сомнение в необходимости массового тестирования на нуклеиновую кислоту в сообществе: {}. Если да, ответьте 'yes', в противном случае ответьте 'no'.",
        # Czech
        "Analyzujte prosím následující příspěvek a určete, zda zmiňuje pochybnosti o nutnosti hromadného testování nukleové kyseliny v komunitě: {}. Pokud ano, odpovězte 'yes', jinak odpovězte 'no'.",
        # Polish
        "Proszę przeanalizować poniższy post i ustalić, czy wspomina o podważeniu konieczności masowych testów na kwas nukleinowy w społeczności: {}. Jeśli tak, odpowiedz 'yes', w przeciwnym razie odpowiedz 'no'.",
        # Arabic
        "يرجى تحليل المنشور أدناه وتحديد ما إذا كان يذكر التشكيك في ضرورة اختبار الحمض النووي واسع النطاق في المجتمع: {}. إذا كان الأمر كذلك، أجب 'yes'، وإلا أجب 'no'.",
        # Persian
        "لطفاً پست زیر را تحلیل کنید و تعیین کنید که آیا اشاره شده است که نسبت به ضرورت انجام آزمایش اسید نوکلئیک در سطح گسترده در جامعه تردید وجود دارد: {}. اگر چنین است، با 'yes' پاسخ دهید، در غیر این صورت 'no' پاسخ دهید.",
        # Hebrew
        "אנא נתחו את הפוסט הבא וקבעו אם הוא מזכיר ספקות לגבי נחיצות בדיקות חומצת גרעין נרחבות בקהילה: {}. אם כן, השב 'yes', אחרת השב 'no'.",
        # Turkish
        "Lütfen aşağıdaki gönderiyi analiz edin ve toplulukta geniş çaplı nükleik asit testlerinin gerekliliğini sorguladığını belirtip belirtmediğini belirleyin: {}. Eğer öyleyse, 'yes' yanıtını verin, aksi takdirde 'no' yanıtını verin.",
        # Japanese
        "以下の投稿を分析し、コミュニティでの大規模な核酸検査の必要性を疑問視しているかどうか記載されているかどうかを判断してください: {}。 そうであれば「yes」と答え、それ以外の場合は「no」と答えてください。",
        # Korean
        "다음 게시물을 분석하고 지역 사회에서 대규모 핵산 검사의 필요성을 의심하고 있는지 여부를 확인하십시오: {}. 그렇다면 'yes'로 답하고, 그렇지 않다면 'no'로 답하십시오.",
        # Vietnamese
        "Vui lòng phân tích bài viết dưới đây và xác định xem nó có đề cập đến việc đặt câu hỏi về sự cần thiết của việc xét nghiệm axit nucleic quy mô lớn trong cộng đồng hay không: {}. Nếu có, hãy trả lời 'yes', nếu không, hãy trả lời 'no'.",
        # Thai
        "โปรดวิเคราะห์โพสต์ด้านล่างนี้และระบุว่ามีการกล่าวถึงการตั้งคำถามเกี่ยวกับความจำเป็นในการทดสอบกรดนิวคลีอิกขนาดใหญ่สำหรับชุมชนหรือไม่: {}. หากเป็นเช่นนั้น ให้ตอบว่า 'yes' มิฉะนั้นให้ตอบว่า 'no'",
        # Indonesian
        "Harap analisis postingan di bawah ini dan tentukan apakah disebutkan mempertanyakan kebutuhan pengujian asam nukleat berskala besar untuk komunitas: {}. Jika demikian, balas dengan 'yes', jika tidak, balas dengan 'no'.",
        # Malay
        "Sila analisis siaran di bawah dan tentukan sama ada ia menyebutkan mempersoalkan keperluan ujian asid nukleik berskala besar untuk komuniti: {}. Jika ya, sila jawab 'yes', jika tidak, jawab 'no'.",
        # Lao
        "โปรดวิเคราะห์โพสต์ด้านล่างนี้และระบุว่ามีการกล่าวถึงการตั้งคำถามเกี่ยวกับความจำเป็นในการทดสอบกรดนิวคลีอิกขนาดใหญ่สำหรับชุมชนหรือไม่: {}. หากเป็นเช่นนั้น ให้ตอบว่า 'yes' มิฉะนั้นให้ตอบว่า 'no'",
        # Burmese
        "ကျေးဇူးပြု၍ အောက်ပါစာမျက်နှာကိုသုံးသပ်ပြီး နျဴကလီဝမ်အက်ဆစ်စစ်ဆေးမှုအတွက် အစီအစဉ်များ သို့မဟုတ် ချိန်းဆိုချက်များကို ရှာဖွေပါ။ ၎င်းသည် စစ်ဆေးမှုကိုမကြာမီစတင်ရန်၊ ချိန်းဆိုချက်တစ်ခုပြုလုပ်ပြီး၊ ချိန်းဆိုချက်တစ်ခုစီစဉ်ထားခြင်း သို့မဟုတ် စစ်ဆေးမှုအတွက်ပြင်ဆင်ထားကြောင်းကိုဖော်ပြသည်။ {}. အကယ်၍အကယ်၍တစ်ခုပြုလုပ်ရန် 'yes' ဖြင့်ပြန်ကြားပါ၊ မဟုတ်ပါက 'no' ဖြင့်ပြန်ကြားပါ။",
        # Cebuano
        "Pagtudlo sa mga plano o mga appointment alang sa nucleic acid testing. Nagpasabot kini nga ang post naghisgot bahin sa pagsugod sa pagsulay sa dili madugay, nga naghimo og appointment, nagplano sa paghimo og appointment, o pagpangandam alang sa pagsulay: {}. Palihug paghimo usa ka paghukom base sa sulod sa ibabaw. Kung mao, tubaga nga 'yes', kung dili, tubaga nga 'no'.",
        # Khmer
        "សូម​វិភាគ​អត្ថបទ​សេដ្ឋកិច្ច​ខាងក្រោម​នេះ ហើយ​សម្រេចថា តើ​វា​មាន​គម្រោង​នោះទេ ឬ​ការណាត់ជួប​សម្រាប់​ការសាកល្បង​អាស៊ីត​នុយក្លេអ៊ីក។ វា​មានន័យថា​ការ​ប្រកាស​បាន​រំឭក​ពី​ការ​ចាប់ផ្តើម​ធ្វើ​តេស្ត​នៅក្នុង​ពេលឆាប់​នេះ ការណាត់ជួប ការធ្វើគម្រោង​នេះ សម្រាប់ការណាត់ជួប ឬ​ការ​រៀបចំ​សម្រាប់​ការសាកល្បង៖ {}. ប្រសិនបើ​បាទ/ចាស សូមឆ្លើយថា 'yes' បើ​មិនមែន ឆ្លើយថា 'no'.",
        # Tagalog
        "Pakisuri ang sumusunod na post at tukuyin kung ito ay binabanggit ang mga plano o mga appointment para sa pagsusuri ng nucleic acid. Ipinapahiwatig nito na ang post ay binanggit na magsisimula ang pagsusuri, nag-set na ng appointment, plano na mag-set ng appointment, o naghahanda na para sa pagsusuri: {}. Kung gayon, sumagot ng 'yes', kung hindi, sumagot ng 'no'.",
        # Hindi
        "कृपया निम्नलिखل پوسٹ کا تجزیہ کریں اور یہ بتائیں کہ کیا اس میں نیوکلک ایسڈ ٹیسٹ کے بارے میں ہچکچاہٹ کا ذکر ہے: {}. اگر ہاں تو 'yes' جواب دیں؛ بصورت دیگر، 'no' جواب دیں۔"
    ],

    '6.1': [
        # Chinese
        "请分析以下微博内容，并判断其是否提到认为核酸检测有效: {}。如果是，请回答'yes'，否则回答'no'。",
        # English
        "Please analyze the following post and determine if it mentions belief in the effectiveness of nucleic acid testing: {}. If so, respond 'yes'; otherwise, respond 'no'.",
        # German
        "Bitte analysieren Sie den folgenden Beitrag und bestimmen Sie, ob er den Glauben an die Wirksamkeit von Nukleinsäuretests erwähnt: {}. Wenn ja, antworten Sie mit 'yes', andernfalls mit 'no'.",
        # French
        "Veuillez analyser le post ci-dessous et déterminer s'il mentionne la croyance en l'efficacité des tests d'acide nucléique: {}. Si c'est le cas, répondez 'yes', sinon répondez 'no'.",
        # Spanish
        "Por favor, analice el siguiente post y determine si menciona la creencia en la eficacia de las pruebas de ácido nucleico: {}. Si es así, responda 'yes', de lo contrario, responda 'no'.",
        # Portuguese
        "Por favor, analise a postagem abaixo e determine se ela menciona a crença na eficácia dos testes de ácido nucleico: {}. Se for o caso, responda 'yes', caso contrário, responda 'no'.",
        # Italian
        "Si prega di analizzare il seguente post e determinare se menziona la fiducia nell'efficacia dei test dell'acido nucleico: {}. In tal caso, rispondi 'yes', altrimenti rispondi 'no'.",
        # Dutch
        "Analyseer alstublieft de volgende post en bepaal of er wordt vermeld dat men gelooft in de effectiviteit van nucleïnezuurtesten: {}. Als dat het geval is, antwoord dan met 'yes', anders met 'no'.",
        # Russian
        "Пожалуйста, проанализируйте следующий пост и определите, упоминается ли в нем вера в эффективность тестирования на нуклеиновую кислоту: {}. Если да, ответьте 'yes', в противном случае ответьте 'no'.",
        # Czech
        "Analyzujte prosím následující příspěvek a určete, zda zmiňuje víru v účinnost testování nukleové kyseliny: {}. Pokud ano, odpovězte 'yes', jinak odpovězte 'no'.",
        # Polish
        "Proszę przeanalizować poniższy post i ustalić, czy wspomina o wierze w skuteczność testów na kwas nukleinowy: {}. Jeśli tak, odpowiedz 'yes', w przeciwnym razie odpowiedz 'no'.",
        # Arabic
        "يرجى تحليل المنشور أدناه وتحديد ما إذا كان يذكر الاعتقاد بفعالية اختبار الحمض النووي: {}. إذا كان الأمر كذلك، أجب 'yes'، وإلا أجب 'no'.",
        # Persian
        "لطفاً پست زیر را تحلیل کنید و تعیین کنید که آیا اشاره شده است که به اثربخشی آزمایش اسید نوکلئیک اعتقاد دارند: {}. اگر چنین است، با 'yes' پاسخ دهید، در غیر این صورت 'no' پاسخ دهید.",
        # Hebrew
        "אנא נתחו את הפוסט הבא וקבעו אם הוא מזכיר אמונה ביעילות בדיקות חומצת גרעין: {}. אם כן, השב 'yes', אחרת השב 'no'.",
        # Turkish
        "Lütfen aşağıdaki gönderiyi analiz edin ve nükleik asit testi etkinliğine inandığını belirtip belirtmediğini belirleyin: {}. Eğer öyleyse, 'yes' yanıtını verin, aksi takdirde 'no' yanıtını verin.",
        # Japanese
        "以下の投稿を分析し、核酸検査の有効性を信じているかどうか記載されているかどうかを判断してください: {}。 そうであれば「yes」と答え、それ以外の場合は「no」と答えてください。",
        # Korean
        "다음 게시물을 분석하고 핵산 검사 효과를 믿는지 여부를 확인하십시오: {}. 그렇다면 'yes'로 답하고, 그렇지 않다면 'no'로 답하십시오.",
        # Vietnamese
        "Vui lòng phân tích bài viết dưới đây và xác định xem nó có đề cập đến niềm tin vào hiệu quả của xét nghiệm axit nucleic hay không: {}. Nếu có, hãy trả lời 'yes', nếu không, hãy trả lời 'no'.",
        # Thai
        "โปรดวิเคราะห์โพสต์ด้านล่างนี้และระบุว่ามีการกล่าวถึงความเชื่อในประสิทธิผลของการทดสอบกรดนิวคลีอิกหรือไม่: {}. หากเป็นเช่นนั้น ให้ตอบว่า 'yes' มิฉะนั้นให้ตอบว่า 'no'",
        # Indonesian
        "Harap analisis postingan di bawah ini dan tentukan apakah disebutkan keyakinan akan efektivitas pengujian asam nukleat: {}. Jika demikian, balas dengan 'yes', jika tidak, balas dengan 'no'.",
        # Malay
        "Sila analisis siaran di bawah dan tentukan sama ada ia menyebutkan kepercayaan terhadap keberkesanan ujian asid nukleik: {}. Jika ya, sila jawab 'yes', jika tidak, jawab 'no'.",
        # Lao
        "โปรดวิเคราะห์โพสต์ด้านล่างนี้และระบุว่ามีการกล่าวถึงความเชื่อในประสิทธิผลของการทดสอบกรดนิวคลีอิกหรือไม่: {}. หากเป็นเช่นนั้น ให้ตอบว่า 'yes' มิฉะนั้นให้ตอบว่า 'no'",
        # Burmese
        "ကျေးဇူးပြု၍ အောက်ပါစာမျက်နှာကိုသုံးသပ်ပြီး နျဴကလီဝမ်အက်ဆစ်စစ်ဆေးမှု၏ ထိရောက်မှုကို ယုံကြည်ကြောင်း ရှာဖွေပါ။ {}. အကယ်၍အကယ်၍တစ်ခုပြုလုပ်ရန် 'yes' ဖြင့်ပြန်ကြားပါ၊ မဟုတ်ပါက 'no' ဖြင့်ပြန်ကြားပါ။",
        # Cebuano
        "Pagtudlo sa mga plano o mga appointment alang sa nucleic acid testing. Nagpasabot kini nga ang post naghisgot bahin sa pagsugod sa pagsulay sa dili madugay, nga naghimo og appointment, nagplano sa paghimo og appointment, o pagpangandam alang sa pagsulay: {}. Palihug paghimo usa ka paghukom base sa sulod sa ibabaw. Kung mao, tubaga nga 'yes', kung dili, tubaga nga 'no'.",
        # Khmer
        "សូម​វិភាគ​អត្ថបទ​សេដ្ឋកិច្ច​ខាងក្រោម​នេះ ហើយ​សម្រេចថា តើ​វា​មាន​គម្រោង​នោះទេ ឬ​ការណាត់ជួប​សម្រាប់​ការសាកល្បង​អាស៊ីត​នុយក្លេអ៊ីក។ វា​មានន័យថា​ការ​ប្រកាស​បាន​រំឭក​ពី​ការ​ចាប់ផ្តើម​ធ្វើ​តេស្ត​នៅក្នុង​ពេលឆាប់​នេះ ការណាត់ជួប ការធ្វើគម្រោង​នេះ សម្រាប់ការណាត់ជួប ឬ​ការ​រៀបចំ​សម្រាប់​ការសាកល្បង៖ {}. ប្រសិនបើ​បាទ/ចាស សូមឆ្លើយថា 'yes' បើ​មិនមែន ឆ្លើយថា 'no'.",
        # Tagalog
        "Pakisuri ang sumusunod na post at tukuyin kung ito ay binabanggit ang mga plano o mga appointment para sa pagsusuri ng nucleic acid. Ipinapahiwatig nito na ang post ay binanggit na magsisimula ang pagsusuri, nag-set na ng appointment, plano na mag-set ng appointment, o naghahanda na para sa pagsusuri: {}. Kung gayon, sumagot ng 'yes', kung hindi, sumagot ng 'no'.",
        # Hindi
        "कृपया निम्नलिखित پوسٹ کا تجزیہ کریں اور یہ بتائیں کہ کیا اس میں نیوکلک ایسڈ ٹیسٹ کے بارے میں ہچکچاہٹ کا ذکر ہے: {}. اگر ہاں تو 'yes' جواب دیں؛ بصورت دیگر، 'no' جواب دیں۔"
    ],

    '6.2': [
        # Chinese
        "请分析以下微博内容，并判断其是否提到质疑核酸检测的有效性: {}。如果是，请回答'yes'，否则回答'no'。",
        # English
        "Please analyze the following post and determine if it mentions doubts about the accuracy of nucleic acid testing: {}. If so, respond 'yes'; otherwise, respond 'no'.",
        # German
        "Bitte analysieren Sie den folgenden Beitrag und bestimmen Sie, ob er Zweifel an der Genauigkeit von Nukleinsäuretests erwähnt: {}. Wenn ja, antworten Sie mit 'yes', andernfalls mit 'no'.",
        # French
        "Veuillez analyser le post ci-dessous et déterminer s'il mentionne des doutes sur la précision des tests d'acide nucléique: {}. Si c'est le cas, répondez 'yes', sinon répondez 'no'.",
        # Spanish
        "Por favor, analice el siguiente post y determine si menciona dudas sobre la precisión de las pruebas de ácido nucleico: {}. Si es así, responda 'yes', de lo contrario, responda 'no'.",
        # Portuguese
        "Por favor, analise a postagem abaixo e determine se ela menciona dúvidas sobre a precisão dos testes de ácido nucleico: {}. Se for o caso, responda 'yes', caso contrário, responda 'no'.",
        # Italian
        "Si prega di analizzare il seguente post e determinare se menziona dubbi sulla precisione dei test dell'acido nucleico: {}. In tal caso, rispondi 'yes', altrimenti rispondi 'no'.",
        # Dutch
        "Analyseer alstublieft de volgende post en bepaal of er wordt vermeld dat er twijfels zijn over de nauwkeurigheid van nucleïnezuurtesten: {}. Als dat het geval is, antwoord dan met 'yes', anders met 'no'.",
        # Russian
        "Пожалуйста, проанализируйте следующий пост и определите, упоминаются ли в нем сомнения в точности тестирования на нуклеиновую кислоту: {}. Если да, ответьте 'yes', в противном случае ответьте 'no'.",
        # Czech
        "Analyzujte prosím následující příspěvek a určete, zda zmiňuje pochybnosti o přesnosti testování nukleové kyseliny: {}. Pokud ano, odpovězte 'yes', jinak odpovězte 'no'.",
        # Polish
        "Proszę przeanalizować poniższy post i ustalić, czy wspomina o wątpliwościach co do dokładności testów na kwas nukleinowy: {}. Jeśli tak, odpowiedz 'yes', w przeciwnym razie odpowiedz 'no'.",
        # Arabic
        "يرجى تحليل المنشور أدناه وتحديد ما إذا كان يذكر الشكوك حول دقة اختبار الحمض النووي: {}. إذا كان الأمر كذلك، أجب 'yes'، وإلا أجب 'no'.",
        # Persian
        "لطفاً پست زیر را تحلیل کنید و تعیین کنید که آیا اشاره شده است که در مورد دقت آزمایش اسید نوکلئیک تردیدهایی وجود دارد: {}. اگر چنین است، با 'yes' پاسخ دهید، در غیر این صورت 'no' پاسخ دهید.",
        # Hebrew
        "אנא נתחו את הפוסט הבא וקבעו אם הוא מזכיר ספקות לגבי דיוק בדיקות חומצת גרעין: {}. אם כן, השב 'yes', אחרת השב 'no'.",
        # Turkish
        "Lütfen aşağıdaki gönderiyi analiz edin ve nükleik asit testi doğruluğu hakkında şüpheler belirtip belirtmediğini belirleyin: {}. Eğer öyleyse, 'yes' yanıtını verin, aksi takdirde 'no' yanıtını verin.",
        # Japanese
        "以下の投稿を分析し、核酸検査の正確性に疑問を呈しているかどうか記載されているかどうかを判断してください: {}。 そうであれば「yes」と答え、それ以外の場合は「no」と答えてください。",
        # Korean
        "다음 게시물을 분석하고 핵산 검사 정확성에 의구심이 있는지 여부를 확인하십시오: {}. 그렇다면 'yes'로 답하고, 그렇지 않다면 'no'로 답하십시오.",
        # Vietnamese
        "Vui lòng phân tích bài viết dưới đây và xác định xem nó có đề cập đến những nghi ngờ về độ chính xác của xét nghiệm axit nucleic hay không: {}. Nếu có, hãy trả lời 'yes', nếu không, hãy trả lời 'no'.",
        # Thai
        "โปรดวิเคราะห์โพสต์ด้านล่างนี้และระบุว่ามีการกล่าวถึงข้อสงสัยเกี่ยวกับความถูกต้องของการทดสอบกรดนิวคลีอิกหรือไม่: {}. หากเป็นเช่นนั้น ให้ตอบว่า 'yes' มิฉะนั้นให้ตอบว่า 'no'",
        # Indonesian
        "Harap analisis postingan di bawah ini dan tentukan apakah disebutkan keraguan tentang keakuratan pengujian asam nukleat: {}. Jika demikian, balas dengan 'yes', jika tidak, balas dengan 'no'.",
        # Malay
        "Sila analisis siaran di bawah dan tentukan sama ada ia menyebutkan keraguan terhadap ketepatan ujian asid nukleik: {}. Jika ya, sila jawab 'yes', jika tidak, jawab 'no'.",
        # Lao
        "โปรดวิเคราะห์โพสต์ด้านล่างนี้และระบุว่ามีการกล่าวถึงข้อสงสัยเกี่ยวกับความถูกต้องของการทดสอบกรดนิวคลีอิกหรือไม่: {}. หากเป็นเช่นนั้น ให้ตอบว่า 'yes' มิฉะนั้นให้ตอบว่า 'no'",
        # Burmese
        "ကျေးဇူးပြု၍ အောက်ပါစာမျက်နှာကိုသုံးသပ်ပြီး နျဴကလီဝမ်အက်ဆစ်စစ်ဆေးမှု၏ ထိရောက်မှုကို ယုံကြည်ကြောင်း ရှာဖွေပါ။ {}. အကယ်၍အကယ်၍တစ်ခုပြုလုပ်ရန် 'yes' ဖြင့်ပြန်ကြားပါ၊ မဟုတ်ပါက 'no' ဖြင့်ပြန်ကြားပါ။",
        # Cebuano
        "Pagtudlo sa mga plano o mga appointment alang sa nucleic acid testing. Nagpasabot kini nga ang post naghisgot bahin sa pagsugod sa pagsulay sa dili madugay, nga naghimo og appointment, nagplano sa paghimo og appointment, o pagpangandam alang sa pagsulay: {}. Palihug paghimo usa ka paghukom base sa sulod sa ibabaw. Kung mao, tubaga nga 'yes', kung dili, tubaga nga 'no'.",
        # Khmer
        "សូម​វិភាគ​អត្ថបទ​សេដ្ឋកិច្ច​ខាងក្រោម​នេះ ហើយ​សម្រេចថា តើ​វា​មាន​គម្រោង​នោះទេ ឬ​ការណាត់ជួប​សម្រាប់​ការសាកល្បង​អាស៊ីត​នុយក្លេអ៊ីក។ វា​មានន័យថា​ការ​ប្រកាស​បាន​រំឭក​ពី​ការ​ចាប់ផ្តើម​ធ្វើ​តេស្ត​នៅក្នុង​ពេលឆាប់​នេះ ការណាត់ជួប ការធ្វើគម្រោង​នេះ សម្រាប់ការណាត់ជួប ឬ​ការ​រៀបចំ​សម្រាប់​ការសាកល្បង៖ {}. ប្រសិនបើ​បាទ/ចាស សូមឆ្លើយថា 'yes' បើ​មិនមែន ឆ្លើយថា 'no'.",
        # Tagalog
        "Pakisuri ang sumusunod na post at tukuyin kung ito ay binabanggit ang mga plano o mga appointment para sa pagsusuri ng nucleic acid. Ipinapahiwatig nito na ang post ay binanggit na magsisimula ang pagsusuri, nag-set na ng appointment, plano na mag-set ng appointment, o naghahanda na para sa pagsusuri: {}. Kung gayon, sumagot ng 'yes', kung hindi, sumagot ng 'no'.",
        # Hindi
        "कृपया निम्नलिखل پوسٹ کا تجزیہ کریں اور یہ بتائیں کہ کیا اس میں نیوکلک ایسڈ ٹیسٹ کے بارے میں ہچکچاہٹ کا ذکر ہے: {}. اگر ہاں تو 'yes' جواب دیں؛ بصورت دیگر، 'no' جواب دیں۔"
    ],

    '7': [
        # Chinese
        "请分析以下微博内容，并判断其是否提到核酸检测有效期: {}。如果是，请回答'yes'，否则回答'no'。",
        # English
        "Please analyze the following post and determine if it mentions the validity period of nucleic acid testing results: {}. If so, respond 'yes'; otherwise, respond 'no'.",
        # German
        "Bitte analysieren Sie den folgenden Beitrag und bestimmen Sie, ob er die Gültigkeitsdauer der Nukleinsäuretestergebnisse erwähnt: {}. Wenn ja, antworten Sie mit 'yes', andernfalls mit 'no'.",
        # French
        "Veuillez analyser le post ci-dessous et déterminer s'il mentionne la période de validité des résultats des tests d'acide nucléique: {}. Si c'est le cas, répondez 'yes', sinon répondez 'no'.",
        # Spanish
        "Por favor, analice el siguiente post y determine si menciona el período de validez de los resultados de las pruebas de ácido nucleico: {}. Si es así, responda 'yes', de lo contrario, responda 'no'.",
        # Portuguese
        "Por favor, analise a postagem abaixo e determine se ela menciona o período de validade dos resultados dos testes de ácido nucleico: {}. Se for o caso, responda 'yes', caso contrário, responda 'no'.",
        # Italian
        "Si prega di analizzare il seguente post e determinare se menziona il periodo di validità dei risultati dei test dell'acido nucleico: {}. In tal caso, rispondi 'yes', altrimenti rispondi 'no'.",
        # Dutch
        "Analyseer alstublieft de volgende post en bepaal of er wordt vermeld dat de geldigheidsduur van de nucleïnezuurtestresultaten wordt vermeld: {}. Als dat het geval is, antwoord dan met 'yes', anders met 'no'.",
        # Russian
        "Пожалуйста, проанализируйте следующий пост и определите, упоминается ли в нем срок действия результатов тестирования на нуклеиновую кислоту: {}. Если да, ответьте 'yes', в противном случае ответьте 'no'.",
        # Czech
        "Analyzujte prosím následující příspěvek a určete, zda zmiňuje dobu platnosti výsledků testování nukleové kyseliny: {}. Pokud ano, odpovězte 'yes', jinak odpovězte 'no'.",
        # Polish
        "Proszę przeanalizować poniższy post i ustalić, czy wspomina o okresie ważności wyników testów na kwas nukleinowy: {}. Jeśli tak, odpowiedz 'yes', w przeciwnym razie odpowiedz 'no'.",
        # Arabic
        "يرجى تحليل المنشور أدناه وتحديد ما إذا كان يذكر فترة صلاحية نتائج اختبار الحمض النووي: {}. إذا كان الأمر كذلك، أجب 'yes'، وإلا أجب 'no'.",
        # Persian
        "لطفاً پست زیر را تحلیل کنید و تعیین کنید که آیا اشاره شده است که مدت اعتبار نتایج آزمایش اسید نوکلئیک ذکر شده است: {}. اگر چنین است، با 'yes' پاسخ دهید، در غیر این صورت 'no' پاسخ دهید.",
        # Hebrew
        "אנא נתחו את הפוסט הבא וקבעו אם הוא מזכיר את תקופת התוקף של תוצאות בדיקות חומצת גרעין: {}. אם כן, השב 'yes', אחרת השב 'no'.",
        # Turkish
        "Lütfen aşağıdaki gönderiyi analiz edin ve nükleik asit testi sonuçlarının geçerlilik süresini belirtip belirtmediğini belirleyin: {}. Eğer öyleyse, 'yes' yanıtını verin, aksi takdirde 'no' yanıtını verin.",
        # Japanese
        "以下の投稿を分析し、核酸検査結果の有効期限が記載されているかどうかを判断してください: {}。 そうであれば「yes」と答え、それ以外の場合は「no」と答えてください。",
        # Korean
        "다음 게시물을 분석하고 핵산 검사 결과의 유효 기간이 언급되어 있는지 여부를 확인하십시오: {}. 그렇다면 'yes'로 답하고, 그렇지 않다면 'no'로 답하십시오.",
        # Vietnamese
        "Vui lòng phân tích bài viết dưới đây và xác định xem nó có đề cập đến thời hạn hiệu lực của kết quả xét nghiệm axit nucleic hay không: {}. Nếu có, hãy trả lời 'yes', nếu không, hãy trả lời 'no'.",
        # Thai
        "โปรดวิเคราะห์โพสต์ด้านล่างนี้และระบุว่ามีการกล่าวถึงระยะเวลาที่ใช้ได้ของผลการทดสอบกรดนิวคลีอิกหรือไม่: {}. หากเป็นเช่นนั้น ให้ตอบว่า 'yes' มิฉะนั้นให้ตอบว่า 'no'",
        # Indonesian
        "Harap analisis postingan di bawah ini dan tentukan apakah disebutkan masa berlaku hasil pengujian asam nukleat: {}. Jika demikian, balas dengan 'yes', jika tidak, balas dengan 'no'.",
        # Malay
        "Sila analisis siaran di bawah dan tentukan sama ada ia menyebutkan tempoh sah keputusan ujian asid nukleik: {}. Jika ya, sila jawab 'yes', jika tidak, jawab 'no'.",
        # Lao
        "โปรดวิเคราะห์โพสต์ด้านล่างนี้และระบุว่ามีการกล่าวถึงระยะเวลาที่ใช้ได้ของผลการทดสอบกรดนิวคลีอิกหรือไม่: {}. หากเป็นเช่นนั้น ให้ตอบว่า 'yes' มิฉะนั้นให้ตอบว่า 'no'",
        # Burmese
        "ကျေးဇူးပြု၍ အောက်ပါစာမျက်နှာကိုသုံးသပ်ပြီး နျဴကလီဝမ်အက်ဆစ်စစ်ဆေးမှု၏ ထိရောက်မှုကို ယုံကြည်ကြောင်း ရှာဖွေပါ။ {}. အကယ်၍အကယ်၍တစ်ခုပြုလုပ်ရန် 'yes' ဖြင့်ပြန်ကြားပါ၊ မဟုတ်ပါက 'no' ဖြင့်ပြန်ကြားပါ။",
        # Cebuano
        "Pagtudlo sa mga plano o mga appointment alang sa nucleic acid testing. Nagpasabot kini nga ang post naghisgot bahin sa pagsugod sa pagsulay sa dili madugay, nga naghimo og appointment, nagplano sa paghimo og appointment, o pagpangandam alang sa pagsulay: {}. Palihug paghimo usa ka paghukom base sa sulod sa ibabaw. Kung mao, tubaga nga 'yes', kung dili, tubaga nga 'no'.",
        # Khmer
        "សូម​វិភាគ​អត្ថបទ​សេដ្ឋកិច្ច​ខាងក្រោម​នេះ ហើយ​សម្រេចថា តើ​វា​មាន​គម្រោង​នោះទេ ឬ​ការណាត់ជួប​សម្រាប់​ការសាកល្បង​អាស៊ីត​នុយក្លេអ៊ីក។ វា​មានន័យថា​ការ​ប្រកាស​បាន​រំឭក​ពី​ការ​ចាប់ផ្តើម​ធ្វើ​តេស្ត​នៅក្នុង​ពេលឆាប់​នេះ ការណាត់ជួប ការធ្វើគម្រោង​នេះ សម្រាប់ការណាត់ជួប ឬ​ការ​រៀបចំ​សម្រាប់​ការសាកល្បង៖ {}. ប្រសិនបើ​បាទ/ចាស សូមឆ្លើយថា 'yes' បើ​មិនមែន ឆ្លើយថា 'no'.",
        # Tagalog
        "Pakisuri ang sumusunod na post at tukuyin kung ito ay binabanggit ang mga plano o mga appointment para sa pagsusuri ng nucleic acid. Ipinapahiwatig nito na ang post ay binanggit na magsisimula ang pagsusuri, nag-set na ng appointment, plano na mag-set ng appointment, o naghahanda na para sa pagsusuri: {}. Kung gayon, sumagot ng 'yes', kung hindi, sumagot ng 'no'.",
        # Hindi
        "कृपया निम्नलिखल پوسٹ کا تجزیہ کریں اور یہ بتائیں کہ کیا اس میں نیوکلک ایسڈ ٹیسٹ کے بارے میں ہچکچاہٹ کا ذکر ہے: {}. اگر ہاں تو 'yes' جواب دیں؛ بصورت دیگر، 'no' جواب دیں۔"
    ],

    '8': [
        # Chinese
        "请分析以下微博内容，并判断其是否提到核酸检测的附带风险（如人群聚集导致交叉感染等）: {}。如果是，请回答'yes'，否则回答'no'。",
        # English
        "Please analyze the following post and determine if it mentions risks associated with nucleic acid testing (such as cross-infection due to crowd gatherings, etc.): {}. If so, respond 'yes'; otherwise, respond 'no'.",
        # German
        "Bitte analysieren Sie den folgenden Beitrag und bestimmen Sie, ob er Risiken im Zusammenhang mit Nukleinsäuretests erwähnt (z. B. Kreuzinfektionen aufgrund von Menschenansammlungen usw.): {}. Wenn ja, antworten Sie mit 'yes', andernfalls mit 'no'.",
        # French
        "Veuillez analyser le post ci-dessous et déterminer s'il mentionne les risques associés aux tests d'acide nucléique (comme la contamination croisée due aux rassemblements, etc.): {}. Si c'est le cas, répondez 'yes', sinon répondez 'no'.",
        # Spanish
        "Por favor, analice el siguiente post y determine si menciona los riesgos asociados con las pruebas de ácido nucleico (como la infección cruzada debido a reuniones multitudinarias, etc.): {}. Si es así, responda 'yes', de lo contrario, responda 'no'.",
        # Portuguese
        "Por favor, analise a postagem abaixo e determine se ela menciona os riscos associados aos testes de ácido nucleico (como a infecção cruzada devido a aglomerações, etc.): {}. Se for o caso, responda 'yes', caso contrário, responda 'no'.",
        # Italian
        "Si prega di analizzare il seguente post e determinare se menziona i rischi associati ai test dell'acido nucleico (come infezioni incrociate dovute a raduni di massa, ecc.): {}. In tal caso, rispondi 'yes', altrimenti rispondi 'no'.",
        # Dutch
        "Analyseer alstublieft de volgende post en bepaal of er wordt vermeld dat er risico's zijn die verband houden met nucleïnezuurtesten (zoals kruisbesmetting door bijeenkomsten, enz.): {}. Als dat het geval is, antwoord dan met 'yes', anders met 'no'.",
        # Russian
        "Пожалуйста, проанализируйте следующий пост и определите, упоминаются ли в нем риски, связанные с тестированием на нуклеиновую кислоту (например, перекрестное заражение из-за скопления людей и т. д.): {}. Если да, ответьте 'yes', в противном случае ответьте 'no'.",
        # Czech
        "Analyzujte prosím následující příspěvek a určete, zda zmiňuje rizika spojená s testováním nukleové kyseliny (např. křížová infekce kvůli shromážděním lidí atd.): {}. Pokud ano, odpovězte 'yes', jinak odpovězte 'no'.",
        # Polish
        "Proszę przeanalizować poniższy post i ustalić, czy wspomina o ryzyku związanym z testami na kwas nukleinowy (np. infekcja krzyżowa spowodowana zgromadzeniami itp.): {}. Jeśli tak, odpowiedz 'yes', w przeciwnym razie odpowiedz 'no'.",
        # Arabic
        "يرجى تحليل المنشور أدناه وتحديد ما إذا كان يذكر المخاطر المرتبطة باختبار الحمض النووي (مثل العدوى المتقاطعة بسبب تجمعات الحشود، إلخ.): {}. إذا كان الأمر كذلك، أجب 'yes'، وإلا أجب 'no'.",
        # Persian
        "لطفاً پست زیر را تحلیل کنید و تعیین کنید که آیا اشاره شده است که نسبت به خطرات مرتبط با آزمایش اسید نوکلئیک (مانند عفونت متقاطع به دلیل تجمعات مردمی، و غیره) اشاره شده است: {}. اگر چنین است، با 'yes' پاسخ دهید، در غیر این صورت 'no' پاسخ دهید.",
        # Hebrew
        "אנא נתחו את הפוסט הבא וקבעו אם הוא מזכיר סיכונים הקשורים לבדיקות חומצת גרעין (כגון זיהום צולב עקב התכנסות המונים, וכו'): {}. אם כן, השב 'yes', אחרת השב 'no'.",
        # Turkish
        "Lütfen aşağıdaki gönderiyi analiz edin ve nükleik asit testi ile ilgili riskleri (örneğin, kalabalık toplanmalardan kaynaklanan çapraz enfeksiyon vb.) belirtip belirtmediğini belirleyin: {}. Eğer öyleyse, 'yes' yanıtını verin, aksi takdirde 'no' yanıtını verin.",
        # Japanese
        "以下の投稿を分析し、核酸検査に伴うリスク（例えば、集団による交差感染など）が記載されているかどうかを判断してください: {}。 そうであれば「yes」と答え、それ以外の場合は「no」と答えてください。",
        # Korean
        "다음 게시물을 분석하고 핵산 검사와 관련된 위험(예: 군중 모임으로 인한 교차 감염 등)이 언급되어 있는지 확인하십시오: {}. 그렇다면 'yes'로 답하고, 그렇지 않다면 'no'로 답하십시오.",
        # Vietnamese
        "Vui lòng phân tích bài viết dưới đây và xác định xem nó có đề cập đến các rủi ro liên quan đến xét nghiệm axit nucleic (chẳng hạn như lây nhiễm chéo do các cuộc tụ tập đông người, v.v.) hay không: {}. Nếu có, hãy trả lời 'yes', nếu không, hãy trả lời 'no'.",
        # Thai
        "โปรดวิเคราะห์โพสต์ด้านล่างนี้และระบุว่ามีการกล่าวถึงความเสี่ยงที่เกี่ยวข้องกับการทดสอบกรดนิวคลีอิก (เช่น การติดเชื้อไขว้เนื่องจากการรวมตัวของฝูงชน เป็นต้น) หรือไม่: {}. หากเป็นเช่นนั้น ให้ตอบว่า 'yes' มิฉะนั้นให้ตอบว่า 'no'",
        # Indonesian
        "Harap analisis postingan di bawah ini dan tentukan apakah disebutkan risiko yang terkait dengan pengujian asam nukleat (seperti infeksi silang karena kerumunan, dll.): {}. Jika demikian, balas dengan 'yes', jika tidak, balas dengan 'no'.",
        # Malay
        "Sila analisis siaran di bawah dan tentukan sama ada ia menyebutkan risiko yang berkaitan dengan ujian asid nukleik (seperti jangkitan silang akibat perhimpunan ramai, dsb.): {}. Jika ya, sila jawab 'yes', jika tidak, jawab 'no'.",
        # Lao
        "โปรดวิเคราะห์โพสต์ด้านล่างนี้และระบุว่ามีการกล่าวถึงความเสี่ยงที่เกี่ยวข้องกับการทดสอบกรดนิวคลีอิก (เช่น การติดเชื้อไขว้เนื่องจากการรวมตัวของฝูงชน เป็นต้น) หรือไม่: {}. หากเป็นเช่นนั้น ให้ตอบว่า 'yes' มิฉะนั้นให้ตอบว่า 'no'",
        # Burmese
        "ကျေးဇူးပြု၍ အောက်ပါစာမျက်နှာကိုသုံးသပ်ပြီး နျဴကလီဝမ်အက်ဆစ်စစ်ဆေးမှုအတွက် အစီအစဉ်များ သို့မဟုတ် ချိန်းဆိုချက်များကို ရှာဖွေပါ။ ၎င်းသည် စစ်ဆေးမှုကိုမကြာမီစတင်ရန်၊ ချိန်းဆိုချက်တစ်ခုပြုလုပ်ပြီး၊ ချိန်းဆိုချက်တစ်ခုစီစဉ်ထားခြင်း သို့မဟုတ် စစ်ဆေးမှုအတွက်ပြင်ဆင်ထားကြောင်းကိုဖော်ပြသည်။ {}. အကယ်၍အကယ်၍တစ်ခုပြုလုပ်ရန် 'yes' ဖြင့်ပြန်ကြားပါ၊ မဟုတ်ပါက 'no' ဖြင့်ပြန်ကြားပါ။",
        # Cebuano
        "Pagtudlo sa mga plano o mga appointment alang sa nucleic acid testing. Nagpasabot kini nga ang post naghisgot bahin sa pagsugod sa pagsulay sa dili madugay, nga naghimo og appointment, nagplano sa paghimo og appointment, o pagpangandam alang sa pagsulay: {}. Palihug paghimo usa ka paghukom base sa sulod sa ibabaw. Kung mao, tubaga nga 'yes', kung dili, tubaga nga 'no'.",
        # Khmer
        "សូម​វិភាគ​អត្ថបទ​សេដ្ឋកិច្ច​ខាងក្រោម​នេះ ហើយ​សម្រេចថា តើ​វា​មាន​គម្រោង​នោះទេ ឬ​ការណាត់ជួប​សម្រាប់​ការសាកល្បង​អាស៊ីត​នុយក្លេអ៊ីក។ វា​មានន័យថា​ការ​ប្រកាស​បាន​រំឭក​ពី​ការ​ចាប់ផ្តើម​ធ្វើ​តេស្ត​នៅក្នុង​ពេលឆាប់​នេះ ការណាត់ជួប ការធ្វើគម្រោង​នេះ សម្រាប់ការណាត់ជួប ឬ​ការ​រៀបចំ​សម្រាប់​ការសាកល្បង៖ {}. ប្រសិនបើ​បាទ/ចាស សូមឆ្លើយថា 'yes' បើ​មិនមែន ឆ្លើយថា 'no'.",
        # Tagalog
        "Pakisuri ang sumusunod na post at tukuyin kung ito ay binabanggit ang mga plano o mga appointment para sa pagsusuri ng nucleic acid. Ipinapahiwatig nito na ang post ay binanggit na magsisimula ang pagsusuri, nag-set na ng appointment, plano na mag-set ng appointment, o naghahanda na para sa pagsusuri: {}. Kung gayon, sumagot ng 'yes', kung hindi, sumagot ng 'no'.",
        # Hindi
        "कृपया निम्नलिखल پوسٹ کا تجزیہ کریں اور یہ بتائیں کہ کیا اس میں نیوکلک ایسڈ ٹیسٹ کے بارے میں ہچکچاہٹ کا ذکر ہے: {}. اگر ہاں تو 'yes' جواب دیں؛ بصورت دیگر، 'no' جواب دیں۔"
    ],

    '9': [
        # Chinese
        "请分析以下微博内容，并判断其是否提到对核酸检测的疑问（通常以问题形式出现）: {}。如果是，请回答'yes'，否则回答'no'。",
        # English
        "Please analyze the following post and determine if it mentions questions about nucleic acid testing (usually in the form of questions): {}. If so, respond 'yes'; otherwise, respond 'no'.",
        # German
        "Bitte analysieren Sie den folgenden Beitrag und bestimmen Sie, ob er Fragen zu Nukleinsäuretests erwähnt (normalerweise in Form von Fragen): {}. Wenn ja, antworten Sie mit 'yes', andernfalls mit 'no'.",
        # French
        "Veuillez analyser le post ci-dessous et déterminer s'il mentionne des questions sur les tests d'acide nucléique (généralement sous forme de questions): {}. Si c'est le cas, répondez 'yes', sinon répondez 'no'.",
        # Spanish
        "Por favor, analice el siguiente post y determine si menciona preguntas sobre las pruebas de ácido nucleico (generalmente en forma de preguntas): {}. Si es así, responda 'yes', de lo contrario, responda 'no'.",
        # Portuguese
        "Por favor, analise a postagem abaixo e determine se ela menciona perguntas sobre testes de ácido nucleico (geralmente em forma de perguntas): {}. Se for o caso, responda 'yes', caso contrário, responda 'no'.",
        # Italian
        "Si prega di analizzare il seguente post e determinare se menziona domande sui test dell'acido nucleico (di solito sotto forma di domande): {}. In tal caso, rispondi 'yes', altrimenti rispondi 'no'.",
        # Dutch
        "Analyseer alstublieft de volgende post en bepaal of er wordt vermeld dat er vragen zijn over nucleïnezuurtesten (meestal in de vorm van vragen): {}. Als dat het geval is, antwoord dan met 'yes', anders met 'no'.",
        # Russian
        "Пожалуйста, проанализируйте следующий пост и определите, упоминаются ли в нем вопросы о тестировании на нуклеиновую кислоту (обычно в форме вопросов): {}. Если да, ответьте 'yes', в противном случае ответьте 'no'.",
        # Czech
        "Analyzujte prosím následující příspěvek a určete, zda zmiňuje otázky týkající se testování nukleové kyseliny (obvykle ve formě otázek): {}. Pokud ano, odpovězte 'yes', jinak odpovězte 'no'.",
        # Polish
        "Proszę przeanalizować poniższy post i ustalić, czy wspomina o pytaniach dotyczących testów na kwas nukleinowy (zwykle w formie pytań): {}. Jeśli tak, odpowiedz 'yes', w przeciwnym razie odpowiedz 'no'.",
        # Arabic
        "يرجى تحليل المنشور أدناه وتحديد ما إذا كان يذكر أسئلة حول اختبار الحمض النووي (عادة في شكل أسئلة): {}. إذا كان الأمر كذلك، أجب 'yes'، وإلا أجب 'no'.",
        # Persian
        "لطفاً پست زیر را تحلیل کنید و تعیین کنید که آیا اشاره شده است که نسبت به آزمایش اسید نوکلئیک سوالاتی مطرح شده است (معمولاً به صورت سوال): {}. اگر چنین است، با 'yes' پاسخ دهید، در غیر این صورت 'no' پاسخ دهید.",
        # Hebrew
        "אנא נתחו את הפוסט הבא וקבעו אם הוא מזכיר שאלות לגבי בדיקות חומצת גרעין (בדרך כלל בצורה של שאלות): {}. אם כן, השב 'yes', אחרת השב 'no'.",
        # Turkish
        "Lütfen aşağıdaki gönderiyi analiz edin ve nükleik asit testi hakkında sorular (genellikle sorular şeklinde) olup olmadığını belirleyin: {}. Eğer öyleyse, 'yes' yanıtını verin, aksi takdirde 'no' yanıtını verin.",
        # Japanese
        "以下の投稿を分析し、核酸検査に関する質問（通常は質問の形式で）が記載されているかどうかを判断してください: {}。 そうであれば「yes」と答え、それ以外の場合は「no」と答えてください。",
        # Korean
        "다음 게시물을 분석하고 핵산 검사에 대한 질문(일반적으로 질문의 형태로)이 언급되어 있는지 확인하십시오: {}. 그렇다면 'yes'로 답하고, 그렇지 않다면 'no'로 답하십시오.",
        # Vietnamese
        "Vui lòng phân tích bài viết dưới đây và xác định xem nó có đề cập đến các câu hỏi về xét nghiệm axit nucleic (thường ở dạng câu hỏi) hay không: {}. Nếu có, hãy trả lời 'yes', nếu không, hãy trả lời 'no'.",
        # Thai
        "โปรดวิเคราะห์โพสต์ด้านล่างนี้และระบุว่ามีการกล่าวถึงคำถามเกี่ยวกับการทดสอบกรดนิวคลีอิก (มักจะอยู่ในรูปแบบของคำถาม) หรือไม่: {}. หากเป็นเช่นนั้น ให้ตอบว่า 'yes' มิฉะนั้นให้ตอบว่า 'no'",
        # Indonesian
        "Harap analisis postingan di bawah ini dan tentukan apakah disebutkan pertanyaan tentang pengujian asam nukleat (biasanya dalam bentuk pertanyaan): {}. Jika demikian, balas dengan 'yes', jika tidak, balas dengan 'no'.",
        # Malay
        "Sila analisis siaran di bawah dan tentukan sama ada ia menyebutkan soalan mengenai ujian asid nukleik (biasanya dalam bentuk soalan): {}. Jika ya, sila jawab 'yes', jika tidak, jawab 'no'.",
        # Lao
        "โปรดวิเคราะห์โพสต์ด้านล่างนี้และระบุว่ามีการกล่าวถึงคำถามเกี่ยวกับการทดสอบกรดนิวคลีอิก (มักจะอยู่ในรูปแบบของคำถาม) หรือไม่: {}. หากเป็นเช่นนั้น ให้ตอบว่า 'yes' มิฉะนั้นให้ตอบว่า 'no'",
        # Burmese
        "ကျေးဇူးပြု၍ အောက်ပါစာမျက်နှာကိုသုံးသပ်ပြီး နျဴကလီဝမ်အက်ဆစ်စစ်ဆေးမှုအတွက် အစီအစဉ်များ သို့မဟုတ် ချိန်းဆိုချက်များကို ရှာဖွေပါ။ ၎င်းသည် စစ်ဆေးမှုကိုမကြာမီစတင်ရန်၊ ချိန်းဆိုချက်တစ်ခုပြုလုပ်ပြီး၊ ချိန်းဆိုချက်တစ်ခုစီစဉ်ထားခြင်း သို့မဟုတ် စစ်ဆေးမှုအတွက်ပြင်ဆင်ထားကြောင်းကိုဖော်ပြသည်။ {}. အကယ်၍အကယ်၍တစ်ခုပြုလုပ်ရန် 'yes' ဖြင့်ပြန်ကြားပါ၊ မဟုတ်ပါက 'no' ဖြင့်ပြန်ကြားပါ။",
        # Cebuano
        "Pagtudlo sa mga plano o mga appointment alang sa nucleic acid testing. Nagpasabot kini nga ang post naghisgot bahin sa pagsugod sa pagsulay sa dili madugay, nga naghimo og appointment, nagplano sa paghimo og appointment, o pagpangandam alang sa pagsulay: {}. Palihug paghimo usa ka paghukom base sa sulod sa ibabaw. Kung mao, tubaga nga 'yes', kung dili, tubaga nga 'no'.",
        # Khmer
        "សូម​វិភាគ​អត្ថបទ​សេដ្ឋកិច្ច​ខាងក្រោម​នេះ ហើយ​សម្រេចថា តើ​វា​មាន​គម្រោង​នោះទេ ឬ​ការណាត់ជួប​សម្រាប់​ការសាកល្បង​អាស៊ីត​នុយក្លេអ៊ីក។ វា​មានន័យថា​ការ​ប្រកាស​បាន​រំឭក​ពី​ការ​ចាប់ផ្តើម​ធ្វើ​តេស្ត​នៅក្នុង​ពេលឆាប់​នេះ ការណាត់ជួប ការធ្វើគម្រោង​នេះ សម្រាប់ការណាត់ជួប ឬ​ការ​រៀបចំ​សម្រាប់​ការសាកល្បង៖ {}. ប្រសិនបើ​បាទ/ចាស សូមឆ្លើយថា 'yes' បើ​មិនមែន ឆ្លើយថា 'no'.",
        # Tagalog
        "Pakisuri ang sumusunod na post at tukuyin kung ito ay binabanggit ang mga plano o mga appointment para sa pagsusuri ng nucleic acid. Ipinapahiwatig nito na ang post ay binanggit na magsisimula ang pagsusuri, nag-set na ng appointment, plano na mag-set ng appointment, o naghahanda na para sa pagsusuri: {}. Kung gayon, sumagot ng 'yes', kung hindi, sumagot ng 'no'.",
        # Hindi
        "कृपया निम्नलिखल پوسٹ کا تجزیہ کریں اور یہ بتائیں کہ کیا اس میں نیوکلک ایسڈ ٹیسٹ کے بارے میں ہچکچاہٹ کا ذکر ہے: {}. اگر ہاں تو 'yes' جواب دیں؛ بصورت دیگر، 'no' جواب دیں۔"
    ],

    '10.1': [
        # Chinese
        "请分析以下微博内容，并判断其是否提到感到对疫情高度恐慌: {}。如果是，请回答'yes'，否则回答'no'。",
        # English
        "Please analyze the following post and determine if it mentions high anxiety or panic about the pandemic: {}. If so, respond 'yes'; otherwise, respond 'no'.",
        # German
        "Bitte analysieren Sie den folgenden Beitrag und bestimmen Sie, ob er von hoher Angst oder Panik in Bezug auf die Pandemie spricht: {}. Wenn ja, antworten Sie mit 'yes', andernfalls mit 'no'.",
        # French
        "Veuillez analyser le post ci-dessous et déterminer s'il mentionne une forte anxiété ou panique à propos de la pandémie: {}. Si c'est le cas, répondez 'yes', sinon répondez 'no'.",
        # Spanish
        "Por favor, analice el siguiente post y determine si menciona una gran ansiedad o pánico sobre la pandemia: {}. Si es así, responda 'yes', de lo contrario, responda 'no'.",
        # Portuguese
        "Por favor, analise a postagem abaixo e determine se ela menciona uma alta ansiedade ou pânico sobre a pandemia: {}. Se for o caso, responda 'yes', caso contrário, responda 'no'.",
        # Italian
        "Si prega di analizzare il seguente post e determinare se menziona una grande ansia o panico riguardo alla pandemia: {}. In tal caso, rispondi 'yes', altrimenti rispondi 'no'.",
        # Dutch
        "Analyseer alstublieft de volgende post en bepaal of er wordt vermeld dat er sprake is van hoge angst of paniek over de pandemie: {}. Als dat het geval is, antwoord dan met 'yes', anders met 'no'.",
        # Russian
        "Пожалуйста, проанализируйте следующий пост и определите, упоминается ли в нем высокая тревога или паника по поводу пандемии: {}. Если да, ответьте 'yes', в противном случае ответьте 'no'.",
        # Czech
        "Analyzujte prosím následující příspěvek a určete, zda zmiňuje vysokou úzkost nebo paniku ohledně pandemie: {}. Pokud ano, odpovězte 'yes', jinak odpovězte 'no'.",
        # Polish
        "Proszę przeanalizować poniższy post i ustalić, czy wspomina o dużym lęku lub panice związanej z pandemią: {}. Jeśli tak, odpowiedz 'yes', w przeciwnym razie odpowiedz 'no'.",
        # Arabic
        "يرجى تحليل المنشور أدناه وتحديد ما إذا كان يذكر القلق الشديد أو الذعر بشأن الوباء: {}. إذا كان الأمر كذلك، أجب 'yes'، وإلا أجب 'no'.",
        # Persian
        "لطفاً پست زیر را تحلیل کنید و تعیین کنید که آیا اشاره شده است که نسبت به همه‌گیری اضطراب یا وحشت شدید وجود دارد: {}. اگر چنین است، با 'yes' پاسخ دهید، در غیر این صورت 'no' پاسخ دهید.",
        # Hebrew
        "אנא נתחו את הפוסט הבא וקבעו אם הוא מזכיר חרדה או פאניקה גבוהה בנוגע למגפה: {}. אם כן, השב 'yes', אחרת השב 'no'.",
        # Turkish
        "Lütfen aşağıdaki gönderiyi analiz edin ve pandemi hakkında yüksek kaygı veya panik olup olmadığını belirtip belirtmediğini belirleyin: {}. Eğer öyleyse, 'yes' yanıtını verin, aksi takdirde 'no' yanıtını verin.",
        # Japanese
        "以下の投稿を分析し、パンデミックに対する高い不安やパニックについて言及しているかどうかを判断してください: {}。 そうであれば「yes」と答え、それ以外の場合は「no」と答えてください。",
        # Korean
        "다음 게시물을 분석하고 팬데믹에 대한 높은 불안 또는 공황이 언급되어 있는지 확인하십시오: {}. 그렇다면 'yes'로 답하고, 그렇지 않다면 'no'로 답하십시오.",
        # Vietnamese
        "Vui lòng phân tích bài viết dưới đây và xác định xem nó có đề cập đến mức độ lo lắng cao hoặc hoảng loạn về đại dịch hay không: {}. Nếu có, hãy trả lời 'yes', nếu không, hãy trả lời 'no'.",
        # Thai
        "โปรดวิเคราะห์โพสต์ด้านล่างนี้และระบุว่ามีการกล่าวถึงความวิตกกังวลสูงหรือความตื่นตระหนกเกี่ยวกับการระบาดใหญ่หรือไม่: {}. หากเป็นเช่นนั้น ให้ตอบว่า 'yes' มิฉะนั้นให้ตอบว่า 'no'",
        # Indonesian
        "Harap analisis postingan di bawah ini dan tentukan apakah disebutkan kecemasan atau kepanikan tinggi tentang pandemi: {}. Jika demikian, balas dengan 'yes', jika tidak, balas dengan 'no'.",
        # Malay
        "Sila analisis siaran di bawah dan tentukan sama ada ia menyebutkan kebimbangan atau panik yang tinggi tentang pandemik: {}. Jika ya, sila jawab 'yes', jika tidak, jawab 'no'.",
        # Lao
        "โปรดวิเคราะห์โพสต์ด้านล่างนี้และระบุว่ามีการกล่าวถึงความวิตกกังวลสูงหรือความตื่นตระหนกเกี่ยวกับการระบาดใหญ่หรือไม่: {}. หากเป็นเช่นนั้น ให้ตอบว่า 'yes' มิฉะนั้นให้ตอบว่า 'no'",
        # Burmese
        "ကျေးဇူးပြု၍ အောက်ပါစာမျက်နှာကိုသုံးသပ်ပြီး ကပ်ရောဂါအကြောင်းကြောင့် စိုးရိမ်ပူပန်မှုများ၊ ဒဏ်ခံနိုင်သောကြောက်ရွံ့မှုများကို ရှာဖွေပါ။ {}. အကယ်၍အကယ်၍တစ်ခုပြုလုပ်ရန် 'yes' ဖြင့်ပြန်ကြားပါ၊ မဟုတ်ပါက 'no' ဖြင့်ပြန်ကြားပါ။",
        # Cebuano
        "Pagtudlo sa mga plano o mga appointment alang sa nucleic acid testing. Nagpasabot kini nga ang post naghisgot bahin sa pagsugod sa pagsulay sa dili madugay, nga naghimo og appointment, nagplano sa paghimo og appointment, o pagpangandam alang sa pagsulay: {}. Palihug paghimo usa ka paghukom base sa sulod sa ibabaw. Kung mao, tubaga nga 'yes', kung dili, tubaga nga 'no'.",
        # Khmer
        "សូម​វិភាគ​អត្ថបទ​សេដ្ឋកិច្ច​ខាងក្រោម​នេះ ហើយ​សម្រេចថា តើ​វា​មាន​គម្រោង​នោះទេ ឬ​ការណាត់ជួប​សម្រាប់​ការសាកល្បង​អាស៊ីត​នុយក្លេអ៊ីក។ វា​មានន័យថា​ការ​ប្រកាស​បាន​រំឭក​ពី​ការ​ចាប់ផ្តើម​ធ្វើ​តេស្ត​នៅក្នុង​ពេលឆាប់​នេះ ការណាត់ជួប ការធ្វើគម្រោង​នេះ សម្រាប់ការណាត់ជួប ឬ​ការ​រៀបចំ​សម្រាប់​ការសាកល្បង៖ {}. ប្រសិនបើ​បាទ/ចាស សូមឆ្លើយថា 'yes' បើ​មិនមែន ឆ្លើយថា 'no'.",
        # Tagalog
        "Pakisuri ang sumusunod na post at tukuyin kung ito ay binabanggit ang mga plano o mga appointment para sa pagsusuri ng nucleic acid. Ipinapahiwatig nito na ang post ay binanggit na magsisimula ang pagsusuri, nag-set na ng appointment, plano na mag-set ng appointment, o naghahanda na para sa pagsusuri: {}. Kung gayon, sumagot ng 'yes', kung hindi, sumagot ng 'no'.",
        # Hindi
        "कृपया निम्नलिखल پوسٹ کا تجزیہ کریں اور یہ بتائیں کہ کیا اس میں نیوکلک ایسڈ ٹیسٹ کے بارے میں ہچکچاہٹ کا ذکر ہے: {}. اگر ہاں تو 'yes' جواب دیں؛ بصورت دیگر، 'no' جواب دیں۔"
    ],

    '10.2': [
        # Chinese
        "请分析以下微博内容，并判断其是否提到认为自己感染风险低: {}。如果是，请回答'yes'，否则回答'no'。",
        # English
        "Please analyze the following post and determine if it mentions the belief that the risk of infection is low: {}. If so, respond 'yes'; otherwise, respond 'no'.",
        # German
        "Bitte analysieren Sie den folgenden Beitrag und bestimmen Sie, ob er den Glauben erwähnt, dass das Infektionsrisiko gering ist: {}. Wenn ja, antworten Sie mit 'yes', andernfalls mit 'no'.",
        # French
        "Veuillez analyser le post ci-dessous et déterminer s'il mentionne la croyance que le risque d'infection est faible: {}. Si c'est le cas, répondez 'yes', sinon répondez 'no'.",
        # Spanish
        "Por favor, analice el siguiente post y determine si menciona la creencia de que el riesgo de infección es bajo: {}. Si es así, responda 'yes', de lo contrario, responda 'no'.",
        # Portuguese
        "Por favor, analise a postagem abaixo e determine se ela menciona a crença de que o risco de infecção é baixo: {}. Se for o caso, responda 'yes', caso contrário, responda 'no'.",
        # Italian
        "Si prega di analizzare il seguente post e determinare se menziona la convinzione che il rischio di infezione sia basso: {}. In tal caso, rispondi 'yes', altrimenti rispondi 'no'.",
        # Dutch
        "Analyseer alstublieft de volgende post en bepaal of er wordt vermeld dat het geloof is dat het risico op infectie laag is: {}. Als dat het geval is, antwoord dan met 'yes', anders met 'no'.",
        # Russian
        "Пожалуйста, проанализируйте следующий пост и определите, упоминается ли в нем вера в то, что риск заражения низкий: {}. Если да, ответьте 'yes', в противном случае ответьте 'no'.",
        # Czech
        "Analyzujte prosím následující příspěvek a určete, zda zmiňuje víru, že riziko infekce je nízké: {}. Pokud ano, odpovězte 'yes', jinak odpovězte 'no'.",
        # Polish
        "Proszę przeanalizować poniższy post i ustalić, czy wspomina o przekonaniu, że ryzyko zakażenia jest niskie: {}. Jeśli tak, odpowiedz 'yes', w przeciwnym razie odpowiedz 'no'.",
        # Arabic
        "يرجى تحليل المنشور أدناه وتحديد ما إذا كان يذكر الاعتقاد بأن خطر الإصابة منخفض: {}. إذا كان الأمر كذلك، أجب 'yes'، وإلا أجب 'no'.",
        # Persian
        "لطفاً پست زیر را تحلیل کنید و تعیین کنید که آیا اشاره شده است که نسبت به اعتقاد به پایین بودن خطر عفونت وجود دارد: {}. اگر چنین است، با 'yes' پاسخ دهید، در غیر این صورت 'no' پاسخ دهید.",
        # Hebrew
        "אנא נתחו את הפוסט הבא וקבעו אם הוא מזכיר אמונה שהסיכון להדבקה הוא נמוך: {}. אם כן, השב 'yes', אחרת השב 'no'.",
        # Turkish
        "Lütfen aşağıdaki gönderiyi analiz edin ve enfeksiyon riskinin düşük olduğuna dair bir inanç olup olmadığını belirtip belirtmediğini belirleyin: {}. Eğer öyleyse, 'yes' yanıtını verin, aksi takdirde 'no' yanıtını verin.",
        # Japanese
        "以下の投稿を分析し、感染リスクが低いという信念について言及しているかどうかを判断してください: {}。 そうであれば「yes」と答え、それ以外の場合は「no」と答えてください。",
        # Korean
        "다음 게시물을 분석하고 감염 위험이 낮다는 믿음이 언급되어 있는지 확인하십시오: {}. 그렇다면 'yes'로 답하고, 그렇지 않다면 'no'로 답하십시오.",
        # Vietnamese
        "Vui lòng phân tích bài viết dưới đây và xác định xem nó có đề cập đến niềm tin rằng nguy cơ nhiễm trùng là thấp hay không: {}. Nếu có, hãy trả lời 'yes', nếu không, hãy trả lời 'no'.",
        # Thai
        "โปรดวิเคราะห์โพสต์ด้านล่างนี้และระบุว่ามีการกล่าวถึงความเชื่อที่ว่าความเสี่ยงในการติดเชื้อต่ำหรือไม่: {}. หากเป็นเช่นนั้น ให้ตอบว่า 'yes' มิฉะนั้นให้ตอบว่า 'no'",
        # Indonesian
        "Harap analisis postingan di bawah ini dan tentukan apakah disebutkan keyakinan bahwa risiko infeksi rendah: {}. Jika demikian, balas dengan 'yes', jika tidak, balas dengan 'no'.",
        # Malay
        "Sila analisis siaran di bawah dan tentukan sama ada ia menyebutkan kepercayaan bahawa risiko jangkitan adalah rendah: {}. Jika ya, sila jawab 'yes', jika tidak, jawab 'no'.",
        # Lao
        "โปรดวิเคราะห์โพสต์ด้านล่างนี้และระบุว่ามีการกล่าวถึงความเชื่อที่ว่าความเสี่ยงในการติดเชื้อต่ำหรือไม่: {}. หากเป็นเช่นนั้น ให้ตอบว่า 'yes' มิฉะนั้นให้ตอบว่า 'no'",
        # Burmese
        "ကျေးဇူးပြု၍ အောက်ပါစာမျက်နှာကိုသုံးသပ်ပြီး ကပ်ရောဂါအကြောင်းကြောင့် စိုးရိမ်ပူပန်မှုများ၊ ဒဏ်ခံနိုင်သောကြောက်ရွံ့မှုများကို ရှာဖွေပါ။ {}. အကယ်၍အကယ်၍တစ်ခုပြုလုပ်ရန် 'yes' ဖြင့်ပြန်ကြားပါ၊ မဟုတ်ပါက 'no' ဖြင့်ပြန်ကြားပါ။",
        # Cebuano
        "Pagtudlo sa mga plano o mga appointment alang sa nucleic acid testing. Nagpasabot kini nga ang post naghisgot bahin sa pagsugod sa pagsulay sa dili madugay, nga naghimo og appointment, nagplano sa paghimo og appointment, o pagpangandam alang sa pagsulay: {}. Palihug paghimo usa ka paghukom base sa sulod sa ibabaw. Kung mao, tubaga nga 'yes', kung dili, tubaga nga 'no'.",
        # Khmer
        "សូម​វិភាគ​អត្ថបទ​សេដ្ឋកិច្ច​ខាងក្រោម​នេះ ហើយ​សម្រេចថា តើ​វា​មាន​គម្រោង​នោះទេ ឬ​ការណាត់ជួប​សម្រាប់​ការសាកល្បង​អាស៊ីត​នុយក្លេអ៊ីក។ វា​មានន័យថា​ការ​ប្រកាស​បាន​រំឭក​ពី​ការ​ចាប់ផ្តើម​ធ្វើ​តេស្ត​នៅក្នុង​ពេលឆាប់​នេះ ការណាត់ជួប ការធ្វើគម្រោង​នេះ សម្រាប់ការណាត់ជួប ឬ​ការ​រៀបចំ​សម្រាប់​ការសាកល្បង៖ {}. ប្រសិនបើ​បាទ/ចាស សូមឆ្លើយថា 'yes' បើ​មិនមែន ឆ្លើយថា 'no'.",
        # Tagalog
        "Pakisuri ang sumusunod na post at tukuyin kung ito ay binabanggit ang mga plano o mga appointment para sa pagsusuri ng nucleic acid. Ipinapahiwatig nito na ang post ay binanggit na magsisimula ang pagsusuri, nag-set na ng appointment, plano na mag-set ng appointment, o naghahanda na para sa pagsusuri: {}. Kung gayon, sumagot ng 'yes', kung hindi, sumagot ng 'no'.",
        # Hindi
        "कृपया निम্নलिखल پوسٹ کا تجزیہ کریں اور یہ بتائیں کہ کیا اس میں نیوکلک ایسڈ ٹیسٹ کے بارے میں ہچکچاہٹ کا ذکر ہے: {}. اگر ہاں تو 'yes' جواب دیں؛ بصورت دیگر، 'no' جواب دیں۔"
    ],

    '11': [
        # Chinese
        "请分析以下微博内容，并判断其是否提到谣言和虚假信息: {}。如果是，请回答'yes'，否则回答'no'。",
        # English
        "Please analyze the following post and determine if it mentions rumors and misinformation: {}. If so, respond 'yes'; otherwise, respond 'no'.",
        # German
        "Bitte analysieren Sie den folgenden Beitrag und bestimmen Sie, ob er Gerüchte und Fehlinformationen erwähnt: {}. Wenn ja, antworten Sie mit 'yes', andernfalls mit 'no'.",
        # French
        "Veuillez analyser le post ci-dessous et déterminer s'il mentionne des rumeurs et des informations erronées: {}. Si c'est le cas, répondez 'yes', sinon répondez 'no'.",
        # Spanish
        "Por favor, analice el siguiente post y determine si menciona rumores e información errónea: {}. Si es así, responda 'yes', de lo contrario, responda 'no'.",
        # Portuguese
        "Por favor, analise a postagem abaixo e determine se ela menciona rumores e desinformação: {}. Se for o caso, responda 'yes', caso contrário, responda 'no'.",
        # Italian
        "Si prega di analizzare il seguente post e determinare se menziona voci e disinformazione: {}. In tal caso, rispondi 'yes', altrimenti rispondi 'no'.",
        # Dutch
        "Analyseer alstublieft de volgende post en bepaal of er wordt vermeld dat er sprake is van geruchten en verkeerde informatie: {}. Als dat het geval is, antwoord dan met 'yes', anders met 'no'.",
        # Russian
        "Пожалуйста, проанализируйте следующий пост и определите, упоминаются ли в нем слухи и дезинформация: {}. Если да, ответьте 'yes', в противном случае ответьте 'no'.",
        # Czech
        "Analyzujte prosím následující příspěvek a určete, zda zmiňuje fámy a dezinformace: {}. Pokud ano, odpovězte 'yes', jinak odpovězte 'no'.",
        # Polish
        "Proszę przeanalizować poniższy post i ustalić, czy wspomina o plotkach i dezinformacji: {}. Jeśli tak, odpowiedz 'yes', w przeciwnym razie odpowiedz 'no'.",
        # Arabic
        "يرجى تحليل المنشور أدناه وتحديد ما إذا كان يذكر الشائعات والمعلومات المضللة: {}. إذا كان الأمر كذلك، أجب 'yes'، وإلا أجب 'no'.",
        # Persian
        "لطفاً پست زیر را تحلیل کنید و تعیین کنید که آیا اشاره شده است که نسبت به شایعات و اطلاعات نادرست وجود دارد: {}. اگر چنین است، با 'yes' پاسخ دهید، در غیر این صورت 'no' پاسخ دهید.",
        # Hebrew
        "אנא נתחו את הפוסט הבא וקבעו אם הוא מזכיר שמועות ומידע שגוי: {}. אם כן, השב 'yes', אחרת השב 'no'.",
        # Turkish
        "Lütfen aşağıdaki gönderiyi analiz edin ve söylentiler ve yanlış bilgiler olup olmadığını belirtip belirtmediğini belirleyin: {}. Eğer öyleyse, 'yes' yanıtını verin, aksi takdirde 'no' yanıtını verin.",
        # Japanese
        "以下の投稿を分析し、噂や誤った情報について言及しているかどうかを判断してください: {}。 そうであれば「yes」と答え、それ以外の場合は「no」と答えてください。",
        # Korean
        "다음 게시물을 분석하고 소문과 잘못된 정보가 언급되어 있는지 확인하십시오: {}. 그렇다면 'yes'로 답하고, 그렇지 않다면 'no'로 답하십시오.",
        # Vietnamese
        "Vui lòng phân tích bài viết dưới đây và xác định xem nó có đề cập đến tin đồn và thông tin sai lệch hay không: {}. Nếu có, hãy trả lời 'yes', nếu không, hãy trả lời 'no'.",
        # Thai
        "โปรดวิเคราะห์โพสต์ด้านล่างนี้และระบุว่ามีการกล่าวถึงข่าวลือและข้อมูลที่ผิดหรือไม่: {}. หากเป็นเช่นนั้น ให้ตอบว่า 'yes' มิฉะนั้นให้ตอบว่า 'no'",
        # Indonesian
        "Harap analisis postingan di bawah ini dan tentukan apakah disebutkan rumor dan disinformasi: {}. Jika demikian, balas dengan 'yes', jika tidak, balas dengan 'no'.",
        # Malay
        "Sila analisis siaran di bawah dan tentukan sama ada ia menyebutkan khabar angin dan maklumat yang salah: {}. Jika ya, sila jawab 'yes', jika tidak, jawab 'no'.",
        # Lao
        "โปรดวิเคราะห์โพสต์ด้านล่างนี้และระบุว่ามีการกล่าวถึงข่าวลือและข้อมูลที่ผิดหรือไม่: {}. หากเป็นเช่นนั้น ให้ตอบว่า 'yes' มิฉะนั้นให้ตอบว่า 'no'",
        # Burmese
        "ကျေးဇူးပြု၍ အောက်ပါစာမျက်နှာကိုသုံးသပ်ပြီး ကပ်ရောဂါအကြောင်းကြောင့် စိုးရိမ်ပူပန်မှုများ၊ ဒဏ်ခံနိုင်သောကြောက်ရွံ့မှုများကို ရှာဖွေပါ။ {}. အကယ်၍အကယ်၍တစ်ခုပြုလုပ်ရန် 'yes' ဖြင့်ပြန်ကြားပါ၊ မဟုတ်ပါက 'no' ဖြင့်ပြန်ကြားပါ။",
        # Cebuano
        "Pagtudlo sa mga plano o mga appointment alang sa nucleic acid testing. Nagpasabot kini nga ang post naghisgot bahin sa pagsugod sa pagsulay sa dili madugay, nga naghimo og appointment, nagplano sa paghimo og appointment, o pagpangandam alang sa pagsulay: {}. Palihug paghimo usa ka paghukom base sa sulod sa ibabaw. Kung mao, tubaga nga 'yes', kung dili, tubaga nga 'no'.",
        # Khmer
        "សូម​វិភាគ​អត្ថបទ​សេដ្ឋកិច្ច​ខាងក្រោម​នេះ ហើយ​សម្រេចថា តើ​វា​មាន​គម្រោង​នោះទេ ឬ​ការណាត់ជួប​សម្រាប់​ការសាកល្បង​អាស៊ីត​នុយក្លេអ៊ីក។ វា​មានន័យថា​ការ​ប្រកាស​បាន​រំឭក​ពី​ការ​ចាប់ផ្តើម​ធ្វើ​តេស្ត​នៅក្នុង​ពេលឆាប់​នេះ ការណាត់ជួប ការធ្វើគម្រោង​នេះ សម្រាប់ការណាត់ជួប ឬ​ការ​រៀបចំ​សម្រាប់​ការសាកល្បង៖ {}. ប្រសិនបើ​បាទ/ចាស សូមឆ្លើយថា 'yes' បើ​មិនមែន ឆ្លើយថា 'no'.",
        # Tagalog
        "Pakisuri ang sumusunod na post at tukuyin kung ito ay binabanggit ang mga plano o mga appointment para sa pagsusuri ng nucleic acid. Ipinapahiwatig nito na ang post ay binanggit na magsisimula ang pagsusuri, nag-set na ng appointment, plano na mag-set ng appointment, o naghahanda na para sa pagsusuri: {}. Kung gayon, sumagot ng 'yes', kung hindi, sumagot ng 'no'.",
        # Hindi
        "कृपया निम्नلिखल پوسٹ کا تجزیہ کریں اور یہ بتائیں کہ کیا اس میں نیوکلک ایسڈ ٹیسٹ کے بارے میں ہچکچاہٹ کا ذکر ہے: {}. اگر ہاں تو 'yes' جواب دیں؛ بصورت دیگر، 'no' جواب دیں۔"
    ],

    '12.1': [
        # Chinese
        "请分析以下微博内容，并判断其是否描述了核酸检测前感到紧张: {}。如果是，请回答'yes'，否则回答'no'。",
        # English
        "Please analyze the following post and determine if it describes feeling nervous before a nucleic acid test: {}. If so, respond 'yes'; otherwise, respond 'no'.",
        # German
        "Bitte analysieren Sie den folgenden Beitrag und bestimmen Sie, ob er beschreibt, dass man sich vor einem Nukleinsäuretest nervös fühlt: {}. Wenn ja, antworten Sie mit 'yes', andernfalls mit 'no'.",
        # French
        "Veuillez analyser le post ci-dessous et déterminer s'il décrit une sensation de nervosité avant un test d'acide nucléique: {}. Si c'est le cas, répondez 'yes', sinon répondez 'no'.",
        # Spanish
        "Por favor, analice el siguiente post y determine si describe sentirse nervioso antes de una prueba de ácido nucleico: {}. Si es así, responda 'yes', de lo contrario, responda 'no'.",
        # Portuguese
        "Por favor, analise a postagem abaixo e determine se ela descreve sentir-se nervoso antes de um teste de ácido nucleico: {}. Se for o caso, responda 'yes', caso contrário, responda 'no'.",
        # Italian
        "Si prega di analizzare il seguente post e determinare se descrive sentirsi nervoso prima di un test dell'acido nucleico: {}. In tal caso, rispondi 'yes', altrimenti rispondi 'no'.",
        # Dutch
        "Analyseer alstublieft de volgende post en bepaal of er wordt vermeld dat men zich nerveus voelt voor een nucleïnezuurtest: {}. Als dat het geval is, antwoord dan met 'yes', anders met 'no'.",
        # Russian
        "Пожалуйста, проанализируйте следующий пост и определите, упоминается ли в нем, что человек чувствует себя нервным перед тестированием на нуклеиновую кислоту: {}. Если да, ответьте 'yes', в противном случае ответьте 'no'.",
        # Czech
        "Analyzujte prosím následující příspěvek a určete, zda zmiňuje, že se člověk před testováním na nukleovou kyselinu cítí nervózní: {}. Pokud ano, odpovězte 'yes', jinak odpovězte 'no'.",
        # Polish
        "Proszę przeanalizować poniższy post i ustalić, czy wspomina o odczuwaniu zdenerwowania przed testem na kwas nukleinowy: {}. Jeśli tak, odpowiedz 'yes', w przeciwnym razie odpowiedz 'no'.",
        # Arabic
        "يرجى تحليل المنشور أدناه وتحديد ما إذا كان يصف الشعور بالتوتر قبل اختبار الحمض النووي: {}. إذا كان الأمر كذلك، أجب 'yes'، وإلا أجب 'no'.",
        # Persian
        "لطفاً پست زیر را تحلیل کنید و تعیین کنید که آیا اشاره شده است که نسبت به احساس عصبی قبل از آزمایش اسید نوکلئیک وجود دارد: {}. اگر چنین است، با 'yes' پاسخ دهید، در غیر این صورت 'no' پاسخ دهید.",
        # Hebrew
        "אנא נתחו את הפוסט הבא וקבעו אם הוא מתאר תחושת עצבנות לפני בדיקת חומצת גרעין: {}. אם כן, השב 'yes', אחרת השב 'no'.",
        # Turkish
        "Lütfen aşağıdaki gönderiyi analiz edin ve nükleik asit testi öncesinde sinirli hissettiğini belirtiyorsa belirleyin: {}. Eğer öyleyse, 'yes' yanıtını verin, aksi takdirde 'no' yanıtını verin.",
        # Japanese
        "以下の投稿を分析し、核酸検査の前に緊張していることを述べているかどうかを判断してください: {}。 そうであれば「yes」と答え、それ以外の場合は「no」と答えてください。",
        # Korean
        "다음 게시물을 분석하고 핵산 검사를 받기 전 긴장감에 대해 설명하는지 여부를 확인하십시오: {}. 그렇다면 'yes'로 답하고, 그렇지 않다면 'no'로 답하십시오.",
        # Vietnamese
        "Vui lòng phân tích bài viết dưới đây và xác định xem nó có mô tả cảm giác lo lắng trước khi xét nghiệm axit nucleic hay không: {}. Nếu có, hãy trả lời 'yes', nếu không, hãy trả lời 'no'.",
        # Thai
        "โปรดวิเคราะห์โพสต์ด้านล่างนี้และระบุว่ามีการอธิบายความรู้สึกกังวลก่อนการทดสอบกรดนิวคลีอิกหรือไม่: {}. หากเป็นเช่นนั้น ให้ตอบว่า 'yes' มิฉะนั้นให้ตอบว่า 'no'",
        # Indonesian
        "Harap analisis postingan di bawah ini dan tentukan apakah disebutkan merasa gugup sebelum tes asam nukleat: {}. Jika demikian, balas dengan 'yes', jika tidak, balas dengan 'no'.",
        # Malay
        "Sila analisis siaran di bawah dan tentukan sama ada ia menyebutkan perasaan gugup sebelum ujian asid nukleik: {}. Jika ya, sila jawab 'yes', jika tidak, jawab 'no'.",
        # Lao
        "โปรดวิเคราะห์โพสต์ด้านล่างนี้และระบุว่ามีการอธิบายความรู้สึกกังวลก่อนการทดสอบกรดนิวคลีอิกหรือไม่: {}. หากเป็นเช่นนั้น ให้ตอบว่า 'yes' มิฉะนั้นให้ตอบว่า 'no'",
        # Burmese
        "ကျေးဇူးပြု၍ အောက်ပါစာမျက်နှာကိုသုံးသပ်ပြီး ကပ်ရောဂါအကြောင်းကြောင့် စိုးရိမ်ပူပန်မှုများ၊ ဒဏ်ခံနိုင်သောကြောက်ရွံ့မှုများကို ရှာဖွေပါ။ {}. အကယ်၍အကယ်၍တစ်ခုပြုလုပ်ရန် 'yes' ဖြင့်ပြန်ကြားပါ၊ မဟုတ်ပါက 'no' ဖြင့်ပြန်ကြားပါ။",
        # Cebuano
        "Pagtudlo sa mga plano o mga appointment alang sa nucleic acid testing. Nagpasabot kini nga ang post naghisgot bahin sa pagsugod sa pagsulay sa dili madugay, nga naghimo og appointment, nagplano sa paghimo og appointment, o pagpangandam alang sa pagsulay: {}. Palihug paghimo usa ka paghukom base sa sulod sa ibabaw. Kung mao, tubaga nga 'yes', kung dili, tubaga nga 'no'.",
        # Khmer
        "សូម​វិភាគ​អត្ថបទ​សេដ្ឋកិច្ច​ខាងក្រោម​នេះ ហើយ​សម្រេចថា តើ​វា​មាន​គម្រោង​នោះទេ ឬ​ការណាត់ជួប​សម្រាប់​ការសាកល្បង​អាស៊ីត​នុយក្លេអ៊ីក។ វា​មានន័យថា​ការ​ប្រកាស​បាន​រំឭក​ពី​ការ​ចាប់ផ្តើម​ធ្វើ​តេស្ត​នៅក្នុង​ពេលឆាប់​នេះ ការណាត់ជួប ការធ្វើគម្រោង​នេះ សម្រាប់ការណាត់ជួប ឬ​ការ​រៀបចំ​សម្រាប់​ការសាកល្បង៖ {}. ប្រសិនបើ​បាទ/ចាស សូមឆ្លើយថា 'yes' បើ​មិនមែន ឆ្លើយថា 'no'.",
        # Tagalog
        "Pakisuri ang sumusunod na post at tukuyin kung ito ay binabanggit ang mga plano o mga appointment para sa pagsusuri ng nucleic acid. Ipinapahiwatig nito na ang post ay binanggit na magsisimula ang pagsusuri, nag-set na ng appointment, plano na mag-set ng appointment, o naghahanda na para sa pagsusuri: {}. Kung gayon, sumagot ng 'yes', kung hindi, sumagot ng 'no'.",
        # Hindi
        "कृपया निम्नलिखल پوسٹ کا تجزیہ کریں اور یہ بتائیں کہ کیا اس میں نیوکلک ایسڈ ٹیسٹ کے بارے میں ہچکچاہٹ کا ذکر ہے: {}. اگر ہاں تو 'yes' جواب دیں؛ بصورت دیگر، 'no' جواب دیں۔"
    ],

    '12.2': [
        # Chinese
        "请分析以下微博内容，并判断其是否描述了核酸检测中未感到不适: {}。如果是，请回答'yes'，否则回答'no'。",
        # English
        "Please analyze the following post and determine if it describes feeling no discomfort during a nucleic acid test: {}. If so, respond 'yes'; otherwise, respond 'no'.",
        # German
        "Bitte analysieren Sie den folgenden Beitrag und bestimmen Sie, ob er beschreibt, dass man sich während eines Nukleinsäuretests nicht unwohl fühlt: {}. Wenn ja, antworten Sie mit 'yes', andernfalls mit 'no'.",
        # French
        "Veuillez analyser le post ci-dessous et déterminer s'il décrit une sensation de confort lors d'un test d'acide nucléique: {}. Si c'est le cas, répondez 'yes', sinon répondez 'no'.",
        # Spanish
        "Por favor, analice el siguiente post y determine si describe no sentir molestias durante una prueba de ácido nucleico: {}. Si es así, responda 'yes', de lo contrario, responda 'no'.",
        # Portuguese
        "Por favor, analise a postagem abaixo e determine se ela descreve não sentir desconforto durante um teste de ácido nucleico: {}. Se for o caso, responda 'yes', caso contrário, responda 'no'.",
        # Italian
        "Si prega di analizzare il seguente post e determinare se descrive di non provare disagio durante un test dell'acido nucleico: {}. In tal caso, rispondi 'yes', altrimenti rispondi 'no'.",
        # Dutch
        "Analyseer alstublieft de volgende post en bepaal of er wordt vermeld dat men zich tijdens een nucleïnezuurtest niet ongemakkelijk voelt: {}. Als dat het geval is, antwoord dan met 'yes', anders met 'no'.",
        # Russian
        "Пожалуйста, проанализируйте следующий пост и определите, упоминается ли в нем, что человек не чувствует дискомфорта во время тестирования на нуклеиновую кислоту: {}. Если да, ответьте 'yes', в противном случае ответьте 'no'.",
        # Czech
        "Analyzujte prosím následující příspěvek a určete, zda zmiňuje, že se člověk během testování na nukleovou kyselinu necítí nepohodlně: {}. Pokud ano, odpovězte 'yes', jinak odpovězte 'no'.",
        # Polish
        "Proszę przeanalizować poniższy post i ustalić, czy wspomina o braku dyskomfortu podczas testu na kwas nukleinowy: {}. Jeśli tak, odpowiedz 'yes', w przeciwnym razie odpowiedz 'no'.",
        # Arabic
        "يرجى تحليل المنشور أدناه وتحديد ما إذا كان يصف عدم الشعور بأي إزعاج أثناء اختبار الحمض النووي: {}. إذا كان الأمر كذلك، أجب 'yes'، وإلا أجب 'no'.",
        # Persian
        "لطفاً پست زیر را تحلیل کنید و تعیین کنید که آیا اشاره شده است که نسبت به عدم احساس ناراحتی در طول آزمایش اسید نوکلئیک وجود دارد: {}. اگر چنین است، با 'yes' پاسخ دهید، در غیر این صورت 'no' پاسخ دهید.",
        # Hebrew
        "אנא נתחו את הפוסט הבא וקבעו אם הוא מתאר תחושה של חוסר אי נוחות במהלך בדיקת חומצת גרעין: {}. אם כן, השב 'yes', אחרת השב 'no'.",
        # Turkish
        "Lütfen aşağıdaki gönderiyi analiz edin ve nükleik asit testi sırasında rahatsızlık hissetmediğini belirtiyorsa belirleyin: {}. Eğer öyleyse, 'yes' yanıtını verin, aksi takdirde 'no' yanıtını verin.",
        # Japanese
        "以下の投稿を分析し、核酸検査中に不快感を感じていないと述べているかどうかを判断してください: {}。 そうであれば「yes」と答え、それ以外の場合は「no」と答えてください。",
        # Korean
        "다음 게시물을 분석하고 핵산 검사 중 불편함을 느끼지 않았다고 설명하는지 여부를 확인하십시오: {}. 그렇다면 'yes'로 답하고, 그렇지 않다면 'no'로 답하십시오.",
        # Vietnamese
        "Vui lòng phân tích bài viết dưới đây và xác định xem nó có mô tả không cảm thấy khó chịu trong quá trình xét nghiệm axit nucleic hay không: {}. Nếu có, hãy trả lời 'yes', nếu không, hãy trả lời 'no'.",
        # Thai
        "โปรดวิเคราะห์โพสต์ด้านล่างนี้และระบุว่ามีการอธิบายถึงความรู้สึกไม่สบายในระหว่างการทดสอบกรดนิวคลีอิกหรือไม่: {}. หากเป็นเช่นนั้น ให้ตอบว่า 'yes' มิฉะนั้นให้ตอบว่า 'no'",
        # Indonesian
        "Harap analisis postingan di bawah ini dan tentukan apakah disebutkan tidak merasa tidak nyaman selama tes asam nukleat: {}. Jika demikian, balas dengan 'yes', jika tidak, balas dengan 'no'.",
        # Malay
        "Sila analisis siaran di bawah dan tentukan sama ada ia menyebutkan tidak berasa tidak selesa semasa ujian asid nukleik: {}. Jika ya, sila jawab 'yes', jika tidak, jawab 'no'.",
        # Lao
        "โปรดวิเคราะห์โพสต์ด้านล่างนี้และระบุว่ามีการอธิบายถึงความรู้สึกไม่สบายในระหว่างการทดสอบกรดนิวคลีอิกหรือไม่: {}. หากเป็นเช่นนั้น ให้ตอบว่า 'yes' มิฉะนั้นให้ตอบว่า 'no'",
        # Burmese
        "ကျေးဇူးပြု၍ အောက်ပါစာမျက်နှာကိုသုံးသပ်ပြီး ကပ်ရောဂါအကြောင်းကြောင့် စိုးရိမ်ပူပန်မှုများ၊ ဒဏ်ခံနိုင်သောကြောက်ရွံ့မှုများကို ရှာဖွေပါ။ {}. အကယ်၍အကယ်၍တစ်ခုပြုလုပ်ရန် 'yes' ဖြင့်ပြန်ကြားပါ၊ မဟုတ်ပါက 'no' ဖြင့်ပြန်ကြားပါ။",
        # Cebuano
        "Pagtudlo sa mga plano o mga appointment alang sa nucleic acid testing. Nagpasabot kini nga ang post naghisgot bahin sa pagsugod sa pagsulay sa dili madugay, nga naghimo og appointment, nagplano sa paghimo og appointment, o pagpangandam alang sa pagsulay: {}. Palihug paghimo usa ka paghukom base sa sulod sa ibabaw. Kung mao, tubaga nga 'yes', kung dili, tubaga nga 'no'.",
        # Khmer
        "សូម​វិភាគ​អត្ថបទ​សេដ្ឋកិច្ច​ខាងក្រោម​នេះ ហើយ​សម្រេចថា តើ​វា​មាន​គម្រោង​នោះទេ ឬ​ការណាត់ជួប​សម្រាប់​ការសាកល្បង​អាស៊ីត​នុយក្លេអ៊ីក។ វា​មានន័យថា​ការ​ប្រកាស​បាន​រំឭក​ពី​ការ​ចាប់ផ្តើម​ធ្វើ​តេស្ត​នៅក្នុង​ពេលឆាប់​នេះ ការណាត់ជួប ការធ្វើគម្រោង​នេះ សម្រាប់ការណាត់ជួប ឬ​ការ​រៀបចំ​សម្រាប់​ការសាកល្បង៖ {}. ប្រសិនបើ​បាទ/ចាស សូមឆ្លើយថា 'yes' បើ​មិនមែន ឆ្លើយថា 'no'.",
        # Tagalog
        "Pakisuri ang sumusunod na post at tukuyin kung ito ay binabanggit ang mga plano o mga appointment para sa pagsusuri ng nucleic acid. Ipinapahiwatig nito na ang post ay binanggit na magsisimula ang pagsusuri, nag-set na ng appointment, plano na mag-set ng appointment, o naghahanda na para sa pagsusuri: {}. Kung gayon, sumagot ng 'yes', kung hindi, sumagot ng 'no'.",
        # Hindi
        "कृपया निम्नلिखल پوسٹ کا تجزیہ کریں اور یہ بتائیں کہ کیا اس میں نیوکلک ایسڈ ٹیسٹ کے بارے میں ہچکچاہٹ کا ذکر ہے: {}. اگر ہاں تو 'yes' جواب دیں؛ بصورت دیگر، 'no' جواب دیں۔"
    ],

    '12.3': [
        # Chinese
        "请分析以下微博内容，并判断其是否描述了核酸检测时感到恶心或不适: {}。如果是，请回答'yes'，否则回答'no'。",
        # English
        "Please analyze the following post and determine if it describes feeling nauseous or uncomfortable during a nucleic acid test: {}. If so, respond 'yes'; otherwise, respond 'no'.",
        # German
        "Bitte analysieren Sie den folgenden Beitrag und bestimmen Sie, ob er beschreibt, dass man sich während eines Nukleinsäuretests übel oder unwohl fühlt: {}. Wenn ja, antworten Sie mit 'yes', andernfalls mit 'no'.",
        # French
        "Veuillez analyser le post ci-dessous et déterminer s'il décrit une sensation de nausée ou de malaise lors d'un test d'acide nucléique: {}. Si c'est le cas, répondez 'yes', sinon répondez 'no'.",
        # Spanish
        "Por favor, analice el siguiente post y determine si describe sentirse nauseabundo o incómodo durante una prueba de ácido nucleico: {}. Si es así, responda 'yes', de lo contrario, responda 'no'.",
        # Portuguese
        "Por favor, analise a postagem abaixo e determine se ela descreve sentir-se enjoado ou desconfortável durante um teste de ácido nucleico: {}. Se for o caso, responda 'yes', caso contrário, responda 'no'.",
        # Italian
        "Si prega di analizzare il seguente post e determinare se descrive di sentirsi nauseato o a disagio durante un test dell'acido nucleico: {}. In tal caso, rispondi 'yes', altrimenti rispondi 'no'.",
        # Dutch
        "Analyseer alstublieft de volgende post en bepaal of er wordt vermeld dat men zich misselijk of ongemakkelijk voelt tijdens een nucleïnezuurtest: {}. Als dat het geval is, antwoord dan met 'yes', anders met 'no'.",
        # Russian
        "Пожалуйста, проанализируйте следующий пост и определите, упоминается ли в нем, что человек чувствует тошноту или дискомфорт во время тестирования на нуклеиновую кислоту: {}. Если да, ответьте 'yes', в противном случае ответьте 'no'.",
        # Czech
        "Analyzujte prosím následující příspěvek a určete, zda zmiňuje, že se člověk během testování na nukleovou kyselinu cítí nevolno nebo nepohodlně: {}. Pokud ano, odpovězte 'yes', jinak odpovězte 'no'.",
        # Polish
        "Proszę przeanalizować poniższy post i ustalić, czy wspomina o odczuwaniu mdłości lub dyskomfortu podczas testu na kwas nukleinowy: {}. Jeśli tak, odpowiedz 'yes', w przeciwnym razie odpowiedz 'no'.",
        # Arabic
        "يرجى تحليل المنشور أدناه وتحديد ما إذا كان يصف الشعور بالغثيان أو عدم الراحة أثناء اختبار الحمض النووي: {}. إذا كان الأمر كذلك، أجب 'yes'، وإلا أجب 'no'.",
        # Persian
        "لطفاً پست زیر را تحلیل کنید و تعیین کنید که آیا اشاره شده است که نسبت به احساس تهوع یا ناراحتی در طول آزمایش اسید نوکلئیک وجود دارد: {}. اگر چنین است، با 'yes' پاسخ دهید، در غیر این صورت 'no' پاسخ دهید.",
        # Hebrew
        "אנא נתחו את הפוסט הבא וקבעו אם הוא מתאר תחושת בחילה או אי נוחות במהלך בדיקת חומצת גרעין: {}. אם כן, השב 'yes', אחרת השב 'no'.",
        # Turkish
        "Lütfen aşağıdaki gönderiyi analiz edin ve nükleik asit testi sırasında mide bulantısı veya rahatsızlık hissettiğini belirtiyorsa belirleyin: {}. Eğer öyleyse, 'yes' yanıtını verin, aksi takdirde 'no' yanıtını verin.",
        # Japanese
        "以下の投稿を分析し、核酸検査中に吐き気や不快感を感じていることを述べているかどうかを判断してください: {}。 そうであれば「yes」と答え、それ以外の場合は「no」と答えてください。",
        # Korean
        "다음 게시물을 분석하고 핵산 검사 중 메스꺼움이나 불편함을 느꼈다고 설명하는지 여부를 확인하십시오: {}. 그렇다면 'yes'로 답하고, 그렇지 않다면 'no'로 답하십시오.",
        # Vietnamese
        "Vui lòng phân tích bài viết dưới đây và xác định xem nó có mô tả cảm giác buồn nôn hoặc khó chịu trong quá trình xét nghiệm axit nucleic hay không: {}. Nếu có, hãy trả lời 'yes', nếu không, hãy trả lời 'no'.",
        # Thai
        "โปรดวิเคราะห์โพสต์ด้านล่างนี้และระบุว่ามีการอธิบายถึงความรู้สึกคลื่นไส้หรือไม่สบายระหว่างการทดสอบกรดนิวคลีอิกหรือไม่: {}. หากเป็นเช่นนั้น ให้ตอบว่า 'yes' มิฉะนั้นให้ตอบว่า 'no'",
        # Indonesian
        "Harap analisis postingan di bawah ini dan tentukan apakah disebutkan merasa mual atau tidak nyaman selama tes asam nukleat: {}. Jika demikian, balas dengan 'yes', jika tidak, balas dengan 'no'.",
        # Malay
        "Sila analisis siaran di bawah dan tentukan sama ada ia menyebutkan merasa mual atau tidak selesa semasa ujian asid nukleik: {}. Jika ya, sila jawab 'yes', jika tidak, jawab 'no'.",
        # Lao
        "โปรดวิเคราะห์โพสต์ด้านล่างนี้และระบุว่ามีการอธิบายถึงความรู้สึกคลื่นไส้หรือไม่สบายระหว่างการทดสอบกรดนิวคลีอิกหรือไม่: {}. หากเป็นเช่นนั้น ให้ตอบว่า 'yes' มิฉะนั้นให้ตอบว่า 'no'",
        # Burmese
        "ကျေးဇူးပြု၍ အောက်ပါစာမျက်နှာကိုသုံးသပ်ပြီး ကပ်ရောဂါအကြောင်းကြောင့် စိုးရိမ်ပူပန်မှုများ၊ ဒဏ်ခံနိုင်သောကြောက်ရွံ့မှုများကို ရှာဖွေပါ။ {}. အကယ်၍အကယ်၍တစ်ခုပြုလုပ်ရန် 'yes' ဖြင့်ပြန်ကြားပါ၊ မဟုတ်ပါက 'no' ဖြင့်ပြန်ကြားပါ။",
        # Cebuano
        "Pagtudlo sa mga plano o mga appointment alang sa nucleic acid testing. Nagpasabot kini nga ang post naghisgot bahin sa pagsugod sa pagsulay sa dili madugay, nga naghimo og appointment, nagplano sa paghimo og appointment, o pagpangandam alang sa pagsulay: {}. Palihug paghimo usa ka paghukom base sa sulod sa ibabaw. Kung mao, tubaga nga 'yes', kung dili, tubaga nga 'no'.",
        # Khmer
        "សូម​វិភាគ​អត្ថបទ​សេដ្ឋកិច្ច​ខាងក្រោម​នេះ ហើយ​សម្រេចថា តើ​វា​មាន​គម្រោង​នោះទេ ឬ​ការណាត់ជួប​សម្រាប់​ការសាកល្បង​អាស៊ីត​នុយក្លេអ៊ីក។ វា​មានន័យថា​ការ​ប្រកាស​បាន​រំឭក​ពី​ការ​ចាប់ផ្តើម​ធ្វើ​តេស្ត​នៅក្នុង​ពេលឆាប់​នេះ ការណាត់ជួប ការធ្វើគម្រោង​នេះ សម្រាប់ការណាត់ជួប ឬ​ការ​រៀបចំ​សម្រាប់​ការសាកល្បង៖ {}. ប្រសិនបើ​បាទ/ចាស សូមឆ្លើយថា 'yes' បើ​មិនមែន ឆ្លើយថា 'no'.",
        # Tagalog
        "Pakisuri ang sumusunod na post at tukuyin kung ito ay binabanggit ang mga plano o mga appointment para sa pagsusuri ng nucleic acid. Ipinapahiwatig nito na ang post ay binanggit na magsisimula ang pagsusuri, nag-set na ng appointment, plano na mag-set ng appointment, o naghahanda na para sa pagsusuri: {}. Kung gayon, sumagot ng 'yes', kung hindi, sumagot ng 'no'.",
        # Hindi
        "कृपया निम्नلिखل پوسٹ کا تجزیہ کریں اور یہ بتائیں کہ کیا اس میں نیوکلک ایسڈ ٹیسٹ کے بارے میں ہچکچاہٹ کا ذکر ہے: {}. اگر ہاں تو 'yes' جواب دیں؛ بصورت دیگر، 'no' جواب دیں۔"
    ],

    '13.1': [
        # Chinese
        "请分析以下微博内容，并判断其是否提到关于核酸检测的预约问题: {}。如果是，请回答'yes'，否则回答'no'。",
        # English
        "Please analyze the following post and determine if it mentions issues related to scheduling nucleic acid testing appointments: {}. If so, respond 'yes'; otherwise, respond 'no'.",
        # German
        "Bitte analysieren Sie den folgenden Beitrag und bestimmen Sie, ob er Probleme im Zusammenhang mit der Terminplanung für Nukleinsäuretests erwähnt: {}. Wenn ja, antworten Sie mit 'yes', andernfalls mit 'no'.",
        # French
        "Veuillez analyser le post ci-dessous et déterminer s'il mentionne des problèmes liés à la planification des rendez-vous pour les tests d'acide nucléique: {}. Si c'est le cas, répondez 'yes', sinon répondez 'no'.",
        # Spanish
        "Por favor, analice el siguiente post y determine si menciona problemas relacionados con la programación de citas para pruebas de ácido nucleico: {}. Si es así, responda 'yes', de lo contrario, responda 'no'.",
        # Portuguese
        "Por favor, analise a postagem abaixo e determine se ela menciona problemas relacionados ao agendamento de consultas para testes de ácido nucleico: {}. Se for o caso, responda 'yes', caso contrário, responda 'no'.",
        # Italian
        "Si prega di analizzare il seguente post e determinare se menziona problemi relativi alla programmazione di appuntamenti per test dell'acido nucleico: {}. In tal caso, rispondi 'yes', altrimenti rispondi 'no'.",
        # Dutch
        "Analyseer alstublieft de volgende post en bepaal of er wordt vermeld dat er problemen zijn met het plannen van afspraken voor nucleïnezuurtesten: {}. Als dat het geval is, antwoord dan met 'yes', anders met 'no'.",
        # Russian
        "Пожалуйста, проанализируйте следующий пост и определите, упоминаются ли в нем проблемы, связанные с назначением тестов на нуклеиновую кислоту: {}. Если да, ответьте 'yes', в противном случае ответьте 'no'.",
        # Czech
        "Analyzujte prosím následující příspěvek a určete, zda zmiňuje problémy související s plánováním testování na nukleovou kyselinu: {}. Pokud ano, odpovězte 'yes', jinak odpovězte 'no'.",
        # Polish
        "Proszę przeanalizować poniższy post i ustalić, czy wspomina o problemach związanych z planowaniem testów na kwas nukleinowy: {}. Jeśli tak, odpowiedz 'yes', w przeciwnym razie odpowiedz 'no'.",
        # Arabic
        "يرجى تحليل المنشور أدناه وتحديد ما إذا كان يذكر مشاكل تتعلق بجدولة مواعيد اختبار الحمض النووي: {}. إذا كان الأمر كذلك، أجب 'yes'، وإلا أجب 'no'.",
        # Persian
        "لطفاً پست زیر را تحلیل کنید و تعیین کنید که آیا اشاره شده است که نسبت به مسائل مربوط به برنامه‌ریزی قرار ملاقات‌های آزمایش اسید نوکلئیک وجود دارد: {}. اگر چنین است، با 'yes' پاسخ دهید، در غیر این صورت 'no' پاسخ دهید.",
        # Hebrew
        "אנא נתחו את הפוסט הבא וקבעו אם הוא מזכיר בעיות הקשורות לתזמון בדיקות חומצת גרעין: {}. אם כן, השב 'yes', אחרת השב 'no'.",
        # Turkish
        "Lütfen aşağıdaki gönderiyi analiz edin ve nükleik asit testi randevularının planlanmasıyla ilgili sorunlar olup olmadığını belirleyin: {}. Eğer öyleyse, 'yes' yanıtını verin, aksi takdirde 'no' yanıtını verin.",
        # Japanese
        "以下の投稿を分析し、核酸検査の予約スケジュールに関連する問題が記載されているかどうかを判断してください: {}。 そうであれば「yes」と答え、それ以外の場合は「no」と答えてください。",
        # Korean
        "다음 게시물을 분석하고 핵산 검사 예약 일정과 관련된 문제가 언급되어 있는지 확인하십시오: {}. 그렇다면 'yes'로 답하고, 그렇지 않다면 'no'로 답하십시오.",
        # Vietnamese
        "Vui lòng phân tích bài viết dưới đây và xác định xem nó có đề cập đến các vấn đề liên quan đến việc lên lịch hẹn xét nghiệm axit nucleic hay không: {}. Nếu có, hãy trả lời 'yes', nếu không, hãy trả lời 'no'.",
        # Thai
        "โปรดวิเคราะห์โพสต์ด้านล่างนี้และระบุว่ามีการกล่าวถึงปัญหาที่เกี่ยวข้องกับการจัดตารางนัดหมายสำหรับการทดสอบกรดนิวคลีอิกหรือไม่: {}. หากเป็นเช่นนั้น ให้ตอบว่า 'yes' มิฉะนั้นให้ตอบว่า 'no'",
        # Indonesian
        "Harap analisis postingan di bawah ini dan tentukan apakah disebutkan masalah terkait penjadwalan janji temu pengujian asam nukleat: {}. Jika demikian, balas dengan 'yes', jika tidak, balas dengan 'no'.",
        # Malay
        "Sila analisis siaran di bawah dan tentukan sama ada ia menyebutkan masalah berkaitan penjadualan janji temu ujian asid nukleik: {}. Jika ya, sila jawab 'yes', jika tidak, jawab 'no'.",
        # Lao
        "โปรดวิเคราะห์โพสต์ด้านล่างนี้และระบุว่ามีการกล่าวถึงปัญหาที่เกี่ยวข้องกับการจัดตารางนัดหมายสำหรับการทดสอบกรดนิวคลีอิกหรือไม่: {}. หากเป็นเช่นนั้น ให้ตอบว่า 'yes' มิฉะนั้นให้ตอบว่า 'no'",
        # Burmese
        "ကျေးဇူးပြု၍ အောက်ပါစာမျက်နှာကိုသုံးသပ်ပြီး ကပ်ရောဂါအကြောင်းကြောင့် စိုးရိမ်ပူပန်မှုများ၊ ဒဏ်ခံနိုင်သောကြောက်ရွံ့မှုများကို ရှာဖွေပါ။ {}. အကယ်၍အကယ်၍တစ်ခုပြုလုပ်ရန် 'yes' ဖြင့်ပြန်ကြားပါ၊ မဟုတ်ပါက 'no' ဖြင့်ပြန်ကြားပါ။",
        # Cebuano
        "Pagtudlo sa mga plano o mga appointment alang sa nucleic acid testing. Nagpasabot kini nga ang post naghisgot bahin sa pagsugod sa pagsulay sa dili madugay, nga naghimo og appointment, nagplano sa paghimo og appointment, o pagpangandam alang sa pagsulay: {}. Palihug paghimo usa ka paghukom base sa sulod sa ibabaw. Kung mao, tubaga nga 'yes', kung dili, tubaga nga 'no'.",
        # Khmer
        "សូម​វិភាគ​អត្ថបទ​សេដ្ឋកិច្ច​ខាងក្រោម​នេះ ហើយ​សម្រេចថា តើ​វា​មាន​គម្រោង​នោះទេ ឬ​ការណាត់ជួប​សម្រាប់​ការសាកល្បង​អាស៊ីត​នុយក្លេអ៊ីក។ វា​មានន័យថា​ការ​ប្រកាស​បាន​រំឭក​ពី​ការ​ចាប់ផ្តើម​ធ្វើ​តេស្ត​នៅក្នុង​ពេលឆាប់​នេះ ការណាត់ជួប ការធ្វើគម្រោង​នេះ សម្រាប់ការណាត់ជួប ឬ​ការ​រៀបចំ​សម្រាប់​ការសាកល្បង៖ {}. ប្រសិនបើ​បាទ/ចាស សូមឆ្លើយថា 'yes' បើ​មិនមែន ឆ្លើយថា 'no'.",
        # Tagalog
        "Pakisuri ang sumusunod na post at tukuyin kung ito ay binabanggit ang mga plano o mga appointment para sa pagsusuri ng nucleic acid. Ipinapahiwatig nito na ang post ay binanggit na magsisimula ang pagsusuri, nag-set na ng appointment, plano na mag-set ng appointment, o naghahanda na para sa pagsusuri: {}. Kung gayon, sumagot ng 'yes', kung hindi, sumagot ng 'no'.",
        # Hindi
        "कृपया निम्नلिखل پوسٹ کا تجزیہ کریں اور یہ بتائیں کہ کیا اس میں نیوکلک ایسڈ ٹیسٹ کے بارے میں ہچکچاہٹ کا ذکر ہے: {}. اگر ہاں تو 'yes' جواب دیں؛ بصورت دیگر، 'no' جواب دیں۔"
    ],

    '13.2': [
        # Chinese
        "请分析以下微博内容，并判断其是否提到核酸检测现场组织情况: {}。如果是，请回答'yes'，否则回答'no'。",
        # English
        "Please analyze the following post and determine if it mentions the organization at nucleic acid testing sites: {}. If so, respond 'yes'; otherwise, respond 'no'.",
        # German
        "Bitte analysieren Sie den folgenden Beitrag und bestimmen Sie, ob er die Organisation an den Nukleinsäureteststellen erwähnt: {}. Wenn ja, antworten Sie mit 'yes', andernfalls mit 'no'.",
        # French
        "Veuillez analyser le post ci-dessous et déterminer s'il mentionne l'organisation des sites de tests d'acide nucléique: {}. Si c'est le cas, répondez 'yes', sinon répondez 'no'.",
        # Spanish
        "Por favor, analice el siguiente post y determine si menciona la organización en los sitios de pruebas de ácido nucleico: {}. Si es así, responda 'yes', de lo contrario, responda 'no'.",
        # Portuguese
        "Por favor, analise a postagem abaixo e determine se ela menciona a organização nos locais de teste de ácido nucleico: {}. Se for o caso, responda 'yes', caso contrário, responda 'no'.",
        # Italian
        "Si prega di analizzare il seguente post e determinare se menziona l'organizzazione nei siti di test dell'acido nucleico: {}. In tal caso, rispondi 'yes', altrimenti rispondi 'no'.",
        # Dutch
        "Analyseer alstublieft de volgende post en bepaal of er wordt vermeld dat er sprake is van organisatie op de nucleïnezuurtestlocaties: {}. Als dat het geval is, antwoord dan met 'yes', anders met 'no'.",
        # Russian
        "Пожалуйста, проанализируйте следующий пост и определите, упоминается ли в нем организация на местах тестирования на нуклеиновую кислоту: {}. Если да, ответьте 'yes', в противном случае ответьте 'no'.",
        # Czech
        "Analyzujte prosím následující příspěvek a určete, zda zmiňuje organizaci na místech testování na nukleovou kyselinu: {}. Pokud ano, odpovězte 'yes', jinak odpovězte 'no'.",
        # Polish
        "Proszę przeanalizować poniższy post i ustalić, czy wspomina o organizacji na miejscach testów na kwas nukleinowy: {}. Jeśli tak, odpowiedz 'yes', w przeciwnym razie odpowiedz 'no'.",
        # Arabic
        "يرجى تحليل المنشور أدناه وتحديد ما إذا كان يذكر التنظيم في مواقع اختبار الحمض النووي: {}. إذا كان الأمر كذلك، أجب 'yes'، وإلا أجب 'no'.",
        # Persian
        "لطفاً پست زیر را تحلیل کنید و تعیین کنید که آیا اشاره شده است که نسبت به سازماندهی در مکان‌های آزمایش اسید نوکلئیک وجود دارد: {}. اگر چنین است، با 'yes' پاسخ دهید، در غیر این صورت 'no' پاسخ دهید.",
        # Hebrew
        "אנא נתחו את הפוסט הבא וקבעו אם הוא מזכיר את הארגון במקומות בדיקות חומצת גרעין: {}. אם כן, השב 'yes', אחרת השב 'no'.",
        # Turkish
        "Lütfen aşağıdaki gönderiyi analiz edin ve nükleik asit testi sitelerinde organizasyon olup olmadığını belirtiyorsa belirleyin: {}. Eğer öyleyse, 'yes' yanıtını verin, aksi takdirde 'no' yanıtını verin.",
        # Japanese
        "以下の投稿を分析し、核酸検査サイトの組織について言及しているかどうかを判断してください: {}。 そうであれば「yes」と答え、それ以外の場合は「no」と答えてください。",
        # Korean
        "다음 게시물을 분석하고 핵산 검사 현장에서의 조직에 대해 언급하는지 여부를 확인하십시오: {}. 그렇다면 'yes'로 답하고, 그렇지 않다면 'no'로 답하십시오.",
        # Vietnamese
        "Vui lòng phân tích bài viết dưới đây và xác định xem nó có đề cập đến việc tổ chức tại các điểm xét nghiệm axit nucleic hay không: {}. Nếu có, hãy trả lời 'yes', nếu không, hãy trả lời 'no'.",
        # Thai
        "โปรดวิเคราะห์โพสต์ด้านล่างนี้และระบุว่ามีการกล่าวถึงการจัดระเบียบที่ไซต์ทดสอบกรดนิวคลีอิกหรือไม่: {}. หากเป็นเช่นนั้น ให้ตอบว่า 'yes' มิฉะนั้นให้ตอบว่า 'no'",
        # Indonesian
        "Harap analisis postingan di bawah ini dan tentukan apakah disebutkan organisasi di situs pengujian asam nukleat: {}. Jika demikian, balas dengan 'yes', jika tidak, balas dengan 'no'.",
        # Malay
        "Sila analisis siaran di bawah dan tentukan sama ada ia menyebutkan organisasi di tapak ujian asid nukleik: {}. Jika ya, sila jawab 'yes', jika tidak, jawab 'no'.",
        # Lao
        "โปรดวิเคราะห์โพสต์ด้านล่างนี้และระบุว่ามีการกล่าวถึงการจัดระเบียบที่ไซต์ทดสอบกรดนิวคลีอิกหรือไม่: {}. หากเป็นเช่นนั้น ให้ตอบว่า 'yes' มิฉะนั้นให้ตอบว่า 'no'",
        # Burmese
        "ကျေးဇူးပြု၍ အောက်ပါစာမျက်နှာကိုသုံးသပ်ပြီး ကပ်ရောဂါအကြောင်းကြောင့် စိုးရိမ်ပူပန်မှုများ၊ ဒဏ်ခံနိုင်သောကြောက်ရွံ့မှုများကို ရှာဖွေပါ။ {}. အကယ်၍အကယ်၍တစ်ခုပြုလုပ်ရန် 'yes' ဖြင့်ပြန်ကြားပါ၊ မဟုတ်ပါက 'no' ဖြင့်ပြန်ကြားပါ။",
        # Cebuano
        "Pagtudlo sa mga plano o mga appointment alang sa nucleic acid testing. Nagpasabot kini nga ang post naghisgot bahin sa pagsugod sa pagsulay sa dili madugay, nga naghimo og appointment, nagplano sa paghimo og appointment, o pagpangandam alang sa pagsulay: {}. Palihug paghimo usa ka paghukom base sa sulod sa ibabaw. Kung mao, tubaga nga 'yes', kung dili, tubaga nga 'no'.",
        # Khmer
        "សូម​វិភាគ​អត្ថបទ​សេដ្ឋកិច្ច​ខាងក្រោម​នេះ ហើយ​សម្រេចថា តើ​វា​មាន​គម្រោង​នោះទេ ឬ​ការណាត់ជួប​សម្រាប់​ការសាកល្បង​អាស៊ីត​នុយក្លេអ៊ីក។ វា​មានន័យថា​ការ​ប្រកាស​បាន​រំឭក​ពី​ការ​ចាប់ផ្តើម​ធ្វើ​តេស្ត​នៅក្នុង​ពេលឆាប់​នេះ ការណាត់ជួប ការធ្វើគម្រោង​នេះ សម្រាប់ការណាត់ជួប ឬ​ការ​រៀបចំ​សម្រាប់​ការសាកល្បង៖ {}. ប្រសិនបើ​បាទ/ចាស សូមឆ្លើយថា 'yes' បើ​មិនមែន ឆ្លើយថា 'no'.",
        # Tagalog
        "Pakisuri ang sumusunod na post at tukuyin kung ito ay binabanggit ang mga plano o mga appointment para sa pagsusuri ng nucleic acid. Ipinapahiwatig nito na ang post ay binanggit na magsisimula ang pagsusuri, nag-set na ng appointment, plano na mag-set ng appointment, o naghahanda na para sa pagsusuri: {}. Kung gayon, sumagot ng 'yes', kung hindi, sumagot ng 'no'.",
        # Hindi
        "कृपया निम्नलिखल پوسٹ کا تجزیہ کریں اور یہ بتائیں کہ کیا اس میں نیوکلک ایسڈ ٹیسٹ کے بارے میں ہچکچاہٹ کا ذکر ہے: {}. اگر ہاں تو 'yes' جواب دیں؛ بصورت دیگر، 'no' جواب دیں۔"
    ],

    '13.3': [
        # Chinese
        "请分析以下微博内容，并判断其是否提到核酸检测出结果速度: {}。如果是，请回答'yes'，否则回答'no'。",
        # English
        "Please analyze the following post and determine if it mentions the speed of nucleic acid test results: {}. If so, respond 'yes'; otherwise, respond 'no'.",
        # German
        "Bitte analysieren Sie den folgenden Beitrag und bestimmen Sie, ob er die Geschwindigkeit der Ergebnisse von Nukleinsäuretests erwähnt: {}. Wenn ja, antworten Sie mit 'yes', andernfalls mit 'no'.",
        # French
        "Veuillez analyser le post ci-dessous et déterminer s'il mentionne la rapidité des résultats des tests d'acide nucléique: {}. Si c'est le cas, répondez 'yes', sinon répondez 'no'.",
        # Spanish
        "Por favor, analice el siguiente post y determine si menciona la velocidad de los resultados de las pruebas de ácido nucleico: {}. Si es así, responda 'yes', de lo contrario, responda 'no'.",
        # Portuguese
        "Por favor, analise a postagem abaixo e determine se ela menciona a velocidade dos resultados dos testes de ácido nucleico: {}. Se for o caso, responda 'yes', caso contrário, responda 'no'.",
        # Italian
        "Si prega di analizzare il seguente post e determinare se menziona la velocità dei risultati dei test dell'acido nucleico: {}. In tal caso, rispondi 'yes', altrimenti rispondi 'no'.",
        # Dutch
        "Analyseer alstublieft de volgende post en bepaal of er wordt vermeld dat er sprake is van de snelheid van nucleïnezuurtestresultaten: {}. Als dat het geval is, antwoord dan met 'yes', anders met 'no'.",
        # Russian
        "Пожалуйста, проанализируйте следующий пост и определите, упоминается ли в нем скорость получения результатов теста на нуклеиновую кислоту: {}. Если да, ответьте 'yes', в противном случае ответьте 'no'.",
        # Czech
        "Analyzujte prosím následující příspěvek a určete, zda zmiňuje rychlost výsledků testování na nukleovou kyselinu: {}. Pokud ano, odpovězte 'yes', jinak odpovězte 'no'.",
        # Polish
        "Proszę przeanalizować poniższy post i ustalić, czy wspomina o szybkości uzyskiwania wyników testów na kwas nukleinowy: {}. Jeśli tak, odpowiedz 'yes', w przeciwnym razie odpowiedz 'no'.",
        # Arabic
        "يرجى تحليل المنشور أدناه وتحديد ما إذا كان يذكر سرعة نتائج اختبار الحمض النووي: {}. إذا كان الأمر كذلك، أجب 'yes'، وإلا أجب 'no'.",
        # Persian
        "لطفاً پست زیر را تحلیل کنید و تعیین کنید که آیا اشاره شده است که نسبت به سرعت نتایج آزمایش اسید نوکلئیک وجود دارد: {}. اگر چنین است، با 'yes' پاسخ دهید، در غیر این صورت 'no' پاسخ دهید.",
        # Hebrew
        "אנא נתחו את הפוסט הבא וקבעו אם הוא מזכיר את מהירות תוצאות בדיקות חומצת גרעין: {}. אם כן, השב 'yes', אחרת השב 'no'.",
        # Turkish
        "Lütfen aşağıdaki gönderiyi analiz edin ve nükleik asit testi sonuçlarının hızını belirtiyorsa belirleyin: {}. Eğer öyleyse, 'yes' yanıtını verin, aksi takdirde 'no' yanıtını verin.",
        # Japanese
        "以下の投稿を分析し、核酸検査結果の速度について言及しているかどうかを判断してください: {}。 そうであれば「yes」と答え、それ以外の場合は「no」と答えてください。",
        # Korean
        "다음 게시물을 분석하고 핵산 검사 결과의 속도에 대해 언급하는지 여부를 확인하십시오: {}. 그렇다면 'yes'로 답하고, 그렇지 않다면 'no'로 답하십시오.",
        # Vietnamese
        "Vui lòng phân tích bài viết dưới đây và xác định xem nó có đề cập đến tốc độ của kết quả xét nghiệm axit nucleic hay không: {}. Nếu có, hãy trả lời 'yes', nếu không, hãy trả lời 'no'.",
        # Thai
        "โปรดวิเคราะห์โพสต์ด้านล่างนี้และระบุว่ามีการกล่าวถึงความเร็วของผลการทดสอบกรดนิวคลีอิกหรือไม่: {}. หากเป็นเช่นนั้น ให้ตอบว่า 'yes' มิฉะนั้นให้ตอบว่า 'no'",
        # Indonesian
        "Harap analisis postingan di bawah ini dan tentukan apakah disebutkan kecepatan hasil tes asam nukleat: {}. Jika demikian, balas dengan 'yes', jika tidak, balas dengan 'no'.",
        # Malay
        "Sila analisis siaran di bawah dan tentukan sama ada ia menyebutkan kelajuan keputusan ujian asid nukleik: {}. Jika ya, sila jawab 'yes', jika tidak, jawab 'no'.",
        # Lao
        "โปรดวิเคราะห์โพสต์ด้านล่างนี้และระบุว่ามีการกล่าวถึงความเร็วของผลการทดสอบกรดนิวคลีอิกหรือไม่: {}. หากเป็นเช่นนั้น ให้ตอบว่า 'yes' มิฉะนั้นให้ตอบว่า 'no'",
        # Burmese
        "ကျေးဇူးပြု၍ အောက်ပါစာမျက်နှာကိုသုံးသပ်ပြီး ကပ်ရောဂါအကြောင်းကြောင့် စိုးရိမ်ပူပန်မှုများ၊ ဒဏ်ခံနိုင်သောကြောက်ရွံ့မှုများကို ရှာဖွေပါ။ {}. အကယ်၍အကယ်၍တစ်ခုပြုလုပ်ရန် 'yes' ဖြင့်ပြန်ကြားပါ၊ မဟုတ်ပါက 'no' ဖြင့်ပြန်ကြားပါ။",
        # Cebuano
        "Pagtudlo sa mga plano o mga appointment alang sa nucleic acid testing. Nagpasabot kini nga ang post naghisgot bahin sa pagsugod sa pagsulay sa dili madugay, nga naghimo og appointment, nagplano sa paghimo og appointment, o pagpangandam alang sa pagsulay: {}. Palihug paghimo usa ka paghukom base sa sulod sa ibabaw. Kung mao, tubaga nga 'yes', kung dili, tubaga nga 'no'.",
        # Khmer
        "សូម​វិភាគ​អត្ថបទ​សេដ្ឋកិច្ច​ខាងក្រោម​នេះ ហើយ​សម្រេចថា តើ​វា​មាន​គម្រោង​នោះទេ ឬ​ការណាត់ជួប​សម្រាប់​ការសាកល្បង​អាស៊ីត​នុយក្លេអ៊ីក។ វា​មានន័យថា​ការ​ប្រកាស​បាន​រំឭក​ពី​ការ​ចាប់ផ្តើម​ធ្វើ​តេស្ត​នៅក្នុង​ពេលឆាប់​នេះ ការណាត់ជួប ការធ្វើគម្រោង​នេះ សម្រាប់ការណាត់ជួប ឬ​ការ​រៀបចំ​សម្រាប់​ការសាកល្បង៖ {}. ប្រសិនបើ​បាទ/ចាស សូមឆ្លើយថា 'yes' បើ​មិនមែន ឆ្លើយថា 'no'.",
        # Tagalog
        "Pakisuri ang sumusunod na post at tukuyin kung ito ay binabanggit ang mga plano o mga appointment para sa pagsusuri ng nucleic acid. Ipinapahiwatig nito na ang post ay binanggit na magsisimula ang pagsusuri, nag-set na ng appointment, plano na mag-set ng appointment, o naghahanda na para sa pagsusuri: {}. Kung gayon, sumagot ng 'yes', kung hindi, sumagot ng 'no'.",
        # Hindi
        "कृपया निम्नलिखل پوسٹ کا تجزیہ کریں اور یہ بتائیں کہ کیا اس میں نیوکلک ایسڈ ٹیسٹ کے بارے میں ہچکچاہٹ کا ذکر ہے: {}. اگر ہاں تو 'yes' جواب دیں؛ بصورت دیگر، 'no' جواب دیں۔"
    ],

    '13.4': [
        # Chinese
        "请分析以下微博内容，并判断其是否提到其他与核酸检测相关的障碍和问题: {}。如果是，请回答'yes'，否则回答'no'。",
        # English
        "Please analyze the following post and determine if it mentions other obstacles and issues related to nucleic acid testing: {}. If so, respond 'yes'; otherwise, respond 'no'.",
        # German
        "Bitte analysieren Sie den folgenden Beitrag und bestimmen Sie, ob er andere Hindernisse und Probleme im Zusammenhang mit Nukleinsäuretests erwähnt: {}. Wenn ja, antworten Sie mit 'yes', andernfalls mit 'no'.",
        # French
        "Veuillez analyser le post ci-dessous et déterminer s'il mentionne d'autres obstacles et problèmes liés aux tests d'acide nucléique: {}. Si c'est le cas, répondez 'yes', sinon répondez 'no'.",
        # Spanish
        "Por favor, analice el siguiente post y determine si menciona otros obstáculos y problemas relacionados con las pruebas de ácido nucleico: {}. Si es así, responda 'yes', de lo contrario, responda 'no'.",
        # Portuguese
        "Por favor, analise a postagem abaixo e determine se ela menciona outros obstáculos e problemas relacionados ao teste de ácido nucleico: {}. Se for o caso, responda 'yes', caso contrário, responda 'no'.",
        # Italian
        "Si prega di analizzare il seguente post e determinare se menziona altri ostacoli e problemi relativi ai test dell'acido nucleico: {}. In tal caso, rispondi 'yes', altrimenti rispondi 'no'.",
        # Dutch
        "Analyseer alstublieft de volgende post en bepaal of er wordt vermeld dat er andere obstakels en problemen zijn met betrekking tot nucleïnezuurtesten: {}. Als dat het geval is, antwoord dan met 'yes', anders met 'no'.",
        # Russian
        "Пожалуйста, проанализируйте следующий пост и определите, упоминаются ли в нем другие препятствия и проблемы, связанные с тестированием на нуклеиновую кислоту: {}. Если да, ответьте 'yes', в противном случае ответьте 'no'.",
        # Czech
        "Analyzujte prosím následující příspěvek a určete, zda zmiňuje další překážky a problémy související s testováním na nukleovou kyselinu: {}. Pokud ano, odpovězte 'yes', jinak odpovězte 'no'.",
        # Polish
        "Proszę przeanalizować poniższy post i ustalić, czy wspomina o innych przeszkodach i problemach związanych z testami na kwas nukleinowy: {}. Jeśli tak, odpowiedz 'yes', w przeciwnym razie odpowiedz 'no'.",
        # Arabic
        "يرجى تحليل المنشور أدناه وتحديد ما إذا كان يذكر عقبات وقضايا أخرى تتعلق باختبارات الحمض النووي: {}. إذا كان الأمر كذلك، أجب 'yes'، وإلا أجب 'no'.",
        # Persian
        "لطفاً پست زیر را تحلیل کنید و تعیین کنید که آیا اشاره شده است که نسبت به موانع و مشکلات دیگر مرتبط با آزمایش اسید نوکلئیک وجود دارد: {}. اگر چنین است، با 'yes' پاسخ دهید، در غیر این صورت 'no' پاسخ دهید.",
        # Hebrew
        "אנא נתחו את הפוסט הבא וקבעו אם הוא מזכיר מכשולים ובעיות אחרות הקשורות לבדיקות חומצת גרעין: {}. אם כן, השב 'yes', אחרת השב 'no'.",
        # Turkish
        "Lütfen aşağıdaki gönderiyi analiz edin ve nükleik asit testi ile ilgili başka engeller ve sorunlar olup olmadığını belirtip belirtmediğini belirleyin: {}. Eğer öyleyse, 'yes' yanıtını verin, aksi takdirde 'no' yanıtını verin.",
        # Japanese
        "以下の投稿を分析し、核酸検査に関連する他の障害や問題について言及しているかどうかを判断してください: {}。 そうであれば「yes」と答え、それ以外の場合は「no」と答えてください。",
        # Korean
        "다음 게시물을 분석하고 핵산 검사를 비롯한 다른 장애물이나 문제가 언급되어 있는지 확인하십시오: {}. 그렇다면 'yes'로 답하고, 그렇지 않다면 'no'로 답하십시오.",
        # Vietnamese
        "Vui lòng phân tích bài viết dưới đây và xác định xem nó có đề cập đến các trở ngại và vấn đề khác liên quan đến việc xét nghiệm axit nucleic hay không: {}. Nếu có, hãy trả lời 'yes', nếu không, hãy trả lời 'no'.",
        # Thai
        "โปรดวิเคราะห์โพสต์ด้านล่างนี้และระบุว่ามีการกล่าวถึงอุปสรรคและปัญหาอื่น ๆ ที่เกี่ยวข้องกับการทดสอบกรดนิวคลีอิกหรือไม่: {}. หากเป็นเช่นนั้น ให้ตอบว่า 'yes' มิฉะนั้นให้ตอบว่า 'no'",
        # Indonesian
        "Harap analisis postingan di bawah ini dan tentukan apakah disebutkan hambatan dan masalah lain terkait pengujian asam nukleat: {}. Jika demikian, balas dengan 'yes', jika tidak, balas dengan 'no'.",
        # Malay
        "Sila analisis siaran di bawah dan tentukan sama ada ia menyebutkan halangan dan masalah lain yang berkaitan dengan ujian asid nukleik: {}. Jika ya, sila jawab 'yes', jika tidak, jawab 'no'.",
        # Lao
        "โปรดวิเคราะห์โพสต์ด้านล่างนี้และระบุว่ามีการกล่าวถึงอุปสรรคและปัญหาอื่น ๆ ที่เกี่ยวข้องกับการทดสอบกรดนิวคลีอิกหรือไม่: {}. หากเป็นเช่นนั้น ให้ตอบว่า 'yes' มิฉะนั้นให้ตอบว่า 'no'",
        # Burmese
        "ကျေးဇူးပြု၍ အောက်ပါစာမျက်နှာကိုသုံးသပ်ပြီး ကပ်ရောဂါအကြောင်းကြောင့် စိုးရိမ်ပူပန်မှုများ၊ ဒဏ်ခံနိုင်သောကြောက်ရွံ့မှုများကို ရှာဖွေပါ။ {}. အကယ်၍အကယ်၍တစ်ခုပြုလုပ်ရန် 'yes' ဖြင့်ပြန်ကြားပါ၊ မဟုတ်ပါက 'no' ဖြင့်ပြန်ကြားပါ။",
        # Cebuano
        "Pagtudlo sa mga plano o mga appointment alang sa nucleic acid testing. Nagpasabot kini nga ang post naghisgot bahin sa pagsugod sa pagsulay sa dili madugay, nga naghimo og appointment, nagplano sa paghimo og appointment, o pagpangandam alang sa pagsulay: {}. Palihug paghimo usa ka paghukom base sa sulod sa ibabaw. Kung mao, tubaga nga 'yes', kung dili, tubaga nga 'no'.",
        # Khmer
        "សូម​វិភាគ​អត្ថបទ​សេដ្ឋកិច្ច​ខាងក្រោម​នេះ ហើយ​សម្រេចថា តើ​វា​មាន​គម្រោង​នោះទេ ឬ​ការណាត់ជួប​សម្រាប់​ការសាកល្បង​អាស៊ីត​នុយក្លេអ៊ីក។ វា​មានន័យថា​ការ​ប្រកាស​បាន​រំឭក​ពី​ការ​ចាប់ផ្តើម​ធ្វើ​តេស្ត​នៅក្នុង​ពេលឆាប់​នេះ ការណាត់ជួប ការធ្វើគម្រោង​នេះ សម្រាប់ការណាត់ជួប ឬ​ការ​រៀបចំ​សម្រាប់​ការសាកល្បង៖ {}. ប្រសិនបើ​បាទ/ចាស សូមឆ្លើយថា 'yes' បើ​មិនមែន ឆ្លើយថា 'no'.",
        # Tagalog
        "Pakisuri ang sumusunod na post at tukuyin kung ito ay binabanggit ang mga plano o mga appointment para sa pagsusuri ng nucleic acid. Ipinapahiwatig nito na ang post ay binanggit na magsisimula ang pagsusuri, nag-set na ng appointment, plano na mag-set ng appointment, o naghahanda na para sa pagsusuri: {}. Kung gayon, sumagot ng 'yes', kung hindi, sumagot ng 'no'.",
        # Hindi
        "कृपया निम्नलिखल پوسٹ کا تجزیہ کریں اور یہ بتائیں کہ کیا اس میں نیوکلک ایسڈ ٹیسٹ کے بارے میں ہچکچاہٹ کا ذکر ہے: {}. اگر ہاں تو 'yes' جواب دیں؛ بصورت دیگر، 'no' جواب دیں۔"
    ],

    '14': [
        # Chinese
        "请分析以下微博内容，并判断其是否提到致敬工作人员: {}。如果是，请回答'yes'，否则回答'no'。",
        # English
        "Please analyze the following post and determine if it mentions paying tribute to staff members: {}. If so, respond 'yes'; otherwise, respond 'no'.",
        # German
        "Bitte analysieren Sie den folgenden Beitrag und bestimmen Sie, ob er eine Würdigung der Mitarbeiter erwähnt: {}. Wenn ja, antworten Sie mit 'yes', andernfalls mit 'no'.",
        # French
        "Veuillez analyser le post ci-dessous et déterminer s'il mentionne un hommage aux membres du personnel: {}. Si c'est le cas, répondez 'yes', sinon répondez 'no'.",
        # Spanish
        "Por favor, analice el siguiente post y determine si menciona un homenaje al personal: {}. Si es así, responda 'yes', de lo contrario, responda 'no'.",
        # Portuguese
        "Por favor, analise a postagem abaixo e determine se ela menciona uma homenagem aos membros da equipe: {}. Se for o caso, responda 'yes', caso contrário, responda 'no'.",
        # Italian
        "Si prega di analizzare il seguente post e determinare se menziona un omaggio ai membri del personale: {}. In tal caso, rispondi 'yes', altrimenti rispondi 'no'.",
        # Dutch
        "Analyseer alstublieft de volgende post en bepaal of er wordt vermeld dat er een eerbetoon wordt gebracht aan medewerkers: {}. Als dat het geval is, antwoord dan met 'yes', anders met 'no'.",
        # Russian
        "Пожалуйста, проанализируйте следующий пост и определите, упоминается ли в нем отдача должного членам персонала: {}. Если да, ответьте 'yes', в противном случае ответьте 'no'.",
        # Czech
        "Analyzujte prosím následující příspěvek a určete, zda zmiňuje poctu pracovníkům: {}. Pokud ano, odpovězte 'yes', jinak odpovězte 'no'.",
        # Polish
        "Proszę przeanalizować poniższy post i ustalić, czy wspomina o oddaniu hołdu członkom personelu: {}. Jeśli tak, odpowiedz 'yes', w przeciwnym razie odpowiedz 'no'.",
        # Arabic
        "يرجى تحليل المنشور أدناه وتحديد ما إذا كان يذكر تكريم الموظفين: {}. إذا كان الأمر كذلك، أجب 'yes'، وإلا أجب 'no'.",
        # Persian
        "لطفاً پست زیر را تحلیل کنید و تعیین کنید که آیا اشاره شده است که نسبت به ادای احترام به اعضای کارکنان وجود دارد: {}. اگر چنین است، با 'yes' پاسخ دهید، در غیر این صورت 'no' پاسخ دهید.",
        # Hebrew
        "אנא נתחו את הפוסט הבא וקבעו אם הוא מזכיר מתן כבוד לצוות: {}. אם כן, השב 'yes', אחרת השב 'no'.",
        # Turkish
        "Lütfen aşağıdaki gönderiyi analiz edin ve personele saygı gösterilip gösterilmediğini belirtip belirtmediğini belirleyin: {}. Eğer öyleyse, 'yes' yanıtını verin, aksi takdirde 'no' yanıtını verin.",
        # Japanese
        "以下の投稿を分析し、スタッフへの敬意が言及されているかどうかを判断してください: {}。 そうであれば「yes」と答え、それ以外の場合は「no」と答えてください。",
        # Korean
        "다음 게시물을 분석하고 직원들에게 경의를 표하는 내용이 있는지 여부를 확인하십시오: {}. 그렇다면 'yes'로 답하고, 그렇지 않다면 'no'로 답하십시오.",
        # Vietnamese
        "Vui lòng phân tích bài viết dưới đây và xác định xem nó có đề cập đến việc tỏ lòng kính trọng đối với các nhân viên hay không: {}. Nếu có, hãy trả lời 'yes', nếu không, hãy trả lời 'no'.",
        # Thai
        "โปรดวิเคราะห์โพสต์ด้านล่างนี้และระบุว่ามีการกล่าวถึงการแสดงความเคารพต่อเจ้าหน้าที่หรือไม่: {}. หากเป็นเช่นนั้น ให้ตอบว่า 'yes' มิฉะนั้นให้ตอบว่า 'no'",
        # Indonesian
        "Harap analisis postingan di bawah ini dan tentukan apakah disebutkan memberikan penghormatan kepada anggota staf: {}. Jika demikian, balas dengan 'yes', jika tidak, balas dengan 'no'.",
        # Malay
        "Sila analisis siaran di bawah dan tentukan sama ada ia menyebutkan memberikan penghormatan kepada kakitangan: {}. Jika ya, sila jawab 'yes', jika tidak, jawab 'no'.",
        # Lao
        "โปรดวิเคราะห์โพสต์ด้านล่างนี้และระบุว่ามีการกล่าวถึงการแสดงความเคารพต่อเจ้าหน้าที่หรือไม่: {}. หากเป็นเช่นนั้น ให้ตอบว่า 'yes' มิฉะนั้นให้ตอบว่า 'no'",
        # Burmese
        "ကျေးဇူးပြု၍ အောက်ပါစာမျက်နှာကိုသုံးသပ်ပြီး ကပ်ရောဂါအကြောင်းကြောင့် စိုးရိမ်ပူပန်မှုများ၊ ဒဏ်ခံနိုင်သောကြောက်ရွံ့မှုများကို ရှာဖွေပါ။ {}. အကယ်၍အကယ်၍တစ်ခုပြုလုပ်ရန် 'yes' ဖြင့်ပြန်ကြားပါ၊ မဟုတ်ပါက 'no' ဖြင့်ပြန်ကြားပါ။",
        # Cebuano
        "Pagtudlo sa mga plano o mga appointment alang sa nucleic acid testing. Nagpasabot kini nga ang post naghisgot bahin sa pagsugod sa pagsulay sa dili madugay, nga naghimo og appointment, nagplano sa paghimo og appointment, o pagpangandam alang sa pagsulay: {}. Palihug paghimo usa ka paghukom base sa sulod sa ibabaw. Kung mao, tubaga nga 'yes', kung dili, tubaga nga 'no'.",
        # Khmer
        "សូម​វិភាគ​អត្ថបទ​សេដ្ឋកិច្ច​ខាងក្រោម​នេះ ហើយ​សម្រេចថា តើ​វា​មាន​គម្រោង​នោះទេ ឬ​ការណាត់ជួប​សម្រាប់​ការសាកល្បង​អាស៊ីត​នុយក្លេអ៊ីក។ វា​មានន័យថា​ការ​ប្រកាស​បាន​រំឭក​ពី​ការ​ចាប់ផ្តើម​ធ្វើ​តេស្ត​នៅក្នុង​ពេលឆាប់​នេះ ការណាត់ជួប ការធ្វើគម្រោង​នេះ សម្រាប់ការណាត់ជួប ឬ​ការ​រៀបចំ​សម្រាប់​ការសាកល្បង៖ {}. ប្រសិនបើ​បាទ/ចាស សូមឆ្លើយថា 'yes' បើ​មិនមែន ឆ្លើយថា 'no'.",
        # Tagalog
        "Pakisuri ang sumusunod na post at tukuyin kung ito ay binabanggit ang mga plano o mga appointment para sa pagsusuri ng nucleic acid. Ipinapahiwatig nito na ang post ay binanggit na magsisimula ang pagsusuri, nag-set na ng appointment, plano na mag-set ng appointment, o naghahanda na para sa pagsusuri: {}. Kung gayon, sumagot ng 'yes', kung hindi, sumagot ng 'no'.",
        # Hindi
        "कृपया निम्नलिखल پوسٹ کا تجزیہ کریں اور یہ بتائیں کہ کیا اس میں نیوکلک ایسڈ ٹیسٹ کے بارے میں ہچکچاہٹ کا ذکر ہے: {}. اگر ہاں तो 'yes' جواب دیں؛ بصورت دیگر، 'no' جواب دیں۔"
    ],

    '15': [
        # Chinese
        "请分析以下微博内容，并判断其是否提到核酸检测价格: {}。如果是，请回答'yes'，否则回答'no'。",
        # English
        "Please analyze the following post and determine if it mentions the cost of nucleic acid testing: {}. If so, respond 'yes'; otherwise, respond 'no'.",
        # German
        "Bitte analysieren Sie den folgenden Beitrag und bestimmen Sie, ob er die Kosten für Nukleinsäuretests erwähnt: {}. Wenn ja, antworten Sie mit 'yes', andernfalls mit 'no'.",
        # French
        "Veuillez analyser le post ci-dessous et déterminer s'il mentionne le coût des tests d'acide nucléique: {}. Si c'est le cas, répondez 'yes', sinon répondez 'no'.",
        # Spanish
        "Por favor, analice el siguiente post y determine si menciona el costo de las pruebas de ácido nucleico: {}. Si es así, responda 'yes', de lo contrario, responda 'no'.",
        # Portuguese
        "Por favor, analise a postagem abaixo e determine se ela menciona o custo dos testes de ácido nucleico: {}. Se for o caso, responda 'yes', caso contrário, responda 'no'.",
        # Italian
        "Si prega di analizzare il seguente post e determinare se menziona il costo dei test dell'acido nucleico: {}. In tal caso, rispondi 'yes', altrimenti rispondi 'no'.",
        # Dutch
        "Analyseer alstublieft de volgende post en bepaal of er wordt vermeld dat er kosten zijn voor nucleïnezuurtesten: {}. Als dat het geval is, antwoord dan met 'yes', anders met 'no'.",
        # Russian
        "Пожалуйста, проанализируйте следующий пост и определите, упоминается ли в нем стоимость тестов на нуклеиновую кислоту: {}. Если да, ответьте 'yes', в противном случае ответьте 'no'.",
        # Czech
        "Analyzujte prosím následující příspěvek a určete, zda zmiňuje náklady na testování na nukleovou kyselinu: {}. Pokud ano, odpovězte 'yes', jinak odpovězte 'no'.",
        # Polish
        "Proszę przeanalizować poniższy post i ustalić, czy wspomina o kosztach testów na kwas nukleinowy: {}. Jeśli tak, odpowiedz 'yes', w przeciwnym razie odpowiedz 'no'.",
        # Arabic
        "يرجى تحليل المنشور أدناه وتحديد ما إذا كان يذكر تكلفة اختبارات الحمض النووي: {}. إذا كان الأمر كذلك، أجب 'yes'، وإلا أجب 'no'.",
        # Persian
        "لطفاً پست زیر را تحلیل کنید و تعیین کنید که آیا اشاره شده است که نسبت به هزینه آزمایش‌های اسید نوکلئیک وجود دارد: {}. اگر چنین است، با 'yes' پاسخ دهید، در غیر این صورت 'no' پاسخ دهید.",
        # Hebrew
        "אנא נתחו את הפוסט הבא וקבעו אם הוא מזכיר את עלות בדיקות חומצת גרעין: {}. אם כן, השב 'yes', אחרת השב 'no'.",
        # Turkish
        "Lütfen aşağıdaki gönderiyi analiz edin ve nükleik asit testlerinin maliyetinden bahsedilip bahsedilmediğini belirtip belirtmediğini belirleyin: {}. Eğer öyleyse, 'yes' yanıtını verin, aksi takdirde 'no' yanıtını verin.",
        # Japanese
        "以下の投稿を分析し、核酸検査の費用について言及しているかどうかを判断してください: {}。 そうであれば「yes」と答え、それ以外の場合は「no」と答えてください。",
        # Korean
        "다음 게시물을 분석하고 핵산 검사 비용에 대해 언급하는지 여부를 확인하십시오: {}. 그렇다면 'yes'로 답하고, 그렇지 않다면 'no'로 답하십시오.",
        # Vietnamese
        "Vui lòng phân tích bài viết dưới đây và xác định xem nó có đề cập đến chi phí của xét nghiệm axit nucleic hay không: {}. Nếu có, hãy trả lời 'yes', nếu không, hãy trả lời 'no'.",
        # Thai
        "โปรดวิเคราะห์โพสต์ด้านล่างนี้และระบุว่ามีการกล่าวถึงค่าใช้จ่ายในการทดสอบกรดนิวคลีอิกหรือไม่: {}. หากเป็นเช่นนั้น ให้ตอบว่า 'yes' มิฉะนั้นให้ตอบว่า 'no'",
        # Indonesian
        "Harap analisis postingan di bawah ini dan tentukan apakah disebutkan biaya pengujian asam nukleat: {}. Jika demikian, balas dengan 'yes', jika tidak, balas dengan 'no'.",
        # Malay
        "Sila analisis siaran di bawah dan tentukan sama ada ia menyebutkan kos ujian asid nukleik: {}. Jika ya, sila jawab 'yes', jika tidak, jawab 'no'.",
        # Lao
        "โปรดวิเคราะห์โพสต์ด้านล่างนี้และระบุว่ามีการกล่าวถึงค่าใช้จ่ายในการทดสอบกรดนิวคลีอิกหรือไม่: {}. หากเป็นเช่นนั้น ให้ตอบว่า 'yes' มิฉะนั้นให้ตอบว่า 'no'",
        # Burmese
        "ကျေးဇူးပြု၍ အောက်ပါစာမျက်နှာကိုသုံးသပ်ပြီး ကပ်ရောဂါအကြောင်းကြောင့် စိုးရိမ်ပူပန်မှုများ၊ ဒဏ်ခံနိုင်သောကြောက်ရွံ့မှုများကို ရှာဖွေပါ။ {}. အကယ်၍အကယ်၍တစ်ခုပြုလုပ်ရန် 'yes' ဖြင့်ပြန်ကြားပါ၊ မဟုတ်ပါက 'no' ဖြင့်ပြန်ကြားပါ။",
        # Cebuano
        "Pagtudlo sa mga plano o mga appointment alang sa nucleic acid testing. Nagpasabot kini nga ang post naghisgot bahin sa pagsugod sa pagsulay sa dili madugay, nga naghimo og appointment, nagplano sa paghimo og appointment, o pagpangandam alang sa pagsulay: {}. Palihug paghimo usa ka paghukom base sa sulod sa ibabaw. Kung mao, tubaga nga 'yes', kung dili, tubaga nga 'no'.",
        # Khmer
        "សូម​វិភាគ​អត្ថបទ​សេដ្ឋកិច្ច​ខាងក្រោម​នេះ ហើយ​សម្រេចថា តើ​វា​មាន​គម្រោង​នោះទេ ឬ​ការណាត់ជួប​សម្រាប់​ការសាកល្បង​អាស៊ីត​នុយក្លេអ៊ីក។ វា​មានន័យថា​ការ​ប្រកាស​បាន​រំឭក​ពី​ការ​ចាប់ផ្តើម​ធ្វើ​តេស្ត​នៅក្នុង​ពេលឆាប់​នេះ ការណាត់ជួប ការធ្វើគម្រោង​នេះ សម្រាប់ការណាត់ជួប ឬ​ការ​រៀបចំ​សម្រាប់​ការសាកល្បង៖ {}. ប្រសិនបើ​បាទ/ចាស សូមឆ្លើយថា 'yes' បើ​មិនមែន ឆ្លើយថា 'no'.",
        # Tagalog
        "Pakisuri ang sumusunod na post at tukuyin kung ito ay binabanggit ang mga plano o mga appointment para sa pagsusuri ng nucleic acid. Ipinapahiwatig nito na ang post ay binanggit na magsisimula ang pagsusuri, nag-set na ng appointment, plano na mag-set ng appointment, o naghahanda na para sa pagsusuri: {}. Kung gayon, sumagot ng 'yes', kung hindi, sumagot ng 'no'.",
        # Hindi
        "कृपया निम्नलिखल پوسٹ کا تجزیہ کریں اور یہ بتائیں کہ کیا اس میں نیوکلک ایسڈ ٹیسٹ کے بارے میں ہچکچاہٹ کا ذکر ہے: {}. اگر ہاں تو 'yes' جواب دیں؛ بصورت دیگر، 'no' جواب دیں۔"
    ]
}

# Create three lists to store the data before converting them to a DataFrame
instructions_list = []
outputs_list = []
categories_list = []

# Function to create instruction based on the selected template
def create_instruction(text, category):
    instruction_template = random.choice(multilingual_templates[category])
    instruction = instruction_template.format(text)
    return instruction

# Iterate over each row in the original DataFrame
for index, row in df.iterrows():
    text = str(row['text'])
    for category in categories:
        # Create instruction for the category
        instruction = create_instruction(text, category)
        if row['topic'].find(" "+category+" ") >= 0:
            output = 'yes'
        else:
            output = 'no'
        
        # Append to lists
        instructions_list.append(instruction)
        outputs_list.append(output)
        categories_list.append(category)

# Once all data is collected, create the DataFrame in one go
instructions_df = pd.DataFrame({
    'instruction': instructions_list,
    'output': outputs_list,
    'category': categories_list
})

# remove those from category 1.4
# instructions_df = instructions_df[instructions_df.category != '1.4']

from sklearn.utils import resample

# Separate the majority and minority classes
df_majority = instructions_df[instructions_df.output == 'no']
df_minority = instructions_df[instructions_df.output == 'yes']

# Downsample the majority class
df_majority_downsampled = resample(df_majority,
                                   replace=False,    # sample without replacement
                                   n_samples=89000, # to match minority class
                                   random_state=42) # reproducible results

# Combine the minority class with the downsampled majority class
instructions_df_balanced = pd.concat([df_majority_downsampled, df_minority])
# print(instructions_df_balanced.output.value_counts())

# Save the DataFrame to a parquet file
instructions_df_balanced[['instruction', 'output']].sample(n=104000, random_state = 42).to_parquet("../../data/WCT/WCT-others.parquet", index=False)
# instructions_df.output.value_counts()
