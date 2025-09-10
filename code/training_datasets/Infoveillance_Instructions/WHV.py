# -*- coding: utf-8 -*-
# WHV (Weibo HPV Vaccine)
# - Paper: https://doi.org/10.1101/2023.12.07.23299667
# - Data: Not publicly available

import pandas as pd
import random
from sklearn.utils import resample
# load data
df = pd.read_csv('../../data/WHV/train_data.csv')
df_result = pd.DataFrame(columns=['instruction', 'output'])

# Define multilingual prompts
multilingual_prompts = {
    "Irrelevant": [
    # English
    "If this Weibo post is made by an individual and related to the HPV vaccine, rather than being sent by an official account or bot like announcements, advertisements, promotions, news, etc., answer yes; otherwise, answer no. The Weibo post said: {text}. Now the Weibo post ends. The answer is:",

    # Chinese (Simplified)
    "如果这条微博帖子是由个人发布并且与HPV疫苗有关，而不是由官方账号或机器人发布的公告、广告、促销、新闻等，则回答 yes；否则回答 no。微博内容如下：{text}。现在微博内容结束。The answer is:",

    # Arabic
    "إذا كان هذا المنشور على Weibo تم بواسطة شخص فردي ويتعلق بلقاح HPV، وليس منشورًا بواسطة حساب رسمي أو بوت مثل الإعلانات أو الترويج أو الأخبار وما إلى ذلك، فاجب yes؛ خلاف ذلك، أجب no. قال منشور Weibo: {text}. الآن ينتهي منشور Weibo. The answer is:",

    # French
    "Si ce post Weibo est fait par un individu et concerne le vaccin HPV, plutôt que par un compte officiel ou un bot comme des annonces, des publicités, des promotions, des nouvelles, etc., répondez yes ; sinon, répondez no. Le post Weibo disait : {text}. Maintenant, le post Weibo est terminé. The answer is:",

    # German
    "Wenn dieser Weibo-Beitrag von einer Einzelperson erstellt wurde und sich auf den HPV-Impfstoff bezieht, nicht aber von einem offiziellen Konto oder einem Bot wie Ankündigungen, Werbung, Promotionen, Nachrichten usw., antworten Sie mit yes; andernfalls antworten Sie mit no. Der Weibo-Beitrag lautete: {text}. Jetzt endet der Weibo-Beitrag. The answer is:",

    # Spanish
    "Si esta publicación de Weibo está hecha por una persona y relacionada con la vacuna contra el VPH, en lugar de ser enviada por una cuenta oficial o un bot como anuncios, promociones, noticias, etc., responda yes; de lo contrario, responda no. La publicación de Weibo decía: {text}. Ahora termina la publicación de Weibo. The answer is:",

    # Portuguese
    "Se esta postagem no Weibo for feita por uma pessoa e estiver relacionada à vacina contra o HPV, em vez de ser enviada por uma conta oficial ou bot como anúncios, promoções, notícias, etc., responda yes; caso contrário, responda no. A postagem no Weibo disse: {text}. Agora a postagem no Weibo termina. The answer is:",

    # Italian
    "Se questo post su Weibo è stato creato da un individuo e riguarda il vaccino HPV, piuttosto che essere inviato da un account ufficiale o bot come annunci, pubblicità, promozioni, notizie, ecc., rispondi yes; altrimenti rispondi no. Il post di Weibo diceva: {text}. Ora il post su Weibo finisce. The answer is:",

    # Dutch
    "Als deze Weibo-post door een individu is gemaakt en betrekking heeft op het HPV-vaccin, en niet door een officieel account of bot zoals aankondigingen, advertenties, promoties, nieuws, enz., antwoord dan yes; anders antwoord no. De Weibo-post zei: {text}. Nu eindigt de Weibo-post. The answer is:",

    # Russian
    "Если этот пост в Weibo сделан физическим лицом и связан с вакциной против ВПЧ, а не отправлен с официального аккаунта или бота, как объявления, реклама, акции, новости и т. д., ответьте yes; в противном случае ответьте no. Пост в Weibo сказал: {text}. Теперь пост в Weibo заканчивается. The answer is:",

    # Persian
    "اگر این پست در Weibo توسط یک فرد ایجاد شده و مربوط به واکسن HPV باشد، و نه توسط یک حساب رسمی یا ربات مانند اطلاعیه‌ها، تبلیغات، ترویجات، اخبار و غیره، پاسخ yes است؛ در غیر این صورت، پاسخ no است. پست Weibo گفت: {text}. اکنون پست Weibo به پایان می‌رسد. The answer is:",

    # Hebrew
    "אם פוסט זה ב-Weibo נכתב על ידי אדם פרטי וקשור לחיסון ה-HPV, ולא על ידי חשבון רשמי או רובוט כמו הודעות, פרסומות, קידומים, חדשות וכו', ענה yes; אחרת, ענה no. הפוסט ב-Weibo אמר: {text}. עכשיו הפוסט ב-Weibo מסתיים. The answer is:",

    # Turkish
    "Bu Weibo gönderisi bir birey tarafından yapıldıysa ve HPV aşısıyla ilgiliyse, resmi bir hesap veya bot tarafından yapılan duyurular, reklamlar, promosyonlar, haberler vb. yerine cevap yes; aksi takdirde no yanıtlayın. Weibo gönderisi dedi ki: {text}. Şimdi Weibo gönderisi bitiyor. The answer is:",

    # Japanese
    "このWeibo投稿が個人によって作成され、HPVワクチンに関連している場合、公式アカウントやボットによるアナウンス、広告、プロモーション、ニュースなどではなく、yesと答えてください。それ以外の場合は、noと答えてください。Weibo投稿には次のように書かれていました: {text}。今、Weibo投稿が終了します。The answer is:",

    # Korean
    "이 Weibo 게시물이 개인에 의해 작성되었고 HPV 백신과 관련이 있는 경우, 공지, 광고, 프로모션, 뉴스 등과 같은 공식 계정 또는 봇에 의해 게시된 것이 아닌 경우, yes로 답변하십시오. 그렇지 않으면 no로 답변하십시오. Weibo 게시물에는 다음과 같이 말했습니다: {text}. 이제 Weibo 게시물이 끝납니다. The answer is:",

    # Vietnamese
    "Nếu bài đăng Weibo này do cá nhân tạo và liên quan đến vắc-xin HPV, không phải từ tài khoản chính thức hoặc bot như thông báo, quảng cáo, khuyến mãi, tin tức, v.v., hãy trả lời yes; nếu không, hãy trả lời no. Bài đăng Weibo đã nói: {text}. Bây giờ bài đăng Weibo kết thúc. The answer is:",

    # Thai
    "หากโพสต์ Weibo นี้สร้างโดยบุคคลและเกี่ยวข้องกับวัคซีน HPV ไม่ใช่โดยบัญชีทางการหรือบอท เช่น การประกาศ โฆษณา การโปรโมต ข่าว ฯลฯ ให้ตอบว่า yes มิฉะนั้นตอบว่า no โพสต์ Weibo กล่าวว่า: {text}. ตอนนี้โพสต์ Weibo สิ้นสุดลงแล้ว. The answer is:",

    # Indonesian
    "Jika postingan Weibo ini dibuat oleh individu dan terkait dengan vaksin HPV, bukan oleh akun resmi atau bot seperti pengumuman, iklan, promosi, berita, dll., jawab yes; jika tidak, jawab no. Postingan Weibo mengatakan: {text}. Sekarang postingan Weibo berakhir. The answer is:",

    # Malay
    "Jika kiriman Weibo ini dibuat oleh individu dan berkaitan dengan vaksin HPV, bukan oleh akaun rasmi atau bot seperti pengumuman, iklan, promosi, berita, dll., jawab yes; jika tidak, jawab no. Kiriman Weibo berkata: {text}. Sekarang kiriman Weibo tamat. The answer is:",

    # Lao
    "ຖ້າການໂພສ Weibo ນີ້ຖືກສ້າງໂດຍບຸກຄົນແລະກ່ຽວຂ້ອງກັບວັກຊີນ HPV, ແທນທີ່ຈະຖືກສົ່ງໂດຍບັນຊີທາງການຫລືບອດຄວາມຄິດເຫັນເຊັ່ນການປະກາດ, ໂຄສະນາ, ການໂຄສະນາ, ຂ່າວສານຕ່າງໆ, ອື່ນໆ, ກະລຸນາຕອບວ່າ yes; ຖ້າບໍ່ຊັ່ນນັ້ນກະລຸນາຕອບວ່າ no. ໂພສ Weibo ກ່າວວ່າ: {text}. ຕອນນີ້ໂພສ Weibo ສິ້ນສຸດແລ້ວ. The answer is:",

    # Burmese
    "ဤ Weibo ပို့စ်သည် တစ်ဦးတစ်ယောက်မှရေးသားပြီး HPV ကာကွယ်ဆေးနှင့်ဆိုင်သော အကောင့်တစ်ခုဖြစ်ပါက ကြော်ငြာများ၊ ကြော်ငြာများ၊ ပရိုမိုးရှင်းများ၊ သတင်းများ စသည်ဖြင့် တရားဝင်အကောင့် သို့မဟုတ် bot မှ ပို့စ်မဟုတ်ပါက yesဖြစ်ပါက၊ မဟုတ်ပါက noဖြေပါ။ Weibo ပို့စ်က: {text}. ယခု Weibo ပို့စ်၏အဆုံးသတ်ဖြစ်သည်။ The answer is:",

    # Cebuano
    "Kung ang kini nga post sa Weibo gihimo sa usa ka indibidwal ug may kalabutan sa bakuna sa HPV, dili gikan sa usa ka opisyal nga account o bot sama sa mga pahibalo, mga ad, mga promosyon, balita, ug uban pa, tubaga ug yes; kung dili, tubaga nga no. Ang Weibo post miingon: {text}. Karun ang Weibo post natapos na. The answer is:",

    # Khmer
    "ប្រសិនបើការប្រកាសនេះត្រូវបានបង្កើតឡើងដោយបុគ្គល និងពាក់ព័ន្ធនឹងវ៉ាក់សាំង HPV មិនមែនត្រូវបានបង្កើតឡើងដោយគណនីផ្លូវការឬបុតនៅក្នុងការផ្សាយពាណិជ្ជកម្ម ការផ្សព្វផ្សាយ ឬព័ត៌មានទេ សូមឆ្លើយថា yes បើមិនមែនទេសូមឆ្លើយថា no។ ប្រកាសនេះបាននិយាយថា {text}។ ឥឡូវការប្រកាសបានបញ្ចប់។ The answer is:",

    # Tagalog
    "Kung ang post na ito sa Weibo ay ginawa ng isang indibidwal at may kaugnayan sa bakuna sa HPV, at hindi ng isang opisyal na account o bot tulad ng mga anunsyo, mga ad, mga promosyon, mga balita, atbp., sagutin ang yes; kung hindi, sagutin ang no. Ang post sa Weibo ay nagsabi: {text}. Ngayon natapos na ang post sa Weibo. The answer is:",

    # Hindi
    "यदि यह Weibo पोस्ट किसी व्यक्ति द्वारा बनाई गई है और HPV वैक्सीन से संबंधित है, न कि किसी आधिकारिक खाते या बॉट द्वारा जैसे घोषणाएँ, विज्ञापन, प्रचार, समाचार आदि, तो उत्तर yes में दें; अन्यथा उत्तर no में दें। Weibo पोस्ट ने कहा: {text}. अब Weibo पोस्ट समाप्त होता है। The answer is:",

    # Bengali
    "যদি এই Weibo পোস্টটি একটি ব্যক্তির দ্বারা তৈরি করা হয় এবং HPV ভ্যাকসিনের সাথে সম্পর্কিত হয়, তাহলে একটি অফিসিয়াল অ্যাকাউন্ট বা বটের মতো ঘোষণা, বিজ্ঞাপন, প্রচার, সংবাদ ইত্যাদির পরিবর্তে yes উত্তর দিন; অন্যথায়, no উত্তর দিন। Weibo পোস্ট বলেছে: {text}. এখন Weibo পোস্ট শেষ। The answer is:",

    # Urdu
    "اگر یہ Weibo پوسٹ کسی فرد نے بنائی ہے اور HPV ویکسین سے متعلق ہے، نہ کہ کسی سرکاری اکاؤنٹ یا بوٹ جیسے اعلانات، اشتہارات، تشہیریں، خبریں وغیرہ کی طرف سے، تو جواب yes میں دیں؛ بصورت دیگر، no میں جواب دیں۔ Weibo پوسٹ نے کہا: {text}. اب Weibo پوسٹ ختم ہوتی ہے۔ The answer is:"

    # Czech
    "Pokud byl tento příspěvek na Weibo vytvořen jednotlivcem a souvisí s vakcínou proti HPV, místo aby byl odeslán oficiálním účtem nebo robotem jako oznámení, reklamy, propagace, zprávy atd., odpovězte ano; jinak odpovězte ne. Příspěvek na Weibo řekl: {text}. Nyní příspěvek na Weibo končí. Odpověď je:",

    # Polish
    "Jeśli ten post na Weibo został napisany przez osobę prywatną i dotyczy szczepionki HPV, a nie został wysłany z oficjalnego konta lub bota, jak ogłoszenia, reklamy, promocje, wiadomości itp., odpowiedz tak; w przeciwnym razie odpowiedz nie. Post na Weibo brzmiał: {text}. Teraz post na Weibo się kończy. Odpowiedź brzmi:"
    ],

    "Attitude": [
        # English
        "Determine which one of these three sentiments this Weibo post expresses about the HPV vaccine: Positive, Neutral, or Negative. The Weibo post said: {text}. Now the Weibo post ends. The answer is:",

        # Chinese (Simplified)
        "判断这条微博帖子对HPV疫苗表达的情感是Positive, Neutral, 还是Negative。微博内容如下：{text}。现在微博内容结束。回答是：",

        # German
        "Bestimmen Sie, welches dieser drei Gefühle dieser Weibo-Beitrag über den HPV-Impfstoff ausdrückt: Positive, Neutral, oder Negative. Der Weibo-Beitrag lautete: {text}. Jetzt endet der Weibo-Beitrag. Die Antwort ist:",

        # French
        "Déterminez lequel de ces trois sentiments ce post Weibo exprime à propos du vaccin HPV : Positive, Neutral, ou Negative. Le post Weibo disait : {text}. Maintenant, le post Weibo est terminé. La réponse est :",

        # Spanish
        "Determine cuál de estos tres sentimientos expresa esta publicación de Weibo sobre la vacuna contra el VPH: Positive, Neutral, o Negative. La publicación de Weibo decía: {text}. Ahora termina la publicación de Weibo. La respuesta es:",

        # Portuguese
        "Determine qual desses três sentimentos este post no Weibo expressa sobre a vacina contra o HPV: Positive, Neutral, ou Negative. A postagem no Weibo disse: {text}. Agora a postagem no Weibo termina. A resposta é:",

        # Italian
        "Determina quale di questi tre sentimenti esprime questo post su Weibo riguardo al vaccino HPV: Positive, Neutral, o Negative. Il post di Weibo diceva: {text}. Ora il post su Weibo finisce. La risposta è:",

        # Dutch
        "Bepaal welke van deze drie gevoelens deze Weibo-post uitdrukt over het HPV-vaccin: Positive, Neutral, of Negative. De Weibo-post zei: {text}. Nu eindigt de Weibo-post. Het antwoord is:",

        # Russian
        "Определите, какое из этих трёх чувств выражает этот пост в Weibo о вакцине против ВПЧ: Positive, Neutral, или Negative. Пост в Weibo сказал: {text}. Теперь пост в Weibo заканчивается. Ответ:",

        # Arabic
        "حدد أي من هذه المشاعر الثلاثة يعبر عنها هذا المنشور على Weibo حول لقاح HPV: Positive, Neutral, أو Negative. قال منشور Weibo: {text}. الآن ينتهي منشور Weibo. الجواب هو:",

        # Persian
        "مشخص کنید که کدام یک از این سه احساس این پست Weibo راجع به واکسن HPV بیان می‌کند: Positive, Neutral، یا Negative. پست Weibo گفت: {text}. اکنون پست Weibo به پایان می‌رسد. پاسخ:",

        # Hebrew
        "קבע איזה מתוך שלושת הרגשות הללו פוסט זה ב-Weibo מבטא לגבי חיסון ה-HPV: Positive, Neutral, או Negative. הפוסט ב-Weibo אמר: {text}. עכשיו הפוסט ב-Weibo מסתיים. התשובה היא:",

        # Turkish
        "Bu Weibo gönderisinin HPV aşısı hakkında hangi üç duygudan birini ifade ettiğini belirleyin: Positive, Neutral veya Negative. Weibo gönderisi dedi ki: {text}. Şimdi Weibo gönderisi bitiyor. Cevap:",

        # Japanese
        "このWeibo投稿がHPVワクチンについてどの感情を表現しているかを決定してください：Positive, Neutral, またはNegative。Weibo投稿には次のように書かれていました: {text}。今、Weibo投稿が終了します。答えは：",

        # Korean
        "이 Weibo 게시물이 HPV 백신에 대해 어떤 감정을 표현하는지 결정하십시오: Positive, Neutral 또는 Negative. Weibo 게시물에는 다음과 같이 말했습니다: {text}. 이제 Weibo 게시물이 끝납니다. 대답은:",

        # Vietnamese
        "Xác định xem bài đăng Weibo này thể hiện cảm xúc nào trong ba cảm xúc sau về vắc-xin HPV: Positive, Neutral, hoặc Negative. Bài đăng Weibo đã nói: {text}. Bây giờ bài đăng Weibo kết thúc. Câu trả lời là:",

        # Thai
        "กำหนดว่าโพสต์ Weibo นี้แสดงอารมณ์ใดจากสามอารมณ์เกี่ยวกับวัคซีน HPV: Positive, Neutral, หรือ Negative โพสต์ Weibo กล่าวว่า: {text}. ตอนนี้โพสต์ Weibo สิ้นสุดลงแล้ว คำตอบคือ:",

        # Indonesian
        "Tentukan mana dari tiga perasaan ini yang diungkapkan oleh postingan Weibo ini tentang vaksin HPV: Positive, Neutral, atau Negative. Postingan Weibo mengatakan: {text}. Sekarang postingan Weibo berakhir. Jawabannya adalah:",

        # Malay
        "Tentukan perasaan mana satu daripada tiga ini yang dinyatakan oleh kiriman Weibo ini tentang vaksin HPV: Positive, Neutral, atau Negative. Kiriman Weibo berkata: {text}. Sekarang kiriman Weibo tamat. Jawapannya ialah:",

        # Lao
        "ກຳນົດຄວາມຮູ້ສຶກສາມຢ່າງນີ້ໃນການໂພສ Weibo ກ່ຽວກັບວັກຊີນ HPV: Positive, Neutral, ຫຼື Negative. ໂພສ Weibo ໄດ້ກ່າວວ່າ: {text}. ຕອນນີ້ໂພສ Weibo ສິ້ນສຸດແລ້ວ. ຄຳຕອບແມ່ນ:",

        # Burmese
        "ဤ Weibo ပို့စ်သည် HPV ကာကွယ်ဆေးနှင့် ပတ်သက်၍ ခံစားချက်သုံးခုထဲမှ Positive, Neutral, Negative တစ်ခုကို စိစစ်ပါ။ Weibo ပို့စ်က: {text}. ယခု Weibo ပို့စ်၏အဆုံးသတ်ဖြစ်သည်။ အဖြေက:",

        # Cebuano
        "Tinoha kung unsa niini nga tulo nga pagbati ang gipahayag niini nga post sa Weibo bahin sa bakuna sa HPV: Positive, Neutral, o Negative. Ang Weibo post miingon: {text}. Karun ang Weibo post natapos na. Ang tubag mao:",

        # Khmer
        "កំណត់ថាមនោសញ្ចេតនាទាំងបីនេះដែលប្រកាសនេះក្នុង Weibo កំពុងបង្ហាញអំពីវ៉ាក់សាំង HPV: Positive, Neutral, ឬ Negative។ ប្រកាស Weibo បាននិយាយថា {text}។ ឥឡូវប្រកាសបានបញ្ចប់។ ការឆ្លើយតបគឺ:",

        # Tagalog
        "Tukuyin kung alin sa tatlong damdamin na ito ang ipinapahayag ng post na ito sa Weibo tungkol sa bakuna sa HPV: Positive, Neutral, o Negative. Ang post sa Weibo ay nagsabi: {text}. Ngayon natapos na ang post sa Weibo. Ang sagot ay:",

        # Hindi
        "निर्धारित करें कि HPV वैक्सीन के बारे में यह Weibo पोस्ट कौन सा तीन भाव व्यक्त करता है: Positive, Neutral, या Negative। Weibo पोस्ट ने कहा: {text}. अब Weibo पोस्ट समाप्त होता है। उत्तर है:",

        # Bengali
        "নির্ধারণ করুন যে এই Weibo পোস্টটি HPV ভ্যাকসিন সম্পর্কে কোন তিনটি অনুভূতির একটি প্রকাশ করে: Positive, Neutral, বা Negative। Weibo পোস্ট বলেছে: {text}. এখন Weibo পোস্ট শেষ। উত্তর হল:",

        # Urdu
        "یہ فیصلہ کریں کہ یہ Weibo پوسٹ HPV ویکسین کے بارے میں کون سا تین جذبات کا اظہار کرتی ہے: Positive, Neutral, یا Negative۔ Weibo پوسٹ نے کہا: {text}. اب Weibo پوسٹ ختم ہوتی ہے۔ جواب ہے:",

        # Czech
        "Určete, který z těchto tří pocitů tento příspěvek na Weibo vyjadřuje o vakcíně proti HPV: Positive (pozitivní), Neutral (neutrální) nebo Negative (negativní). Příspěvek na Weibo říkal: {text}. Nyní příspěvek na Weibo končí. Odpověď je:",

        # Polish
        "Określ, którą z tych trzech emocji wyraża ten post na Weibo na temat szczepionki przeciw HPV: Positive (pozytywna), Neutral (neutralna) lub Negative (negatywna). Post na Weibo brzmiał: {text}. Teraz post na Weibo się kończy. Odpowiedź brzmi:",
],

    "Behavior": [
        # English
        "Determine if the Weibo post indicates that the person posting or their immediate family members have been vaccinated or have an appointment for the HPV vaccine. If so, respond with 'yes'; otherwise, respond with 'no'. The Weibo post said: {text}. Now the Weibo post ends. The answer is:",

        # Chinese (Simplified)
        "确定微博帖子是否表明发布者或其直系亲属已接种HPV疫苗或预约了HPV疫苗。如果是，请回答 yes；否则请回答 no。微博内容如下：{text}。现在微博内容结束。The answer is:",

        # German
        "Bestimmen Sie, ob der Weibo-Beitrag darauf hindeutet, dass die Person, die postet, oder ihre unmittelbaren Familienmitglieder gegen HPV geimpft wurden oder einen Termin für die HPV-Impfung haben. Wenn ja, antworten Sie mit 'yes'; andernfalls antworten Sie mit 'no'. Der Weibo-Beitrag lautete: {text}. Jetzt endet der Weibo-Beitrag. The answer is:",

        # French
        "Déterminez si le post Weibo indique que la personne qui poste ou les membres de sa famille immédiate ont été vaccinés ou ont un rendez-vous pour le vaccin HPV. Si c'est le cas, répondez par 'yes'; sinon, répondez par 'no'. Le post Weibo disait : {text}. Maintenant, le post Weibo est terminé. The answer is:",

        # Spanish
        "Determine si la publicación de Weibo indica que la persona que publica o los miembros de su familia inmediata han sido vacunados o tienen una cita para la vacuna contra el VPH. Si es así, responda con 'yes'; de lo contrario, responda con 'no'. La publicación de Weibo decía: {text}. Ahora termina la publicación de Weibo. The answer is:",

        # Portuguese
        "Determine se a postagem no Weibo indica que a pessoa que postou ou os membros imediatos da família foram vacinados ou têm um compromisso para a vacina contra o HPV. Se sim, responda com 'yes'; caso contrário, responda com 'no'. A postagem no Weibo disse: {text}. Agora a postagem no Weibo termina. The answer is:",

        # Italian
        "Determina se il post su Weibo indica che la persona che ha postato o i membri immediati della sua famiglia sono stati vaccinati o hanno un appuntamento per il vaccino HPV. Se sì, rispondi 'yes'; altrimenti rispondi 'no'. Il post di Weibo diceva: {text}. Ora il post su Weibo finisce. The answer is:",

        # Dutch
        "Bepaal of de Weibo-post aangeeft dat de persoon die post of hun naaste familieleden zijn gevaccineerd of een afspraak hebben voor het HPV-vaccin. Zo ja, antwoord dan met 'yes'; anders antwoord met 'no'. De Weibo-post zei: {text}. Nu eindigt de Weibo-post. The answer is:",

        # Russian
        "Определите, указывает ли пост в Weibo на то, что лицо, размещающее сообщение, или их ближайшие родственники были вакцинированы или имеют назначение на вакцинацию против ВПЧ. Если да, ответьте «yes»; в противном случае ответьте «no». Пост в Weibo сказал: {text}. Теперь пост в Weibo заканчивается. The answer is:",

        # Arabic
        "حدد ما إذا كان منشور Weibo يشير إلى أن الشخص الذي ينشر أو أفراد عائلته المقربين قد تم تطعيمهم أو لديهم موعد للحصول على لقاح HPV. إذا كان الأمر كذلك، استجب بـ 'yes'؛ خلاف ذلك، استجب بـ 'no'. قال منشور Weibo: {text}. الآن ينتهي منشور Weibo. The answer is:",

        # Persian
        "مشخص کنید که آیا پست Weibo نشان می‌دهد که شخص پست‌کننده یا اعضای نزدیک خانواده آن‌ها واکسینه شده‌اند یا برای واکسن HPV وقت گرفته‌اند. اگر چنین است، با «yes» پاسخ دهید؛ در غیر این صورت، پاسخ «no» دهید. پست Weibo گفت: {text}. اکنون پست Weibo به پایان می‌رسد. The answer is:",

        # Hebrew
        "קבע אם הפוסט ב-Weibo מציין שהאדם שמפרסם או בני משפחתו הקרובים חוסנו או שיש להם תור לחיסון HPV. אם כן, ענה 'yes'; אחרת, ענה 'no'. הפוסט ב-Weibo אמר: {text}. עכשיו הפוסט ב-Weibo מסתיים. The answer is:",

        # Turkish
        "Weibo gönderisinin, gönderiyi yapan kişinin veya birinci dereceden aile üyelerinin HPV aşısı olduğunu veya HPV aşısı için randevu aldıklarını belirtip belirtmediğini belirleyin. Eğer öyleyse, 'yes' yanıtlayın; aksi takdirde 'no' yanıtlayın. Weibo gönderisi dedi ki: {text}. Şimdi Weibo gönderisi bitiyor. The answer is:",

        # Japanese
        "Weibo投稿が投稿者またはその直系家族がHPVワクチンを接種したか、HPVワクチンの予約をしていることを示しているかどうかを判断してください。もしそうなら、「yes」と答えてください。それ以外の場合は、「no」と答えてください。Weibo投稿には次のように書かれていました: {text}。今、Weibo投稿が終了します。The answer is:",

        # Korean
        "Weibo 게시물이 게시한 사람 또는 그 직계 가족이 HPV 백신을 접종했거나 HPV 백신 예약을 했는지 여부를 결정하십시오. 그렇다면 'yes'로 답변하십시오. 그렇지 않으면 'no'로 답변하십시오. Weibo 게시물에는 다음과 같이 말했습니다: {text}. 이제 Weibo 게시물이 끝납니다. The answer is:",

        # Vietnamese
        "Nếu bài đăng Weibo này cho biết rằng người đăng hoặc các thành viên gia đình của họ đã được tiêm phòng hoặc đã có hẹn để tiêm vắc-xin HPV, nếu vậy, hãy trả lời 'yes'; nếu không, hãy trả lời 'no'. Bài đăng Weibo đã nói: {text}. Bây giờ bài đăng Weibo kết thúc. The answer is:",

        # Thai
        "กำหนดว่าโพสต์ Weibo นี้ระบุว่าผู้ที่โพสต์หรือสมาชิกในครอบครัวโดยตรงของพวกเขาได้รับการฉีดวัคซีนหรือมีนัดสำหรับวัคซีน HPV หรือไม่ หากเป็นเช่นนั้น ให้ตอบว่า 'yes'; มิฉะนั้น ให้ตอบว่า 'no'. โพสต์ Weibo กล่าวว่า: {text}. ตอนนี้โพสต์ Weibo สิ้นสุดลงแล้ว. The answer is:",

        # Indonesian
        "Tentukan apakah postingan Weibo ini menunjukkan bahwa orang yang memposting atau anggota keluarga dekat mereka telah divaksinasi atau memiliki janji untuk vaksin HPV. Jika demikian, jawab 'yes'; jika tidak, jawab 'no'. Postingan Weibo mengatakan: {text}. Sekarang postingan Weibo berakhir. The answer is:",

        # Malay
        "Tentukan sama ada kiriman Weibo ini menunjukkan bahawa orang yang membuat kiriman atau ahli keluarga terdekat mereka telah diberi vaksin atau mempunyai janji temu untuk vaksin HPV. Jika ya, jawab 'yes'; jika tidak, jawab 'no'. Kiriman Weibo berkata: {text}. Sekarang kiriman Weibo tamat. The answer is:",

        # Lao
        "ກຳນົດວ່າການໂພສ Weibo ນີ້ລະບຸວ່າບຸກຄົນທີ່ໂພສຫຼືສະມາຊິກຄອບຄົວຂອງເຂົາເຈົ້າທີ່ເຄີຍຮັບການຉີດວັກຊີນຫຼືມີນັດສໍາລັບການຉີດວັກຊີນ HPV ຫຼືບໍ່. ຖ້າແມ່ນແລ້ວ, ກະລຸນາຕອບວ່າ 'yes'; ຖ້າບໍ່ແມ່ນ, ກະລຸນາຕອບວ່າ 'no'. ໂພສ Weibo ກ່າວວ່າ: {text}. ຕອນນີ້ໂພສ Weibo ສິ້ນສຸດແລ້ວ. The answer is:",

        # Burmese
        "ဤ Weibo ပို့စ်သည် ပို့စ်ရေးသားသူ သို့မဟုတ် သူ၏မိသားစုဝင်များသည် HPV ကာကွယ်ဆေးကို ထိုးပြီးကြောင်း သို့မဟုတ် HPV ကာကွယ်ဆေးထိုးရန် ခန့်အပ်ထားကြောင်း ပြသသည် ဟုတ်လျှင် 'yes' ဖြင့် ဖြေပါ၊ မဟုတ်လျှင် 'no' ဖြင့် ဖြေပါ။ Weibo ပို့စ်က: {text}. ယခု Weibo ပို့စ်၏အဆုံးသတ်ဖြစ်သည်။ The answer is:",

        # Cebuano
        "Tinoha kung ang post sa Weibo nagpakita nga ang tawo nga nag-post o ang ilang kasagarang mga miyembro sa pamilya nabakunahan na o adunay appointment alang sa bakuna sa HPV. Kung mao, tubaga ug 'yes'; kung dili, tubaga nga 'no'. Ang Weibo post miingon: {text}. Karun ang Weibo post natapos na. The answer is:",

        # Khmer
        "កំណត់ថាប្រសិនបើបុគ្គលដែលប្រកាសឬសមាជិកគ្រួសាររបស់ពួកគេបានបាញ់វ៉ាក់សាំងហើយ ឬមានការណាត់ជួបសម្រាប់វ៉ាក់សាំង HPV។ ប្រសិនបើបាទ សូមឆ្លើយថា 'yes'; ប្រសិនបើមិនមែនទេ សូមឆ្លើយថា 'no'។ ប្រកាស Weibo បាននិយាយថា {text}។ ឥឡូវប្រកាសបានបញ្ចប់។ The answer is:",

        # Tagalog
        "Tukuyin kung ang Weibo post ay nagpapahiwatig na ang taong nag-post o ang kanilang mga agarang miyembro ng pamilya ay nabakunahan o may appointment para sa HPV vaccine. Kung gayon, sumagot ng 'yes'; kung hindi, sumagot ng 'no'. Ang post sa Weibo ay nagsabi: {text}. Ngayon natapos na ang post sa Weibo. The answer is:",

        # Hindi
        "निर्धारित करें कि Weibo पोस्ट यह संकेत करता है कि पोस्ट करने वाला व्यक्ति या उनके निकटतम परिवार के सदस्य टीका लगवा चुके हैं या HPV वैक्सीन के लिए अपॉइंटमेंट किया है। यदि हां, तो 'yes' में उत्तर दें; अन्यथा, 'no' में उत्तर दें। Weibo पोस्ट ने कहा: {text}. अब Weibo पोस्ट समाप्त होता है। The answer is:",

        # Bengali
        "নির্ধারণ করুন যে Weibo পোস্টটি এই নির্দেশ করে যে পোস্টকারী বা তাদের নিকটতম পরিবার সদস্যরা টিকা পেয়েছে বা HPV ভ্যাকসিনের জন্য অ্যাপয়েন্টমেন্ট করেছে। যদি হ্যাঁ, তাহলে 'yes' উত্তর দিন; অন্যথায়, 'no' উত্তর দিন। Weibo পোস্ট বলেছে: {text}. এখন Weibo পোস্ট শেষ। The answer is:",

        # Urdu
        "یہ تعین کریں کہ Weibo پوسٹ سے پتہ چلتا ہے کہ پوسٹ کرنے والے شخص یا ان کے قریبی خاندان کے افراد نے HPV ویکسین لگوائی ہے یا HPV ویکسین کے لیے وقت مقرر کیا ہے۔ اگر ایسا ہے تو 'yes' میں جواب دیں؛ ورنہ 'no' میں جواب دیں۔ Weibo پوسٹ نے کہا: {text}. اب Weibo پوسٹ ختم ہوتی ہے۔ The answer is:",

        # Czech
        "Určete, zda příspěvek na Weibo naznačuje, že osoba, která příspěvek zveřejnila, nebo její bezprostřední rodinní příslušníci byli očkováni nebo mají termín na očkování proti HPV. Pokud ano, odpovězte 'yes'; v opačném případě odpovězte 'no'. Příspěvek na Weibo říkal: {text}. Nyní příspěvek na Weibo končí. The answer is:",

        # Polish
        "Określ, czy post na Weibo wskazuje, że osoba publikująca lub jej najbliżsi członkowie rodziny zostali zaszczepieni lub mają umówioną wizytę na szczepienie przeciw HPV. Jeśli tak, odpowiedz 'yes'; w przeciwnym razie odpowiedz 'no'. Post na Weibo brzmiał: {text}. Teraz post na Weibo się kończy. The answer is:",
],

    "Perceived Disease Risk (+)": [
        # English
        "If the post subjectively assesses the risk of human papillomavirus (HPV) infection, the severity of such infection, or its potential consequences, respond 'yes'. Otherwise, respond 'no'. The Weibo post said: {text}. Now the Weibo post ends. The answer is:",

        # Chinese (Simplified)
        "如果帖子主观评估了人乳头瘤病毒（HPV）感染的风险、此类感染的严重性或其潜在后果，请回答 'yes'。否则，请回答 'no'。微博内容如下：{text}。现在微博内容结束。The answer is:",

        # German
        "Wenn der Beitrag das Risiko einer humanen Papillomavirus (HPV)-Infektion, die Schwere einer solchen Infektion oder deren potenzielle Folgen subjektiv bewertet, antworten Sie mit 'yes'. Andernfalls antworten Sie mit 'no'. Der Weibo-Beitrag lautete: {text}. Jetzt endet der Weibo-Beitrag. The answer is:",

        # French
        "Si le post évalue subjectivement le risque d'infection par le papillomavirus humain (HPV), la gravité de cette infection ou ses conséquences potentielles, répondez 'yes'. Sinon, répondez 'no'. Le post Weibo disait : {text}. Maintenant, le post Weibo est terminé. The answer is:",

        # Spanish
        "Si la publicación evalúa subjetivamente el riesgo de infección por el virus del papiloma humano (VPH), la gravedad de dicha infección o sus posibles consecuencias, responda 'yes'. De lo contrario, responda 'no'. La publicación de Weibo decía: {text}. Ahora termina la publicación de Weibo. The answer is:",

        # Portuguese
        "Se a postagem avalia subjetivamente o risco de infecção pelo papilomavírus humano (HPV), a gravidade dessa infecção ou suas potenciais consequências, responda 'yes'. Caso contrário, responda 'no'. A postagem no Weibo disse: {text}. Agora a postagem no Weibo termina. The answer is:",

        # Italian
        "Se il post valuta soggettivamente il rischio di infezione da papillomavirus umano (HPV), la gravità di tale infezione o le sue potenziali conseguenze, rispondi 'yes'. Altrimenti, rispondi 'no'. Il post di Weibo diceva: {text}. Ora il post su Weibo finisce. The answer is:",

        # Dutch
        "Als de post subjectief het risico op een infectie met het humaan papillomavirus (HPV), de ernst van een dergelijke infectie of de mogelijke gevolgen beoordeelt, antwoord dan met 'yes'. Anders antwoord met 'no'. De Weibo-post zei: {text}. Nu eindigt de Weibo-post. The answer is:",

        # Russian
        "Если в посте субъективно оценивается риск заражения вирусом папилломы человека (ВПЧ), тяжесть такого заражения или его потенциальные последствия, ответьте «yes». В противном случае ответьте «no». Пост в Weibo сказал: {text}. Теперь пост в Weibo заканчивается. The answer is:",

        # Arabic
        "إذا قام المنشور بتقييم المخاطر المحتملة للإصابة بفيروس الورم الحليمي البشري (HPV)، وشدة الإصابة، أو العواقب المحتملة بشكل ذاتي، استجب بـ 'yes'. خلاف ذلك، استجب بـ 'no'. قال منشور Weibo: {text}. الآن ينتهي منشور Weibo. The answer is:",

        # Persian
        "اگر پست به صورت ذهنی خطر ابتلا به عفونت ویروس پاپیلومای انسانی (HPV)، شدت این عفونت یا عواقب احتمالی آن را ارزیابی کرده است، با «yes» پاسخ دهید. در غیر این صورت، پاسخ «no» دهید. پست Weibo گفت: {text}. اکنون پست Weibo به پایان می‌رسد. The answer is:",

        # Hebrew
        "אם הפוסט מעריך באופן סובייקטיבי את הסיכון להדבקה בנגיף הפפילומה האנושי (HPV), את חומרת ההדבקה או את ההשלכות הפוטנציאליות שלה, ענה 'yes'. אחרת, ענה 'no'. הפוסט ב-Weibo אמר: {text}. עכשיו הפוסט ב-Weibo מסתיים. The answer is:",

        # Turkish
        "Gönderi, insan papilloma virüsü (HPV) enfeksiyonu riskini, böyle bir enfeksiyonun ciddiyetini veya olası sonuçlarını öznel olarak değerlendiriyorsa, 'yes' yanıtlayın. Aksi takdirde 'no' yanıtlayın. Weibo gönderisi dedi ki: {text}. Şimdi Weibo gönderisi bitiyor. The answer is:",

        # Japanese
        "投稿がヒトパピローマウイルス（HPV）感染のリスク、その感染の重症度、またはその潜在的な結果を主観的に評価している場合は、「yes」と答えてください。それ以外の場合は、「no」と答えてください。Weibo投稿には次のように書かれていました: {text}。今、Weibo投稿が終了します。The answer is:",

        # Korean
        "이 게시물이 사람 유두종 바이러스 (HPV) 감염의 위험성, 그러한 감염의 심각성 또는 잠재적 결과를 주관적으로 평가하는 경우 'yes'로 답변하십시오. 그렇지 않으면 'no'로 답변하십시오. Weibo 게시물에는 다음과 같이 말했습니다: {text}. 이제 Weibo 게시물이 끝납니다. The answer is:",

        # Vietnamese
        "Nếu bài đăng đánh giá chủ quan về nguy cơ nhiễm vi-rút gây u nhú ở người (HPV), mức độ nghiêm trọng của nhiễm trùng hoặc những hậu quả tiềm ẩn của nó, hãy trả lời 'yes'. Nếu không, hãy trả lời 'no'. Bài đăng Weibo đã nói: {text}. Bây giờ bài đăng Weibo kết thúc. The answer is:",

        # Thai
        "หากโพสต์ประเมินความเสี่ยงในการติดเชื้อไวรัส human papillomavirus (HPV) ความรุนแรงของการติดเชื้อดังกล่าว หรือผลที่ตามมาในเชิงอัตวิสัย ให้ตอบว่า 'yes' มิฉะนั้น ให้ตอบว่า 'no'. โพสต์ Weibo กล่าวว่า: {text}. ตอนนี้โพสต์ Weibo สิ้นสุดลงแล้ว. The answer is:",

        # Indonesian
        "Jika postingan tersebut menilai secara subyektif risiko infeksi human papillomavirus (HPV), tingkat keparahan infeksi tersebut, atau konsekuensi potensialnya, jawab 'yes'. Jika tidak, jawab 'no'. Postingan Weibo mengatakan: {text}. Sekarang postingan Weibo berakhir. The answer is:",

        # Malay
        "Jika kiriman tersebut secara subyektif menilai risiko jangkitan papillomavirus manusia (HPV), keterukan jangkitan tersebut, atau akibat berpotensi, jawab 'yes'. Jika tidak, jawab 'no'. Kiriman Weibo berkata: {text}. Sekarang kiriman Weibo tamat. The answer is:",

        # Lao
        "ຖ້າໂພສດັ່ງກ່າວມີການປະເມີນຄວາມສ່ຽງຂອງການຕິດເຊື້ອຫມັກທັງຫມົດ (HPV) ຄວາມຮຸນແຮງຂອງການຕິດເຊື້ອດັ່ງກ່າວ ຫຼືຜົນສົ່ງເສີມທີ່ສາມາດເກີດຂຶ້ນໄດ້ຢ່າງຫຼວງພະຣາວົງໄດ້ຖ້າກ່ຽວຂ້ອງ ແລະ ຄວາມສ່ຽງສາມາດຈະໄດ້ການປະເມີນສ່ວນຕົວແບບເອັມພີເວສກົດໂທລົດຕອບວ່າ 'yes' ຖ້າບໍ່ແມ່ນ ກະລຸນາຕອບວ່າ 'no' ໂພສ Weibo ກ່າວວ່າ: {text} ຕອນນີ້ໂພສ Weibo ສິ້ນສຸດແລ້ວ. The answer is:",

        # Burmese
        "ဤပို့စ်တွင် လူသားမှု့ရင်ခွဲဝင်းပိုးမအားကျင်းခံခြင်း၏ အန္တရာယ်အခြေအနေကို အသွင်ပြောင်းနိုင်သော ဘေးကင်းစွာ စီစဥ်ထားသည်ဟု လက်ခံရပါသည်။ အကယ်၍ ပို့စ်သည် ထိုကဲ့သို့ ဤအန္တရာယ်သည် 'yes' ဟုဖြေပါ၊ ထိုဟာအန္တရာယ်မရှိသည်ဟု မသံသယဖြစ်ပါက 'no' ဟုဖြေပါ။ Weibo ပို့စ်က: {text} ယခု Weibo ပို့စ်၏အဆုံးသတ်ဖြစ်သည်။ The answer is:",

        # Cebuano
        "Kung ang post subjectively nagtantiya sa risgo sa human papillomavirus (HPV) infection, ang kalig-on sa kini nga infection, o ang mga potensyal nga kahimuan, tubaga ug 'yes'. Kung dili, tubaga ug 'no'. Ang Weibo post miingon: {text}. Karun ang Weibo post natapos na. The answer is:",

        # Khmer
        "ប្រសិនបើការប្រកាសនេះបានវាយតម្លៃយ៉ាងជាក់លាក់អំពីហានិភ័យនៃការឆ្លងមេរោគហួមភាពពីរបស់មនុស្ស (HPV) ភាពធ្ងន់ធ្ងរការឆ្លងឬឧបករណ៍ដែលអាចឆ្លង បានពេញប៉ុណ្ណោះ សូមឆ្លើយថា 'yes'. បើមិនមែនទេសូមឆ្លើយថា 'no'. ប្រកាស Weibo បាននិយាយថា {text}។ ឥឡូវប្រកាសបានបញ្ចប់។ The answer is:",

        # Tagalog
        "Kung ang post ay nagtataya sa panganib ng human papillomavirus (HPV) infection, ang kalubhaan ng impeksiyon na ito, o ang mga potensyal na epekto nito, sagutin ang 'yes'. Kung hindi, sagutin ang 'no'. Ang post sa Weibo ay nagsabi: {text}. Ngayon natapos na ang post sa Weibo. The answer is:",

        # Hindi
        "यदि पोस्ट मानव पेपिलोमावायरस (HPV) संक्रमण के जोखिम, इस तरह के संक्रमण की गंभीरता, या इसके संभावित परिणामों का व्यक्तिपरक आकलन करती है, तो 'yes' में उत्तर दें। अन्यथा, 'no' में उत्तर दें। Weibo पोस्ट ने कहा: {text}. अब Weibo पोस्ट समाप्त होता है। The answer is:",

        # Bengali
        "যদি পোস্টটি মানব প্যাপিলোমাভাইরাস (HPV) সংক্রমণের ঝুঁকি, এমন সংক্রমণের তীব্রতা, বা এর সম্ভাব্য পরিণতি সম্পর্কিত বিষয়গত মূল্যায়ন করে, তবে 'yes' উত্তর দিন। অন্যথায়, 'no' উত্তর দিন। Weibo পোস্ট বলেছে: {text}. এখন Weibo পোস্ট শেষ। The answer is:",

        # Urdu
        "اگر پوسٹ انسان پیپیلوما وائرس (HPV) انفیکشن کے خطرے، اس انفیکشن کی شدت، یا اس کے ممکنہ نتائج کے بارے میں ذاتی رائے سے جائزہ لیتی ہے، تو 'yes' میں جواب دیں؛ بصورت دیگر 'no' میں جواب دیں۔ Weibo پوسٹ نے کہا: {text}. اب Weibo پوسٹ ختم ہوتی ہے۔ The answer is:",

        # Czech
        "Pokud příspěvek subjektivně hodnotí riziko infekce lidským papilomavirem (HPV), závažnost této infekce nebo její potenciální následky, odpovězte 'yes'. Jinak odpovězte 'no'. Příspěvek na Weibo říkal: {text}. Nyní příspěvek na Weibo končí. Odpověď je:",

        # Polish
        "Jeśli post subiektywnie ocenia ryzyko zakażenia wirusem brodawczaka ludzkiego (HPV), powagę takiej infekcji lub jej potencjalne konsekwencje, odpowiedz 'yes'. W przeciwnym razie odpowiedz 'no'. Post na Weibo brzmiał: {text}. Teraz post na Weibo się kończy. Odpowiedź brzmi:",
],

    "Perceived benefits (+)": [
        # English
        "If the Weibo post mentions the perceived benefits of HPV vaccination and expresses confidence in the efficacy of HPV vaccination, such as the benefit or efficacy of the HPV vaccine in protecting against HPV infection and relevant cancers, respond 'yes'. Otherwise, respond 'no'. The Weibo post said: {text}. Now the Weibo post ends. The answer is:",

        # Chinese (Simplified)
        "如果微博帖子提到HPV疫苗接种的好处，并对HPV疫苗的有效性表达信心，例如HPV疫苗在预防HPV感染和相关癌症方面的好处或有效性，请回答 'yes'。否则，请回答 'no'。微博内容如下：{text}。现在微博内容结束。The answer is:",

        # German
        "Wenn der Weibo-Beitrag die wahrgenommenen Vorteile der HPV-Impfung erwähnt und Vertrauen in die Wirksamkeit der HPV-Impfung ausdrückt, wie zum Beispiel den Nutzen oder die Wirksamkeit des HPV-Impfstoffs zum Schutz vor HPV-Infektionen und relevanten Krebsarten, antworten Sie mit 'yes'. Andernfalls antworten Sie mit 'no'. Der Weibo-Beitrag lautete: {text}. Jetzt endet der Weibo-Beitrag. The answer is:",

        # French
        "Si le post Weibo mentionne les avantages perçus de la vaccination contre le VPH et exprime sa confiance dans l'efficacité de la vaccination contre le VPH, tels que le bénéfice ou l'efficacité du vaccin contre le VPH dans la protection contre l'infection par le VPH et les cancers associés, répondez 'yes'. Sinon, répondez 'no'. Le post Weibo disait : {text}. Maintenant, le post Weibo est terminé. The answer is:",

        # Spanish
        "Si la publicación de Weibo menciona los beneficios percibidos de la vacunación contra el VPH y expresa confianza en la eficacia de la vacunación contra el VPH, como el beneficio o la eficacia de la vacuna contra el VPH en la protección contra la infección por VPH y los cánceres asociados, responda 'yes'. De lo contrario, responda 'no'. La publicación de Weibo decía: {text}. Ahora termina la publicación de Weibo. The answer is:",

        # Portuguese
        "Se a postagem no Weibo menciona os benefícios percebidos da vacinação contra o HPV e expressa confiança na eficácia da vacinação contra o HPV, como o benefício ou a eficácia da vacina contra o HPV na proteção contra a infecção pelo HPV e os cânceres associados, responda 'yes'. Caso contrário, responda 'no'. A postagem no Weibo disse: {text}. Agora a postagem no Weibo termina. The answer is:",

        # Italian
        "Se il post su Weibo menziona i benefici percepiti della vaccinazione contro l'HPV ed esprime fiducia nell'efficacia della vaccinazione contro l'HPV, come il beneficio o l'efficacia del vaccino HPV nella protezione contro l'infezione da HPV e i relativi tumori, rispondi 'yes'. Altrimenti, rispondi 'no'. Il post di Weibo diceva: {text}. Ora il post su Weibo finisce. The answer is:",

        # Dutch
        "Als de Weibo-post de waargenomen voordelen van HPV-vaccinatie noemt en vertrouwen uitspreekt in de effectiviteit van HPV-vaccinatie, zoals het voordeel of de effectiviteit van het HPV-vaccin bij het beschermen tegen HPV-infectie en relevante vormen van kanker, antwoord dan met 'yes'. Anders antwoord met 'no'. De Weibo-post zei: {text}. Nu eindigt de Weibo-post. The answer is:",

        # Russian
        "Если в посте на Weibo упоминаются предполагаемые преимущества вакцинации против ВПЧ и выражается уверенность в эффективности вакцинации против ВПЧ, например, польза или эффективность вакцины против ВПЧ в защите от заражения ВПЧ и соответствующих видов рака, ответьте «yes». В противном случае ответьте «no». Пост в Weibo сказал: {text}. Теперь пост в Weibo заканчивается. The answer is:",

        # Arabic
        "إذا ذكر المنشور على Weibo الفوائد المحتملة لتطعيم HPV وأعرب عن ثقته في فعالية تطعيم HPV، مثل فائدة أو فعالية لقاح HPV في الحماية من عدوى HPV والسرطانات المرتبطة به، فاستجب بـ 'yes'. خلاف ذلك، استجب بـ 'no'. قال منشور Weibo: {text}. الآن ينتهي منشور Weibo. The answer is:",

        # Persian
        "اگر در پست Weibo به مزایای درک‌شده واکسیناسیون HPV اشاره شده و اعتماد به کارایی واکسیناسیون HPV ابراز شده است، مانند مزیت یا کارایی واکسن HPV در محافظت در برابر عفونت HPV و سرطان‌های مرتبط، با «yes» پاسخ دهید. در غیر این صورت، پاسخ «no» دهید. پست Weibo گفت: {text}. اکنون پست Weibo به پایان می‌رسد. The answer is:",

        # Hebrew
        "אם הפוסט ב-Weibo מזכיר את היתרונות הנתפסים של חיסון ה-HPV ומביע אמון ביעילות חיסון ה-HPV, כגון התועלת או היעילות של חיסון ה-HPV בהגנה מפני זיהום ב-HPV וסרטן קשור, ענה 'yes'. אחרת, ענה 'no'. הפוסט ב-Weibo אמר: {text}. עכשיו הפוסט ב-Weibo מסתיים. The answer is:",

        # Turkish
        "Weibo gönderisi HPV aşısının algılanan faydalarından bahsediyorsa ve HPV aşısının etkinliğine güven duyuyorsa, örneğin HPV enfeksiyonuna ve ilgili kanserlere karşı korumadaki faydası veya etkinliği gibi, 'yes' yanıtlayın. Aksi takdirde 'no' yanıtlayın. Weibo gönderisi dedi ki: {text}. Şimdi Weibo gönderisi bitiyor. The answer is:",

        # Japanese
        "Weibo投稿がHPVワクチン接種の利点を認識し、HPVワクチン接種の有効性に自信を表明している場合、たとえばHPV感染や関連するがんに対する保護におけるHPVワクチンの利点や有効性などについては、「yes」と答えてください。それ以外の場合は、「no」と答えてください。Weibo投稿には次のように書かれていました: {text}。今、Weibo投稿が終了します。The answer is:",

        # Korean
        "Weibo 게시물이 HPV 백신 접종의 이점을 언급하고 HPV 백신 접종의 효능에 대한 자신감을 표현하는 경우, 예를 들어 HPV 감염 및 관련 암에 대한 보호에서 HPV 백신의 이점 또는 효능, 'yes'로 답변하십시오. 그렇지 않으면 'no'로 답변하십시오. Weibo 게시물에는 다음과 같이 말했습니다: {text}. 이제 Weibo 게시물이 끝납니다. The answer is:",

        # Vietnamese
        "Nếu bài đăng trên Weibo đề cập đến lợi ích được cảm nhận của việc tiêm vắc-xin HPV và bày tỏ sự tin tưởng vào hiệu quả của việc tiêm vắc-xin HPV, chẳng hạn như lợi ích hoặc hiệu quả của vắc-xin HPV trong việc bảo vệ chống lại nhiễm trùng HPV và các bệnh ung thư có liên quan, hãy trả lời 'yes'. Nếu không, hãy trả lời 'no'. Bài đăng Weibo đã nói: {text}. Bây giờ bài đăng Weibo kết thúc. The answer is:",

        # Thai
        "หากโพสต์ Weibo กล่าวถึงประโยชน์ที่รับรู้ของการฉีดวัคซีน HPV และแสดงความมั่นใจในประสิทธิภาพของการฉีดวัคซีน HPV เช่น ประโยชน์หรือประสิทธิภาพของวัคซีน HPV ในการป้องกันการติดเชื้อ HPV และมะเร็งที่เกี่ยวข้อง ให้ตอบว่า 'yes' มิฉะนั้น ให้ตอบว่า 'no'. โพสต์ Weibo กล่าวว่า: {text}. ตอนนี้โพสต์ Weibo สิ้นสุดลงแล้ว. The answer is:",

        # Indonesian
        "Jika postingan Weibo menyebutkan manfaat yang dirasakan dari vaksinasi HPV dan mengungkapkan kepercayaan pada kemanjuran vaksinasi HPV, seperti manfaat atau kemanjuran vaksin HPV dalam melindungi terhadap infeksi HPV dan kanker terkait, jawab 'yes'. Jika tidak, jawab 'no'. Postingan Weibo mengatakan: {text}. Sekarang postingan Weibo berakhir. The answer is:",

        # Malay
        "Jika kiriman Weibo menyebutkan manfaat yang dirasai daripada vaksinasi HPV dan menyatakan keyakinan terhadap keberkesanan vaksinasi HPV, seperti manfaat atau keberkesanan vaksin HPV dalam melindungi daripada jangkitan HPV dan kanser yang berkaitan, jawab 'yes'. Jika tidak, jawab 'no'. Kiriman Weibo berkata: {text}. Sekarang kiriman Weibo tamat. The answer is:",

        # Lao
        "ຖ້າໂພສ Weibo ກ່າວເຖິງປະໂຫຍດທີ່ຮັບຮູ້ຂອງການສິ້ນຊີວິດວັກຊີນ HPV ແລະໄດ້ພົງໄພໃນພວມເຊື່ອຖືຂອງວັກຊີນ HPV ໃນການປົກປ້ອງບາງປະສົບການທີ່ມີຄວາມຊື່ມຊອມຂອງ HPV ແລະຄວາມເຂັ້ມຂອງເຊື້ອທາງການແກກາກ່ຽວເນັ້ນໂຕປົກກັນພາຍໃຕ້ການສົ່ງເພງໃນໄລຍະທີ່ພວມເປັນໄປສົມບູນໂພມຕອບວ່າ 'yes' ຖ້າບໍ່ມີຄວາມເສົາດາບໍ່ຕອບວ່າ 'no' ໂພສ Weibo ກ່າວວ່າ: {text}. ຕອນນີ້ໂພສ Weibo ສິ້ນສຸດແລ້ວ. The answer is:",

        # Burmese
        "ဤ Weibo ပို့စ်တွင် HPV ကာကွယ်ဆေးပျက်ကွက်မှုမှ အကျိုးကျေးဇူးများကို ပြီးပြည့်စုံစွာ ဖော်ပြခဲ့ပြီး ကျန်းမာရေးကာကွယ်မှုဆိုင်ရာအားဖြင့် HPV ကာကွယ်ဆေး၏ ကျိုးဖြစ်စေမှုကို ယုံကြည်မှုရှိစွာ အထောက်အကူပြုခဲ့သည်။ ဖော်ပြချက်တစ်ခုလုံး၏ အသုံးချမှုတွင် 'yes' ဖြင့် ဖြေပါ၊ ဟုတ်မဟုတ် 'no' ဖြင့် ဖြေပါ။ Weibo ပို့စ်က: {text}. ယခု Weibo ပို့စ်၏အဆုံးသတ်ဖြစ်သည်။ The answer is:",

        # Cebuano
        "Kung ang post sa Weibo naghisgot sa mga benepisyo nga nadawat sa HPV nga bakuna ug gipahayag ang pagsalig sa pagka-epektibo sa HPV nga bakuna, sama sa benepisyo o pagka-epektibo sa HPV nga bakuna sa pagpanalipod batok sa HPV nga impeksyon ug mga kaangay nga cancer, tubaga ug 'yes'. Kung dili, tubaga nga 'no'. Ang Weibo post miingon: {text}. Karun ang Weibo post natapos na. The answer is:",

        # Khmer
        "ប្រសិនបើប្រកាស Weibo បាននិយាយអំពីអត្ថប្រយោជន៍ដែលទទួលបានពីការចាក់វ៉ាក់សាំង HPV និងបង្ហាញពីក្តីទុកចិត្តលើប្រសិទ្ធភាពនៃការចាក់វ៉ាក់សាំង HPV ដូចជាអត្ថប្រយោជន៍ឬប្រសិទ្ធភាពនៃវ៉ាក់សាំង HPV ក្នុងការការពារចំពោះការឆ្លងមេរោគ HPV និងមហារីកដែលពាក់ព័ន្ធសូមឆ្លើយថា 'yes'. ប្រសិនបើមិនមែនទេសូមឆ្លើយថា 'no'. ប្រកាស Weibo បាននិយាយថា {text}. ឥឡូវប្រកាសបានបញ្ចប់។ The answer is:",

        # Tagalog
        "Kung binabanggit ng post sa Weibo ang mga pakinabang na nadarama ng pagbabakuna ng HPV at nagpapahayag ng kumpiyansa sa bisa ng pagbabakuna ng HPV, tulad ng pakinabang o bisa ng bakuna sa HPV sa proteksyon laban sa impeksyon sa HPV at mga kaugnay na kanser, sagutin ang 'yes'. Kung hindi, sagutin ang 'no'. Ang post sa Weibo ay nagsabi: {text}. Ngayon natapos na ang post sa Weibo. The answer is:",

        # Hindi
        "यदि Weibo पोस्ट में HPV टीकाकरण के संभावित लाभों का उल्लेख किया गया है और HPV टीकाकरण की प्रभावकारिता में विश्वास व्यक्त किया गया है, जैसे कि HPV संक्रमण और संबंधित कैंसर से बचाव में HPV टीके का लाभ या प्रभावकारिता, तो 'yes' में उत्तर दें। अन्यथा, 'no' में उत्तर दें। Weibo पोस्ट ने कहा: {text}. अब Weibo पोस्ट समाप्त होता है। The answer is:",

        # Bengali
        "যদি Weibo পোস্টে HPV টিকাদানের অনুভূত সুবিধাগুলি উল্লেখ করা হয় এবং HPV টিকাদানের কার্যকারিতার প্রতি আস্থা প্রকাশ করা হয়, যেমন HPV সংক্রমণ এবং প্রাসঙ্গিক ক্যান্সার থেকে সুরক্ষার ক্ষেত্রে HPV টিকার সুবিধা বা কার্যকারিতা, তবে 'yes' উত্তর দিন। অন্যথায়, 'no' উত্তর দিন। Weibo পোস্ট বলেছে: {text}. এখন Weibo পোস্ট শেষ। The answer is:",

        # Urdu
        "اگر Weibo پوسٹ میں HPV ویکسینیشن کے متوقع فوائد کا ذکر کیا گیا ہے اور HPV ویکسینیشن کی تاثیر پر اعتماد کا اظہار کیا گیا ہے، جیسے کہ HPV انفیکشن اور متعلقہ کینسر کے خلاف HPV ویکسین کا فائدہ یا تاثیر، تو 'yes' میں جواب دیں؛ بصورت دیگر 'no' میں جواب دیں۔ Weibo پوسٹ نے کہا: {text}. اب Weibo پوسٹ ختم ہوتی ہے۔ The answer is:",

        # Czech
        "Pokud příspěvek na Weibo zmiňuje vnímané výhody očkování proti HPV a vyjadřuje důvěru v účinnost očkování proti HPV, jako jsou výhody nebo účinnost vakcíny proti HPV při ochraně před infekcí HPV a souvisejícími rakovinami, odpovězte 'yes'. Jinak odpovězte 'no'. Příspěvek na Weibo říkal: {text}. Nyní příspěvek na Weibo končí. Odpověď je:",

        # Polish
        "Jeśli post na Weibo wspomina o postrzeganych korzyściach ze szczepienia przeciwko HPV i wyraża zaufanie do skuteczności szczepionki przeciwko HPV, na przykład o korzyściach lub skuteczności szczepionki HPV w ochronie przed zakażeniem HPV i odpowiednimi nowotworami, odpowiedz 'yes'. W przeciwnym razie odpowiedz 'no'. Post na Weibo brzmiał: {text}. Teraz post na Weibo się kończy. Odpowiedź brzmi:",
],

    "Perceived barriers to accepting vaccines (-)": [
        # English
        "If the Weibo post mentions perceived barriers to accepting vaccines, including safety issues, side effects, discomfort, or lack of confidence in the safety and efficacy of the HPV vaccine, respond 'yes'. Otherwise, respond 'no'. The Weibo post said: {text}. Now the Weibo post ends. The answer is:",

        # Chinese (Simplified)
        "如果微博帖子提到接种疫苗的障碍，包括安全问题、副作用、不适，或对HPV疫苗安全性和有效性的信心不足，请回答 'yes'。否则，请回答 'no'。微博内容如下：{text}。现在微博内容结束。The answer is:",

        # German
        "Wenn der Weibo-Beitrag wahrgenommene Hindernisse für die Akzeptanz von Impfstoffen, einschließlich Sicherheitsprobleme, Nebenwirkungen, Unbehagen oder mangelndes Vertrauen in die Sicherheit und Wirksamkeit des HPV-Impfstoffs, erwähnt, antworten Sie mit 'yes'. Andernfalls antworten Sie mit 'no'. Der Weibo-Beitrag lautete: {text}. Jetzt endet der Weibo-Beitrag. The answer is:",

        # French
        "Si le post Weibo mentionne des obstacles perçus à l'acceptation des vaccins, y compris des problèmes de sécurité, des effets secondaires, de l'inconfort ou un manque de confiance dans la sécurité et l'efficacité du vaccin contre le VPH, répondez 'yes'. Sinon, répondez 'no'. Le post Weibo disait : {text}. Maintenant, le post Weibo est terminé. The answer is:",

        # Spanish
        "Si la publicación de Weibo menciona barreras percibidas para aceptar vacunas, incluidos problemas de seguridad, efectos secundarios, malestar o falta de confianza en la seguridad y eficacia de la vacuna contra el VPH, responda 'yes'. De lo contrario, responda 'no'. La publicación de Weibo decía: {text}. Ahora termina la publicación de Weibo. The answer is:",

        # Portuguese
        "Se a postagem no Weibo menciona barreiras percebidas para aceitar vacinas, incluindo problemas de segurança, efeitos colaterais, desconforto ou falta de confiança na segurança e eficácia da vacina contra o HPV, responda 'yes'. Caso contrário, responda 'no'. A postagem no Weibo disse: {text}. Agora a postagem no Weibo termina. The answer is:",

        # Italian
        "Se il post su Weibo menziona ostacoli percepiti all'accettazione dei vaccini, inclusi problemi di sicurezza, effetti collaterali, disagio o mancanza di fiducia nella sicurezza e nell'efficacia del vaccino HPV, rispondi 'yes'. Altrimenti, rispondi 'no'. Il post di Weibo diceva: {text}. Ora il post su Weibo finisce. The answer is:",

        # Dutch
        "Als de Weibo-post waargenomen barrières noemt voor het accepteren van vaccins, waaronder veiligheidsproblemen, bijwerkingen, ongemak of gebrek aan vertrouwen in de veiligheid en effectiviteit van het HPV-vaccin, antwoord dan met 'yes'. Anders antwoord met 'no'. De Weibo-post zei: {text}. Nu eindigt de Weibo-post. The answer is:",

        # Russian
        "Если в посте на Weibo упоминаются предполагаемые барьеры для принятия вакцин, включая проблемы безопасности, побочные эффекты, дискомфорт или отсутствие уверенности в безопасности и эффективности вакцины против ВПЧ, ответьте «yes». В противном случае ответьте «no». Пост в Weibo сказал: {text}. Теперь пост в Weibo заканчивается. The answer is:",

        # Arabic
        "إذا ذكر المنشور على Weibo العوائق المتصورة لقبول اللقاحات، بما في ذلك مشاكل السلامة، الآثار الجانبية، الانزعاج، أو عدم الثقة في سلامة وفعالية لقاح HPV، فاستجب بـ 'yes'. خلاف ذلك، استجب بـ 'no'. قال منشور Weibo: {text}. الآن ينتهي منشور Weibo. The answer is:",

        # Persian
        "اگر در پست Weibo به موانع ادراک‌شده برای پذیرش واکسن‌ها، از جمله مسائل ایمنی، عوارض جانبی، ناراحتی، یا کمبود اطمینان به ایمنی و کارایی واکسن HPV اشاره شده است، با «yes» پاسخ دهید. در غیر این صورت، پاسخ «no» دهید. پست Weibo گفت: {text}. اکنون پست Weibo به پایان می‌رسد. The answer is:",

        # Hebrew
        "אם הפוסט ב-Weibo מזכיר מכשולים נתפסים לקבלת חיסונים, כולל בעיות בטיחות, תופעות לוואי, חוסר נוחות או חוסר אמון בבטיחות וביעילות של חיסון ה-HPV, ענה 'yes'. אחרת, ענה 'no'. הפוסט ב-Weibo אמר: {text}. עכשיו הפוסט ב-Weibo מסתיים. The answer is:",

        # Turkish
        "Weibo gönderisi, aşıların kabulü için algılanan engelleri, güvenlik sorunları, yan etkiler, rahatsızlık veya HPV aşısının güvenliği ve etkinliğine olan güven eksikliği de dahil olmak üzere, bahsediyorsa, 'yes' yanıtlayın. Aksi takdirde 'no' yanıtlayın. Weibo gönderisi dedi ki: {text}. Şimdi Weibo gönderisi bitiyor. The answer is:",

        # Japanese
        "Weibo投稿がワクチン受け入れの障壁として認識されるもの、たとえば安全性の問題、副作用、不快感、またはHPVワクチンの安全性と有効性に対する信頼の欠如について言及している場合、「yes」と答えてください。それ以外の場合は、「no」と答えてください。Weibo投稿には次のように書かれていました: {text}。今、Weibo投稿が終了します。The answer is:",

        # Korean
        "Weibo 게시물이 백신 수용에 대한 인식된 장벽, 안전성 문제, 부작용, 불편함 또는 HPV 백신의 안전성과 효능에 대한 자신감 부족을 언급하는 경우 'yes'로 답변하십시오. 그렇지 않으면 'no'로 답변하십시오. Weibo 게시물에는 다음과 같이 말했습니다: {text}. 이제 Weibo 게시물이 끝납니다. The answer is:",

        # Vietnamese
        "Nếu bài đăng trên Weibo đề cập đến những rào cản được nhận thức trong việc chấp nhận vắc-xin, bao gồm các vấn đề về an toàn, tác dụng phụ, sự khó chịu hoặc thiếu tự tin vào sự an toàn và hiệu quả của vắc-xin HPV, hãy trả lời 'yes'. Nếu không, hãy trả lời 'no'. Bài đăng Weibo đã nói: {text}. Bây giờ bài đăng Weibo kết thúc. The answer is:",

        # Thai
        "หากโพสต์ Weibo กล่าวถึงอุปสรรคที่รับรู้ในการยอมรับวัคซีน รวมถึงปัญหาด้านความปลอดภัย ผลข้างเคียง ความรู้สึกไม่สบาย หรือความไม่มั่นใจในความปลอดภัยและประสิทธิภาพของวัคซีน HPV ให้ตอบว่า 'yes' มิฉะนั้น ให้ตอบว่า 'no'. โพสต์ Weibo กล่าวว่า: {text}. ตอนนี้โพสต์ Weibo สิ้นสุดลงแล้ว. The answer is:",

        # Indonesian
        "Jika postingan Weibo menyebutkan hambatan yang dirasakan untuk menerima vaksin, termasuk masalah keamanan, efek samping, ketidaknyamanan, atau kurangnya kepercayaan pada keamanan dan kemanjuran vaksin HPV, jawab 'yes'. Jika tidak, jawab 'no'. Postingan Weibo mengatakan: {text}. Sekarang postingan Weibo berakhir. The answer is:",

        # Malay
        "Jika kiriman Weibo menyebutkan halangan yang dirasai untuk menerima vaksin, termasuk isu keselamatan, kesan sampingan, ketidakselesaan, atau kekurangan keyakinan terhadap keselamatan dan keberkesanan vaksin HPV, jawab 'yes'. Jika tidak, jawab 'no'. Kiriman Weibo berkata: {text}. Sekarang kiriman Weibo tamat. The answer is:",

        # Lao
        "ຖ້າໂພສ Weibo ກ່າວເຖິງອຸປະສັກທີ່ສັງກັດມີການປັກວັກຊີນຢ່າງປານລະຄືງຄວາມປອດໄພຂອງຄວາມປອດໄພການປິດຕົກຄືການປິດລັດການສົງປະເດືອນວັກຊີນ HPV ແລະຄວາມບົກບືນໃນຄວາມສະທານາດີບັນທັດອົງທີ່ຈະມີຄວາມຂັດທຽມອັນຄວາມວິຕົດທີ່ສົງອຸປະສັກທີ່ຈະຂະຫຍາຍສາມາດຕອບວ່າ 'yes' ບໍ່ແມ່ນທໍາລາກໄດ້ກະລຸນາຕອບວ່າ 'no' ໂພສ Weibo ກ່າວວ່າ: {text}. ຕອນນີ້ໂພສ Weibo ສິ້ນສຸດແລ້ວ. The answer is:",

        # Burmese
        "ဤ Weibo ပို့စ်တွင် မျှော်မှန်းထားသော ကာကွယ်ရေးဆေးပေါင်းများကို လက်ခံရာတွင် သဘောကောင်းမှုမှ ပိတ်ဆို့မှုများပါဝင်သည်မှာ အန္တရာယ်ကင်းရာများကို ရှောင်လွှဲနိုင်သည်ဟုဆိုသည်။ 'yes' ဖြင့် ဖြေပါ၊ မဟုတ်ပါက 'no' ဖြင့် ဖြေပါ။ Weibo ပို့စ်က: {text}. ယခု Weibo ပို့စ်၏အဆုံးသတ်ဖြစ်သည်။ The answer is:",

        # Cebuano
        "Kung ang post sa Weibo naghisgot sa mga babag nga nadawat sa pagdawat sa bakuna, lakip na ang mga isyu sa kahilwasan, mga epekto, kakulian, o kakulang sa pagsalig sa kaluwasan ug pagka-epektibo sa HPV nga bakuna, tubaga ug 'yes'. Kung dili, tubaga nga 'no'. Ang Weibo post miingon: {text}. Karun ang Weibo post natapos na. The answer is:",

        # Khmer
        "ប្រសិនបើប្រកាស Weibo បាននិយាយអំពីឧបសគ្គដែលត្រូវបានយល់ព្រមចំពោះការទទួលយកវ៉ាក់សាំង រួមមានបញ្ហាសុវត្ថិភាព ផលប៉ះពាល់ ភាពមិនសប្បាយ រឺការខ្វះនៃការទុកចិត្តលើសុវត្ថិភាពនិងប្រសិទ្ធភាពនៃវ៉ាក់សាំង HPV សូមឆ្លើយថា 'yes'. ប្រសិនបើមិនមែនទេសូមឆ្លើយថា 'no'. ប្រកាស Weibo បាននិយាយថា {text}. ឥឡូវប្រកាសបានបញ្ចប់។ The answer is:",

        # Tagalog
        "Kung binabanggit ng post sa Weibo ang mga hadlang na nadarama sa pagtanggap ng mga bakuna, kabilang ang mga isyu sa kaligtasan, mga side effect, pagkadiskomportable, o kawalan ng kumpiyansa sa kaligtasan at bisa ng HPV na bakuna, sagutin ang 'yes'. Kung hindi, sagutin ang 'no'. Ang post sa Weibo ay nagsabi: {text}. Ngayon natapos na ang post sa Weibo. The answer is:",

        # Hindi
        "यदि Weibo पोस्ट में टीकों को स्वीकार करने में मानी जाने वाली बाधाओं का उल्लेख किया गया है, जिसमें सुरक्षा मुद्दे, दुष्प्रभाव, असुविधा या HPV टीके की सुरक्षा और प्रभावकारिता में विश्वास की कमी शामिल है, तो 'yes' में उत्तर दें। अन्यथा, 'no' में उत्तर दें। Weibo पोस्ट ने कहा: {text}. अब Weibo पोस्ट समाप्त होता है। The answer is:",

        # Bengali
        "যদি Weibo পোস্টে টিকা গ্রহণে বাধা হিসাবে বিবেচিত বিষয়গুলি উল্লেখ করা হয়, যার মধ্যে নিরাপত্তা সমস্যাগুলি, পার্শ্বপ্রতিক্রিয়া, অস্বস্তি বা HPV টিকার নিরাপত্তা এবং কার্যকারিতা নিয়ে আত্মবিশ্বাসের অভাব অন্তর্ভুক্ত থাকে, তবে 'yes' উত্তর দিন। অন্যথায়, 'no' উত্তর দিন। Weibo পোস্ট বলেছে: {text}. এখন Weibo পোস্ট শেষ। The answer is:",

        # Urdu
        "اگر Weibo پوسٹ میں ویکسین قبول کرنے میں متوقع رکاوٹوں کا ذکر ہے، بشمول حفاظتی مسائل، مضر اثرات، عدم اطمینان، یا HPV ویکسین کی حفاظت اور افادیت پر اعتماد کی کمی، تو 'yes' میں جواب دیں؛ بصورت دیگر 'no' میں جواب دیں۔ Weibo پوسٹ نے کہا: {text}. اب Weibo پوسٹ ختم ہوتی ہے۔ The answer is:",

        # Czech
        "Pokud příspěvek na Weibo zmiňuje vnímané překážky při přijímání vakcín, včetně bezpečnostních problémů, vedlejších účinků, nepohodlí nebo nedostatku důvěry v bezpečnost a účinnost vakcíny proti HPV, odpovězte 'yes'. Jinak odpovězte 'no'. Příspěvek na Weibo říkal: {text}. Nyní příspěvek na Weibo končí. Odpověď je:",

        # Polish
        "Jeśli post na Weibo wspomina o postrzeganych barierach w akceptacji szczepionek, w tym o kwestiach bezpieczeństwa, skutkach ubocznych, dyskomforcie lub braku zaufania do bezpieczeństwa i skuteczności szczepionki przeciwko HPV, odpowiedz 'yes'. W przeciwnym razie odpowiedz 'no'. Post na Weibo brzmiał: {text}. Teraz post na Weibo się kończy. Odpowiedź brzmi:",
],

    "Practical barriers to vaccination (-)": [
        # English
        "If the post mentions practical barriers to HPV vaccination, such as being busy, lack of vaccine supply, inconvenience, high cost, distance to vaccination sites, scheduling conflicts, or poor quality of service, respond 'yes'. Otherwise, respond 'no'. The Weibo post said: {text}. Now the Weibo post ends. The answer is:",

        # Chinese (Simplified)
        "如果帖子提到接种HPV疫苗的实际障碍，例如忙碌、疫苗供应不足、不便、高成本、距离接种地点远、时间冲突或服务质量差，请回答 'yes'。否则，请回答 'no'。微博内容如下：{text}。现在微博内容结束。The answer is:",

        # German
        "Wenn der Beitrag praktische Hindernisse für die HPV-Impfung erwähnt, wie zum Beispiel Zeitmangel, fehlende Impfstoffversorgung, Unannehmlichkeiten, hohe Kosten, Entfernung zu Impfstätten, Terminkonflikte oder schlechte Servicequalität, antworten Sie mit 'yes'. Andernfalls antworten Sie mit 'no'. Der Weibo-Beitrag lautete: {text}. Jetzt endet der Weibo-Beitrag. The answer is:",

        # French
        "Si le post mentionne des obstacles pratiques à la vaccination contre le VPH, comme être occupé, le manque de disponibilité des vaccins, l'inconvénient, le coût élevé, la distance des sites de vaccination, les conflits d'horaires ou la mauvaise qualité des services, répondez 'yes'. Sinon, répondez 'no'. Le post Weibo disait : {text}. Maintenant, le post Weibo est terminé. The answer is:",

        # Spanish
        "Si la publicación menciona barreras prácticas para la vacunación contra el VPH, como estar ocupado, falta de suministro de vacunas, inconvenientes, alto costo, distancia a los sitios de vacunación, conflictos de programación o mala calidad del servicio, responda 'yes'. De lo contrario, responda 'no'. La publicación de Weibo decía: {text}. Ahora termina la publicación de Weibo. The answer is:",

        # Portuguese
        "Se a postagem menciona barreiras práticas para a vacinação contra o HPV, como estar ocupado, falta de fornecimento de vacinas, inconveniência, alto custo, distância aos locais de vacinação, conflitos de agenda ou má qualidade do serviço, responda 'yes'. Caso contrário, responda 'no'. A postagem no Weibo disse: {text}. Agora a postagem no Weibo termina. The answer is:",

        # Italian
        "Se il post menziona ostacoli pratici alla vaccinazione contro l'HPV, come essere occupati, mancanza di fornitura di vaccini, inconvenienti, costi elevati, distanza dai centri vaccinali, conflitti di orario o scarsa qualità del servizio, rispondi 'yes'. Altrimenti, rispondi 'no'. Il post di Weibo diceva: {text}. Ora il post su Weibo finisce. The answer is:",

        # Dutch
        "Als de post praktische barrières voor HPV-vaccinatie noemt, zoals druk zijn, gebrek aan vaccinvoorziening, ongemak, hoge kosten, afstand tot vaccinatielocaties, planningsconflicten of slechte servicekwaliteit, antwoord dan met 'yes'. Anders antwoord met 'no'. De Weibo-post zei: {text}. Nu eindigt de Weibo-post. The answer is:",

        # Russian
        "Если в посте упоминаются практические препятствия для вакцинации против ВПЧ, такие как занятость, нехватка вакцины, неудобства, высокая стоимость, удаленность от пунктов вакцинации, конфликтное расписание или низкое качество обслуживания, ответьте «yes». В противном случае ответьте «no». Пост в Weibo сказал: {text}. Теперь пост в Weibo заканчивается. The answer is:",

        # Arabic
        "إذا ذكر المنشور على Weibo عوائق عملية أمام تطعيم HPV، مثل الانشغال، نقص إمدادات اللقاحات، عدم الراحة، التكلفة العالية، البعد عن مواقع التطعيم، تعارض الجداول الزمنية، أو سوء جودة الخدمة، فاستجب بـ 'yes'. خلاف ذلك، استجب بـ 'no'. قال منشور Weibo: {text}. الآن ينتهي منشور Weibo. The answer is:",

        # Persian
        "اگر در پست Weibo به موانع عملی واکسیناسیون HPV، مانند مشغول بودن، کمبود عرضه واکسن، ناراحتی، هزینه بالا، فاصله زیاد تا مراکز واکسیناسیون، تداخل زمان‌بندی یا کیفیت پایین خدمات اشاره شده است، با «yes» پاسخ دهید. در غیر این صورت، پاسخ «no» دهید. پست Weibo گفت: {text}. اکنون پست Weibo به پایان می‌رسد. The answer is:",

        # Hebrew
        "אם הפוסט ב-Weibo מזכיר מכשולים מעשיים לחיסון ה-HPV, כמו להיות עסוק, מחסור באספקת חיסונים, אי נוחות, עלות גבוהה, מרחק מאתרי החיסון, התנגשויות בלוחות הזמנים או איכות שירות ירודה, ענה 'yes'. אחרת, ענה 'no'. הפוסט ב-Weibo אמר: {text}. עכשיו הפוסט ב-Weibo מסתיים. The answer is:",

        # Turkish
        "Gönderide HPV aşısına yönelik pratik engellerden, örneğin meşgul olma, aşı arzının yetersizliği, rahatsızlık, yüksek maliyet, aşılama yerlerine uzaklık, zamanlama çatışmaları veya hizmet kalitesinin düşüklüğünden bahsediliyorsa, 'yes' yanıtlayın. Aksi takdirde 'no' yanıtlayın. Weibo gönderisi dedi ki: {text}. Şimdi Weibo gönderisi bitiyor. The answer is:",

        # Japanese
        "投稿でHPVワクチン接種に対する実際の障壁、たとえば多忙、ワクチン供給の不足、不便さ、高額な費用、ワクチン接種会場までの距離、スケジュールの競合、またはサービスの質の低さが言及されている場合、「yes」と答えてください。それ以外の場合は、「no」と答えてください。Weibo投稿には次のように書かれていました: {text}。今、Weibo投稿が終了します。The answer is:",

        # Korean
        "게시물에서 바쁜 것, 백신 공급 부족, 불편함, 높은 비용, 접종 장소까지의 거리, 일정 충돌, 서비스 품질 저하 등 HPV 백신 접종에 대한 실질적인 장벽을 언급하는 경우 'yes'로 답변하십시오. 그렇지 않으면 'no'로 답변하십시오. Weibo 게시물에는 다음과 같이 말했습니다: {text}. 이제 Weibo 게시물이 끝납니다. The answer is:",

        # Vietnamese
        "Nếu bài đăng đề cập đến những rào cản thực tế đối với việc tiêm vắc-xin HPV, chẳng hạn như bận rộn, thiếu nguồn cung vắc-xin, bất tiện, chi phí cao, khoảng cách đến các địa điểm tiêm chủng, xung đột lịch trình hoặc chất lượng dịch vụ kém, hãy trả lời 'yes'. Nếu không, hãy trả lời 'no'. Bài đăng Weibo đã nói: {text}. Bây giờ bài đăng Weibo kết thúc. The answer is:",

        # Thai
        "หากโพสต์กล่าวถึงอุปสรรคในการฉีดวัคซีน HPV เช่น การยุ่ง การขาดแคลนอุปทานวัคซีน ความไม่สะดวก ต้นทุนสูง ระยะทางไปยังสถานที่ฉีดวัคซีน ความขัดแย้งในการจัดตารางเวลา หรือคุณภาพการบริการที่ไม่ดี ให้ตอบว่า 'yes' มิฉะนั้น ให้ตอบว่า 'no'. โพสต์ Weibo กล่าวว่า: {text}. ตอนนี้โพสต์ Weibo สิ้นสุดลงแล้ว. The answer is:",

        # Indonesian
        "Jika postingan menyebutkan hambatan praktis untuk vaksinasi HPV, seperti sibuk, kurangnya pasokan vaksin, ketidaknyamanan, biaya tinggi, jarak ke tempat vaksinasi, konflik jadwal, atau kualitas layanan yang buruk, jawab 'yes'. Jika tidak, jawab 'no'. Postingan Weibo mengatakan: {text}. Sekarang postingan Weibo berakhir. The answer is:",

        # Malay
        "Jika kiriman menyebutkan halangan praktikal untuk vaksinasi HPV, seperti sibuk, kekurangan bekalan vaksin, ketidakselesaan, kos tinggi, jarak ke lokasi vaksinasi, konflik penjadualan, atau kualiti perkhidmatan yang rendah, jawab 'yes'. Jika tidak, jawab 'no'. Kiriman Weibo berkata: {text}. Sekarang kiriman Weibo tamat. The answer is:",

        # Lao
        "ຖ້າໂພສບອກວ່າຂົງເຂດຂອງວັກຊີນ HPV ເຫລົ່ານີ້ມີຄວາມຄິດຄົມຂອງການສໍາລັບປະກອບດ້ານການຈັດຕາລາງຄວາມໄວລົງຕະຄົມຄືການຈັດຕາລາງສົ່ງເພື່ອລະຫວ່າງວິທະຍາການການທໍາລາດປະເດືອນແລະຄວາມວິຕົດຂົງເຂດຂອງທີ່ໃນຂະຫຍາຍຕອບວ່າ 'yes' ບໍ່ແມ່ນທໍາລາກໄດ້ກະລຸນາຕອບວ່າ 'no' ໂພສ Weibo ກ່າວວ່າ: {text}. ຕອນນີ້ໂພສ Weibo ສິ້ນສຸດແລ້ວ. The answer is:",

        # Burmese
        "ဤ Weibo ပို့စ်တွင် HPV ကာကွယ်ဆေးကို အချိန်လျော်လည်တည်ဆောက်မှုဖြင့် အကျုံးဝင်မှုကိန်းကျင့်မှုဖြင့် ကာကွယ်နိုင်သော ဘာရည်မှန်းချက်များစွာကို ပြီးပြည့်စုံစွာ သို့မဟုတ် အကျိုးဖြစ်စေမှုနှင့်အတူ ဖော်ပြသည်။ 'yes' ဖြင့် ဖြေပါ၊ မဟုတ်ပါက 'no' ဖြင့် ဖြေပါ။ Weibo ပို့စ်က: {text}. ယခု Weibo ပို့စ်၏အဆုံးသတ်ဖြစ်သည်။ The answer is:",

        # Cebuano
        "Kung ang post sa Weibo naghisgot sa mga praktikal nga babag sa HPV nga bakuna, sama sa pagkabu-otan, kakulang sa suplay sa bakuna, kakulian, kataas sa gasto, gilay-on sa mga site sa pagbakuna, pagbangga sa iskedyul, o dili maayo nga kalidad sa serbisyo, tubaga ug 'yes'. Kung dili, tubaga nga 'no'. Ang Weibo post miingon: {text}. Karun ang Weibo post natapos na. The answer is:",

        # Khmer
        "ប្រសិនបើប្រកាសបាននិយាយអំពីឧបសគ្គផ្នែកឧត្តមមួយចំនួនក្នុងការចាក់វ៉ាក់សាំង HPV ដូចជាការរវល់ ការខ្វះខាតនៃការផ្គត់ផ្គង់វ៉ាក់សាំង ភាពមិនស្រួល ឬតម្លៃថ្លៃ ការរវល់កាលវិភាគ ឬគុណភាពសេវាកម្មដែលមិនល្អ សូមឆ្លើយថា 'yes'. ប្រសិនបើមិនមែនទេសូមឆ្លើយថា 'no'. ប្រកាស Weibo បាននិយាយថា {text}. ឥឡូវប្រកាសបានបញ្ចប់។ The answer is:",

        # Tagalog
        "Kung binabanggit ng post ang mga praktikal na hadlang sa pagbabakuna ng HPV, tulad ng pagiging abala, kakulangan ng suplay ng bakuna, pagkadiskomportable, mataas na halaga, distansya sa mga lugar ng pagbabakuna, mga salungat na iskedyul, o mahinang kalidad ng serbisyo, sagutin ang 'yes'. Kung hindi, sagutin ang 'no'. Ang post sa Weibo ay nagsabi: {text}. Ngayon natapos na ang post sa Weibo. The answer is:",

        # Hindi
        "यदि पोस्ट में HPV टीकाकरण के लिए व्यावहारिक बाधाओं का उल्लेख किया गया है, जैसे व्यस्तता, टीके की आपूर्ति की कमी, असुविधा, उच्च लागत, टीकाकरण स्थलों की दूरी, समय निर्धारण में बाधाएँ, या सेवा की खराब गुणवत्ता, तो 'yes' में उत्तर दें। अन्यथा, 'no' में उत्तर दें। Weibo पोस्ट ने कहा: {text}. अब Weibo पोस्ट समाप्त होता है। The answer is:",

        # Bengali
        "যদি পোস্টে HPV টিকাদানের জন্য ব্যবহারিক বাধাগুলির উল্লেখ করা হয়, যেমন ব্যস্ততা, ভ্যাকসিন সরবরাহের অভাব, অসুবিধা, উচ্চ খরচ, টিকাদান সাইটের দূরত্ব, সময় নির্ধারণের সংঘাত, বা পরিষেবার নিম্নমান, তবে 'yes' উত্তর দিন। অন্যথায়, 'no' উত্তর দিন। Weibo পোস্ট বলেছে: {text}. এখন Weibo পোস্ট শেষ। The answer is:",

        # Urdu
        "اگر پوسٹ میں HPV ویکسینیشن کے لئے عملی رکاوٹوں کا ذکر ہے، جیسے کہ مصروفیت، ویکسین کی فراہمی کی کمی، تکلیف، اعلی قیمت، ویکسینیشن مقامات تک فاصلہ، شیڈولنگ میں رکاوٹیں، یا خدمت کی کم معیار، تو 'yes' میں جواب دیں؛ بصورت دیگر 'no' میں جواب دیں۔ Weibo پوسٹ نے کہا: {text}. اب Weibo پوسٹ ختم ہوتی ہے۔ The answer is:",

        # Czech
        "Pokud příspěvek zmiňuje praktické překážky očkování proti HPV, jako jsou zaneprázdněnost, nedostatek vakcín, nepohodlí, vysoké náklady, vzdálenost k očkovacím místům, časové konflikty nebo špatná kvalita služeb, odpovězte 'yes'. Jinak odpovězte 'no'. Příspěvek na Weibo říkal: {text}. Nyní příspěvek na Weibo končí. Odpověď je:",

        # Polish
        "Jeśli post wspomina o praktycznych przeszkodach w szczepieniu przeciwko HPV, takich jak bycie zajętym, brak dostaw szczepionki, niedogodności, wysokie koszty, odległość do punktów szczepień, konflikty harmonogramu lub niska jakość usług, odpowiedz 'yes'. W przeciwnym razie odpowiedz 'no'. Post na Weibo brzmiał: {text}. Teraz post na Weibo się kończy. Odpowiedź brzmi:",
],

    "Misinformation (-)": [
        # English
        "If the post contains false, inaccurate, or negative information about HPV infection and vaccines, such as rumors, anti-vaccine messages, vaccine inefficacy, alternative medicine, civil liberties, conspiracy theories, falsehoods, or negative reports, respond 'yes'. Otherwise, respond 'no'. The Weibo post said: {text}. Now the Weibo post ends. The answer is:",

        # Chinese (Simplified)
        "如果帖子包含有关HPV感染和疫苗的虚假、不准确或负面信息，例如谣言、反疫苗言论、疫苗无效性、替代医学、公民自由、阴谋论、虚假信息或负面报告，请回答 'yes'。否则，请回答 'no'。微博内容如下：{text}。现在微博内容结束。The answer is:",

        # German
        "Wenn der Beitrag falsche, ungenaue oder negative Informationen über HPV-Infektionen und Impfstoffe enthält, wie zum Beispiel Gerüchte, Anti-Impf-Nachrichten, Impfstoffineffizienz, alternative Medizin, Bürgerrechte, Verschwörungstheorien, Unwahrheiten oder negative Berichte, antworten Sie mit 'yes'. Andernfalls antworten Sie mit 'no'. Der Weibo-Beitrag lautete: {text}. Jetzt endet der Weibo-Beitrag. The answer is:",

        # French
        "Si le post contient des informations fausses, inexactes ou négatives sur l'infection au VPH et les vaccins, telles que des rumeurs, des messages anti-vaccins, l'inefficacité des vaccins, la médecine alternative, les libertés civiles, les théories du complot, des mensonges ou des rapports négatifs, répondez 'yes'. Sinon, répondez 'no'. Le post Weibo disait : {text}. Maintenant, le post Weibo est terminé. The answer is:",

        # Spanish
        "Si la publicación contiene información falsa, inexacta o negativa sobre la infección por VPH y las vacunas, como rumores, mensajes antivacunas, ineficacia de las vacunas, medicina alternativa, libertades civiles, teorías conspirativas, falsedades o informes negativos, responda 'yes'. De lo contrario, responda 'no'. La publicación de Weibo decía: {text}. Ahora termina la publicación de Weibo. The answer is:",

        # Portuguese
        "Se a postagem contém informações falsas, imprecisas ou negativas sobre a infecção por HPV e vacinas, como boatos, mensagens antivacinas, ineficácia da vacina, medicina alternativa, liberdades civis, teorias da conspiração, inverdades ou relatos negativos, responda 'yes'. Caso contrário, responda 'no'. A postagem no Weibo disse: {text}. Agora a postagem no Weibo termina. The answer is:",

        # Italian
        "Se il post contiene informazioni false, imprecise o negative sull'infezione da HPV e sui vaccini, come voci, messaggi anti-vaccino, inefficacia del vaccino, medicina alternativa, libertà civili, teorie del complotto, falsità o rapporti negativi, rispondi 'yes'. Altrimenti, rispondi 'no'. Il post di Weibo diceva: {text}. Ora il post su Weibo finisce. The answer is:",

        # Dutch
        "Als de post onjuiste, onnauwkeurige of negatieve informatie bevat over HPV-infectie en vaccins, zoals geruchten, anti-vaccin berichten, vaccin ineffectiviteit, alternatieve geneeskunde, burgerlijke vrijheden, complottheorieën, onwaarheden of negatieve rapporten, antwoord dan met 'yes'. Anders antwoord met 'no'. De Weibo-post zei: {text}. Nu eindigt de Weibo-post. The answer is:",

        # Russian
        "Если пост содержит ложную, неточную или негативную информацию о ВПЧ-инфекции и вакцинах, такую как слухи, сообщения против вакцинации, неэффективность вакцины, альтернативную медицину, гражданские свободы, теории заговора, ложные сведения или негативные отчеты, ответьте «yes». В противном случае ответьте «no». Пост в Weibo сказал: {text}. Теперь пост в Weibo заканчивается. The answer is:",

        # Arabic
        "إذا كان المنشور يحتوي على معلومات خاطئة أو غير دقيقة أو سلبية حول عدوى فيروس الورم الحليمي البشري واللقاحات، مثل الشائعات، الرسائل المضادة للقاحات، عدم فعالية اللقاح، الطب البديل، الحريات المدنية، نظريات المؤامرة، الأكاذيب أو التقارير السلبية، فاستجب بـ 'yes'. خلاف ذلك، استجب بـ 'no'. قال منشور Weibo: {text}. الآن ينتهي منشور Weibo. The answer is:",

        # Persian
        "اگر پست حاوی اطلاعات نادرست، نادرست یا منفی درباره عفونت HPV و واکسن‌ها است، مانند شایعات، پیام‌های ضد واکسن، ناکارآمدی واکسن، طب جایگزین، آزادی‌های مدنی، تئوری‌های توطئه، دروغ‌ها یا گزارش‌های منفی، با «yes» پاسخ دهید. در غیر این صورت، پاسخ «no» دهید. پست Weibo گفت: {text}. اکنون پست Weibo به پایان می‌رسد. The answer is:",

        # Hebrew
        "אם הפוסט מכיל מידע שקרי, לא מדויק או שלילי על זיהום HPV וחיסונים, כגון שמועות, מסרי אנטי-חיסונים, חוסר יעילות של החיסון, רפואה אלטרנטיבית, זכויות אזרח, תאוריות קונספירציה, שקרים או דוחות שליליים, ענה 'yes'. אחרת, ענה 'no'. הפוסט ב-Weibo אמר: {text}. עכשיו הפוסט ב-Weibo מסתיים. The answer is:",

        # Turkish
        "Gönderi, HPV enfeksiyonu ve aşılar hakkında yanlış, hatalı veya olumsuz bilgiler, örneğin söylentiler, aşı karşıtı mesajlar, aşının etkisizliği, alternatif tıp, sivil özgürlükler, komplo teorileri, yalanlar veya olumsuz raporlar içeriyorsa, 'yes' yanıtlayın. Aksi takdirde 'no' yanıtlayın. Weibo gönderisi dedi ki: {text}. Şimdi Weibo gönderisi bitiyor. The answer is:",

        # Japanese
        "投稿にHPV感染症やワクチンに関する虚偽、不正確、または否定的な情報、たとえば噂、反ワクチンメッセージ、ワクチンの無効性、代替医療、公民権、陰謀論、虚偽または否定的な報告が含まれている場合、「yes」と答えてください。それ以外の場合は、「no」と答えてください。Weibo投稿には次のように書かれていました: {text}。今、Weibo投稿が終了します。The answer is:",

        # Korean
        "게시물에 HPV 감염 및 백신에 대한 허위, 부정확하거나 부정적인 정보, 예를 들어 소문, 백신 반대 메시지, 백신 비효율성, 대체 의학, 시민 자유, 음모 이론, 거짓말 또는 부정적인 보고가 포함되어 있으면 'yes'로 답변하십시오. 그렇지 않으면 'no'로 답변하십시오. Weibo 게시물에는 다음과 같이 말했습니다: {text}. 이제 Weibo 게시물이 끝납니다. The answer is:",

        # Vietnamese
        "Nếu bài đăng chứa thông tin sai lệch, không chính xác hoặc tiêu cực về nhiễm trùng HPV và vắc-xin, chẳng hạn như tin đồn, thông điệp chống vắc-xin, sự kém hiệu quả của vắc-xin, y học thay thế, quyền tự do dân sự, thuyết âm mưu, thông tin sai lệch hoặc báo cáo tiêu cực, hãy trả lời 'yes'. Nếu không, hãy trả lời 'no'. Bài đăng Weibo đã nói: {text}. Bây giờ bài đăng Weibo kết thúc. The answer is:",

        # Thai
        "หากโพสต์มีข้อมูลที่ผิด ไม่ถูกต้อง หรือเป็นลบเกี่ยวกับการติดเชื้อ HPV และวัคซีน เช่น ข่าวลือ ข้อความต่อต้านวัคซีน ประสิทธิภาพของวัคซีนต่ำ การแพทย์ทางเลือก สิทธิเสรีภาพทางแพ่ง ทฤษฎีสมคบคิด ข้อเท็จจริงที่ผิดพลาด หรือรายงานเชิงลบ ให้ตอบว่า 'yes' มิฉะนั้น ให้ตอบว่า 'no'. โพสต์ Weibo กล่าวว่า: {text}. ตอนนี้โพสต์ Weibo สิ้นสุดลงแล้ว. The answer is:",

        # Indonesian
        "Jika postingan tersebut berisi informasi palsu, tidak akurat, atau negatif tentang infeksi HPV dan vaksin, seperti rumor, pesan anti-vaksin, ketidakefektifan vaksin, pengobatan alternatif, kebebasan sipil, teori konspirasi, kebohongan, atau laporan negatif, jawab 'yes'. Jika tidak, jawab 'no'. Postingan Weibo mengatakan: {text}. Sekarang postingan Weibo berakhir. The answer is:",

        # Malay
        "Jika kiriman tersebut mengandungi maklumat palsu, tidak tepat, atau negatif mengenai jangkitan HPV dan vaksin, seperti khabar angin, mesej anti-vaksin, keberkesanan vaksin yang rendah, perubatan alternatif, kebebasan awam, teori konspirasi, pembohongan, atau laporan negatif, jawab 'yes'. Jika tidak, jawab 'no'. Kiriman Weibo berkata: {text}. Sekarang kiriman Weibo tamat. The answer is:",

        # Lao
        "ຖ້າໂພສນີ້ມີຂໍ້ມູນທີ່ບໍ່ຖືກຕ້ອງ ບໍ່ແມ່ນຄວາມຈິງ ຫລືວ່າບໍ່ຖືກຕ້ອງກ່ຽວກັບການຕິດເຊື້ອ HPV ແລະວັກຊີນເຊັ່ນຂ່າວລື ຂໍ້ຄວາມຕໍ່ຕ້ານວັກຊີນ ຜົນການໃຊ້ວັກຊີນ ການແພທິດທາງເລືອກ ສິດທິດ້ານເສດຖານາການ ການແລ້ວເລີຍວິດຖານ ຫລືຄວາມເປັນປະເດືອນຫລືບໍ່ແມ່ນບໍ່ຈິງກະລຸນາຕອບວ່າ 'yes' ບໍ່ແມ່ນທໍາລາກໄດ້ກະລຸນາຕອບວ່າ 'no' ໂພສ Weibo ກ່າວວ່າ: {text}. ຕອນນີ້ໂພສ Weibo ສິ້ນສຸດແລ້ວ. The answer is:",

        # Burmese
        "ဤ Weibo ပို့စ်တွင် HPV ကာကွယ်ဆေးများနှင့်ပတ်သက်သည့် မွားယွင်းခြင်း၊ မှားယွင်းမှု သို့မဟုတ် အနုတ်အဖျားသော သတင်းအချက်အလက်များ ပါရှိသောပြောဆိုမှုများကို အသုံးပြုသည့် ကျင့်ကြံတည်ဆောက်မှုများပါဝင်ပါက 'yes' ဖြင့် ဖြေပါ၊ မဟုတ်ပါက 'no' ဖြင့် ဖြေပါ။ Weibo ပို့စ်က: {text}. ယခု Weibo ပို့စ်၏အဆုံးသတ်ဖြစ်သည်။ The answer is:",

        # Cebuano
        "Kung ang post naglakip sa sayop, dili tukma, o negatibong impormasyon bahin sa HPV nga impeksyon ug mga bakuna, sama sa mga hungihong, anti-bakuna nga mensahe, wala’y bisa ang bakuna, alternatibo nga medisina, sibil nga kagawasan, teorya sa pagpanlimbong, sayop nga impormasyon, o negatibong taho, tubaga ug 'yes'. Kung dili, tubaga nga 'no'. Ang Weibo post miingon: {text}. Karun ang Weibo post natapos na. The answer is:",

        # Khmer
        "ប្រសិនបើប្រកាសនេះមានព័ត៌មានក្លែងក្លាយ មិនត្រឹមត្រូវ ឬអវិជ្ជមានអំពីការឆ្លងមេរោគ HPV និងវ៉ាក់សាំង ដូចជាអំពើព្រាងអ្វីៗព្យាបាលពេទ្យ ផ្សព្វផ្សាយការបង្កើតឱសថផ្សេងៗ ប្រជាជនភាព ប្រព័ន្ធប្រដាប់ប្រដា សុខភាពបន្សាបឬបើអត់មានការសម្គាល់ពីឱសថផ្សេងៗនោះ សូមឆ្លើយថា 'yes'. បើមិនមែនទេសូមឆ្លើយថា 'no'. ប្រកាស Weibo បាននិយាយថា {text}. ឥឡូវប្រកាសបានបញ្ចប់។ The answer is:",

        # Tagalog
        "Kung naglalaman ang post ng mali, hindi tumpak, o negatibong impormasyon tungkol sa impeksyon ng HPV at mga bakuna, tulad ng mga tsismis, mensaheng kontra-bakuna, hindi epektibong bakuna, alternatibong gamot, mga karapatang sibil, mga teorya ng sabwatan, kasinungalingan, o mga negatibong ulat, sagutin ang 'yes'. Kung hindi, sagutin ang 'no'. Ang post sa Weibo ay nagsabi: {text}. Ngayon natapos na ang post sa Weibo. The answer is:",

        # Hindi
        "यदि पोस्ट में HPV संक्रमण और टीकों के बारे में झूठी, गलत या नकारात्मक जानकारी है, जैसे अफवाहें, एंटी-वैक्सीन संदेश, टीके की विफलता, वैकल्पिक चिकित्सा, नागरिक स्वतंत्रता, साजिश के सिद्धांत, झूठ या नकारात्मक रिपोर्ट, तो 'yes' में उत्तर दें। अन्यथा, 'no' में उत्तर दें। Weibo पोस्ट ने कहा: {text}. अब Weibo पोस्ट समाप्त होता है। The answer is:",

        # Bengali
        "যদি পোস্টে HPV সংক্রমণ এবং টিকা সম্পর্কিত মিথ্যা, ভুল বা নেতিবাচক তথ্য থাকে, যেমন গুজব, অ্যান্টি-ভ্যাকসিন বার্তা, টিকার অকার্যকারিতা, বিকল্প চিকিৎসা, নাগরিক স্বাধীনতা, ষড়যন্ত্র তত্ত্ব, মিথ্যা বা নেতিবাচক প্রতিবেদন, তবে 'yes' উত্তর দিন। অন্যথায়, 'no' উত্তর দিন। Weibo পোস্ট বলেছে: {text}. এখন Weibo পোস্ট শেষ। The answer is:",

        # Urdu
        "اگر پوسٹ میں HPV انفیکشن اور ویکسینز کے بارے میں غلط، غلط یا منفی معلومات ہیں، جیسے افواہیں، اینٹی ویکسین پیغامات، ویکسین کی ناکامی، متبادل طب، شہری آزادیوں، سازشی نظریات، جھوٹ یا منفی رپورٹیں، تو 'yes' میں جواب دیں؛ بصورت دیگر 'no' میں جواب دیں۔ Weibo پوسٹ نے کہا: {text}. اب Weibo پوسٹ ختم ہوتی ہے۔ The answer is:",

        # Czech
        "Pokud příspěvek obsahuje falešné, nepřesné nebo negativní informace o infekci HPV a vakcínách, například fámy, antivakcinační zprávy, neúčinnost vakcín, alternativní medicínu, občanské svobody, konspirační teorie, nepravdy nebo negativní zprávy, odpovězte 'yes'. Jinak odpovězte 'no'. Příspěvek na Weibo říkal: {text}. Nyní příspěvek na Weibo končí. Odpověď je:",

        # Polish
        "Jeśli post zawiera fałszywe, niedokładne lub negatywne informacje na temat zakażeń HPV i szczepionek, takie jak plotki, wiadomości przeciwko szczepionkom, nieskuteczność szczepionek, medycyna alternatywna, swobody obywatelskie, teorie spiskowe, kłamstwa lub negatywne raporty, odpowiedz 'yes'. W przeciwnym razie odpowiedz 'no'. Post na Weibo brzmiał: {text}. Teraz post na Weibo się kończy. Odpowiedź brzmi:",
    ],

    "Social norms / cues to action (+)": [
        # English
        "If the social media post mentions events or information from external sources, such as close others, the media, or health care providers promoting HPV vaccination, respond 'yes'. Otherwise, respond 'no'. The Weibo post said: {text}. Now the Weibo post ends. The answer is:",

        # Chinese (Simplified)
        "如果社交媒体帖子提到来自外部来源的事件或信息，例如亲密的他人、媒体或医疗保健提供者推广HPV疫苗接种，请回答 'yes'。否则，请回答 'no'。微博内容如下：{text}。现在微博内容结束。The answer is:",

        # German
        "Wenn der Social-Media-Beitrag Ereignisse oder Informationen aus externen Quellen erwähnt, wie z. B. enge Bekannte, die Medien oder Gesundheitsdienstleister, die die HPV-Impfung fördern, antworten Sie mit 'yes'. Andernfalls antworten Sie mit 'no'. Der Weibo-Beitrag lautete: {text}. Jetzt endet der Weibo-Beitrag. The answer is:",

        # French
        "Si le post sur les réseaux sociaux mentionne des événements ou des informations provenant de sources externes, telles que des proches, les médias ou des prestataires de soins de santé qui promeuvent la vaccination contre le VPH, répondez 'yes'. Sinon, répondez 'no'. Le post Weibo disait : {text}. Maintenant, le post Weibo est terminé. The answer is:",

        # Spanish
        "Si la publicación en redes sociales menciona eventos o información de fuentes externas, como personas cercanas, los medios de comunicación o proveedores de atención médica que promueven la vacunación contra el VPH, responda 'yes'. De lo contrario, responda 'no'. La publicación de Weibo decía: {text}. Ahora termina la publicación de Weibo. The answer is:",

        # Portuguese
        "Se a postagem nas redes sociais menciona eventos ou informações de fontes externas, como pessoas próximas, a mídia ou prestadores de serviços de saúde que promovem a vacinação contra o HPV, responda 'yes'. Caso contrário, responda 'no'. A postagem no Weibo disse: {text}. Agora a postagem no Weibo termina. The answer is:",

        # Italian
        "Se il post sui social media menziona eventi o informazioni provenienti da fonti esterne, come persone vicine, i media o fornitori di assistenza sanitaria che promuovono la vaccinazione contro l'HPV, rispondi 'yes'. Altrimenti, rispondi 'no'. Il post di Weibo diceva: {text}. Ora il post su Weibo finisce. The answer is:",

        # Dutch
        "Als de sociale mediapost gebeurtenissen of informatie van externe bronnen vermeldt, zoals nauwe anderen, de media of zorgverleners die HPV-vaccinatie promoten, antwoord dan met 'yes'. Anders antwoord met 'no'. De Weibo-post zei: {text}. Nu eindigt de Weibo-post. The answer is:",

        # Russian
        "Если в посте в социальной сети упоминаются события или информация из внешних источников, таких как близкие, СМИ или медицинские работники, которые продвигают вакцинацию против ВПЧ, ответьте «yes». В противном случае ответьте «no». Пост в Weibo сказал: {text}. Теперь пост в Weibo заканчивается. The answer is:",

        # Arabic
        "إذا ذكر المنشور على وسائل التواصل الاجتماعي أحداثًا أو معلومات من مصادر خارجية، مثل الأشخاص المقربين، وسائل الإعلام، أو مقدمي الرعاية الصحية الذين يروجون لتطعيم HPV، فاستجب بـ 'yes'. خلاف ذلك، استجب بـ 'no'. قال منشور Weibo: {text}. الآن ينتهي منشور Weibo. The answer is:",

        # Persian
        "اگر در پست شبکه‌های اجتماعی به رویدادها یا اطلاعاتی از منابع خارجی، مانند نزدیکان، رسانه‌ها، یا ارائه‌دهندگان خدمات بهداشتی که واکسیناسیون HPV را تبلیغ می‌کنند، اشاره شده است، با «yes» پاسخ دهید. در غیر این صورت، پاسخ «no» دهید. پست Weibo گفت: {text}. اکنون پست Weibo به پایان می‌رسد. The answer is:",

        # Hebrew
        "אם הפוסט במדיה החברתית מזכיר אירועים או מידע ממקורות חיצוניים, כגון אחרים קרובים, התקשורת או נותני שירותי בריאות שמקדמים חיסון HPV, ענה 'yes'. אחרת, ענה 'no'. הפוסט ב-Weibo אמר: {text}. עכשיו הפוסט ב-Weibo מסתיים. The answer is:",

        # Turkish
        "Sosyal medya gönderisi, yakın diğer kişiler, medya veya HPV aşısını teşvik eden sağlık hizmeti sağlayıcıları gibi harici kaynaklardan olaylardan veya bilgilerden bahsediyorsa, 'yes' yanıtlayın. Aksi takdirde 'no' yanıtlayın. Weibo gönderisi dedi ki: {text}. Şimdi Weibo gönderisi bitiyor. The answer is:",

        # Japanese
        "ソーシャルメディアの投稿で、HPVワクチン接種を促進する外部の情報源、たとえば親しい他者、メディア、医療提供者などに言及している場合は、「yes」と答えてください。それ以外の場合は、「no」と答えてください。Weibo投稿には次のように書かれていました: {text}。今、Weibo投稿が終了します。The answer is:",

        # Korean
        "소셜 미디어 게시물에서 HPV 백신 접종을 홍보하는 가까운 다른 사람들, 미디어 또는 의료 제공자와 같은 외부 출처의 이벤트나 정보를 언급하는 경우 'yes'로 답변하십시오. 그렇지 않으면 'no'로 답변하십시오. Weibo 게시물에는 다음과 같이 말했습니다: {text}. 이제 Weibo 게시물이 끝납니다. The answer is:",

        # Vietnamese
        "Nếu bài đăng trên mạng xã hội đề cập đến các sự kiện hoặc thông tin từ các nguồn bên ngoài, chẳng hạn như những người thân cận, giới truyền thông hoặc nhà cung cấp dịch vụ chăm sóc sức khỏe quảng bá việc tiêm vắc-xin HPV, hãy trả lời 'yes'. Nếu không, hãy trả lời 'no'. Bài đăng Weibo đã nói: {text}. Bây giờ bài đăng Weibo kết thúc. The answer is:",

        # Thai
        "หากโพสต์บนโซเชียลมีเดียกล่าวถึงเหตุการณ์หรือข้อมูลจากแหล่งข้อมูลภายนอก เช่น คนใกล้ชิด สื่อ หรือผู้ให้บริการด้านสุขภาพที่ส่งเสริมการฉีดวัคซีน HPV ให้ตอบว่า 'yes' มิฉะนั้น ให้ตอบว่า 'no'. โพสต์ Weibo กล่าวว่า: {text}. ตอนนี้โพสต์ Weibo สิ้นสุดลงแล้ว. The answer is:",

        # Indonesian
        "Jika postingan media sosial menyebutkan acara atau informasi dari sumber eksternal, seperti orang dekat lainnya, media, atau penyedia layanan kesehatan yang mempromosikan vaksinasi HPV, jawab 'yes'. Jika tidak, jawab 'no'. Postingan Weibo mengatakan: {text}. Sekarang postingan Weibo berakhir. The answer is:",

        # Malay
        "Jika kiriman media sosial menyebutkan peristiwa atau maklumat daripada sumber luar, seperti orang yang rapat, media, atau penyedia penjagaan kesihatan yang mempromosikan vaksinasi HPV, jawab 'yes'. Jika tidak, jawab 'no'. Kiriman Weibo berkata: {text}. Sekarang kiriman Weibo tamat. The answer is:",

        # Lao
        "ຖ້າໂພສສື່ສັງຄົມກ່າວເຖິງກິດຈະການຫລືຂໍ້ມູນຈາກແຫລ່ງອື່ນໆດັ່ງເຊັ່ນຄົນໃກ້ຕົວອື່ນໆສື່ຫລືຜູ້ໃຫ້ບໍລິການດ້ານສຸຂະພາບທີ່ໄດ້ຮັບການສົ່ງເສີມການສັກວັກຊີນ HPV ກະລຸນາຕອບວ່າ 'yes' ບໍ່ແມ່ນທໍາລາກໄດ້ກະລຸນາຕອບວ່າ 'no' ໂພສ Weibo ກ່າວວ່າ: {text}. ຕອນນີ້ໂພສ Weibo ສິ້ນສຸດແລ້ວ. The answer is:",

        # Burmese
        "အကယ်၍ ပို့စ်တွင် HPV ကာကွယ်ဆေးပေးခြင်းကို အထောက်အပံ့ပြုသော အခြားသူများ၊ မီဒီယာများ သို့မဟုတ် ကျန်းမာရေးထောက်ပံ့သူများအကြောင်းကို အပြင်ပုံရိပ်များနှင့်တကွ ရှင်းပြထားပါက 'yes' ဖြင့်ဖြေပါ။ မဟုတ်လျှင် 'no' ဖြင့် ဖြေပါ။ Weibo ပို့စ်က: {text}. ယခု Weibo ပို့စ်၏အဆုံးသတ်ဖြစ်သည်။ The answer is:",

        # Cebuano
        "Kung ang post sa social media naghisgot sa mga panghitabo o impormasyon gikan sa gawas nga mga tinubdan, sama sa mga duol nga uban, media, o mga maghahatag og serbisyo sa panglawas nga nagpalambo sa HPV nga bakuna, tubaga ug 'yes'. Kung dili, tubaga nga 'no'. Ang Weibo post miingon: {text}. Karun ang Weibo post natapos na. The answer is:",

        # Khmer
        "ប្រសិនបើប្រកាសបណ្តាញសង្គមបាននិយាយអំពីព្រឹត្តិការណ៍ ឬព័ត៌មានពីប្រភពខាងក្រៅ ដូចជាផ្សេងៗ ទូរទស្សន៍ ឬអ្នកផ្តល់សេវាថែទាំសុខភាព ដែលបានផ្សព្វផ្សាយការចាក់វ៉ាក់សាំង HPV សូមឆ្លើយថា 'yes'. ប្រសិនបើមិនមែនទេសូមឆ្លើយថា 'no'. ប្រកាស Weibo បាននិយាយថា {text}. ឥឡូវប្រកាសបានបញ្ចប់។ The answer is:",

        # Tagalog
        "Kung binabanggit ng post sa social media ang mga kaganapan o impormasyon mula sa mga panlabas na mapagkukunan, tulad ng mga taong malapit sa iba, media, o mga tagapagbigay ng pangangalagang pangkalusugan na nagsusulong ng pagbabakuna ng HPV, sagutin ang 'yes'. Kung hindi, sagutin ang 'no'. Ang post sa Weibo ay nagsabi: {text}. Ngayon natapos na ang post sa Weibo. The answer is:",

        # Hindi
        "यदि सोशल मीडिया पोस्ट में बाहरी स्रोतों से घटनाओं या जानकारी का उल्लेख है, जैसे कि करीबी अन्य, मीडिया, या स्वास्थ्य सेवा प्रदाता जो HPV टीकाकरण को बढ़ावा देते हैं, तो 'yes' में उत्तर दें। अन्यथा, 'no' में उत्तर दें। Weibo पोस्ट ने कहा: {text}. अब Weibo पोस्ट समाप्त होता है। The answer is:",

        # Bengali
        "যদি সোশ্যাল মিডিয়া পোস্টে বাইরের উৎস থেকে ইভেন্ট বা তথ্য উল্লেখ করা হয়, যেমন ঘনিষ্ঠ অন্যান্য, মিডিয়া, বা স্বাস্থ্যসেবা প্রদানকারী যারা HPV টিকাদান প্রচার করে, তাহলে 'yes' উত্তর দিন। অন্যথায়, 'no' উত্তর দিন। Weibo পোস্ট বলেছে: {text}. এখন Weibo পোস্ট শেষ। The answer is:",

        # Urdu
        "اگر سوشل میڈیا پوسٹ میں بیرونی ذرائع سے واقعات یا معلومات کا ذکر ہے، جیسے قریبی دیگر، میڈیا، یا صحت کی دیکھ بھال کرنے والے فراہم کنندہ جو HPV ویکسینیشن کو فروغ دیتے ہیں، تو 'yes' میں جواب دیں؛ بصورت دیگر 'no' میں جواب دیں۔ Weibo پوسٹ نے کہا: {text}. اب Weibo پوسٹ ختم ہوتی ہے۔ The answer is:",

        # Czech
        "Pokud příspěvek na sociálních sítích zmiňuje události nebo informace z externích zdrojů, jako jsou blízcí lidé, média nebo poskytovatelé zdravotní péče podporující očkování proti HPV, odpovězte 'yes'. Jinak odpovězte 'no'. Příspěvek na Weibo říkal: {text}. Nyní příspěvek na Weibo končí. Odpověď je:",

        # Polish
        "Jeśli post w mediach społecznościowych wspomina o wydarzeniach lub informacjach z zewnętrznych źródeł, takich jak bliskie osoby, media lub świadczeniodawcy opieki zdrowotnej promujący szczepienia przeciwko HPV, odpowiedz 'yes'. W przeciwnym razie odpowiedz 'no'. Post na Weibo brzmiał: {text}. Teraz post na Weibo się kończy. Odpowiedź brzmi:",
]

}


def construct_instruction(text, category):
    instruction = random.choice(multilingual_prompts[category]).format(text=text)
    return instruction

def construct_output(row, category):
    if category == 'Attitude':
        if row["Attitude"] == 1:
            output = "Positive"
        elif row["Attitude"] == 2:
            output = "Negative"
        elif row["Attitude"] == 0:
            output = "Neutral"
    elif category == 'Irrelevant':
        if row[category] == 0:
            output = "yes"
        else:
            output = 'no'
    else:
        if row[category] == 1:
            output = "yes"
        else:
            output = 'no'
    return output

categories = [
    "Behavior",
    "Attitude",
    "Perceived Disease Risk (+)",
    "Perceived benefits (+)",
    "Perceived barriers to accepting vaccines (-)",
    "Practical barriers to vaccination (-)",
    "Misinformation (-)",
    "Social norms / cues to action (+)",
]

# first layer classification
for index, row in df.iterrows():
    instruction = construct_instruction(row['微博内容'], 'Irrelevant')
    output = construct_output(row, 'Irrelevant')
    df_result = pd.concat([df_result, pd.DataFrame({'instruction': [instruction], 'output': [output]})], ignore_index=True)

# second layer classification
df = df[df['Irrelevant']==0]
for index, row in df.iterrows():
    for current_category in categories:
        instruction = construct_instruction(row['微博内容'], current_category)
        output = construct_output(row, current_category)
        df_result = pd.concat([df_result, pd.DataFrame({'instruction': [instruction], 'output': [output]})], ignore_index=True)

# Separate the classes
no_class = df_result[df_result['output'] == 'no']
yes_class = df_result[df_result['output'] == 'yes']
positive_class = df_result[df_result['output'] == 'Positive']
neutral_class = df_result[df_result['output'] == 'Neutral']
negative_class = df_result[df_result['output'] == 'Negative']


no_downsampled = resample(no_class,
                          replace=False,   # sample without replacement
                          n_samples=14000, # to get 13k samples
                          random_state=42)

yes_downsampled = resample(yes_class,
                           replace=True,    # sample with replacement
                           n_samples=8000,
                           random_state=42)


positive_upsampled = resample(positive_class,
                              replace=False,
                              n_samples=600,   # to get 600 samples
                              random_state=42)

neutral_upsampled = resample(neutral_class,
                             replace=True,
                             n_samples=250,   # to get 250 samples
                             random_state=42)

negative_upsampled = resample(negative_class,
                              replace=True,
                              n_samples=150,   # to get 150 samples
                              random_state=42)

# Combine the downsampled and upsampled data
df_balanced = pd.concat([no_downsampled,
                         yes_downsampled,
                         positive_upsampled,
                         neutral_upsampled,
                         negative_upsampled])

df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)


df_balanced.to_parquet("../../data/WHV/WHV.parquet", index=False)