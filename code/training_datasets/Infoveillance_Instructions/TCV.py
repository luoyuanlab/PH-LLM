# -*- coding: utf-8 -*-
# TCV (Twitter COVID-19 vaccine)
# - Paper: https://doi.org/10.2471/BLT.23.289682
# - Data: Not publicly available

import pandas as pd
import random
from sklearn.utils import resample

df1 = pd.read_csv("../../data/TCV/TrainWithOutAug.csv", encoding='cp1252')
df2 = pd.read_csv("../../data/TCV/test.csv", encoding='cp1252')
df = pd.concat([df1, df2])
df = df.drop(['Class_1_3_AND_1_4', 'Class_1_else', 'Class_3_1_AND_3_2', 'Class_6_2_AND_8.2', 'Class_15_all', 'tag'], axis=1)

df = df.reindex(columns=['text', 'Class_1_1', 'Class_1_2', 'Class_1_3', 'Class_1_4', 'Class_2_1',
                         'Class_2_2', 'Class_3_1', 'Class_4_1', 'Class_6_2', 'Class_10',
                         'Class_13', 'Class_14.1', 'Class_15_1', 'Class_18'])
df.columns = [col.replace('Class_', '').replace('_', '.') for col in df.columns]

# Category list
categories = ['18']

# Define prompts in all required languages
prompts = [
    # Chinese
    "推特帖子内容：{text}。\n\n请标记这条推特帖子是否是普通用户关于新冠疫苗的发帖。这里指表达了个人经历、态度、想法等等的帖子，而区别于政府、企业、社区等组织或机器人发送的不包含个人态度的帖子。完全无关新冠疫苗的帖子也要排除出这一类。如果是，回答 'yes'; 否则，请回答 'no' (请用英文作答)。",

    # English
    "Tweet content: {text}.\n\nPlease mark whether this tweet is a post by an ordinary user about the COVID-19 vaccine. This refers to posts expressing personal experiences, attitudes, thoughts, etc., as opposed to posts sent by governments, companies, communities, or bots that do not contain personal attitudes. Posts that are completely unrelated to the COVID-19 vaccine should also be excluded from this category. If so, answer 'yes'; otherwise, respond with 'no' (please answer in English).",

    # German
    "Tweet-Inhalt: {text}.\n\nBitte markieren Sie, ob dieser Tweet ein Beitrag eines normalen Benutzers über den COVID-19-Impfstoff ist. Dies bezieht sich auf Beiträge, die persönliche Erfahrungen, Einstellungen, Gedanken usw. ausdrücken, im Gegensatz zu Beiträgen, die von Regierungen, Unternehmen, Gemeinschaften oder Bots gesendet wurden und keine persönlichen Einstellungen enthalten. Beiträge, die überhaupt nicht mit dem COVID-19-Impfstoff zusammenhängen, sollten ebenfalls von dieser Kategorie ausgeschlossen werden. Wenn ja, antworten Sie mit 'yes'; andernfalls antworten Sie mit 'no' (bitte antworten Sie auf Englisch).",

    # French
    "Contenu du tweet : {text}.\n\nVeuillez indiquer si ce tweet est un post d'un utilisateur ordinaire sur le vaccin COVID-19. Cela fait référence à des posts exprimant des expériences personnelles, des attitudes, des pensées, etc., par opposition à des posts envoyés par des gouvernements, des entreprises, des communautés ou des robots qui ne contiennent pas d'attitudes personnelles. Les posts qui ne sont absolument pas liés au vaccin COVID-19 doivent également être exclus de cette catégorie. Si oui, répondez par 'yes'; sinon, répondez par 'no' (veuillez répondre en anglais).",

    # Spanish
    "Contenido del tweet: {text}.\n\nPor favor, marque si este tweet es una publicación de un usuario ordinario sobre la vacuna COVID-19. Esto se refiere a publicaciones que expresan experiencias personales, actitudes, pensamientos, etc., en contraste con publicaciones enviadas por gobiernos, empresas, comunidades o bots que no contienen actitudes personales. Las publicaciones que no están relacionadas en absoluto con la vacuna COVID-19 también deben excluirse de esta categoría. Si es así, responda con 'yes'; de lo contrario, responda con 'no' (por favor, responda en inglés).",

    # Portuguese
    "Conteúdo do tweet: {text}.\n\nMarque se este tweet é uma postagem de um usuário comum sobre a vacina COVID-19. Isso se refere a postagens que expressam experiências pessoais, atitudes, pensamentos, etc., em oposição a postagens enviadas por governos, empresas, comunidades ou bots que não contêm atitudes pessoais. Postagens que não estão absolutamente relacionadas à vacina COVID-19 também devem ser excluídas desta categoria. Se sim, responda 'yes'; caso contrário, responda 'no' (responda em inglês, por favor).",

    # Italian
    "Contenuto del tweet: {text}.\n\nSi prega di indicare se questo tweet è un post di un utente comune sul vaccino COVID-19. Ciò si riferisce a post che esprimono esperienze personali, atteggiamenti, pensieri, ecc., a differenza di post inviati da governi, aziende, comunità o bot che non contengono atteggiamenti personali. I post che non sono assolutamente correlati al vaccino COVID-19 devono essere esclusi da questa categoria. Se sì, rispondi 'yes'; altrimenti, rispondi 'no' (rispondi in inglese, per favore).",

    # Dutch
    "Tweet-inhoud: {text}.\n\nGeef aan of deze tweet een bericht is van een gewone gebruiker over het COVID-19-vaccin. Dit verwijst naar berichten die persoonlijke ervaringen, houdingen, gedachten, enz. uitdrukken, in tegenstelling tot berichten die zijn verzonden door regeringen, bedrijven, gemeenschappen of bots die geen persoonlijke houdingen bevatten. Berichten die helemaal niet gerelateerd zijn aan het COVID-19-vaccin moeten ook worden uitgesloten van deze categorie. Als dat het geval is, antwoord dan met 'yes'; anders antwoord met 'no' (antwoord alstublieft in het Engels).",

    # Russian
    "Содержание твита: {text}.\n\nОтметьте, является ли этот твит постом обычного пользователя о вакцине от COVID-19. Это относится к постам, выражающим личный опыт, взгляды, мысли и т. д., в отличие от постов, отправленных правительствами, компаниями, сообществами или ботами, не содержащими личных взглядов. Посты, которые вообще не связаны с вакциной от COVID-19, также должны быть исключены из этой категории. Если это так, ответьте 'yes'; в противном случае ответьте 'no' (ответьте, пожалуйста, на английском).",

    # Czech
    "Obsah tweetu: {text}.\n\nOznačte, zda je tento tweet příspěvkem běžného uživatele o vakcíně COVID-19. To se týká příspěvků, které vyjadřují osobní zkušenosti, postoje, myšlenky atd., na rozdíl od příspěvků zaslaných vládami, společnostmi, komunitami nebo roboty, které neobsahují osobní postoje. Příspěvky, které vůbec nesouvisejí s vakcínou COVID-19, by měly být z této kategorie rovněž vyloučeny. Pokud ano, odpovězte 'yes'; jinak odpovězte 'no' (odpovězte prosím anglicky).",

    # Polish
    "Treść tweeta: {text}.\n\nOznacz, czy ten tweet jest postem zwykłego użytkownika o szczepionce COVID-19. Dotyczy to postów wyrażających osobiste doświadczenia, postawy, myśli itp., w przeciwieństwie do postów wysyłanych przez rządy, firmy, społeczności lub boty, które nie zawierają osobistych postaw. Posty, które w ogóle nie są związane ze szczepionką COVID-19, powinny być również wykluczone z tej kategorii. Jeśli tak, odpowiedz 'yes'; w przeciwnym razie odpowiedz 'no' (odpowiedz proszę po angielsku).",

    # Arabic
    "محتوى التغريدة: {text}.\n\nيرجى تحديد ما إذا كانت هذه التغريدة عبارة عن منشور لمستخدم عادي حول لقاح COVID-19. يشير هذا إلى المنشورات التي تعبر عن التجارب الشخصية والمواقف والأفكار وما إلى ذلك، على عكس المنشورات التي ترسلها الحكومات أو الشركات أو المجتمعات أو الروبوتات التي لا تحتوي على مواقف شخصية. يجب أيضًا استبعاد المنشورات غير المتعلقة بلقاح COVID-19 تمامًا من هذه الفئة. إذا كان الأمر كذلك، فأجب بـ 'yes'; خلاف ذلك، أجب بـ 'no' (يرجى الإجابة باللغة الإنجليزية).",

    # Persian
    "محتوای توییت: {text}.\n\nلطفاً مشخص کنید که آیا این توییت پستی از یک کاربر عادی در مورد واکسن COVID-19 است یا خیر. این به پست‌هایی اشاره دارد که تجربیات شخصی، نگرش‌ها، افکار و غیره را بیان می‌کنند، در مقابل پست‌هایی که توسط دولت‌ها، شرکت‌ها، جوامع یا ربات‌ها ارسال می‌شوند و حاوی نگرش‌های شخصی نیستند. پست‌هایی که به هیچ وجه با واکسن COVID-19 مرتبط نیستند نیز باید از این دسته حذف شوند. اگر چنین است، پاسخ 'yes'; در غیر این صورت، با 'no' پاسخ دهید (لطفاً به انگلیسی پاسخ دهید).",

    # Hebrew
    "תוכן הציוץ: {text}.\n\nאנא ציינו האם הציוץ הזה הוא פוסט של משתמש רגיל בנוגע לחיסון COVID-19. זה מתייחס לפוסטים המביעים חוויות אישיות, עמדות, מחשבות וכו', בניגוד לפוסטים שנשלחים על ידי ממשלות, חברות, קהילות או רובוטים שאינם מכילים עמדות אישיות. יש גם לכלול פוסטים שאינם קשורים כלל לחיסון COVID-19. אם כן, השב 'yes'; אחרת, השב 'no' (אנא השב באנגלית).",

    # Turkish
    "Tweet içeriği: {text}.\n\nLütfen bu tweet'in COVID-19 aşısı hakkında sıradan bir kullanıcı tarafından yapılan bir gönderi olup olmadığını belirtin. Bu, hükümetler, şirketler, topluluklar veya kişisel tutumlar içermeyen botlar tarafından gönderilen gönderiler yerine kişisel deneyimleri, tutumları, düşünceleri vb. ifade eden gönderilere atıfta bulunur. COVID-19 aşısıyla tamamen ilgisiz gönderiler de bu kategoriden hariç tutulmalıdır. Eğer öyleyse, 'yes' cevabını verin; aksi takdirde, 'no' ile yanıtlayın (lütfen İngilizce cevaplayın).",

    # Japanese
    "ツイート内容：{text}。\n\nこのツイートが、COVID-19ワクチンについての一般ユーザーによる投稿かどうかをマークしてください。これは、政府、企業、コミュニティ、または個人的な態度を含まないボットによって送信された投稿とは対照的に、個人的な経験、態度、考えなどを表現する投稿を指します。COVID-19ワクチンとはまったく関係のない投稿も、このカテゴリから除外する必要があります。その場合は「yes」と答えてください。それ以外の場合は、「no」で答えてください（英語でお答えください）。",

    # Korean
    "트윗 내용: {text}.\n\n이 트윗이 COVID-19 백신에 관한 일반 사용자의 게시물인지 표시하세요. 이는 정부, 기업, 커뮤니티 또는 개인적인 태도를 포함하지 않는 봇이 보낸 게시물이 아닌 개인 경험, 태도, 생각 등을 표현한 게시물을 의미합니다. COVID-19 백신과 전혀 관련이 없는 게시물도 이 범주에서 제외되어야 합니다. 그렇다면 'yes'라고 답하십시오. 그렇지 않으면 'no'로 응답하십시오(영어로 답변해 주세요).",

    # Vietnamese
    "Nội dung tweet: {text}.\n\nVui lòng đánh dấu xem tweet này có phải là bài đăng của người dùng thông thường về vắc-xin COVID-19 hay không. Điều này đề cập đến các bài đăng thể hiện trải nghiệm cá nhân, thái độ, suy nghĩ, v.v., trái ngược với các bài đăng do chính phủ, công ty, cộng đồng hoặc bot gửi không chứa thái độ cá nhân. Các bài đăng hoàn toàn không liên quan đến vắc-xin COVID-19 cũng nên được loại trừ khỏi danh mục này. Nếu vậy, hãy trả lời 'yes'; nếu không, hãy trả lời 'no' (vui lòng trả lời bằng tiếng Anh).",

    # Thai
    "เนื้อหาทวีต: {text}.\n\nโปรดระบุว่าทวีตนี้เป็นโพสต์ของผู้ใช้ทั่วไปเกี่ยวกับวัคซีน COVID-19 หรือไม่ ซึ่งหมายถึงโพสต์ที่แสดงถึงประสบการณ์ส่วนตัว ทัศนคติ ความคิด ฯลฯ ตรงกันข้ามกับโพสต์ที่ส่งโดยรัฐบาล บริษัท ชุมชน หรือบอทที่ไม่แสดงทัศนคติส่วนตัว โพสต์ที่ไม่เกี่ยวข้องกับวัคซีน COVID-19 เลย ควรถูกแยกออกจากหมวดหมู่นี้ด้วยเช่นกัน หากเป็นเช่นนั้น ให้ตอบว่า 'yes' มิฉะนั้น ให้ตอบว่า 'no' (โปรดตอบเป็นภาษาอังกฤษ)",

    # Indonesian
    "Konten tweet: {text}.\n\nHarap tandai apakah tweet ini adalah postingan oleh pengguna biasa tentang vaksin COVID-19. Ini mengacu pada postingan yang mengekspresikan pengalaman pribadi, sikap, pemikiran, dll., berbeda dengan postingan yang dikirim oleh pemerintah, perusahaan, komunitas, atau bot yang tidak mengandung sikap pribadi. Postingan yang sama sekali tidak terkait dengan vaksin COVID-19 juga harus dikecualikan dari kategori ini. Jika demikian, jawab 'yes'; jika tidak, jawab 'no' (silakan jawab dalam bahasa Inggris).",

    # Malay
    "Kandungan tweet: {text}.\n\nSila tandakan sama ada tweet ini adalah siaran oleh pengguna biasa mengenai vaksin COVID-19. Ini merujuk kepada siaran yang menyatakan pengalaman peribadi, sikap, pemikiran, dan sebagainya, berbanding dengan siaran yang dihantar oleh kerajaan, syarikat, komuniti atau bot yang tidak mengandungi sikap peribadi. Siaran yang sama sekali tidak berkaitan dengan vaksin COVID-19 juga harus dikecualikan daripada kategori ini. Jika ya, jawab 'yes'; jika tidak, jawab 'no' (sila jawab dalam Bahasa Inggeris).",

    # Lao
    "ເນື້ອຫາຂອງທີ່ສົ່ງມາເພື່ອບໍ່ມີໃຜຕອບກັບວັກຊີນ COVID-19: {text}.\n\nກະລຸນາຕິດຕາມວ່າທີ່ສົ່ງມານີ້ເປັນການສົ່ງຂໍ້ມູນຫວ່າງເລັດນັກທີ່ມີສະເປັນແນວນີ້ກໍຂອງວັກຊີນ COVID-19. ເຫັນວ່າການບອກຄໍາກອງວ່າເລື່ອງນີ້ຈະໄດ້ຮັບການບອກຂອງທີ່ສົ່ງມາຫາພວກເຮົາ. ແມ່ນຫວ່າງການຄວບຄຸມ, ອົດທົນ, ຄວາມຄິດເຫັນ, ຂໍຄໍາເຫັນ, ຄໍາເຫັນທີ່ສ້າງຄວາມເຫັນວ່າສ່ວນຕົວກໍເລື່ອງກວມວັກຊີນ COVID-19. ກຸ່ມບົດຄວາມທີ່ບໍ່ຕ່າງຫລາຍຈາກວັກຊີນ COVID-19 ກໍເພື່ອຂໍຄຳຕອບ 'yes'; ບໍ່ຊັ່ນນັ້ນ, ຕອບ 'no' (ກະລຸນາຕອບເປັນພາສາອັງກິດ)。",

    # Burmese
    "တူစ်၏အကြောင်းအရာ: {text}.\n\nCOVID-19 ကူးစက်ရောဂါဆိုင်ရာ ဆေးနှင့်စုံလုံးသည် သင်ပြုလုပ်မည့် တူစ်များနှင့်ဆက်စပ်နေသောတူစ်များကို ခွဲခြားအတွက် ဖြစ်သည်။အကယ်၍လည်း ပုံမှန်စမ်းသပ်မှုပြီးနောက်၌ ဆေးကူးစက်ရောဂါများနှင့်ဆက်စပ်နေသောကြောင့် တူစ်တွင်ဖော်ပြမည့် ဆေးများကို ခွဲခြားပေးပါသည်။ အကယ်၍သင်ထောက်ခံပါက, 'yes' ဖြင့်တုန့်ပြန်ပါ; မဟုတ်လျှင် 'no' ဖြင့်တုန့်ပြန်ပါ (ကျေးဇူးပြု၍ အင်္ဂလိပ်ဘာသာဖြင့် တုံ့ပြန်ပါ)。",

    # Cebuano
    "Sulod sa tweet: {text}.\n\nPalihug tandaan kung kini nga tweet usa ka post sa usa ka ordinaryong tiggamit bahin sa COVID-19 nga bakuna. Kini nagtumong sa mga post nga nagpadayag sa personal nga kasinatian, mga baruganan, mga hunahuna, ug uban pa., nga sukwahi sa mga post nga gipadala sa mga gobyerno, mga kompanya, mga komunidad, o mga bot nga wala maglakip sa personal nga mga baruganan. Ang mga post nga wala gyud kalabotan sa COVID-19 nga bakuna kinahanglan usab nga tangtangon gikan niini nga kategoriya. Kung mao, tubaga 'yes'; kon dili, tubaga 'no' (palihog pagtubag sa Iningles).",

    # Khmer
    "មាតិកានៃការបញ្ចូលអត្ថបទ: {text}.\n\nសូមសម្គាល់ថាតើអត្ថបទនេះគឺជាការប្រកាសដោយអ្នកប្រើប្រាស់ធម្មតាអំពីវ៉ាក់សាំង COVID-19 ឬយ៉ាងណា។ នេះគឺជាអត្ថបទដែលបង្ហាញពីបទពិសោធន៍ផ្ទាល់ខ្លួន មនោសញ្ចេតនា គំនិត បទសម្ភាសន៍និងសំណុំគំនិត ដែលខុសពីការបញ្ចូលអត្ថបទរបស់រដ្ឋាភិបាល ក្រុមហ៊ុន សហគមន៍ ឬប្លុកដែលមិនមានមនោសញ្ចេតនាផ្ទាល់ខ្លួន។ អត្ថបទដែលមិនទាក់ទងនឹងវ៉ាក់សាំង COVID-19 គួរតែត្រូវលុបចេញពីប្រភេទនេះផងដែរ។ ប្រសិនបើពិតប្រាកដ សូមឆ្លើយ 'yes'; បើមិនមែន ឆ្លើយតប 'no' (សូមឆ្លើយជាភាសាអង់គ្លេស)។",

    # Tagalog
    "Nilalaman ng tweet: {text}.\n\nMangyaring tukuyin kung ang tweet na ito ay isang post ng isang ordinaryong gumagamit tungkol sa bakuna sa COVID-19. Ito ay tumutukoy sa mga post na nagpapahayag ng mga personal na karanasan, saloobin, pag-iisip, atbp., taliwas sa mga post na ipinadala ng mga gobyerno, kumpanya, komunidad, o bot na walang nilalaman na personal na saloobin. Ang mga post na walang kinalaman sa bakuna sa COVID-19 ay dapat ding ibukod mula sa kategoryang ito. Kung gayon, sumagot ng 'yes'; kung hindi, sumagot ng 'no' (mangyaring sagutin sa Ingles).",

    # Hindi
    "ट्वीट सामग्री: {text}.\n\nकृपया चिह्नित करें कि यह ट्वीट एक सामान्य उपयोगकर्ता द्वारा COVID-19 वैक्सीन के बारे में पोस्ट है या नहीं। यह उन पोस्टों को संदर्भित करता है जो व्यक्तिगत अनुभवों, दृष्टिकोणों, विचारों आदि को व्यक्त करती हैं, न कि सरकारों, कंपनियों, समुदायों या बॉट्स द्वारा भेजी गई पोस्ट जिन्हें व्यक्तिगत दृष्टिकोण शामिल नहीं हैं। COVID-19 वैक्सीन से पूरी तरह से असंबंधित पोस्ट को भी इस श्रेणी से बाहर रखा जाना चाहिए। यदि हां, तो 'yes' के साथ उत्तर दें; अन्यथा, 'no' के साथ उत्तर दें (कृपया उत्तर अंग्रेजी में दें)।",

    # Bengali
    "টুইটের বিষয়বস্তু: {text}.\n\nঅনুগ্রহ করে চিহ্নিত করুন যে এই টুইটটি COVID-19 ভ্যাকসিন সম্পর্কে একজন সাধারণ ব্যবহারকারীর পোস্ট কিনা। এটি সেই পোস্টগুলিকে বোঝায় যা ব্যক্তিগত অভিজ্ঞতা, মনোভাব, চিন্তাধারা ইত্যাদি প্রকাশ করে, সরকার, কোম্পানি, সম্প্রদায় বা বট দ্বারা পাঠানো পোস্টগুলির বিপরীতে যা ব্যক্তিগত মনোভাব ধারণ করে না। COVID-19 ভ্যাকসিনের সাথে সম্পর্কহীন পোস্টগুলিকেও এই বিভাগ থেকে বাদ দেওয়া উচিত। যদি তাই হয়, 'yes' দিয়ে উত্তর দিন; অন্যথায়, 'no' দিয়ে উত্তর দিন (দয়া করে ইংরেজিতে উত্তর দিন)।",

    # Urdu
    "ٹویٹ مواد: {text}.\n\nبراہ کرم نشان زد کریں کہ آیا یہ ٹویٹ COVID-19 ویکسین کے بارے میں ایک عام صارف کی پوسٹ ہے۔ اس کا مطلب ہے کہ وہ پوسٹس جو ذاتی تجربات، رویوں، خیالات وغیرہ کا اظہار کرتی ہیں، ان پوسٹس کے برعکس جو حکومتوں، کمپنیوں، کمیونٹیز یا بوٹس نے بھیجی ہیں جن میں ذاتی رویے شامل نہیں ہیں۔ COVID-19 ویکسین سے مکمل طور پر غیر متعلق پوسٹس کو بھی اس زمرے سے خارج کر دینا چاہیے۔ اگر ایسا ہے تو 'yes' کے ساتھ جواب دیں؛ بصورت دیگر، 'no' کے ساتھ جواب دیں (براہ کرم انگریزی میں جواب دیں)۔"
]

# Function to create instruction for a given row and category
def create_instruction(text, category):
    instruction_template = random.choice(prompts)
    instruction = instruction_template.format(text=text)
    return instruction

# Create a new DataFrame that will contain the instructions for each label
instructions_df = pd.DataFrame(columns=['instruction', 'output', 'category'])

# Iterate over each row in the original DataFrame
for index, row in df.iterrows():
    text = row['text']
    for category in categories:
        # Create instruction for the category
        instruction = create_instruction(text, category)
        output = 'yes' if row[category] == 0 else 'no'
        instructions_df = pd.concat([instructions_df, pd.DataFrame({'instruction': [instruction], 'output': [output], 'category':[category]})], ignore_index=True)

instructions_df[['instruction', 'output']].sample(n=8000).to_parquet("../../data/TCV/TCV-subtaskA.parquet", index=False)

categories = [
    '1.1', '1.2', '1.3', '1.4',
    '2.1', '2.2',
    '3.1',
    '4.1',
    '6.2',
    '10',
    '13',
    '14.1',
    '15.1'
]

multilingual_prompts = {
    '1.1': [
        # Chinese
        "推特帖子内容：{text}。请标记这条推特帖子是否提到接种新冠疫苗的意愿。这里指的是表达支持、接受或愿意接种新冠疫苗。如果是，回答 'yes'；否则，回答 'no' (请用英语回答).",
        # English
        "Tweet content: {text}. Please mark whether this tweet mentions the willingness to receive the COVID-19 vaccine. This refers to expressing support, acceptance, or willingness to get vaccinated. If so, answer 'yes'; otherwise, respond with 'no' (please answer in English).",
        # German
        "Tweet-Inhalt: {text}. Bitte markieren Sie, ob dieser Tweet die Bereitschaft zur COVID-19-Impfung erwähnt. Dies bezieht sich auf die Bereitschaft, Unterstützung oder Akzeptanz für die Impfung auszudrücken. Wenn ja, antworten Sie mit 'yes'; andernfalls antworten Sie mit 'no' (bitte antworten Sie auf Englisch).",
        # French
        "Contenu du tweet : {text}. Veuillez indiquer si ce tweet mentionne la volonté de recevoir le vaccin contre la COVID-19. Cela signifie exprimer un soutien, une acceptation ou une volonté de se faire vacciner. Si oui, répondez 'yes'; sinon, répondez 'no' (veuillez répondre en anglais).",
        # Spanish
        "Contenido del tweet: {text}. Por favor, marque si este tweet menciona la disposición a recibir la vacuna contra la COVID-19. Esto se refiere a expresar apoyo, aceptación o disposición a vacunarse. Si es así, responda 'yes'; de lo contrario, responda 'no' (por favor responda en inglés).",
        # Portuguese
        "Conteúdo do tweet: {text}. Por favor, indique se este tweet menciona a disposição de receber a vacina contra a COVID-19. Isto refere-se a expressar apoio, aceitação ou disposição para ser vacinado. Se sim, responda 'yes'; caso contrário, responda 'no' (por favor, responda em inglês).",
        # Italian
        "Contenuto del tweet: {text}. Si prega di indicare se questo tweet menziona la volontà di ricevere il vaccino contro il COVID-19. Questo si riferisce a esprimere sostegno, accettazione o disponibilità a farsi vaccinare. Se sì, rispondi 'yes'; altrimenti rispondi 'no' (si prega di rispondere in inglese).",
        # Dutch
        "Tweet-inhoud: {text}. Markeer of deze tweet de bereidheid tot het ontvangen van het COVID-19-vaccin vermeldt. Dit verwijst naar het uiten van steun, acceptatie of bereidheid om gevaccineerd te worden. Als dat zo is, antwoord dan 'yes'; anders antwoord 'no' (antwoord alstublieft in het Engels).",
        # Russian
        "Содержание твита: {text}. Отметьте, упоминает ли этот твит готовность получить вакцину от COVID-19. Это относится к выражению поддержки, принятия или желания вакцинироваться. Если да, ответьте 'yes'; в противном случае ответьте 'no' (пожалуйста, ответьте на английском языке).",
        # Czech
        "Obsah tweetu: {text}. Označte, zda tento tweet zmiňuje ochotu přijmout vakcínu proti COVID-19. Toto se týká vyjádření podpory, přijetí nebo ochoty nechat se očkovat. Pokud ano, odpovězte 'yes'; v opačném případě odpovězte 'no' (prosím, odpovězte v angličtině).",
        # Polish
        "Treść tweeta: {text}. Proszę zaznaczyć, czy ten tweet wspomina o gotowości do przyjęcia szczepionki na COVID-19. To odnosi się do wyrażenia poparcia, akceptacji lub chęci zaszczepienia się. Jeśli tak, odpowiedz 'yes'; w przeciwnym razie odpowiedz 'no' (proszę odpowiedzieć po angielsku).",
        # Arabic
        "محتوى التغريدة: {text}. يُرجى تحديد ما إذا كانت هذه التغريدة تذكر الاستعداد لتلقي لقاح COVID-19. يشير هذا إلى التعبير عن الدعم أو القبول أو الرغبة في التطعيم. إذا كان الأمر كذلك، أجب 'yes'؛ وإلا، أجب 'no' (يرجى الرد بالإنجليزية).",
        # Persian
        "محتوای توییت: {text}. لطفاً مشخص کنید آیا این توییت تمایل به دریافت واکسن COVID-19 را ذکر می‌کند. این به معنای ابراز حمایت، پذیرش یا تمایل به واکسیناسیون است. اگر بله، با 'yes' پاسخ دهید؛ در غیر این صورت، با 'no' پاسخ دهید (لطفاً به انگلیسی پاسخ دهید).",
        # Hebrew
        "תוכן הציוץ: {text}. נא לסמן אם הציוץ הזה מזכיר את הרצון לקבל את חיסון ה-COVID-19. זה מתייחס להבעת תמיכה, קבלה או רצון להתחסן. אם כן, ענה 'yes'; אחרת ענה 'no' (אנא ענה באנגלית).",
        # Turkish
        "Tweet içeriği: {text}. Lütfen bu tweetin COVID-19 aşısı alma isteğini belirtip belirtmediğini işaretleyin. Bu, aşı olma isteğini, kabulünü veya desteğini ifade etmeye atıfta bulunur. Eğer öyleyse, 'yes' ile cevap verin; aksi takdirde, 'no' ile cevap verin (lütfen İngilizce cevaplayın).",
        # Japanese
        "ツイート内容：{text}。このツイートがCOVID-19ワクチンを受ける意志を示しているかどうかをマークしてください。これは、ワクチン接種への支持、受け入れ、または意欲を表明することを指します。もしそうなら、「yes」と答えてください。そうでない場合は「no」と答えてください（英語でお答えください）。",
        # Korean
        "트윗 내용: {text}. 이 트윗이 COVID-19 백신을 맞겠다는 의사를 언급하는지 표시해 주세요. 이는 백신 접종에 대한 지지, 수락 또는 의사를 표현하는 것을 의미합니다. 그렇다면 'yes'로 대답해 주세요; 그렇지 않으면 'no'로 대답해 주세요 (영어로 대답해 주세요).",
        # Vietnamese
        "Nội dung tweet: {text}. Vui lòng đánh dấu xem tweet này có đề cập đến mong muốn nhận vắc-xin COVID-19 hay không. Điều này ám chỉ việc bày tỏ sự ủng hộ, chấp nhận hoặc sẵn sàng tiêm vắc-xin. Nếu có, hãy trả lời 'yes'; nếu không, hãy trả lời 'no' (vui lòng trả lời bằng tiếng Anh).",
        # Thai
        "เนื้อหาทวีต: {text}. กรุณาระบุว่าทวีตนี้กล่าวถึงความเต็มใจที่จะรับวัคซีน COVID-19 หรือไม่ ซึ่งหมายถึงการแสดงการสนับสนุน การยอมรับ หรือความเต็มใจที่จะรับการฉีดวัคซีน หากเป็นเช่นนั้น ให้ตอบ 'yes' มิฉะนั้น ให้ตอบ 'no' (โปรดตอบเป็นภาษาอังกฤษ).",
        # Indonesian
        "Konten tweet: {text}. Silakan tandai apakah tweet ini menyebutkan kesediaan untuk menerima vaksin COVID-19. Ini mengacu pada mengungkapkan dukungan, penerimaan, atau kesediaan untuk divaksinasi. Jika demikian, jawab 'yes'; jika tidak, jawab 'no' (silakan jawab dalam bahasa Inggris).",
        # Malay
        "Kandungan tweet: {text}. Sila tandakan sama ada tweet ini menyebut kesediaan untuk menerima vaksin COVID-19. Ini merujuk kepada menyatakan sokongan, penerimaan, atau kesediaan untuk divaksin. Jika ya, jawab 'yes'; jika tidak, jawab 'no' (sila jawab dalam Bahasa Inggeris).",
        # Lao
        "ເນື້ອຫາທວິດ: {text}. ກະລຸນາໝາຍວ່າທວິດນີ້ໄດ້ພາກົດເອົາຄວາມຕັ້ງໃຈທີ່ຈະຮັບການວັກຊີນ COVID-19 ຫຼືບໍ່. ນີ້ແມ່ນການທີ່ຈະພາກົດເອົາການສະໜອງທັງຫມົດ, ການຍອມຮັບຫຼືຄວາມຕັ້ງໃຈທີ່ຈະຮັບການສັກວັກຊີນ. ຖ້າແມ່ນແລ້ວ, ກະລຸນາຕອບວ່າ 'yes'; ຖ້າບໍ່ແມ່ນ, ກະລຸນາຕອບວ່າ 'no' (ກະລຸນາຕອບໃນພາສາອັງກິດ).",
        # Burmese
        "တွစ်အကြောင်းအရာ: {text}။ ဒီတွစ်ဟာ COVID-19 ကာကွယ်ဆေးကို အဆင်သင့်ရှိကြောင်းကို ဖော်ပြခဲ့ပါသလားဆိုတာကို မှတ်ပေးပါ။ ဒါက ကာကွယ်ဆေးထိုးခြင်း၊ လက်ခံခြင်း၊ ဒါမှမဟုတ် ဆန့်ကျင်ဖို့အဆင်သင့်ရှိခြင်းကို ဖော်ပြထားတာပါ။ အကယ်၍ရှိပါက 'yes' ဖြင့် ဖြေပါ၊ မဟုတ်ပါက 'no' ဖြင့် ဖြေပါ (ကျေးဇူးပြု၍ အင်္ဂလိပ်ဘာသာဖြင့် ပြန်ဆိုပါ).",
        # Cebuano
        "Sulud sa tweet: {text}. Palihug markahi kung kini nga tweet naghisgot sa kahanduraw sa pagdawat sa bakuna sa COVID-19. Kini nagtumong sa pagpakita og suporta, pagdawat, o kahanduraw sa pagpabakuna. Kung mao, tubaga og 'yes'; kung dili, tubaga og 'no' (palihug tubaga sa English).",
        # Khmer
        "ខ្លឹមសារទីចេញផ្សាយ: {text}។ សូមសម្គាល់មើលថាតើការផ្សាយនេះបាននិយាយអំពីចិត្តនឹងទទួលការចាក់វ៉ាក់សាំង COVID-19 ឬអត់។ នេះមានន័យថាសំដៅទៅលើការសំអាងអំពីការគាំទ្រ ការទទួលយកឬក្តីប្រាថ្នាដែលនឹងចាក់វ៉ាក់សាំង COVID-19។ ប្រសិនបើចង់និយាយពីចិត្តនឹងទទួល ក្រាបបាទឬអត់ត្រូវបានគេបាននិយាយថា yes។ បើមិនបានបញ្ចប់សូមឆ្លើយតបដោយនិយាយថា 'no'។ (សូមឆ្លើយតបជាភាសាអង់គ្លេស).",
        # Tagalog
        "Nilalaman ng tweet: {text}. Paki-markahan kung ang tweet na ito ay nagpapahayag ng kahandaan na makatanggap ng bakuna sa COVID-19. Ito ay tumutukoy sa pagpapahayag ng suporta, pagtanggap, o kahandaan na magpabakuna. Kung oo, sagutin ang 'yes'; kung hindi, sagutin ang 'no' (mangyaring sagutin sa Ingles).",
        # Hindi
        "ट्वीट सामग्री: {text}। कृपया चिह्नित करें कि यह ट्वीट COVID-19 वैक्सीन प्राप्त करने की इच्छा का उल्लेख करता है या नहीं। इसका अर्थ है टीकाकरण के प्रति समर्थन, स्वीकृति या इच्छा को व्यक्त करना। यदि हाँ, तो 'yes' में उत्तर दें; अन्यथा, 'no' में उत्तर दें (कृपया अंग्रेजी में उत्तर दें)।",
        # Bengali
        "টুইটের বিষয়বস্তু: {text}। অনুগ্রহ করে চিহ্নিত করুন যে এই টুইটটি COVID-19 ভ্যাকসিন পাওয়ার ইচ্ছার উল্লেখ করে কিনা। এটি টিকাদানের প্রতি সমর্থন, গ্রহণযোগ্যতা বা ইচ্ছা প্রকাশের উল্লেখ করে। যদি হ্যাঁ, তাহলে 'yes' দিয়ে উত্তর দিন; অন্যথায়, 'no' দিয়ে উত্তর দিন (অনুগ্রহ করে ইংরেজিতে উত্তর দিন)।",
        # Urdu
        "ٹویٹ کا مواد: {text}. براہ کرم نشان لگائیں کہ آیا یہ ٹویٹ COVID-19 ویکسین حاصل کرنے کی خواہش کا ذکر کرتا ہے۔ اس کا مطلب ویکسین لگانے کی حمایت، قبولیت یا خواہش کا اظہار ہے۔ اگر ایسا ہے تو، 'yes' میں جواب دیں؛ ورنہ، 'no' میں جواب دیں (براہ کرم انگریزی میں جواب دیں)."
    ],

    '1.2': [
        # Chinese
        "推特帖子内容：{text}。请标记这条推特帖子是否表达中立态度。这里指的是对新冠疫苗的接种表达了总体中立的态度（或者没有表达任何态度），也就是说，我们无法将这个帖子判断为支持或反对新冠疫苗接种。如果是，回答 'yes'；否则，回答 'no' (请用英语回答).",
        # English
        "Tweet content: {text}. Please mark whether this tweet expresses a neutral attitude. This refers to expressing a generally neutral attitude towards COVID-19 vaccination (or not expressing any attitude at all), meaning we cannot determine whether this tweet supports or opposes COVID-19 vaccination. If so, answer 'yes'; otherwise, respond with 'no' (please answer in English).",
        # German
        "Tweet-Inhalt: {text}. Bitte markieren Sie, ob dieser Tweet eine neutrale Haltung ausdrückt. Dies bezieht sich auf eine allgemein neutrale Haltung gegenüber der COVID-19-Impfung (oder gar keine Haltung), was bedeutet, dass wir nicht feststellen können, ob dieser Tweet die COVID-19-Impfung unterstützt oder ablehnt. Wenn ja, antworten Sie mit 'yes'; andernfalls antworten Sie mit 'no' (bitte antworten Sie auf Englisch).",
        # French
        "Contenu du tweet : {text}. Veuillez indiquer si ce tweet exprime une attitude neutre. Cela signifie exprimer une attitude généralement neutre à l'égard de la vaccination contre la COVID-19 (ou ne pas exprimer d'attitude du tout), ce qui signifie que nous ne pouvons pas déterminer si ce tweet soutient ou s'oppose à la vaccination contre la COVID-19. Si oui, répondez 'yes'; sinon, répondez 'no' (veuillez répondre en anglais).",
        # Spanish
        "Contenido del tweet: {text}. Por favor, marque si este tweet expresa una actitud neutral. Esto se refiere a expresar una actitud generalmente neutral hacia la vacunación contra la COVID-19 (o no expresar ninguna actitud), lo que significa que no podemos determinar si este tweet apoya o se opone a la vacunación contra la COVID-19. Si es así, responda 'yes'; de lo contrario, responda 'no' (por favor responda en inglés).",
        # Portuguese
        "Conteúdo do tweet: {text}. Por favor, indique se este tweet expressa uma atitude neutra. Isso se refere a expressar uma atitude geralmente neutra em relação à vacinação contra a COVID-19 (ou não expressar nenhuma atitude), o que significa que não podemos determinar se este tweet apoia ou se opõe à vacinação contra a COVID-19. Se sim, responda 'yes'; caso contrário, responda 'no' (por favor, responda em inglês).",
        # Italian
        "Contenuto del tweet: {text}. Si prega di indicare se questo tweet esprime un atteggiamento neutrale. Questo si riferisce a esprimere un atteggiamento generalmente neutrale nei confronti della vaccinazione contro il COVID-19 (o non esprimere alcun atteggiamento), il che significa che non possiamo determinare se questo tweet supporta o si oppone alla vaccinazione contro il COVID-19. Se sì, rispondi 'yes'; altrimenti rispondi 'no' (si prega di rispondere in inglese).",
        # Dutch
        "Tweet-inhoud: {text}. Markeer of deze tweet een neutrale houding uitdrukt. Dit verwijst naar het uiten van een algemeen neutrale houding ten opzichte van COVID-19-vaccinatie (of geen houding), wat betekent dat we niet kunnen bepalen of deze tweet COVID-19-vaccinatie ondersteunt of ertegen is. Als dat zo is, antwoord dan 'yes'; anders antwoord 'no' (antwoord alstublieft in het Engels).",
        # Russian
        "Содержание твита: {text}. Отметьте, выражает ли этот твит нейтральное отношение. Это относится к выражению общей нейтральной позиции по отношению к вакцинации от COVID-19 (или отсутствия какого-либо отношения), что означает, что мы не можем определить, поддерживает ли этот твит вакцинацию от COVID-19 или выступает против нее. Если да, ответьте 'yes'; в противном случае ответьте 'no' (пожалуйста, ответьте на английском языке).",
        # Czech
        "Obsah tweetu: {text}. Označte, zda tento tweet vyjadřuje neutrální postoj. To se týká vyjádření obecně neutrálního postoje k očkování proti COVID-19 (nebo nevyjádření žádného postoje), což znamená, že nemůžeme určit, zda tento tweet podporuje nebo je proti očkování proti COVID-19. Pokud ano, odpovězte 'yes'; v opačném případě odpovězte 'no' (prosím, odpovězte v angličtině).",
        # Polish
        "Treść tweeta: {text}. Proszę zaznaczyć, czy ten tweet wyraża neutralną postawę. Odnosi się to do wyrażenia ogólnie neutralnej postawy wobec szczepień przeciwko COVID-19 (lub nie wyrażania żadnej postawy), co oznacza, że nie możemy określić, czy ten tweet popiera, czy sprzeciwia się szczepieniom przeciwko COVID-19. Jeśli tak, odpowiedz 'yes'; w przeciwnym razie odpowiedz 'no' (proszę odpowiedzieć po angielsku).",
        # Arabic
        "محتوى التغريدة: {text}. يُرجى تحديد ما إذا كانت هذه التغريدة تعبر عن موقف محايد. يشير هذا إلى التعبير عن موقف محايد بشكل عام تجاه التطعيم ضد COVID-19 (أو عدم التعبير عن أي موقف على الإطلاق)، مما يعني أننا لا نستطيع تحديد ما إذا كانت هذه التغريدة تدعم أو تعارض التطعيم ضد COVID-19. إذا كان الأمر كذلك، أجب 'yes'؛ وإلا، أجب 'no' (يرجى الرد بالإنجليزية).",
        # Persian
        "محتوای توییت: {text}. لطفاً مشخص کنید که آیا این توییت نگرش خنثی را بیان می‌کند. این به معنای بیان یک نگرش کلی خنثی نسبت به واکسیناسیون COVID-19 (یا عدم بیان هیچ نگرشی) است، به این معنی که ما نمی‌توانیم تعیین کنیم که آیا این توییت از واکسیناسیون COVID-19 حمایت می‌کند یا مخالف آن است. اگر بله، با 'yes' پاسخ دهید؛ در غیر این صورت، با 'no' پاسخ دهید (لطفاً به انگلیسی پاسخ دهید).",
        # Hebrew
        "תוכן הציוץ: {text}. נא לסמן אם הציוץ הזה מבטא עמדה נייטרלית. זה מתייחס להבעת עמדה נייטרלית בדרך כלל כלפי חיסון ה-COVID-19 (או לא להביע עמדה כלל), מה שאומר שאנחנו לא יכולים לקבוע אם הציוץ הזה תומך או מתנגד לחיסון ה-COVID-19. אם כן, ענה 'yes'; אחרת ענה 'no' (אנא ענה באנגלית).",
        # Turkish
        "Tweet içeriği: {text}. Lütfen bu tweetin nötr bir tutum sergileyip sergilemediğini işaretleyin. Bu, COVID-19 aşısına karşı genel olarak nötr bir tutum sergilemeyi (veya herhangi bir tutum sergilemeyi) ifade eder; bu, bu tweetin COVID-19 aşısını destekleyip desteklemediğini belirleyemeyeceğimiz anlamına gelir. Eğer öyleyse, 'yes' ile cevap verin; aksi takdirde, 'no' ile cevap verin (lütfen İngilizce cevaplayın).",
        # Japanese
        "ツイート内容：{text}。このツイートが中立的な態度を示しているかどうかをマークしてください。これは、COVID-19ワクチン接種に対して一般的に中立的な態度を示すこと（またはまったく態度を示さないこと）を指します。したがって、このツイートがCOVID-19ワクチン接種を支持しているか、反対しているかを判断できません。もしそうなら、「yes」と答えてください。そうでない場合は「no」と答えてください（英語でお答えください）。",
        # Korean
        "트윗 내용: {text}. 이 트윗이 중립적인 태도를 표현하는지 표시해 주세요. 이는 COVID-19 백신 접종에 대해 일반적으로 중립적인 태도를 표현하는 것(또는 전혀 태도를 표현하지 않는 것)을 의미합니다. 따라서 이 트윗이 COVID-19 백신 접종을 지지하는지 반대하는지 알 수 없습니다. 그렇다면 'yes'로 대답해 주세요; 그렇지 않으면 'no'로 대답해 주세요 (영어로 대답해 주세요).",
        # Vietnamese
        "Nội dung tweet: {text}. Vui lòng đánh dấu xem tweet này có thể hiện thái độ trung lập hay không. Điều này có nghĩa là bày tỏ thái độ trung lập đối với việc tiêm vắc-xin COVID-19 (hoặc không thể hiện bất kỳ thái độ nào), nghĩa là chúng tôi không thể xác định liệu tweet này có ủng hộ hay phản đối việc tiêm vắc-xin COVID-19 hay không. Nếu có, hãy trả lời 'yes'; nếu không, hãy trả lời 'no' (vui lòng trả lời bằng tiếng Anh).",
        # Thai
        "เนื้อหาทวีต: {text}. กรุณาระบุว่าทวีตนี้แสดงท่าทีเป็นกลางหรือไม่ ซึ่งหมายถึงการแสดงท่าทีเป็นกลางโดยทั่วไปต่อการฉีดวัคซีน COVID-19 (หรือไม่ได้แสดงท่าทีใดๆ เลย) หมายความว่าเราไม่สามารถระบุได้ว่าทวีตนี้สนับสนุนหรือคัดค้านการฉีดวัคซีน COVID-19 หากเป็นเช่นนั้น ให้ตอบ 'yes' มิฉะนั้น ให้ตอบ 'no' (โปรดตอบเป็นภาษาอังกฤษ).",
        # Indonesian
        "Konten tweet: {text}. Silakan tandai apakah tweet ini mengungkapkan sikap netral. Ini mengacu pada sikap netral secara umum terhadap vaksinasi COVID-19 (atau tidak menyatakan sikap apa pun), yang berarti kami tidak dapat menentukan apakah tweet ini mendukung atau menentang vaksinasi COVID-19. Jika demikian, jawab 'yes'; jika tidak, jawab 'no' (silakan jawab dalam bahasa Inggris).",
        # Malay
        "Kandungan tweet: {text}. Sila tandakan sama ada tweet ini menyatakan sikap neutral. Ini merujuk kepada menyatakan sikap neutral secara umum terhadap vaksinasi COVID-19 (atau tidak menyatakan apa-apa sikap langsung), yang bermaksud kita tidak dapat menentukan sama ada tweet ini menyokong atau menentang vaksinasi COVID-19. Jika ya, jawab 'yes'; jika tidak, jawab 'no' (sila jawab dalam Bahasa Inggeris).",
        # Lao
        "ເນື້ອຫາທວິດ: {text}. ກະລຸນາໝາຍວ່າທວິດນີ້ໄດ້ເຜີຍແຈ້ງວ່າມີຄວາມຄິດເຫັນທີ່ເປັນການສະແດງຄວາມຄິດເຫັນຕໍ່ການເຂົ້າຮັບວັກຊີນ COVID-19 ຫຼືບໍ່. ຖ້າເປັນຄວາມຄິດເຫັນທີ່ເປັນການຂັດຄ້ານວັກຊີນ COVID-19 ຫຼືບໍ່ມີການເຫັນຊອບໃຈແລະບໍ່ມີການແສງຄວາມຫລືແຂ້ງຄວາມເຫັນຕໍ່ໃຫ້ຫລຸດລົງ. ຖ້າແມ່ນເຫັນແຕກຕ່າງ, ກະລຸນາຕອບ 'yes'; ຖ້າບໍ່ແມ່ນ, ກະລຸນາຕອບ 'no' (ກະລຸນາຕອບໃນພາສາອັງກິດ).",

        # Burmese
        "တွစ်အကြောင်းအရာ: {text}။ ဒီတွစ်က COVID-19 ကာကွယ်ဆေးအပေါ်မည်သည့် အဆင့်အတန်းမှ မပြသဘဲ အပစ်အခတ်အပြစ်မှုမပါဝင်ဘဲ ဦးတည်ချက်ဟုတ်ဟုတ်ကို ဖြေရှင်းနိုင်ပါသလားဆိုတာ မှတ်ပေးပါ။ အကယ်၍ 'yes'ဖြင့်ဖြေပါ၊ မဟုတ်ပါက 'no' ဖြင့်ဖြေပါ (ကျေးဇူးပြု၍ အင်္ဂလိပ်ဘာသာဖြင့် ပြန်ဆိုပါ).",

        # Cebuano
        "Sulud sa tweet: {text}. Palihug markahi kung kini nga tweet nagpakita sa usa ka neutral nga postura bahin sa bakuna sa COVID-19 (o wala gyud postura nga gipakita). Kung mao, tubaga og 'yes'; kung dili, tubaga og 'no' (palihug tubaga sa English).",

        # Khmer
        "ខ្លឹមសារទីចេញផ្សាយ: {text}។ សូមសម្គាល់មើលថាតើការផ្សាយនេះបាននិយាយអំពីចិត្តតុល្យភាព ទាក់ទងនឹងការចាក់វ៉ាក់សាំង COVID-19 ឬអត់។ ប្រសិនបើចិត្តមានតុល្យភាព ឬមិនបានបង្ហាញចិត្តតុល្យភាព ចម្លើយគឺ 'yes'។ ប្រសិនបើមិនទេ ចម្លើយគឺ 'no'។ (សូមឆ្លើយជាភាសាអង់គ្លេស).",

        # Tagalog
        "Nilalaman ng tweet: {text}. Paki-markahan kung ang tweet na ito ay nagpapahayag ng isang neutral na saloobin hinggil sa pagbabakuna ng COVID-19. Kung ito ay nagpapahayag ng neutral na saloobin, sagutin ang 'yes'; kung hindi, sagutin ang 'no' (mangyaring sagutin sa Ingles).",

        # Hindi
        "ट्वीट सामग्री: {text}। कृपया चिह्नित करें कि यह ट्वीट COVID-19 टीकाकरण के बारे में तटस्थ दृष्टिकोण व्यक्त करता है या नहीं। यदि हाँ, तो 'yes' में उत्तर दें; अन्यथा, 'no' में उत्तर दें (कृपया अंग्रेजी में उत्तर दें)।",

        # Bengali
        "টুইটের বিষয়বস্তু: {text}। অনুগ্রহ করে চিহ্নিত করুন যে এই টুইটটি COVID-19 টিকাদানের বিষয়ে নিরপেক্ষ মনোভাব প্রকাশ করে কিনা। যদি হ্যাঁ, তাহলে 'yes' দিয়ে উত্তর দিন; অন্যথায় 'no' দিয়ে উত্তর দিন (দয়া করে ইংরেজিতে উত্তর দিন)।",

        # Urdu
        "ٹویٹ کا مواد: {text}. براہ کرم نشان لگائیں کہ آیا یہ ٹویٹ COVID-19 ویکسینیشن کے بارے میں غیرجانبدار رویہ کا اظہار کرتا ہے۔ اگر ایسا ہے تو، 'yes' میں جواب دیں؛ ورنہ، 'no' میں جواب دیں (براہ کرم انگریزی میں جواب دیں)."
    ],
    '1.3': [
        # Chinese
        "推特帖子内容：{text}。请标记这条推特帖子是否提到对新冠疫苗的担心或怀疑。这里指对新冠疫苗担心或持怀疑态度，但没有直接表达拒绝、反对、不支持新冠疫苗。担心不安全、没效果等都应算在这里。如果是，回答 'yes'；否则，回答 'no' (请用英语回答).",

        # English
        "Tweet content: {text}. Please mark whether this tweet mentions concerns or doubts about the COVID-19 vaccine. This refers to concerns or doubts about the COVID-19 vaccine but without directly expressing refusal, opposition, or lack of support for the vaccine. Concerns about safety, efficacy, etc., should be included here. If so, answer 'yes'; otherwise, respond with 'no' (please answer in English).",

        # German
        "Tweet-Inhalt: {text}. Bitte markieren Sie, ob dieser Tweet Bedenken oder Zweifel am COVID-19-Impfstoff erwähnt. Dies bezieht sich auf Bedenken oder Zweifel am COVID-19-Impfstoff, ohne direkt eine Ablehnung, einen Widerspruch oder eine fehlende Unterstützung für den Impfstoff auszudrücken. Bedenken hinsichtlich der Sicherheit, Wirksamkeit usw. sollten hier einbezogen werden. Wenn ja, antworten Sie mit 'yes'; andernfalls antworten Sie mit 'no' (bitte auf Englisch antworten).",

        # French
        "Contenu du tweet : {text}. Veuillez indiquer si ce tweet mentionne des préoccupations ou des doutes concernant le vaccin COVID-19. Cela fait référence à des préoccupations ou des doutes concernant le vaccin COVID-19, sans exprimer directement un refus, une opposition ou un manque de soutien pour le vaccin. Les préoccupations concernant la sécurité, l'efficacité, etc., doivent être incluses ici. Si oui, répondez 'yes'; sinon, répondez 'no' (veuillez répondre en anglais).",

        # Spanish
        "Contenido del tweet: {text}. Por favor, marque si este tweet menciona preocupaciones o dudas sobre la vacuna COVID-19. Esto se refiere a preocupaciones o dudas sobre la vacuna COVID-19, pero sin expresar directamente rechazo, oposición o falta de apoyo a la vacuna. Las preocupaciones sobre la seguridad, la eficacia, etc., deben incluirse aquí. Si es así, responda 'yes'; de lo contrario, responda 'no' (por favor, responda en inglés).",

        # Portuguese
        "Conteúdo do tweet: {text}. Por favor, indique se este tweet menciona preocupações ou dúvidas sobre a vacina COVID-19. Isso se refere a preocupações ou dúvidas sobre a vacina COVID-19, mas sem expressar diretamente recusa, oposição ou falta de apoio à vacina. Preocupações sobre segurança, eficácia, etc., devem ser incluídas aqui. Se sim, responda 'yes'; caso contrário, responda 'no' (responda em inglês, por favor).",

        # Italian
        "Contenuto del tweet: {text}. Si prega di indicare se questo tweet menziona preoccupazioni o dubbi sul vaccino COVID-19. Questo si riferisce a preoccupazioni o dubbi sul vaccino COVID-19, senza esprimere direttamente un rifiuto, un'opposizione o una mancanza di supporto per il vaccino. Le preoccupazioni sulla sicurezza, efficacia, ecc. dovrebbero essere incluse qui. Se sì, rispondi 'yes'; altrimenti rispondi 'no' (rispondi in inglese, per favore).",

        # Dutch
        "Tweet-inhoud: {text}. Markeer of deze tweet zorgen of twijfels over het COVID-19-vaccin vermeldt. Dit verwijst naar zorgen of twijfels over het COVID-19-vaccin, zonder direct afwijzing, oppositie of gebrek aan steun voor het vaccin te uiten. Zorgen over veiligheid, werkzaamheid, enz., moeten hier worden opgenomen. Als dat zo is, antwoord dan 'yes'; anders antwoord 'no' (antwoord alstublieft in het Engels).",

        # Russian
        "Содержание твита: {text}. Отметьте, упоминает ли этот твит опасения или сомнения по поводу вакцины от COVID-19. Это относится к опасениям или сомнениям по поводу вакцины от COVID-19, но без прямого выражения отказа, оппозиции или отсутствия поддержки вакцины. Опасения по поводу безопасности, эффективности и т. д. следует включить сюда. Если да, ответьте 'yes'; в противном случае ответьте 'no' (пожалуйста, ответьте на английском языке).",

        # Czech
        "Obsah tweetu: {text}. Označte, zda tento tweet zmiňuje obavy nebo pochybnosti o vakcíně proti COVID-19. To se týká obav nebo pochybností o vakcíně proti COVID-19, ale bez přímého vyjádření odmítnutí, opozice nebo nedostatečné podpory vakcíny. Obavy o bezpečnost, účinnost atd. by měly být zahrnuty zde. Pokud ano, odpovězte 'yes'; v opačném případě odpovězte 'no' (odpovězte prosím anglicky).",

        # Polish
        "Treść tweeta: {text}. Proszę zaznaczyć, czy ten tweet wspomina o obawach lub wątpliwościach dotyczących szczepionki na COVID-19. To odnosi się do obaw lub wątpliwości dotyczących szczepionki na COVID-19, ale bez bezpośredniego wyrażania odmowy, sprzeciwu lub braku poparcia dla szczepionki. Obawy dotyczące bezpieczeństwa, skuteczności itp. powinny być tutaj uwzględnione. Jeśli tak, odpowiedz 'yes'; w przeciwnym razie odpowiedz 'no' (proszę odpowiedzieć po angielsku).",

        # Arabic
        "محتوى التغريدة: {text}. يُرجى تحديد ما إذا كانت هذه التغريدة تذكر مخاوف أو شكوكًا بشأن لقاح COVID-19. يشير هذا إلى مخاوف أو شكوك بشأن لقاح COVID-19 دون التعبير مباشرة عن الرفض أو المعارضة أو عدم دعم اللقاح. يجب تضمين المخاوف المتعلقة بالسلامة والفعالية وما إلى ذلك هنا. إذا كان الأمر كذلك، أجب 'yes'؛ وإلا، أجب 'no' (يرجى الرد بالإنجليزية).",

        # Persian
        "محتوای توییت: {text}. لطفاً مشخص کنید که آیا این توییت به نگرانی‌ها یا تردیدها درباره واکسن COVID-19 اشاره دارد یا خیر. این اشاره به نگرانی‌ها یا تردیدها در مورد واکسن COVID-19 دارد، بدون اینکه مستقیماً از عدم حمایت، مخالفت یا رد واکسن صحبت کند. نگرانی‌ها در مورد ایمنی، اثربخشی و غیره باید در اینجا گنجانده شوند. اگر بله، با 'yes' پاسخ دهید؛ در غیر این صورت، با 'no' پاسخ دهید (لطفاً به انگلیسی پاسخ دهید).",

        # Hebrew
        "תוכן הציוץ: {text}. נא לסמן אם הציוץ הזה מזכיר חששות או ספקות לגבי חיסון ה-COVID-19. זה מתייחס לחששות או ספקות לגבי חיסון ה-COVID-19, מבלי להביע ישירות סירוב, התנגדות או חוסר תמיכה בחיסון. חששות לגבי בטיחות, יעילות וכו' צריכים להיכלל כאן. אם כן, ענה 'yes'; אחרת ענה 'no' (אנא ענה באנגלית).",

        # Turkish
        "Tweet içeriği: {text}. Lütfen bu tweetin COVID-19 aşısı hakkındaki endişeleri veya şüpheleri belirtip belirtmediğini işaretleyin. Bu, COVID-19 aşısıyla ilgili endişeler veya şüphelerle ilgilidir, ancak aşıya doğrudan bir ret, muhalefet veya destek eksikliği ifade etmeden. Güvenlik, etkililik vb. ile ilgili endişeler burada dahil edilmelidir. Eğer öyleyse, 'yes' ile cevap verin; aksi takdirde, 'no' ile cevap verin (lütfen İngilizce cevaplayın).",

        # Japanese
        "ツイート内容：{text}。このツイートがCOVID-19ワクチンに関する懸念や疑念を示しているかどうかをマークしてください。これは、ワクチンに対する懸念や疑念を指しますが、ワクチンに対する拒否、反対、またはサポートの欠如を直接表明しているわけではありません。安全性、効果などの懸念はここに含まれるべきです。もしそうなら、「yes」と答えてください。そうでない場合は「no」と答えてください（英語でお答えください）。",

        # Korean
        "트윗 내용: {text}. 이 트윗이 COVID-19 백신에 대한 우려나 의구심을 언급하는지 표시해 주세요. 이는 백신에 대한 우려나 의구심을 말하지만 백신에 대한 거부, 반대 또는 지지 부족을 직접적으로 표현하지는 않습니다. 안전성, 효능 등에 대한 우려는 여기에 포함되어야 합니다. 그렇다면 'yes'로 대답해 주세요; 그렇지 않으면 'no'로 대답해 주세요 (영어로 대답해 주세요).",

        # Vietnamese
        "Nội dung tweet: {text}. Vui lòng đánh dấu xem tweet này có đề cập đến những lo ngại hoặc nghi ngờ về vắc-xin COVID-19 hay không. Điều này đề cập đến những lo ngại hoặc nghi ngờ về vắc-xin COVID-19 nhưng không trực tiếp bày tỏ sự từ chối, phản đối hoặc thiếu hỗ trợ cho vắc-xin. Những lo ngại về an toàn, hiệu quả, v.v., nên được đưa vào đây. Nếu có, hãy trả lời 'yes'; nếu không, hãy trả lời 'no' (vui lòng trả lời bằng tiếng Anh).",

        # Thai
        "เนื้อหาทวีต: {text}. กรุณาระบุว่าทวีตนี้กล่าวถึงความกังวลหรือข้อสงสัยเกี่ยวกับวัคซีน COVID-19 หรือไม่ ซึ่งหมายถึงความกังวลหรือข้อสงสัยเกี่ยวกับวัคซีน COVID-19 โดยไม่แสดงออกถึงการปฏิเสธ การคัดค้าน หรือการขาดการสนับสนุนวัคซีนโดยตรง ความกังวลเกี่ยวกับความปลอดภัย ประสิทธิภาพ ฯลฯ ควรรวมไว้ที่นี่ หากเป็นเช่นนั้น ให้ตอบ 'yes' มิฉะนั้น ให้ตอบ 'no' (โปรดตอบเป็นภาษาอังกฤษ).",

        # Indonesian
        "Konten tweet: {text}. Silakan tandai apakah tweet ini menyebutkan kekhawatiran atau keraguan tentang vaksin COVID-19. Ini mengacu pada kekhawatiran atau keraguan tentang vaksin COVID-19 tetapi tanpa secara langsung menyatakan penolakan, oposisi, atau kurangnya dukungan untuk vaksin. Kekhawatiran tentang keamanan, kemanjuran, dll. harus dimasukkan di sini. Jika demikian, jawab 'yes'; jika tidak, jawab 'no' (silakan jawab dalam bahasa Inggris).",

        # Malay
        "Kandungan tweet: {text}. Sila tandakan sama ada tweet ini menyebutkan kebimbangan atau keraguan tentang vaksin COVID-19. Ini merujuk kepada kebimbangan atau keraguan tentang vaksin COVID-19 tetapi tanpa menyatakan secara langsung penolakan, tentangan, atau kekurangan sokongan untuk vaksin. Kebimbangan tentang keselamatan, keberkesanan, dll. harus dimasukkan di sini. Jika ya, jawab 'yes'; jika tidak, jawab 'no' (sila jawab dalam Bahasa Inggeris).",

        # Lao
        "ເນື້ອຫາທວິດ: {text}. ກະລຸນາໝາຍວ່າທວິດນີ້ໄດ້ພາກົດເອົາຄວາມກັງວົນຫຼືຄວາມສົມເຫັນຕໍ່ການຮັກສາວັກຊີນ COVID-19 ຫຼືບໍ່. ນີ້ແມ່ນຄວາມກັງວົນກ່ຽວກັບຄວາມປອດໄພ ຄວາມມີປະສິດທິພາບ ແລະອື່ນໆຄວາມຫຍັງທີ່ຄວນໄດ້ຮັບການພິຈາລະນາ. ຖ້າແມ່ນ, ກະລຸນາຕອບ 'yes'; ຖ້າບໍ່ແມ່ນ, ກະລຸນາຕອບ 'no' (ກະລຸນາຕອບໃນພາສາອັງກິດ).",

        # Burmese
        "တွစ်အကြောင်းအရာ: {text}။ ဒီတွစ်က COVID-19 ကာကွယ်ဆေးနှင့်ပတ်သက်သည့် စိုးရိမ်မှုများနှင့် စပ်လျဉ်းနေသလားဆိုတာကို မှတ်ပေးပါ။ ဒါဟာ လုံခြုံမှု၊ ထိရောက်မှုစသဖြင့် စိုးရိမ်မှုများကို ထည့်သွင်းဆင်ခြင်ရမည်ဖြစ်ပါသည်။ အကယ်၍ရှိပါက 'yes' ဖြင့် ဖြေပါ၊ မဟုတ်ပါက 'no' ဖြင့် ဖြေပါ (ကျေးဇူးပြု၍ အင်္ဂလိပ်ဘာသာဖြင့် ပြန်ဆိုပါ).",

        # Cebuano
        "Sulud sa tweet: {text}. Palihug markahi kung kini nga tweet naghisgot sa mga kabalaka o mga pagduhaduha bahin sa bakuna sa COVID-19. Kini nagtumong sa mga kabalaka o mga pagduhaduha bahin sa bakuna sa COVID-19 apan wala maghisgot sa direktang pagdumili, pagsupak, o kakulang sa suporta alang sa bakuna. Ang mga kabalaka bahin sa kaluwasan, pagkaepektibo, ug uban pa, kinahanglan nga ilakip dinhi. Kung mao, tubaga 'yes'; kung dili, tubaga 'no' (palihug tubaga sa English).",

        # Khmer
        "ខ្លឹមសារទីចេញផ្សាយ: {text}។ សូមសម្គាល់មើលថាតើការផ្សាយនេះបាននិយាយអំពីការព្រួយបារម្ភ ឬការមានចិត្តខ្មាស់អៀនចំពោះវ៉ាក់សាំង COVID-19 ឬអត់។ នេះមានន័យថាសំដៅទៅលើការព្រួយបារម្ភ ឬការមានចិត្តខ្មាស់អៀនចំពោះវ៉ាក់សាំង COVID-19 ប៉ុន្តែដោយគ្មានការលើកឡើងបញ្ចេញការគំទ្រ ឬការទុកចិត្តពីលើវ៉ាក់សាំង COVID-19។ ការព្រួយបារម្ភអំពីសុវត្ថិភាព ប្រសិទ្ធភាព លទ្ធផលជាដើម គួរត្រូវបានបញ្ចូលនៅទីនេះ។ ប្រសិនបើពិតប្រាកដ សូមឆ្លើយតប 'yes'; បើមិនមែន ឆ្លើយ 'no' (សូមឆ្លើយតបជាភាសាអង់គ្លេស).",

        # Tagalog
        "Nilalaman ng tweet: {text}. Paki-markahan kung ang tweet na ito ay nagpapahayag ng mga alalahanin o pagdududa tungkol sa bakuna sa COVID-19. Ito ay tumutukoy sa mga alalahanin o pagdududa tungkol sa bakuna sa COVID-19 ngunit hindi direktang nagpapahayag ng pagtanggi, pagsalungat, o kawalan ng suporta para sa bakuna. Ang mga alalahanin tungkol sa kaligtasan, pagiging epektibo, atbp., ay dapat isama rito. Kung oo, sagutin ang 'yes'; kung hindi, sagutin ang 'no' (mangyaring sagutin sa Ingles).",

        # Hindi
        "ट्वीट सामग्री: {text}। कृपया चिह्नित करें कि यह ट्वीट COVID-19 वैक्सीन के बारे में चिंता या संदेह का उल्लेख करता है या नहीं। इसका अर्थ COVID-19 वैक्सीन के बारे में चिंताओं या संदेहों का उल्लेख करना है, लेकिन वैक्सीन के लिए सीधे अस्वीकृति, विरोध या समर्थन की कमी व्यक्त नहीं करना। सुरक्षा, प्रभावशीलता आदि के बारे में चिंताओं को यहाँ शामिल किया जाना चाहिए। यदि हाँ, तो 'yes' में उत्तर दें; अन्यथा, 'no' में उत्तर दें (कृपया अंग्रेजी में उत्तर दें)।",

        # Bengali
        "টুইটের বিষয়বস্তু: {text}। অনুগ্রহ করে চিহ্নিত করুন যে এই টুইটটি COVID-19 ভ্যাকসিন সম্পর্কে উদ্বেগ বা সন্দেহের উল্লেখ করে কিনা। এটি COVID-19 ভ্যাকসিন সম্পর্কে উদ্বেগ বা সন্দেহের উল্লেখ করে, তবে ভ্যাকসিনের জন্য সরাসরি প্রত্যাখ্যান, বিরোধিতা বা সমর্থনের অভাব প্রকাশ করে না। নিরাপত্তা, কার্যকারিতা ইত্যাদি সম্পর্কে উদ্বেগ এখানে অন্তর্ভুক্ত করা উচিত। যদি হ্যাঁ, তাহলে 'yes' দিয়ে উত্তর দিন; অন্যথায়, 'no' দিয়ে উত্তর দিন (অনুগ্রহ করে ইংরেজিতে উত্তর দিন)।",

        # Urdu
        "ٹویٹ کا مواد: {text}. براہ کرم نشان لگائیں کہ آیا یہ ٹویٹ COVID-19 ویکسین کے بارے میں خدشات یا شکوک کا ذکر کرتا ہے۔ اس کا مطلب COVID-19 ویکسین کے بارے میں خدشات یا شکوک کا ذکر کرنا ہے، لیکن ویکسین کے لیے براہ راست انکار، مخالفت یا حمایت کی کمی کا اظہار کیے بغیر۔ یہاں حفاظت، تاثیر وغیرہ کے بارے میں خدشات شامل کیے جانے چاہئیں۔ اگر ایسا ہے تو، 'yes' میں جواب دیں؛ ورنہ، 'no' میں جواب دیں (براہ کرم انگریزی میں جواب دیں)."
    ],
    '1.4': [
        # Chinese
        "推特帖子内容：{text}。请标记这条推特帖子是否提到拒绝新冠疫苗的意愿。这里指的是拒绝接种新冠疫苗，或明确反对、不支持新冠疫苗的态度。如果是，回答 'yes'；否则，回答 'no' (请用英语回答)。",

        # English
        "Tweet content: {text}. Please mark whether this tweet mentions the intention to refuse the COVID-19 vaccine. This refers to rejecting vaccination or explicitly opposing or not supporting the COVID-19 vaccine. If so, answer 'yes'; otherwise, respond with 'no' (please answer in English).",

        # German
        "Tweet-Inhalt: {text}. Bitte markieren Sie, ob dieser Tweet die Absicht erwähnt, den COVID-19-Impfstoff abzulehnen. Dies bezieht sich auf die Ablehnung der Impfung oder die ausdrückliche Opposition oder Nichtunterstützung des COVID-19-Impfstoffs. Wenn ja, antworten Sie mit 'yes'; andernfalls antworten Sie mit 'no' (bitte antworten Sie auf Englisch).",

        # French
        "Contenu du tweet : {text}. Veuillez indiquer si ce tweet mentionne l'intention de refuser le vaccin COVID-19. Cela fait référence au refus de la vaccination ou à l'opposition explicite ou au manque de soutien pour le vaccin COVID-19. Si oui, répondez 'yes'; sinon, répondez 'no' (veuillez répondre en anglais).",

        # Spanish
        "Contenido del tweet: {text}. Por favor, marque si este tweet menciona la intención de rechazar la vacuna COVID-19. Esto se refiere al rechazo de la vacunación o a la oposición explícita o a la falta de apoyo a la vacuna COVID-19. Si es así, responda 'yes'; de lo contrario, responda 'no' (por favor responda en inglés).",

        # Portuguese
        "Conteúdo do tweet: {text}. Por favor, indique se este tweet menciona a intenção de recusar a vacina COVID-19. Isto refere-se à rejeição da vacinação ou à oposição explícita ou à falta de apoio à vacina COVID-19. Se sim, responda 'yes'; caso contrário, responda 'no' (por favor, responda em inglês).",

        # Italian
        "Contenuto del tweet: {text}. Si prega di indicare se questo tweet menziona l'intenzione di rifiutare il vaccino COVID-19. Questo si riferisce al rifiuto della vaccinazione o all'opposizione esplicita o alla mancanza di supporto per il vaccino COVID-19. Se sì, rispondi 'yes'; altrimenti rispondi 'no' (si prega di rispondere in inglese).",

        # Dutch
        "Tweet-inhoud: {text}. Markeer of deze tweet de intentie vermeldt om het COVID-19-vaccin te weigeren. Dit verwijst naar het afwijzen van vaccinatie of het expliciet tegenwerken of niet ondersteunen van het COVID-19-vaccin. Als dat zo is, antwoord dan 'yes'; anders antwoord 'no' (antwoord alstublieft in het Engels).",

        # Russian
        "Содержание твита: {text}. Отметьте, упоминает ли этот твит намерение отказаться от вакцины от COVID-19. Это относится к отказу от вакцинации или явному противодействию или неподдержке вакцины от COVID-19. Если да, ответьте 'yes'; в противном случае ответьте 'no' (пожалуйста, ответьте на английском языке).",

        # Czech
        "Obsah tweetu: {text}. Označte, zda tento tweet zmiňuje úmysl odmítnout vakcínu proti COVID-19. To se týká odmítnutí očkování nebo výslovného odporu nebo nepodporování vakcíny proti COVID-19. Pokud ano, odpovězte 'yes'; v opačném případě odpovězte 'no' (prosím, odpovězte v angličtině).",

        # Polish
        "Treść tweeta: {text}. Proszę zaznaczyć, czy ten tweet wspomina o zamiarze odmowy przyjęcia szczepionki na COVID-19. To odnosi się do odmowy szczepienia lub wyraźnego sprzeciwu lub braku poparcia dla szczepionki na COVID-19. Jeśli tak, odpowiedz 'yes'; w przeciwnym razie odpowiedz 'no' (proszę odpowiedzieć po angielsku).",

        # Arabic
        "محتوى التغريدة: {text}. يُرجى تحديد ما إذا كانت هذه التغريدة تذكر النية لرفض لقاح COVID-19. يشير هذا إلى رفض التطعيم أو معارضة اللقاح COVID-19 صراحةً أو عدم دعمه. إذا كان الأمر كذلك، أجب 'yes'؛ وإلا، أجب 'no' (يرجى الرد بالإنجليزية).",

        # Persian
        "محتوای توییت: {text}. لطفاً مشخص کنید که آیا این توییت به قصد رد واکسن COVID-19 اشاره دارد یا خیر. این به معنای رد واکسیناسیون یا مخالفت صریح یا عدم حمایت از واکسن COVID-19 است. اگر بله، با 'yes' پاسخ دهید؛ در غیر این صورت، با 'no' پاسخ دهید (لطفاً به انگلیسی پاسخ دهید).",

        # Hebrew
        "תוכן הציוץ: {text}. נא לסמן אם הציוץ הזה מזכיר את הכוונה לסרב לחיסון ה-COVID-19. זה מתייחס לסירוב להתחסן או להתנגדות מפורשת או לא לתמוך בחיסון ה-COVID-19. אם כן, ענה 'yes'; אחרת ענה 'no' (אנא ענה באנגלית).",

        # Turkish
        "Tweet içeriği: {text}. Lütfen bu tweetin COVID-19 aşısını reddetme niyetini belirtip belirtmediğini işaretleyin. Bu, aşıyı reddetmek veya COVID-19 aşısına açıkça karşı çıkmak veya desteklememek anlamına gelir. Eğer öyleyse, 'yes' ile cevap verin; aksi takdirde, 'no' ile cevap verin (lütfen İngilizce cevaplayın).",

        # Japanese
        "ツイート内容：{text}。このツイートがCOVID-19ワクチンを拒否する意図を示しているかどうかをマークしてください。これは、ワクチン接種を拒否したり、COVID-19ワクチンに明確に反対したり、支持しないことを指します。もしそうなら、「yes」と答えてください。そうでない場合は「no」と答えてください（英語でお答えください）。",

        # Korean
        "트윗 내용: {text}. 이 트윗이 COVID-19 백신을 거부할 의사를 언급하는지 표시해 주세요. 이는 백신 접종을 거부하거나 COVID-19 백신에 명시적으로 반대하거나 지지하지 않음을 의미합니다. 그렇다면 'yes'로 대답해 주세요; 그렇지 않으면 'no'로 대답해 주세요 (영어로 대답해 주세요).",

        # Vietnamese
        "Nội dung tweet: {text}. Vui lòng đánh dấu xem tweet này có đề cập đến ý định từ chối vắc-xin COVID-19 hay không. Điều này đề cập đến việc từ chối tiêm vắc-xin hoặc phản đối rõ ràng hoặc không ủng hộ vắc-xin COVID-19. Nếu có, hãy trả lời 'yes'; nếu không, hãy trả lời 'no' (vui lòng trả lời bằng tiếng Anh).",

        # Thai
        "เนื้อหาทวีต: {text}. กรุณาระบุว่าทวีตนี้กล่าวถึงเจตนาที่จะปฏิเสธวัคซีน COVID-19 หรือไม่ ซึ่งหมายถึงการปฏิเสธการฉีดวัคซีนหรือคัดค้านหรือไม่สนับสนุนวัคซีน COVID-19 โดยชัดแจ้ง หากเป็นเช่นนั้น ให้ตอบ 'yes' มิฉะนั้น ให้ตอบ 'no' (โปรดตอบเป็นภาษาอังกฤษ).",

        # Indonesian
        "Konten tweet: {text}. Silakan tandai apakah tweet ini menyebutkan niat untuk menolak vaksin COVID-19. Ini mengacu pada penolakan vaksinasi atau secara eksplisit menentang atau tidak mendukung vaksin COVID-19. Jika demikian, jawab 'yes'; jika tidak, jawab 'no' (silakan jawab dalam bahasa Inggris).",

        # Malay
        "Kandungan tweet: {text}. Sila tandakan sama ada tweet ini menyebut niat untuk menolak vaksin COVID-19. Ini merujuk kepada penolakan vaksinasi atau secara terang-terangan menentang atau tidak menyokong vaksin COVID-19. Jika ya, jawab 'yes'; jika tidak, jawab 'no' (sila jawab dalam Bahasa Inggeris).",

        # Lao
        "ເນື້ອຫາທວິດ: {text}. ກະລຸນາໝາຍວ່າທວິດນີ້ໄດ້ພາກົດເອົາຄວາມຕັ້ງໃຈທີ່ຈະປະຕິເສດວັກຊີນ COVID-19 ຫຼືບໍ່. ນີ້ແມ່ນການປະຕິເສດການສັກວັກຊີນ ຫລືການຕໍ່ຕ້ານຫຼືບໍ່ຮັບຮອງວັກຊີນ COVID-19 ຢ່າງຊັດເຈນ. ຖ້າແມ່ນແລ້ວ, ກະລຸນາຕອບວ່າ 'yes'; ຖ້າບໍ່ແມ່ນ, ກະລຸນາຕອບວ່າ 'no' (ກະລຸນາຕອບໃນພາສາອັງກິດ).",

        # Burmese
        "တွစ်အကြောင်းအရာ: {text}။ COVID-19 ကာကွယ်ဆေးထိုးခြင်းကို လက်မခံဘူးလို့ ဒီတွစ်ထဲမှာ ဖော်ပြထားလားဆိုတာကို မှတ်ပေးပါ။ ဒါဟာ ကာကွယ်ဆေးထိုးခြင်းကို ရှောင်ရှားတာ၊ ကာကွယ်ဆေးကို ပယ်ချတာ၊ ဒါမှမဟုတ် ကာကွယ်ဆေးအပေါ်မပံ့ပိုးတဲ့အမြင်တွေအပါအဝင်ဖြစ်ပါတယ်။ အကယ်၍ရှိပါက 'yes' ဖြင့် ဖြေပါ၊ မဟုတ်ပါက 'no' ဖြင့် ဖြေပါ (ကျေးဇူးပြု၍ အင်္ဂလိပ်ဘာသာဖြင့် ဖြေပါ).",

        # Cebuano
        "Sulud sa tweet: {text}. Palihug markahi kung kini nga tweet naghisgot sa tinguha nga dili magpabakuna sa COVID-19. Kini nagtumong sa pagdumili sa pagbakuna o sa pagbatok sa bakuna sa COVID-19 nga tin-aw o sa dili pagsuporta niini. Kung mao, tubaga og 'yes'; kung dili, tubaga og 'no' (palihug tubaga sa English).",

        # Khmer
        "ខ្លឹមសារទីចេញផ្សាយ: {text}។ សូមសម្គាល់មើលថាតើការផ្សាយនេះបាននិយាយអំពីចិត្តចង់ច្រានចោលការចាក់វ៉ាក់សាំង COVID-19 ឬអត់។ នេះសំដៅទៅលើការបដិសេធការចាក់វ៉ាក់សាំង COVID-19 ឬពោលច្បាស់ៗថាមិនគាំទ្រទៅនឹងវ៉ាក់សាំង COVID-19។ ប្រសិនបើចង់និយាយបញ្ចប់សូមឆ្លើយតបដោយនិយាយថា 'yes'។ បើមិនបានបញ្ចប់សូមឆ្លើយតបដោយនិយាយថា 'no'។ (សូមឆ្លើយតបជាភាសាអង់គ្លេស).",

        # Tagalog
        "Nilalaman ng tweet: {text}. Paki-markahan kung ang tweet na ito ay nagpapahayag ng hangarin na tanggihan ang bakuna sa COVID-19. Ito ay tumutukoy sa pagtanggi na magpabakuna o tahasang pagsalungat o hindi pagsuporta sa bakuna sa COVID-19. Kung oo, sagutin ang 'yes'; kung hindi, sagutin ang 'no' (mangyaring sagutin sa Ingles).",

        # Hindi
        "ट्वीट सामग्री: {text}। कृपया चिह्नित करें कि यह ट्वीट COVID-19 वैक्सीन को अस्वीकार करने के इरादे का उल्लेख करता है या नहीं। इसका अर्थ है टीकाकरण को अस्वीकार करना या COVID-19 वैक्सीन का स्पष्ट रूप से विरोध करना या समर्थन नहीं करना। यदि हाँ, तो 'yes' में उत्तर दें; अन्यथा, 'no' में उत्तर दें (कृपया अंग्रेजी में उत्तर दें)।",

        # Bengali
        "টুইটের বিষয়বস্তু: {text}। অনুগ্রহ করে চিহ্নিত করুন যে এই টুইটটি COVID-19 ভ্যাকসিন প্রত্যাখ্যানের ইচ্ছার উল্লেখ করে কিনা। এটি টিকাদান প্রত্যাখ্যান বা COVID-19 ভ্যাকসিনের স্পষ্ট বিরোধিতা বা সমর্থনের অভাবের উল্লেখ করে। যদি হ্যাঁ, তাহলে 'yes' দিয়ে উত্তর দিন; অন্যথায়, 'no' দিয়ে উত্তর দিন (অনুগ্রহ করে ইংরেজিতে উত্তর দিন)।",

        # Urdu
        "ٹویٹ کا مواد: {text}. براہ کرم نشان لگائیں کہ آیا یہ ٹویٹ COVID-19 ویکسین کو مسترد کرنے کے ارادے کا ذکر کرتا ہے۔ اس کا مطلب ویکسینیشن کو مسترد کرنا یا COVID-19 ویکسین کی واضح مخالفت یا حمایت نہ کرنا ہے۔ اگر ایسا ہے تو، 'yes' میں جواب دیں؛ ورنہ، 'no' میں جواب دیں (براہ کرم انگریزی میں جواب دیں)."
    ],
    '2.1': [
        # Chinese
        "推特帖子内容：{text}。请标记这条推特帖子是否提到认为新冠疫苗安全。这里指认为新冠疫苗较为安全可靠，无不良反应。如果是，回答 'yes'；否则，回答 'no' (请用英语回答)。",

        # English
        "Tweet content: {text}. Please mark whether this tweet mentions that the COVID-19 vaccine is safe. This refers to the belief that the COVID-19 vaccine is relatively safe and reliable, with no adverse effects. If so, answer 'yes'; otherwise, respond with 'no' (please answer in English).",

        # German
        "Tweet-Inhalt: {text}. Bitte markieren Sie, ob dieser Tweet erwähnt, dass der COVID-19-Impfstoff sicher ist. Dies bezieht sich auf die Ansicht, dass der COVID-19-Impfstoff relativ sicher und zuverlässig ist, ohne unerwünschte Wirkungen. Wenn ja, antworten Sie mit 'yes'; andernfalls antworten Sie mit 'no' (bitte antworten Sie auf Englisch).",

        # French
        "Contenu du tweet : {text}. Veuillez indiquer si ce tweet mentionne que le vaccin COVID-19 est sûr. Cela fait référence à la croyance que le vaccin COVID-19 est relativement sûr et fiable, sans effets indésirables. Si oui, répondez 'yes'; sinon, répondez 'no' (veuillez répondre en anglais).",

        # Spanish
        "Contenido del tweet: {text}. Por favor, marque si este tweet menciona que la vacuna COVID-19 es segura. Esto se refiere a la creencia de que la vacuna COVID-19 es relativamente segura y confiable, sin efectos adversos. Si es así, responda 'yes'; de lo contrario, responda 'no' (por favor responda en inglés).",

        # Portuguese
        "Conteúdo do tweet: {text}. Por favor, indique se este tweet menciona que a vacina COVID-19 é segura. Isso se refere à crença de que a vacina COVID-19 é relativamente segura e confiável, sem efeitos adversos. Se sim, responda 'yes'; caso contrário, responda 'no' (por favor, responda em inglês).",

        # Italian
        "Contenuto del tweet: {text}. Si prega di indicare se questo tweet menziona che il vaccino COVID-19 è sicuro. Questo si riferisce alla convinzione che il vaccino COVID-19 sia relativamente sicuro e affidabile, senza effetti avversi. Se sì, rispondi 'yes'; altrimenti rispondi 'no' (si prega di rispondere in inglese).",

        # Dutch
        "Tweet-inhoud: {text}. Markeer of deze tweet vermeldt dat het COVID-19-vaccin veilig is. Dit verwijst naar de overtuiging dat het COVID-19-vaccin relatief veilig en betrouwbaar is, zonder nadelige effecten. Als dat zo is, antwoord dan 'yes'; anders antwoord 'no' (antwoord alstublieft in het Engels).",

        # Russian
        "Содержание твита: {text}. Отметьте, упоминает ли этот твит, что вакцина от COVID-19 безопасна. Это относится к мнению, что вакцина от COVID-19 относительно безопасна и надежна, без побочных эффектов. Если да, ответьте 'yes'; в противном случае ответьте 'no' (пожалуйста, ответьте на английском языке).",

        # Czech
        "Obsah tweetu: {text}. Označte, zda tento tweet zmiňuje, že vakcína proti COVID-19 je bezpečná. To se týká přesvědčení, že vakcína proti COVID-19 je relativně bezpečná a spolehlivá, bez nežádoucích účinků. Pokud ano, odpovězte 'yes'; v opačném případě odpovězte 'no' (prosím, odpovězte v angličtině).",

        # Polish
        "Treść tweeta: {text}. Proszę zaznaczyć, czy ten tweet wspomina, że ​​szczepionka na COVID-19 jest bezpieczna. To odnosi się do przekonania, że ​​szczepionka na COVID-19 jest stosunkowo bezpieczna i niezawodna, bez skutków ubocznych. Jeśli tak, odpowiedz 'yes'; w przeciwnym razie odpowiedz 'no' (proszę odpowiedzieć po angielsku).",

        # Arabic
        "محتوى التغريدة: {text}. يُرجى تحديد ما إذا كانت هذه التغريدة تذكر أن لقاح COVID-19 آمن. يشير هذا إلى الاعتقاد بأن لقاح COVID-19 آمن وموثوق نسبيًا، دون أي آثار سلبية. إذا كان الأمر كذلك، أجب 'yes'؛ وإلا، أجب 'no' (يرجى الرد بالإنجليزية).",

        # Persian
        "محتوای توییت: {text}. لطفاً مشخص کنید که آیا این توییت به ایمن بودن واکسن COVID-19 اشاره دارد یا خیر. این به این باور اشاره دارد که واکسن COVID-19 نسبتاً ایمن و قابل اعتماد است و هیچ عوارض جانبی ندارد. اگر بله، با 'yes' پاسخ دهید؛ در غیر این صورت، با 'no' پاسخ دهید (لطفاً به انگلیسی پاسخ دهید).",

        # Hebrew
        "תוכן הציוץ: {text}. נא לסמן אם הציוץ הזה מזכיר שהחיסון ל-COVID-19 בטוח. זה מתייחס לאמונה שהחיסון ל-COVID-19 הוא בטוח ואמין יחסית, ללא תופעות לוואי. אם כן, ענה 'yes'; אחרת ענה 'no' (אנא ענה באנגלית).",

        # Turkish
        "Tweet içeriği: {text}. Lütfen bu tweetin COVID-19 aşısının güvenli olduğunu belirtip belirtmediğini işaretleyin. Bu, COVID-19 aşısının nispeten güvenli ve güvenilir olduğuna ve olumsuz etkilerinin olmadığına inanmayı ifade eder. Eğer öyleyse, 'yes' ile cevap verin; aksi takdirde, 'no' ile cevap verin (lütfen İngilizce cevaplayın).",

        # Japanese
        "ツイート内容：{text}。このツイートがCOVID-19ワクチンが安全であると示しているかどうかをマークしてください。これは、COVID-19ワクチンが比較的安全で信頼性が高く、副作用がないという信念を指します。もしそうなら、「yes」と答えてください。そうでない場合は「no」と答えてください（英語でお答えください）。",

        # Korean
        "트윗 내용: {text}. 이 트윗이 COVID-19 백신이 안전하다고 언급하는지 표시해 주세요. 이는 COVID-19 백신이 비교적 안전하고 신뢰할 수 있으며 부작용이 없다고 믿는 것을 의미합니다. 그렇다면 'yes'로 대답해 주세요; 그렇지 않으면 'no'로 대답해 주세요 (영어로 대답해 주세요).",

        # Vietnamese
        "Nội dung tweet: {text}. Vui lòng đánh dấu xem tweet này có đề cập đến việc vắc-xin COVID-19 an toàn hay không. Điều này ám chỉ rằng vắc-xin COVID-19 tương đối an toàn và đáng tin cậy, không có tác dụng phụ. Nếu có, hãy trả lời 'yes'; nếu không, hãy trả lời 'no' (vui lòng trả lời bằng tiếng Anh).",

        # Thai
        "เนื้อหาทวีต: {text}. กรุณาระบุว่าทวีตนี้กล่าวถึงความปลอดภัยของวัคซีน COVID-19 หรือไม่ ซึ่งหมายถึงความเชื่อว่าวัคซีน COVID-19 ปลอดภัยและเชื่อถือได้โดยไม่มีผลข้างเคียง หากเป็นเช่นนั้น ให้ตอบ 'yes' มิฉะนั้น ให้ตอบ 'no' (โปรดตอบเป็นภาษาอังกฤษ).",

        # Indonesian
        "Konten tweet: {text}. Silakan tandai apakah tweet ini menyebutkan bahwa vaksin COVID-19 aman. Ini mengacu pada keyakinan bahwa vaksin COVID-19 relatif aman dan andal, tanpa efek samping. Jika demikian, jawab 'yes'; jika tidak, jawab 'no' (silakan jawab dalam bahasa Inggris).",

        # Malay
        "Kandungan tweet: {text}. Sila tandakan sama ada tweet ini menyebutkan bahawa vaksin COVID-19 selamat. Ini merujuk kepada kepercayaan bahawa vaksin COVID-19 adalah agak selamat dan boleh dipercayai, tanpa kesan sampingan. Jika ya, jawab 'yes'; jika tidak, jawab 'no' (sila jawab dalam Bahasa Inggeris).",

        # Lao
        "ເນື້ອຫາທວິດ: {text}. ກະລຸນາໝາຍວ່າທວິດນີ້ໄດ້ກ່າວເຖິງການທີ່ວັກຊີນ COVID-19 ປອດໄພຫຼືບໍ່. ນີ້ແມ່ນການເຊື່ອທີ່ວ່າວັກຊີນ COVID-19 ປອດໄພແລະເຊື່ອໄດ້, ບໍ່ມີຜົນຂ້າງເຄຽງ. ຖ້າແມ່ນແລ້ວ, ກະລຸນາຕອບ 'yes'; ຖ້າບໍ່ແມ່ນ, ກະລຸນາຕອບ 'no' (ກະລຸນາຕອບເປັນພາສາອັງກິດ).",

        # Burmese
        "တွစ်အကြောင်းအရာ: {text}။ ဒီတွစ်ဟာ COVID-19 ကာကွယ်ဆေးကို လုံခြုံတယ်လို့ ပြောထားလားဆိုတာကို မှတ်ပေးပါ။ ဒါဟာ COVID-19 ကာကွယ်ဆေးကို လုံခြုံပြီး ယုံကြည်စိတ်ချရတဲ့အဖြစ် ယူဆထားတဲ့ အမြင်ကို ဖော်ပြတာဖြစ်ပါတယ်။ အကယ်၍ရှိပါက 'yes' ဖြင့် ဖြေပါ၊ မဟုတ်ပါက 'no' ဖြင့် ဖြေပါ (ကျေးဇူးပြု၍ အင်္ဂလိပ်ဘာသာဖြင့် ပြန်ဆိုပါ).",

        # Cebuano
        "Sulud sa tweet: {text}. Palihug markahi kung kini nga tweet naghisgot nga ang bakuna sa COVID-19 luwas. Kini nagtumong sa pagtuo nga ang bakuna sa COVID-19 luwas ug kasaligan, nga walay mga epekto. Kung mao, tubaga og 'yes'; kung dili, tubaga og 'no' (palihug tubaga sa English).",

        # Khmer
        "ខ្លឹមសារទីចេញផ្សាយ: {text}។ សូមសម្គាល់មើលថាតើការផ្សាយនេះបាននិយាយអំពីសុវត្ថិភាពនៃវ៉ាក់សាំង COVID-19 ឬអត់។ នេះសំដៅទៅលើការជឿជាក់ថាវ៉ាក់សាំង COVID-19 មានសុវត្ថិភាព និងមានការជឿជាក់។ ប្រសិនបើចង់និយាយបញ្ចប់សូមឆ្លើយតបដោយនិយាយថា 'yes'។ បើមិនបានបញ្ចប់សូមឆ្លើយតបដោយនិយាយថា 'no'។ (សូមឆ្លើយតបជាភាសាអង់គ្លេស).",

        # Tagalog
        "Nilalaman ng tweet: {text}. Paki-markahan kung ang tweet na ito ay nagpapahayag na ang bakuna sa COVID-19 ay ligtas. Ito ay tumutukoy sa paniniwalang ang bakuna sa COVID-19 ay medyo ligtas at maaasahan, na walang mga masamang epekto. Kung oo, sagutin ang 'yes'; kung hindi, sagutin ang 'no' (mangyaring sagutin sa Ingles).",

        # Hindi
        "ट्वीट सामग्री: {text}। कृपया चिह्नित करें कि यह ट्वीट COVID-19 वैक्सीन को सुरक्षित मानता है या नहीं। इसका अर्थ है कि COVID-19 वैक्सीन को अपेक्षाकृत सुरक्षित और विश्वसनीय माना जाता है, और इसके कोई प्रतिकूल प्रभाव नहीं हैं। यदि हाँ, तो 'yes' में उत्तर दें; अन्यथा, 'no' में उत्तर दें (कृपया अंग्रेजी में उत्तर दें)।",

        # Bengali
        "টুইটের বিষয়বস্তু: {text}। অনুগ্রহ করে চিহ্নিত করুন যে এই টুইটটি COVID-19 ভ্যাকসিনকে নিরাপদ বলে উল্লেখ করে কিনা। এর অর্থ হল COVID-19 ভ্যাকসিনকে তুলনামূলকভাবে নিরাপদ এবং নির্ভরযোগ্য বলে মনে করা হয়, কোনও প্রতিকূল প্রভাব ছাড়াই। যদি হ্যাঁ, তাহলে 'yes' দিয়ে উত্তর দিন; অন্যথায়, 'no' দিয়ে উত্তর দিন (অনুগ্রহ করে ইংরেজিতে উত্তর দিন)।",

        # Urdu
        "ٹویٹ کا مواد: {text}. براہ کرم نشان لگائیں کہ آیا یہ ٹویٹ COVID-19 ویکسین کو محفوظ سمجھتا ہے۔ اس کا مطلب ہے کہ COVID-19 ویکسین کو نسبتا محفوظ اور قابل اعتماد سمجھا جاتا ہے، اور اس کے کوئی منفی اثرات نہیں ہیں۔ اگر ایسا ہے تو، 'yes' میں جواب دیں؛ ورنہ، 'no' میں جواب دیں (براہ کرم انگریزی میں جواب دیں)."
    ],
    '2.2': [
        # Chinese
        "推特帖子内容：{text}。请标记这条推特帖子是否提到认为新冠疫苗不安全。这里指怀疑新冠疫苗的安全性，认为可能有不良反应，或对健康带来损害。如果是，回答 'yes'；否则，回答 'no' (请用英语回答)。",

        # English
        "Tweet content: {text}. Please mark whether this tweet mentions that the COVID-19 vaccine is unsafe. This refers to doubts about the vaccine's safety, the possibility of adverse reactions, or potential harm to health. If so, answer 'yes'; otherwise, respond with 'no' (please answer in English).",

        # German
        "Tweet-Inhalt: {text}. Bitte markieren Sie, ob dieser Tweet erwähnt, dass der COVID-19-Impfstoff unsicher ist. Dies bezieht sich auf Zweifel an der Sicherheit des Impfstoffs, die Möglichkeit von Nebenwirkungen oder potenzielle Gesundheitsschäden. Wenn ja, antworten Sie mit 'yes'; andernfalls antworten Sie mit 'no' (bitte antworten Sie auf Englisch).",

        # French
        "Contenu du tweet : {text}. Veuillez indiquer si ce tweet mentionne que le vaccin COVID-19 n'est pas sûr. Cela fait référence aux doutes sur la sécurité du vaccin, la possibilité de réactions indésirables ou les dommages potentiels pour la santé. Si oui, répondez 'yes'; sinon, répondez 'no' (veuillez répondre en anglais).",

        # Spanish
        "Contenido del tweet: {text}. Por favor, marque si este tweet menciona que la vacuna COVID-19 no es segura. Esto se refiere a las dudas sobre la seguridad de la vacuna, la posibilidad de reacciones adversas o el daño potencial para la salud. Si es así, responda 'yes'; de lo contrario, responda 'no' (por favor responda en inglés).",

        # Portuguese
        "Conteúdo do tweet: {text}. Por favor, indique se este tweet menciona que a vacina COVID-19 não é segura. Isso se refere a dúvidas sobre a segurança da vacina, a possibilidade de reações adversas ou danos potenciais à saúde. Se sim, responda 'yes'; caso contrário, responda 'no' (por favor, responda em inglês).",

        # Italian
        "Contenuto del tweet: {text}. Si prega di indicare se questo tweet menziona che il vaccino COVID-19 non è sicuro. Questo si riferisce ai dubbi sulla sicurezza del vaccino, alla possibilità di reazioni avverse o ai potenziali danni alla salute. Se sì, rispondi 'yes'; altrimenti rispondi 'no' (si prega di rispondere in inglese).",

        # Dutch
        "Tweet-inhoud: {text}. Markeer of deze tweet vermeldt dat het COVID-19-vaccin onveilig is. Dit verwijst naar twijfels over de veiligheid van het vaccin, de mogelijkheid van bijwerkingen of mogelijke schade aan de gezondheid. Als dat zo is, antwoord dan 'yes'; anders antwoord 'no' (antwoord alstublieft in het Engels).",

        # Russian
        "Содержание твита: {text}. Отметьте, упоминает ли этот твит, что вакцина от COVID-19 небезопасна. Это относится к сомнениям в безопасности вакцины, возможности побочных реакций или потенциальному вреду для здоровья. Если да, ответьте 'yes'; в противном случае ответьте 'no' (пожалуйста, ответьте на английском языке).",

        # Czech
        "Obsah tweetu: {text}. Označte, zda tento tweet zmiňuje, že vakcína proti COVID-19 není bezpečná. To se týká pochybností o bezpečnosti vakcíny, možnosti nežádoucích účinků nebo potenciálního poškození zdraví. Pokud ano, odpovězte 'yes'; v opačném případě odpovězte 'no' (prosím, odpovězte v angličtině).",

        # Polish
        "Treść tweeta: {text}. Proszę zaznaczyć, czy ten tweet wspomina, że ​​szczepionka na COVID-19 jest niebezpieczna. To odnosi się do wątpliwości dotyczących bezpieczeństwa szczepionki, możliwości wystąpienia działań niepożądanych lub potencjalnych szkód dla zdrowia. Jeśli tak, odpowiedz 'yes'; w przeciwnym razie odpowiedz 'no' (proszę odpowiedzieć po angielsku).",

        # Arabic
        "محتوى التغريدة: {text}. يُرجى تحديد ما إذا كانت هذه التغريدة تذكر أن لقاح COVID-19 غير آمن. يشير هذا إلى الشكوك حول سلامة اللقاح، وإمكانية حدوث تفاعلات سلبية، أو الضرر المحتمل على الصحة. إذا كان الأمر كذلك، أجب 'yes'؛ وإلا، أجب 'no' (يرجى الرد بالإنجليزية).",

        # Persian
        "محتوای توییت: {text}. لطفاً مشخص کنید که آیا این توییت به ایمن نبودن واکسن COVID-19 اشاره دارد یا خیر. این به شک و تردید در مورد ایمنی واکسن، امکان بروز عوارض جانبی یا آسیب‌های احتمالی به سلامت اشاره دارد. اگر بله، با 'yes' پاسخ دهید؛ در غیر این صورت، با 'no' پاسخ دهید (لطفاً به انگلیسی پاسخ دهید).",

        # Hebrew
        "תוכן הציוץ: {text}. נא לסמן אם הציוץ הזה מזכיר שהחיסון ל-COVID-19 לא בטוח. זה מתייחס לספקות בנוגע לבטיחות החיסון, האפשרות לתגובות שליליות, או נזק פוטנציאלי לבריאות. אם כן, ענה 'yes'; אחרת ענה 'no' (אנא ענה באנגלית).",

        # Turkish
        "Tweet içeriği: {text}. Lütfen bu tweetin COVID-19 aşısının güvenli olmadığını belirtip belirtmediğini işaretleyin. Bu, aşının güvenliği, olumsuz reaksiyonların olasılığı veya sağlığa potansiyel zarar hakkında şüpheleri ifade eder. Eğer öyleyse, 'yes' ile cevap verin; aksi takdirde, 'no' ile cevap verin (lütfen İngilizce cevaplayın).",

        # Japanese
        "ツイート内容：{text}。このツイートがCOVID-19ワクチンが安全ではないと示しているかどうかをマークしてください。これは、ワクチンの安全性についての疑念、副作用の可能性、または健康への潜在的な危害についての懸念を指します。もしそうなら、「yes」と答えてください。そうでない場合は「no」と答えてください（英語でお答えください）。",

        # Korean
        "트윗 내용: {text}. 이 트윗이 COVID-19 백신이 안전하지 않다고 언급하는지 표시해 주세요. 이는 백신의 안전성에 대한 의구심, 부작용의 가능성 또는 건강에 대한 잠재적 피해에 대한 우려를 의미합니다. 그렇다면 'yes'로 대답해 주세요; 그렇지 않으면 'no'로 대답해 주세요 (영어로 대답해 주세요).",

        # Vietnamese
        "Nội dung tweet: {text}. Vui lòng đánh dấu xem tweet này có đề cập đến việc vắc-xin COVID-19 không an toàn hay không. Điều này ám chỉ rằng có những nghi ngờ về tính an toàn của vắc-xin, khả năng xảy ra phản ứng bất lợi hoặc tổn hại tiềm tàng đến sức khỏe. Nếu có, hãy trả lời 'yes'; nếu không, hãy trả lời 'no' (vui lòng trả lời bằng tiếng Anh).",

        # Thai
        "เนื้อหาทวีต: {text}. กรุณาระบุว่าทวีตนี้กล่าวถึงวัคซีน COVID-19 ว่าไม่ปลอดภัยหรือไม่ ซึ่งหมายถึงข้อสงสัยเกี่ยวกับความปลอดภัยของวัคซีน ความเป็นไปได้ของปฏิกิริยาไม่พึงประสงค์ หรืออันตรายที่อาจเกิดขึ้นต่อสุขภาพ หากเป็นเช่นนั้น ให้ตอบ 'yes' มิฉะนั้น ให้ตอบ 'no' (โปรดตอบเป็นภาษาอังกฤษ).",

        # Indonesian
        "Konten tweet: {text}. Silakan tandai apakah tweet ini menyebutkan bahwa vaksin COVID-19 tidak aman. Ini mengacu pada keraguan tentang keamanan vaksin, kemungkinan reaksi merugikan, atau potensi bahaya bagi kesehatan. Jika demikian, jawab 'yes'; jika tidak, jawab 'no' (silakan jawab dalam bahasa Inggris).",

        # Malay
        "Kandungan tweet: {text}. Sila tandakan sama ada tweet ini menyebutkan bahawa vaksin COVID-19 tidak selamat. Ini merujuk kepada keraguan tentang keselamatan vaksin, kemungkinan reaksi buruk, atau potensi bahaya kepada kesihatan. Jika ya, jawab 'yes'; jika tidak, jawab 'no' (sila jawab dalam Bahasa Inggeris).",

        # Lao
        "ເນື້ອຫາທວິດ: {text}. ກະລຸນາໝາຍວ່າທວິດນີ້ໄດ້ພາກົດເອົາຄວາມຄິດຄົມຂອງຄວາມບໍ່ປອດໄພຂອງວັກຊີນ COVID-19 ຫຼືບໍ່. ນີ້ແມ່ນການທີ່ຈະສະແດງເຫັນວ່າມີຄວາມສົມຄວນຂອງຄວາມບໍ່ປອດໄພຂອງວັກຊີນ, ຄວາມເປັນໄປໄດ້ຂອງການຕອບໂຕທີ່ອາດມີຜົນຂ້າງເຄຽງຫຼືບໍ່ດີທາງສຸຂະພາບ. ຖ້າແມ່ນແລ້ວ, ກະລຸນາຕອບ 'yes'; ຖ້າບໍ່ແມ່ນ, ກະລຸນາຕອບ 'no' (ກະລຸນາຕອບເປັນພາສາອັງກິດ).",

        # Burmese
        "တွစ်အကြောင်းအရာ: {text}။ ဒီတွစ်ဟာ COVID-19 ကာကွယ်ဆေးထိုးခြင်းမှာ လုံခြုံမှုမရှိဘူးလို့ ဖော်ပြထားတယ်ဆိုတာကို မှတ်ပေးပါ။ ဒါဟာ COVID-19 ကာကွယ်ဆေးထိုးခြင်းမှာ အန္တရာယ်ရှိမယ်၊ ဘေးထွက်ဆိုးကျိုးရှိနိုင်တယ်၊ ဒါမှမဟုတ် ကျန်းမာရေးကို ထိခိုက်စေမယ်ဆိုတဲ့စိုးရိမ်ချက်ကို ဖော်ပြထားတာပါ။ အကယ်၍ရှိပါက 'yes' ဖြင့် ဖြေပါ၊ မဟုတ်ပါက 'no' ဖြင့် ဖြေပါ (ကျေးဇူးပြု၍ အင်္ဂလိပ်ဘာသာဖြင့် ပြန်ဆိုပါ).",

        # Cebuano
        "Sulud sa tweet: {text}. Palihug markahi kung kini nga tweet naghisgot nga ang bakuna sa COVID-19 dili luwas. Kini nagtumong sa mga pagduhaduha bahin sa kaluwasan sa bakuna, ang posibilidad nga adunay mga negatibong reaksyon, o ang potensyal nga kadaot sa panglawas. Kung mao, tubaga og 'yes'; kung dili, tubaga og 'no' (palihug tubaga sa English).",

        # Khmer
        "ខ្លឹមសារទីចេញផ្សាយ: {text}។ សូមសម្គាល់មើលថាតើការផ្សាយនេះបាននិយាយអំពីវ៉ាក់សាំង COVID-19 មិនមានសុវត្ថិភាព ឬអត់។ នេះមានន័យថាសំដៅទៅលើការចាប់អារម្មណ៍លើសុវត្ថិភាពរបស់វ៉ាក់សាំង COVID-19 និងអាចមានបញ្ហាផ្សេងៗចំការប្រតិកម្មផ្ទុយទាំងនិងជាពន្យល់ដ៏ធំមួយ។ ប្រសិនបើចង់និយាយបញ្ចប់សូមឆ្លើយតបដោយនិយាយថា 'yes'។ បើមិនបានបញ្ចប់សូមឆ្លើយតបដោយនិយាយថា 'no'។ (សូមឆ្លើយតបជាភាសាអង់គ្លេស).",

        # Tagalog
        "Nilalaman ng tweet: {text}. Paki-markahan kung ang tweet na ito ay nagpapahayag na ang bakuna sa COVID-19 ay hindi ligtas. Ito ay tumutukoy sa mga pagdududa tungkol sa kaligtasan ng bakuna, ang posibilidad ng mga negatibong reaksyon, o potensyal na pinsala sa kalusugan. Kung oo, sagutin ang 'yes'; kung hindi, sagutin ang 'no' (mangyaring sagutin sa Ingles).",

        # Hindi
        "ट्वीट सामग्री: {text}। कृपया चिह्नित करें कि यह ट्वीट COVID-19 वैक्सीन को असुरक्षित मानता है या नहीं। इसका अर्थ है कि वैक्सीन की सुरक्षा के बारे में संदेह, प्रतिकूल प्रतिक्रियाओं की संभावना, या स्वास्थ्य को संभावित नुकसान की आशंका है। यदि हाँ, तो 'yes' में उत्तर दें; अन्यथा, 'no' में उत्तर दें (कृपया अंग्रेजी में उत्तर दें)।",

        # Bengali
        "টুইটের বিষয়বস্তু: {text}। অনুগ্রহ করে চিহ্নিত করুন যে এই টুইটটি COVID-19 ভ্যাকসিনকে অনিরাপদ বলে উল্লেখ করে কিনা। এর অর্থ হল ভ্যাকসিনের সুরক্ষা সম্পর্কে সন্দেহ, প্রতিকূল প্রতিক্রিয়ার সম্ভাবনা বা স্বাস্থ্যের সম্ভাব্য ক্ষতি। যদি হ্যাঁ, তাহলে 'yes' দিয়ে উত্তর দিন; অন্যথায়, 'no' দিয়ে উত্তর দিন (অনুগ্রহ করে ইংরেজিতে উত্তর দিন)।",

        # Urdu
        "ٹویٹ کا مواد: {text}. براہ کرم نشان لگائیں کہ آیا یہ ٹویٹ COVID-19 ویکسین کو غیر محفوظ سمجھتا ہے۔ اس کا مطلب ہے ویکسین کی حفاظت کے بارے میں شکوک و شبہات، منفی ردعمل کے امکانات، یا صحت کو ممکنہ نقصان کا خدشہ ہے۔ اگر ایسا ہے تو، 'yes' میں جواب دیں؛ ورنہ، 'no' میں جواب دیں (براہ کرم انگریزی میں جواب دیں)."
    ],
    '3.1': [
        # Chinese
        "推特帖子内容：{text}。请标记这条推特帖子是否提到认为新冠疫苗有效。这里指认为新冠疫苗是有效的，可产生抗体，或具有预防新冠肺炎的效果（有效性正面评价）。如果是，回答 'yes'；否则，回答 'no' (请用英语回答)。",

        # English
        "Tweet content: {text}. Please mark whether this tweet mentions that the COVID-19 vaccine is effective. This refers to the belief that the vaccine is effective, can produce antibodies, or has the effect of preventing COVID-19 (positive evaluation of efficacy). If so, answer 'yes'; otherwise, respond with 'no' (please answer in English).",

        # German
        "Tweet-Inhalt: {text}. Bitte markieren Sie, ob dieser Tweet erwähnt, dass der COVID-19-Impfstoff wirksam ist. Dies bezieht sich auf die Ansicht, dass der Impfstoff wirksam ist, Antikörper produzieren kann oder eine Wirkung zur Vorbeugung von COVID-19 hat (positive Bewertung der Wirksamkeit). Wenn ja, antworten Sie mit 'yes'; andernfalls antworten Sie mit 'no' (bitte antworten Sie auf Englisch).",

        # French
        "Contenu du tweet : {text}. Veuillez indiquer si ce tweet mentionne que le vaccin COVID-19 est efficace. Cela fait référence à la croyance que le vaccin est efficace, peut produire des anticorps ou a l'effet de prévenir le COVID-19 (évaluation positive de l'efficacité). Si oui, répondez 'yes'; sinon, répondez 'no' (veuillez répondre en anglais).",

        # Spanish
        "Contenido del tweet: {text}. Por favor, marque si este tweet menciona que la vacuna COVID-19 es efectiva. Esto se refiere a la creencia de que la vacuna es efectiva, puede producir anticuerpos o tiene el efecto de prevenir el COVID-19 (evaluación positiva de la eficacia). Si es así, responda 'yes'; de lo contrario, responda 'no' (por favor responda en inglés).",

        # Portuguese
        "Conteúdo do tweet: {text}. Por favor, indique se este tweet menciona que a vacina COVID-19 é eficaz. Isso se refere à crença de que a vacina é eficaz, pode produzir anticorpos ou tem o efeito de prevenir o COVID-19 (avaliação positiva da eficácia). Se sim, responda 'yes'; caso contrário, responda 'no' (por favor, responda em inglês).",

        # Italian
        "Contenuto del tweet: {text}. Si prega di indicare se questo tweet menziona che il vaccino COVID-19 è efficace. Questo si riferisce alla convinzione che il vaccino sia efficace, possa produrre anticorpi o abbia l'effetto di prevenire il COVID-19 (valutazione positiva dell'efficacia). Se sì, rispondi 'yes'; altrimenti rispondi 'no' (si prega di rispondere in inglese).",

        # Dutch
        "Tweet-inhoud: {text}. Markeer of deze tweet vermeldt dat het COVID-19-vaccin effectief is. Dit verwijst naar de overtuiging dat het vaccin effectief is, antilichamen kan produceren of het effect heeft om COVID-19 te voorkomen (positieve evaluatie van de werkzaamheid). Als dat zo is, antwoord dan 'yes'; anders antwoord 'no' (antwoord alstublieft in het Engels).",

        # Russian
        "Содержание твита: {text}. Отметьте, упоминает ли этот твит, что вакцина от COVID-19 эффективна. Это относится к мнению, что вакцина эффективна, может производить антитела или оказывает действие по предотвращению COVID-19 (положительная оценка эффективности). Если да, ответьте 'yes'; в противном случае ответьте 'no' (пожалуйста, ответьте на английском языке).",

        # Czech
        "Obsah tweetu: {text}. Označte, zda tento tweet zmiňuje, že vakcína proti COVID-19 je účinná. To se týká přesvědčení, že vakcína je účinná, může produkovat protilátky nebo má účinek při prevenci COVID-19 (pozitivní hodnocení účinnosti). Pokud ano, odpovězte 'yes'; v opačném případě odpovězte 'no' (prosím, odpovězte v angličtině).",

        # Polish
        "Treść tweeta: {text}. Proszę zaznaczyć, czy ten tweet wspomina, że ​​szczepionka na COVID-19 jest skuteczna. To odnosi się do przekonania, że ​​szczepionka jest skuteczna, może wytwarzać przeciwciała lub działać w zapobieganiu COVID-19 (pozytywna ocena skuteczności). Jeśli tak, odpowiedz 'yes'; w przeciwnym razie odpowiedz 'no' (proszę odpowiedzieć po angielsku).",

        # Arabic
        "محتوى التغريدة: {text}. يُرجى تحديد ما إذا كانت هذه التغريدة تذكر أن لقاح COVID-19 فعال. يشير هذا إلى الاعتقاد بأن اللقاح فعال، ويمكن أن ينتج أجسامًا مضادة، أو له تأثير في الوقاية من COVID-19 (تقييم إيجابي للفعالية). إذا كان الأمر كذلك، أجب 'yes'؛ وإلا، أجب 'no' (يرجى الرد بالإنجليزية).",

        # Persian
        "محتوای توییت: {text}. لطفاً مشخص کنید که آیا این توییت به مؤثر بودن واکسن COVID-19 اشاره دارد یا خیر. این به این باور اشاره دارد که واکسن مؤثر است، می‌تواند آنتی‌بادی تولید کند یا در پیشگیری از COVID-19 مؤثر باشد (ارزیابی مثبت از اثربخشی). اگر بله، با 'yes' پاسخ دهید؛ در غیر این صورت، با 'no' پاسخ دهید (لطفاً به انگلیسی پاسخ دهید).",

        # Hebrew
        "תוכן הציוץ: {text}. נא לסמן אם הציוץ הזה מזכיר שהחיסון ל-COVID-19 יעיל. זה מתייחס לאמונה שהחיסון יעיל, יכול לייצר נוגדנים, או שיש לו השפעה במניעת COVID-19 (הערכה חיובית של היעילות). אם כן, ענה 'yes'; אחרת ענה 'no' (אנא ענה באנגלית).",

        # Turkish
        "Tweet içeriği: {text}. Lütfen bu tweetin COVID-19 aşısının etkili olduğunu belirtip belirtmediğini işaretleyin. Bu, aşının etkili olduğu, antikor üretebileceği veya COVID-19'u önlemede etkisi olduğu inancını ifade eder (etkinliğin olumlu değerlendirilmesi). Eğer öyleyse, 'yes' ile cevap verin; aksi takdirde, 'no' ile cevap verin (lütfen İngilizce cevaplayın).",

        # Japanese
        "ツイート内容：{text}。このツイートがCOVID-19ワクチンが効果的であると示しているかどうかをマークしてください。これは、ワクチンが効果的であり、抗体を生成できるか、COVID-19を予防する効果があると信じられていることを指します（有効性の肯定的な評価）。もしそうなら、「yes」と答えてください。そうでない場合は「no」と答えてください（英語でお答えください）。",

        # Korean
        "트윗 내용: {text}. 이 트윗이 COVID-19 백신이 효과적이라고 언급하는지 표시해 주세요. 이는 백신이 효과적이며 항체를 생성하거나 COVID-19를 예방하는 효과가 있다는 믿음을 의미합니다 (효능에 대한 긍정적인 평가). 그렇다면 'yes'로 대답해 주세요; 그렇지 않으면 'no'로 대답해 주세요 (영어로 대답해 주세요).",

        # Vietnamese
        "Nội dung tweet: {text}. Vui lòng đánh dấu xem tweet này có đề cập đến việc vắc-xin COVID-19 có hiệu quả hay không. Điều này ám chỉ rằng vắc-xin có hiệu quả, có thể tạo ra kháng thể hoặc có tác dụng ngăn ngừa COVID-19 (đánh giá tích cực về hiệu quả). Nếu có, hãy trả lời 'yes'; nếu không, hãy trả lời 'no' (vui lòng trả lời bằng tiếng Anh).",

        # Thai
        "เนื้อหาทวีต: {text}. กรุณาระบุว่าทวีตนี้กล่าวถึงความมีประสิทธิผลของวัคซีน COVID-19 หรือไม่ ซึ่งหมายถึงความเชื่อว่าวัคซีนมีประสิทธิผล สามารถสร้างแอนติบอดี หรือมีผลในการป้องกัน COVID-19 (การประเมินเชิงบวกเกี่ยวกับประสิทธิผล) หากเป็นเช่นนั้น ให้ตอบ 'yes' มิฉะนั้น ให้ตอบ 'no' (โปรดตอบเป็นภาษาอังกฤษ).",

        # Indonesian
        "Konten tweet: {text}. Silakan tandai apakah tweet ini menyebutkan bahwa vaksin COVID-19 efektif. Ini mengacu pada keyakinan bahwa vaksin itu efektif, dapat menghasilkan antibodi, atau memiliki efek mencegah COVID-19 (evaluasi positif dari kemanjuran). Jika demikian, jawab 'yes'; jika tidak, jawab 'no' (silakan jawab dalam bahasa Inggris).",

        # Malay
        "Kandungan tweet: {text}. Sila tandakan sama ada tweet ini menyebutkan bahawa vaksin COVID-19 berkesan. Ini merujuk kepada kepercayaan bahawa vaksin ini berkesan, boleh menghasilkan antibodi atau mempunyai kesan dalam mencegah COVID-19 (penilaian positif keberkesanan). Jika ya, jawab 'yes'; jika tidak, jawab 'no' (sila jawab dalam Bahasa Inggeris).",

        # Lao
        "ເນື້ອຫາທວິດ: {text}. ກະລຸນາໝາຍວ່າທວິດນີ້ໄດ້ພາກົດເອົາຄວາມຄິດຄົມຂອງຄວາມມີປະສິດທິພາບຂອງວັກຊີນ COVID-19 ຫຼືບໍ່. ນີ້ແມ່ນການສະແດງເຫັນວ່າມີຄວາມເຊື່ອໃນປະສິດທິພາບຂອງວັກຊີນທີ່ສາມາດຜະລິດພັນທານໄດ້ຫລືມີຜົນສໍາລັບການປ້ອງກັນ COVID-19 (ການປະເມີນຜົນງານຊອງຫຼັງທີ່ດີ). ຖ້າແມ່ນແລ້ວ, ກະລຸນາຕອບວ່າ 'yes'; ຖ້າບໍ່ແມ່ນ, ກະລຸນາຕອບວ່າ 'no' (ກະລຸນາຕອບໃນພາສາອັງກິດ).",

        # Burmese
        "တွစ်အကြောင်းအရာ: {text}။ ဒီတွစ်ဟာ COVID-19 ကာကွယ်ဆေးထိုးခြင်းမှာ အကျိုးရှိတယ်လို့ ဖော်ပြထားတယ်ဆိုတာကို မှတ်ပေးပါ။ ဒါဟာ COVID-19 ကာကွယ်ဆေးထိုးခြင်းမှာ အကျိုးရှိတယ်၊ အန္တရာယ်ကင်းစေတယ်၊ ဒါမှမဟုတ် COVID-19 ကို ကာကွယ်နိုင်မယ်ဆိုတဲ့ ယုံကြည်ချက်ကို ဖော်ပြထားတာပါ (ထိရောက်မှုအပေါ် အကောင်းဘက်သို့ ညွှန်းချက်များ). အကယ်၍ရှိပါက 'yes' ဖြင့် ဖြေပါ၊ မဟုတ်ပါက 'no' ဖြင့် ဖြေပါ (ကျေးဇူးပြု၍ အင်္ဂလိပ်ဘာသာဖြင့် ပြန်ဆိုပါ).",

        # Cebuano
        "Sulud sa tweet: {text}. Palihug markahi kung kini nga tweet naghisgot nga ang bakuna sa COVID-19 epektibo. Kini nagtumong sa pagtuo nga ang bakuna epektibo, makaproduce og antibodies, o adunay epekto sa pagpugong sa COVID-19 (positibo nga ebalwasyon sa pagkaepektibo). Kung mao, tubaga og 'yes'; kung dili, tubaga og 'no' (palihug tubaga sa English).",

        # Khmer
        "ខ្លឹមសារទីចេញផ្សាយ: {text}។ សូមសម្គាល់មើលថាតើការផ្សាយនេះបាននិយាយអំពីសក្ដានុពលនៃវ៉ាក់សាំង COVID-19 ឬអត់។ នេះមានន័យថាសំដៅទៅលើការជឿជាក់ថាវ៉ាក់សាំងមានសក្ដានុពលក្នុងការផលិតវ៉ាក់សាំង ឬវាអាចមានប្រសិទ្ធភាពក្នុងការការពាររោគ COVID-19 (ការវាយតម្លៃវិជ្ជមានអំពីសក្ដានុពល)។ ប្រសិនបើចង់និយាយបញ្ចប់សូមឆ្លើយតបដោយនិយាយថា 'yes'។ បើមិនបានបញ្ចប់សូមឆ្លើយតបដោយនិយាយថា 'no'។ (សូមឆ្លើយតបជាភាសាអង់គ្លេស).",

        # Tagalog
        "Nilalaman ng tweet: {text}. Paki-markahan kung ang tweet na ito ay nagpapahayag na ang bakuna sa COVID-19 ay epektibo. Ito ay tumutukoy sa paniniwala na ang bakuna ay epektibo, maaaring makabuo ng mga antibodies, o may epekto sa pag-iwas sa COVID-19 (positibong pagsusuri ng pagiging epektibo). Kung oo, sagutin ang 'yes'; kung hindi, sagutin ang 'no' (mangyaring sagutin sa Ingles).",

        # Hindi
        "ट्वीट सामग्री: {text}। कृपया चिह्नित करें कि यह ट्वीट COVID-19 वैक्सीन को प्रभावी मानता है या नहीं। इसका अर्थ है कि वैक्सीन को प्रभावी माना जाता है, एंटीबॉडी उत्पन्न कर सकता है, या COVID-19 को रोकने का प्रभाव रखता है (प्रभावशीलता का सकारात्मक मूल्यांकन)। यदि हाँ, तो 'yes' में उत्तर दें; अन्यथा, 'no' में उत्तर दें (कृपया अंग्रेजी में उत्तर दें)।",

        # Bengali
        "টুইটের বিষয়বস্তু: {text}। অনুগ্রহ করে চিহ্নিত করুন যে এই টুইটটি COVID-19 ভ্যাকসিনকে কার্যকর বলে উল্লেখ করে কিনা। এর অর্থ হল ভ্যাকসিনকে কার্যকর হিসাবে মনে করা হয়, অ্যান্টিবডি উত্পাদন করতে পারে বা COVID-19 কে প্রতিরোধ করার প্রভাব রয়েছে (কার্যকারিতার ইতিবাচক মূল্যায়ন)। যদি হ্যাঁ, তাহলে 'yes' দিয়ে উত্তর দিন; অন্যথায়, 'no' দিয়ে উত্তর দিন (অনুগ্রহ করে ইংরেজিতে উত্তর দিন)।",

        # Urdu
        "ٹویٹ کا مواد: {text}. براہ کرم نشان لگائیں کہ آیا یہ ٹویٹ COVID-19 ویکسین کو مؤثر سمجھتا ہے۔ اس کا مطلب ہے کہ ویکسین کو مؤثر سمجھا جاتا ہے، اینٹی باڈیز پیدا کر سکتا ہے، یا COVID-19 کو روکنے کا اثر رکھتا ہے (افادیت کا مثبت جائزہ). اگر ایسا ہے تو، 'yes' میں جواب دیں؛ ورنہ، 'no' میں جواب دیں (براہ کرم انگریزی میں جواب دیں)."
    ],
    '4.1': [
        # Chinese
        "推特帖子内容：{text}。请标记这条推特帖子是否提到认为新冠疫苗是重要的、必要的或必须的。如果是，回答 'yes'；否则，回答 'no' (请用英语回答)。",

        # English
        "Tweet content: {text}. Please mark whether this tweet mentions that the COVID-19 vaccine is important, necessary, or essential. If so, answer 'yes'; otherwise, respond with 'no' (please answer in English).",

        # German
        "Tweet-Inhalt: {text}. Bitte markieren Sie, ob dieser Tweet erwähnt, dass der COVID-19-Impfstoff wichtig, notwendig oder unverzichtbar ist. Wenn ja, antworten Sie mit 'yes'; andernfalls antworten Sie mit 'no' (bitte antworten Sie auf Englisch).",

        # French
        "Contenu du tweet : {text}. Veuillez indiquer si ce tweet mentionne que le vaccin COVID-19 est important, nécessaire ou essentiel. Si oui, répondez 'yes'; sinon, répondez 'no' (veuillez répondre en anglais).",

        # Spanish
        "Contenido del tweet: {text}. Por favor, marque si este tweet menciona que la vacuna COVID-19 es importante, necesaria o esencial. Si es así, responda 'yes'; de lo contrario, responda 'no' (por favor responda en inglés).",

        # Portuguese
        "Conteúdo do tweet: {text}. Por favor, indique se este tweet menciona que a vacina COVID-19 é importante, necessária ou essencial. Se sim, responda 'yes'; caso contrário, responda 'no' (por favor, responda em inglês).",

        # Italian
        "Contenuto del tweet: {text}. Si prega di indicare se questo tweet menziona che il vaccino COVID-19 è importante, necessario o essenziale. Se sì, rispondi 'yes'; altrimenti rispondi 'no' (si prega di rispondere in inglese).",

        # Dutch
        "Tweet-inhoud: {text}. Markeer of deze tweet vermeldt dat het COVID-19-vaccin belangrijk, noodzakelijk of essentieel is. Als dat zo is, antwoord dan 'yes'; anders antwoord 'no' (antwoord alstublieft in het Engels).",

        # Russian
        "Содержание твита: {text}. Отметьте, упоминает ли этот твит, что вакцина от COVID-19 важна, необходима или существенна. Если да, ответьте 'yes'; в противном случае ответьте 'no' (пожалуйста, ответьте на английском языке).",

        # Czech
        "Obsah tweetu: {text}. Označte, zda tento tweet zmiňuje, že vakcína proti COVID-19 je důležitá, nezbytná nebo zásadní. Pokud ano, odpovězte 'yes'; v opačném případě odpovězte 'no' (prosím, odpovězte v angličtině).",

        # Polish
        "Treść tweeta: {text}. Proszę zaznaczyć, czy ten tweet wspomina, że ​​szczepionka na COVID-19 jest ważna, konieczna lub niezbędna. Jeśli tak, odpowiedz 'yes'; w przeciwnym razie odpowiedz 'no' (proszę odpowiedzieć po angielsku).",

        # Arabic
        "محتوى التغريدة: {text}. يُرجى تحديد ما إذا كانت هذه التغريدة تذكر أن لقاح COVID-19 مهم أو ضروري أو أساسي. إذا كان الأمر كذلك، أجب 'yes'؛ وإلا، أجب 'no' (يرجى الرد بالإنجليزية).",

        # Persian
        "محتوای توییت: {text}. لطفاً مشخص کنید که آیا این توییت به مهم، ضروری یا اساسی بودن واکسن COVID-19 اشاره دارد یا خیر. اگر بله، با 'yes' پاسخ دهید؛ در غیر این صورت، با 'no' پاسخ دهید (لطفاً به انگلیسی پاسخ دهید).",

        # Hebrew
        "תוכן הציוץ: {text}. נא לסמן אם הציוץ הזה מזכיר שהחיסון ל-COVID-19 חשוב, נחוץ או חיוני. אם כן, ענה 'yes'; אחרת ענה 'no' (אנא ענה באנגלית).",

        # Turkish
        "Tweet içeriği: {text}. Lütfen bu tweetin COVID-19 aşısının önemli, gerekli veya hayati olduğunu belirtip belirtmediğini işaretleyin. Eğer öyleyse, 'yes' ile cevap verin; aksi takdirde, 'no' ile cevap verin (lütfen İngilizce cevaplayın).",

        # Japanese
        "ツイート内容：{text}。このツイートがCOVID-19ワクチンが重要、必要、または不可欠であると示しているかどうかをマークしてください。もしそうなら、「yes」と答えてください。そうでない場合は「no」と答えてください（英語でお答えください）。",

        # Korean
        "트윗 내용: {text}. 이 트윗이 COVID-19 백신이 중요하거나, 필요하거나, 필수적이라고 언급하는지 표시해 주세요. 그렇다면 'yes'로 대답해 주세요; 그렇지 않으면 'no'로 대답해 주세요 (영어로 대답해 주세요).",

        # Vietnamese
        "Nội dung tweet: {text}. Vui lòng đánh dấu xem tweet này có đề cập đến việc vắc-xin COVID-19 là quan trọng, cần thiết hay thiết yếu không. Nếu có, hãy trả lời 'yes'; nếu không, hãy trả lời 'no' (vui lòng trả lời bằng tiếng Anh).",

        # Thai
        "เนื้อหาทวีต: {text}. กรุณาระบุว่าทวีตนี้กล่าวถึงวัคซีน COVID-19 ว่าสำคัญ จำเป็น หรือจำเป็นหรือไม่ หากเป็นเช่นนั้น ให้ตอบ 'yes' มิฉะนั้น ให้ตอบ 'no' (โปรดตอบเป็นภาษาอังกฤษ).",

        # Indonesian
        "Konten tweet: {text}. Silakan tandai apakah tweet ini menyebutkan bahwa vaksin COVID-19 penting, perlu, atau penting. Jika demikian, jawab 'yes'; jika tidak, jawab 'no' (silakan jawab dalam bahasa Inggris).",

        # Malay
        "Kandungan tweet: {text}. Sila tandakan sama ada tweet ini menyebutkan bahawa vaksin COVID-19 penting, perlu atau penting. Jika ya, jawab 'yes'; jika tidak, jawab 'no' (sila jawab dalam Bahasa Inggeris).",

        # Lao
        "ເນື້ອຫາທວິດ: {text}. ກະລຸນາໝາຍວ່າທວິດນີ້ໄດ້ພາກົດເອົາຄວາມສຳຄັນຂອງວັກຊີນ COVID-19 ຫຼືບໍ່. ນີ້ແມ່ນການສະແດງໃຫ້ເຫັນຄວາມສຳຄັນ, ຄວາມຈຳເປັນ, ຫຼືຄວາມຕ້ອງການຂອງວັກຊີນ COVID-19. ຖ້າແມ່ນແລ້ວ, ກະລຸນາຕອບວ່າ 'yes'; ຖ້າບໍ່ແມ່ນ, ກະລຸນາຕອບວ່າ 'no' (ກະລຸນາຕອບໃນພາສາອັງກິດ).",

        # Burmese
        "တွစ်အကြောင်းအရာ: {text}။ ဒီတွစ်ဟာ COVID-19 ကာကွယ်ဆေးထိုးခြင်းမှာ အရေးကြီးတယ်၊ လိုအပ်တယ်၊ ဒါမှမဟုတ် မရှိမျှော်လင့်မရအောင် ဖြစ်နေတယ်ဆိုတာကို ဖော်ပြခဲ့ပါသလားဆိုတာကို မှတ်ပေးပါ။ အကယ်၍ရှိပါက 'yes' ဖြင့် ဖြေပါ၊ မဟုတ်ပါက 'no' ဖြင့် ဖြေပါ (ကျေးဇူးပြု၍ အင်္ဂလိပ်ဘာသာဖြင့် ပြန်ဆိုပါ).",

        # Cebuano
        "Sulud sa tweet: {text}. Palihug markahi kung kini nga tweet naghisgot nga ang bakuna sa COVID-19 importante, kinahanglan, o kinahanglang. Kung mao, tubaga og 'yes'; kung dili, tubaga og 'no' (palihug tubaga sa English).",

        # Khmer
        "ខ្លឹមសារទីចេញផ្សាយ: {text}។ សូមសម្គាល់មើលថាតើការផ្សាយនេះបាននិយាយអំពីសារៈសំខាន់នៃវ៉ាក់សាំង COVID-19 ឬអត់។ ប្រសិនបើចង់និយាយបញ្ចប់សូមឆ្លើយតបដោយនិយាយថា 'yes'។ បើមិនបានបញ្ចប់សូមឆ្លើយតបដោយនិយាយថា 'no'។ (សូមឆ្លើយតបជាភាសាអង់គ្លេស).",

        # Tagalog
        "Nilalaman ng tweet: {text}. Paki-markahan kung ang tweet na ito ay nagpapahayag na ang bakuna sa COVID-19 ay mahalaga, kinakailangan, o mahahalaga. Kung oo, sagutin ang 'yes'; kung hindi, sagutin ang 'no' (mangyaring sagutin sa Ingles).",

        # Hindi
        "ट्वीट सामग्री: {text}। कृपया चिह्नित करें कि यह ट्वीट COVID-19 वैक्सीन को महत्वपूर्ण, आवश्यक या आवश्यक मानता है या नहीं। यदि हाँ, तो 'yes' में उत्तर दें; अन्यथा, 'no' में उत्तर दें (कृपया अंग्रेजी में उत्तर दें)।",

        # Bengali
        "টুইটের বিষয়বস্তু: {text}। অনুগ্রহ করে চিহ্নিত করুন যে এই টুইটটি COVID-19 ভ্যাকসিনকে গুরুত্বপূর্ণ, প্রয়োজনীয় বা অপরিহার্য বলে উল্লেখ করে কিনা। যদি হ্যাঁ, তাহলে 'yes' দিয়ে উত্তর দিন; অন্যথায়, 'no' দিয়ে উত্তর দিন (অনুগ্রহ করে ইংরেজিতে উত্তর দিন)।",

        # Urdu
        "ٹویٹ کا مواد: {text}. براہ کرم نشان لگائیں کہ آیا یہ ٹویٹ COVID-19 ویکسین کو اہم، ضروری یا ضروری سمجھتا ہے۔ اگر ایسا ہے تو، 'yes' میں جواب دیں؛ ورنہ، 'no' میں جواب دیں (براہ کرم انگریزی میں جواب دیں)."
    ],
    '6.2': [
        # Chinese
        "推特帖子内容：{text}。请标记这条推特帖子是否提到对国家、政府决策者或机构的不信任或不满。如果是，回答 'yes'；否则，回答 'no' (请用英语回答)。",

        # English
        "Tweet content: {text}. Please mark whether this tweet mentions distrust or dissatisfaction with national and government decision-makers or institutions. If so, answer 'yes'; otherwise, respond with 'no' (please answer in English).",

        # German
        "Tweet-Inhalt: {text}. Bitte markieren Sie, ob dieser Tweet Misstrauen oder Unzufriedenheit gegenüber nationalen und staatlichen Entscheidungsträgern oder Institutionen erwähnt. Wenn ja, antworten Sie mit 'yes'; andernfalls antworten Sie mit 'no' (bitte antworten Sie auf Englisch).",

        # French
        "Contenu du tweet : {text}. Veuillez indiquer si ce tweet mentionne la méfiance ou l'insatisfaction à l'égard des décideurs nationaux et gouvernementaux ou des institutions. Si oui, répondez 'yes'; sinon, répondez 'no' (veuillez répondre en anglais).",

        # Spanish
        "Contenido del tweet: {text}. Por favor, marque si este tweet menciona desconfianza o insatisfacción con los responsables políticos nacionales y gubernamentales o con las instituciones. Si es así, responda 'yes'; de lo contrario, responda 'no' (por favor responda en inglés).",

        # Portuguese
        "Conteúdo do tweet: {text}. Por favor, indique se este tweet menciona desconfiança ou insatisfação com os responsáveis ​​políticos ou instituições nacionais e governamentais. Se sim, responda 'yes'; caso contrário, responda 'no' (por favor, responda em inglês).",

        # Italian
        "Contenuto del tweet: {text}. Si prega di indicare se questo tweet menziona sfiducia o insoddisfazione nei confronti dei responsabili politici nazionali e governativi o delle istituzioni. Se sì, rispondi 'yes'; altrimenti rispondi 'no' (si prega di rispondere in inglese).",

        # Dutch
        "Tweet-inhoud: {text}. Markeer of deze tweet wantrouwen of ontevredenheid vermeldt met nationale en regeringsbesluitvormers of instellingen. Als dat zo is, antwoord dan 'yes'; anders antwoord 'no' (antwoord alstublieft in het Engels).",

        # Russian
        "Содержание твита: {text}. Отметьте, упоминает ли этот твит недоверие или неудовлетворенность национальными и правительственными органами или учреждениями. Если да, ответьте 'yes'; в противном случае ответьте 'no' (пожалуйста, ответьте на английском языке).",

        # Czech
        "Obsah tweetu: {text}. Označte, zda tento tweet zmiňuje nedůvěru nebo nespokojenost s národními a vládními rozhodovacími orgány nebo institucemi. Pokud ano, odpovězte 'yes'; v opačném případě odpovězte 'no' (prosím, odpovězte v angličtině).",

        # Polish
        "Treść tweeta: {text}. Proszę zaznaczyć, czy ten tweet wspomina o braku zaufania lub niezadowoleniu z decydentów narodowych i rządowych lub instytucji. Jeśli tak, odpowiedz 'yes'; w przeciwnym razie odpowiedz 'no' (proszę odpowiedzieć po angielsku).",

        # Arabic
        "محتوى التغريدة: {text}. يُرجى تحديد ما إذا كانت هذه التغريدة تذكر عدم الثقة أو عدم الرضا عن صانعي القرار الوطنيين والحكوميين أو المؤسسات. إذا كان الأمر كذلك، أجب 'yes'؛ وإلا، أجب 'no' (يرجى الرد بالإنجليزية).",

        # Persian
        "محتوای توییت: {text}. لطفاً مشخص کنید که آیا این توییت به بی‌اعتمادی یا نارضایتی از تصمیم‌گیرندگان ملی و دولتی یا مؤسسات اشاره دارد یا خیر. اگر بله، با 'yes' پاسخ دهید؛ در غیر این صورت، با 'no' پاسخ دهید (لطفاً به انگلیسی پاسخ دهید).",

        # Hebrew
        "תוכן הציוץ: {text}. נא לסמן אם הציוץ הזה מזכיר חוסר אמון או חוסר שביעות רצון כלפי קובעי מדיניות לאומיים וממשלתיים או מוסדות. אם כן, ענה 'yes'; אחרת ענה 'no' (אנא ענה באנגלית).",

        # Turkish
        "Tweet içeriği: {text}. Lütfen bu tweetin ulusal ve hükümet karar vericilerine veya kurumlarına karşı güvensizlik veya memnuniyetsizlik belirtip belirtmediğini işaretleyin. Eğer öyleyse, 'yes' ile cevap verin; aksi takdirde, 'no' ile cevap verin (lütfen İngilizce cevaplayın).",

        # Japanese
        "ツイート内容：{text}。このツイートが国や政府の意思決定者や機関に対する不信感や不満を示しているかどうかをマークしてください。もしそうなら、「yes」と答えてください。そうでない場合は「no」と答えてください（英語でお答えください）。",

        # Korean
        "트윗 내용: {text}. 이 트윗이 국가 및 정부 결정권자 또는 기관에 대한 불신 또는 불만을 언급하는지 표시해 주세요. 그렇다면 'yes'로 대답해 주세요; 그렇지 않으면 'no'로 대답해 주세요 (영어로 대답해 주세요).",

        # Vietnamese
        "Nội dung tweet: {text}. Vui lòng đánh dấu xem tweet này có đề cập đến sự thiếu tin tưởng hoặc không hài lòng với các nhà hoạch định chính sách quốc gia và chính phủ hoặc các tổ chức hay không. Nếu có, hãy trả lời 'yes'; nếu không, hãy trả lời 'no' (vui lòng trả lời bằng tiếng Anh).",

        # Thai
        "เนื้อหาทวีต: {text}. กรุณาระบุว่าทวีตนี้กล่าวถึงความไม่ไว้วางใจหรือความไม่พอใจกับผู้มีอำนาจตัดสินใจระดับชาติและรัฐบาลหรือสถาบันต่างๆ หรือไม่ หากเป็นเช่นนั้น ให้ตอบ 'yes' มิฉะนั้น ให้ตอบ 'no' (โปรดตอบเป็นภาษาอังกฤษ).",

        # Indonesian
        "Konten tweet: {text}. Silakan tandai apakah tweet ini menyebutkan ketidakpercayaan atau ketidakpuasan dengan pembuat keputusan nasional dan pemerintah atau institusi. Jika demikian, jawab 'yes'; jika tidak, jawab 'no' (silakan jawab dalam bahasa Inggris).",

        # Malay
        "Kandungan tweet: {text}. Sila tandakan sama ada tweet ini menyebutkan ketidakpercayaan atau ketidakpuasan hati dengan pembuat keputusan dan institusi negara serta kerajaan. Jika ya, jawab 'yes'; jika tidak, jawab 'no' (sila jawab dalam Bahasa Inggeris).",

        # Lao
        "ເນື້ອຫາທວິດ: {text}. ກະລຸນາໝາຍວ່າທວິດນີ້ໄດ້ພາກົດເອົາຄວາມບໍ່ໄວ້ໃຈຫລືຄວາມບໍ່ພໍໃຈຕໍ່ນາຍຫົວເຮັດການຕໍ່ແຫ່ງຊາດ ຫລືຕໍ່ສະຖາບັນຂອງລັດຖະບານ. ຖ້າແມ່ນແລ້ວ, ກະລຸນາຕອບວ່າ 'yes'; ຖ້າບໍ່ແມ່ນ, ກະລຸນາຕອບວ່າ 'no' (ກະລຸນາຕອບໃນພາສາອັງກິດ).",

        # Burmese
        "တွစ်အကြောင်းအရာ: {text}။ ဒီတွစ်ဟာ နိုင်ငံနဲ့ဆိုင်တဲ့ အမှုဆောင်တောင်းဆိုချက်တွေ၊ နိုင်ငံတကာ အဖွဲ့အစည်းတွေကို မယုံကြည်ဘူးဆိုတာကို ဖော်ပြထားပါသလားဆိုတာကို မှတ်ပေးပါ။ အကယ်၍ရှိပါက 'yes' ဖြင့် ဖြေပါ၊ မဟုတ်ပါက 'no' ဖြင့် ဖြေပါ (ကျေးဇူးပြု၍ အင်္ဂလိပ်ဘာသာဖြင့် ပြန်ဆိုပါ).",

        # Cebuano
        "Sulud sa tweet: {text}. Palihug markahi kung kini nga tweet naghisgot sa kasaligan o kawalay kasigurohan sa mga nasudnon ug gobyerno nga mga tigdumala o mga institusyon. Kung mao, tubaga og 'yes'; kung dili, tubaga og 'no' (palihug tubaga sa English).",

        # Khmer
        "ខ្លឹមសារទីចេញផ្សាយ: {text}។ សូមសម្គាល់មើលថាតើការផ្សាយនេះបាននិយាយអំពីការមិនទុកចិត្ត ឬអំពីភាពមិនពេញចិត្តចំពោះ អ្នកសំរេចចិត្តជាន់ខ្ពស់ឬអង្គភាពរបស់រដ្ឋ ឬអត់។ ប្រសិនបើចង់និយាយបញ្ចប់សូមឆ្លើយតបដោយនិយាយថា 'yes'។ បើមិនបានបញ្ចប់សូមឆ្លើយតបដោយនិយាយថា 'no'។ (សូមឆ្លើយតបជាភាសាអង់គ្លេស).",

        # Tagalog
        "Nilalaman ng tweet: {text}. Paki-markahan kung ang tweet na ito ay nagpapahayag ng kawalan ng tiwala o hindi kasiyahan sa mga gumagawa ng desisyon sa pambansang at gobyerno o mga institusyon. Kung oo, sagutin ang 'yes'; kung hindi, sagutin ang 'no' (mangyaring sagutin sa Ingles).",

        # Hindi
        "ट्वीट सामग्री: {text}। कृपया चिह्नित करें कि यह ट्वीट राष्ट्रीय और सरकारी निर्णय निर्माताओं या संस्थानों पर अविश्वास या असंतोष का उल्लेख करता है या नहीं। यदि हाँ, तो 'yes' में उत्तर दें; अन्यथा, 'no' में उत्तर दें (कृपया अंग्रेजी में उत्तर दें)।",

        # Bengali
        "টুইটের বিষয়বস্তু: {text}। অনুগ্রহ করে চিহ্নিত করুন যে এই টুইটটি জাতীয় এবং সরকারি সিদ্ধান্ত গ্রহণকারীদের বা প্রতিষ্ঠানের প্রতি অবিশ্বাস বা অসন্তোষের উল্লেখ করে কিনা। যদি হ্যাঁ, তাহলে 'yes' দিয়ে উত্তর দিন; অন্যথায়, 'no' দিয়ে উত্তর দিন (অনুগ্রহ করে ইংরেজিতে উত্তর দিন)।",

        # Urdu
        "ٹویٹ کا مواد: {text}. براہ کرم نشان لگائیں کہ آیا یہ ٹویٹ قومی اور حکومتی پالیسی سازوں یا اداروں پر عدم اعتماد یا عدم اطمینان کا ذکر کرتا ہے۔ اگر ایسا ہے تو، 'yes' میں جواب دیں؛ ورنہ، 'no' میں جواب دیں (براہ کرم انگریزی میں جواب دیں)."
    ],
    '10': [
        # Chinese
        "推特帖子内容：{text}。请标记这条推特帖子是否提及可及性（生产与供给）。这里指新冠疫苗的生产能力、生产情况，或供应能力、供应情况等。如果是，回答 'yes'；否则，回答 'no' (请用英语回答)。",

        # English
        "Tweet content: {text}. Please mark whether this tweet mentions accessibility (production and supply). This refers to the production capacity, production situation, or supply capacity, supply situation of the COVID-19 vaccine. If so, answer 'yes'; otherwise, respond with 'no' (please answer in English).",

        # German
        "Tweet-Inhalt: {text}. Bitte markieren Sie, ob dieser Tweet die Zugänglichkeit (Produktion und Versorgung) erwähnt. Dies bezieht sich auf die Produktionskapazität, die Produktionssituation oder die Versorgungskapazität und Versorgungssituation des COVID-19-Impfstoffs. Wenn ja, antworten Sie mit 'yes'; andernfalls antworten Sie mit 'no' (bitte auf Englisch antworten).",

        # French
        "Contenu du tweet : {text}. Veuillez indiquer si ce tweet mentionne l'accessibilité (production et approvisionnement). Cela fait référence à la capacité de production, à la situation de production ou à la capacité d'approvisionnement, à la situation d'approvisionnement du vaccin COVID-19. Si oui, répondez 'yes'; sinon, répondez 'no' (veuillez répondre en anglais).",

        # Spanish
        "Contenido del tweet: {text}. Por favor, marque si este tweet menciona la accesibilidad (producción y suministro). Esto se refiere a la capacidad de producción, la situación de producción o la capacidad de suministro, la situación de suministro de la vacuna COVID-19. Si es así, responda 'yes'; de lo contrario, responda 'no' (por favor responda en inglés).",

        # Portuguese
        "Conteúdo do tweet: {text}. Por favor, indique se este tweet menciona acessibilidade (produção e fornecimento). Isso se refere à capacidade de produção, situação de produção ou capacidade de fornecimento, situação de fornecimento da vacina COVID-19. Se sim, responda 'yes'; caso contrário, responda 'no' (por favor, responda em inglês).",

        # Italian
        "Contenuto del tweet: {text}. Si prega di indicare se questo tweet menziona l'accessibilità (produzione e fornitura). Questo si riferisce alla capacità produttiva, alla situazione della produzione o alla capacità di fornitura, situazione della fornitura del vaccino COVID-19. Se sì, rispondi 'yes'; altrimenti rispondi 'no' (si prega di rispondere in inglese).",

        # Dutch
        "Tweet-inhoud: {text}. Markeer of deze tweet toegankelijkheid (productie en levering) vermeldt. Dit verwijst naar de productiecapaciteit, productiesituatie of leveringscapaciteit, leveringssituatie van het COVID-19-vaccin. Als dat zo is, antwoord dan 'yes'; anders antwoord 'no' (antwoord alstublieft in het Engels).",

        # Russian
        "Содержание твита: {text}. Отметьте, упоминает ли этот твит доступность (производство и поставка). Это относится к производственной мощности, ситуации с производством или поставочной способности, ситуации с поставками вакцины от COVID-19. Если да, ответьте 'yes'; в противном случае ответьте 'no' (пожалуйста, ответьте на английском языке).",

        # Czech
        "Obsah tweetu: {text}. Označte, zda tento tweet zmiňuje přístupnost (výroba a dodávky). To se týká výrobní kapacity, výrobní situace nebo dodavatelské kapacity, situace s dodávkami vakcíny proti COVID-19. Pokud ano, odpovězte 'yes'; v opačném případě odpovězte 'no' (prosím, odpovězte v angličtině).",

        # Polish
        "Treść tweeta: {text}. Proszę zaznaczyć, czy ten tweet wspomina o dostępności (produkcja i dostawa). To odnosi się do zdolności produkcyjnych, sytuacji produkcyjnej lub zdolności dostawczych, sytuacji z dostawą szczepionki na COVID-19. Jeśli tak, odpowiedz 'yes'; w przeciwnym razie odpowiedz 'no' (proszę odpowiedzieć po angielsku).",

        # Arabic
        "محتوى التغريدة: {text}. يُرجى تحديد ما إذا كانت هذه التغريدة تذكر إمكانية الوصول (الإنتاج والإمداد). يشير هذا إلى قدرة الإنتاج، أو وضع الإنتاج، أو قدرة الإمداد، أو وضع الإمداد للقاح COVID-19. إذا كان الأمر كذلك، أجب 'yes'؛ وإلا، أجب 'no' (يرجى الرد بالإنجليزية).",

        # Persian
        "محتوای توییت: {text}. لطفاً مشخص کنید که آیا این توییت به دسترسی‌پذیری (تولید و عرضه) اشاره دارد یا خیر. این اشاره به ظرفیت تولید، وضعیت تولید یا ظرفیت عرضه، وضعیت عرضه واکسن COVID-19 دارد. اگر بله، با 'yes' پاسخ دهید؛ در غیر این صورت، با 'no' پاسخ دهید (لطفاً به انگلیسی پاسخ دهید).",

        # Hebrew
        "תוכן הציוץ: {text}. נא לסמן אם הציוץ הזה מזכיר נגישות (ייצור ואספקה). זה מתייחס לקיבולת ייצור, מצב ייצור או קיבולת אספקה, מצב אספקה של חיסון ה-COVID-19. אם כן, ענה 'yes'; אחרת ענה 'no' (אנא ענה באנגלית).",

        # Turkish
        "Tweet içeriği: {text}. Lütfen bu tweetin erişilebilirliği (üretim ve tedarik) belirtip belirtmediğini işaretleyin. Bu, üretim kapasitesi, üretim durumu veya tedarik kapasitesi, COVID-19 aşısının tedarik durumu anlamına gelir. Eğer öyleyse, 'yes' ile cevap verin; aksi takdirde, 'no' ile cevap verin (lütfen İngilizce cevaplayın).",

        # Japanese
        "ツイート内容：{text}。このツイートがアクセス可能性（生産と供給）について言及しているかどうかをマークしてください。これは、COVID-19ワクチンの生産能力、生産状況、または供給能力、供給状況を指します。もしそうなら、「yes」と答えてください。そうでない場合は「no」と答えてください（英語でお答えください）。",

        # Korean
        "트윗 내용: {text}. 이 트윗이 접근성(생산 및 공급)에 대해 언급하는지 표시해 주세요. 이는 COVID-19 백신의 생산 능력, 생산 상황 또는 공급 능력, 공급 상황을 의미합니다. 그렇다면 'yes'로 대답해 주세요; 그렇지 않으면 'no'로 대답해 주세요 (영어로 대답해 주세요).",

        # Vietnamese
        "Nội dung tweet: {text}. Vui lòng đánh dấu xem tweet này có đề cập đến khả năng tiếp cận (sản xuất và cung ứng) hay không. Điều này ám chỉ năng lực sản xuất, tình hình sản xuất hoặc năng lực cung ứng, tình hình cung ứng của vắc-xin COVID-19. Nếu có, hãy trả lời 'yes'; nếu không, hãy trả lời 'no' (vui lòng trả lời bằng tiếng Anh).",

        # Thai
        "เนื้อหาทวีต: {text}. กรุณาระบุว่าทวีตนี้กล่าวถึงการเข้าถึง (การผลิตและการจัดหา) หรือไม่ ซึ่งหมายถึงความสามารถในการผลิต สถานการณ์การผลิต หรือความสามารถในการจัดหา สถานการณ์การจัดหาวัคซีน COVID-19 หากเป็นเช่นนั้น ให้ตอบ 'yes' มิฉะนั้น ให้ตอบ 'no' (โปรดตอบเป็นภาษาอังกฤษ).",

        # Indonesian
        "Konten tweet: {text}. Silakan tandai apakah tweet ini menyebutkan aksesibilitas (produksi dan pasokan). Ini mengacu pada kapasitas produksi, situasi produksi, atau kapasitas pasokan, situasi pasokan vaksin COVID-19. Jika demikian, jawab 'yes'; jika tidak, jawab 'no' (silakan jawab dalam bahasa Inggris).",

        # Malay
        "Kandungan tweet: {text}. Sila tandakan sama ada tweet ini menyebutkan kebolehcapaian (pengeluaran dan bekalan). Ini merujuk kepada kapasiti pengeluaran, situasi pengeluaran, atau kapasiti bekalan, situasi bekalan vaksin COVID-19. Jika ya, jawab 'yes'; jika tidak, jawab 'no' (sila jawab dalam Bahasa Inggeris).",

        # Lao
        "ເນື້ອຫາທວິດ: {text}. ກະລຸນາໝາຍວ່າທວິດນີ້ໄດ້ພາກົດເອົາຄວາມສາມາດໃນການເຂົ້າເຖິງ (ການຜະລິດແລະການສົ່ງເສີມ) ຫຼືບໍ່. ນີ້ແມ່ນການສະແດງເຫັນຄວາມສາມາດໃນການຜະລິດ, ສະພາບຂອງການຜະລິດ, ຫຼືຄວາມສາມາດໃນການສົ່ງເສີມ, ສະພາບຂອງການສົ່ງເສີມວັກຊີນ COVID-19. ຖ້າແມ່ນແລ້ວ, ກະລຸນາຕອບວ່າ 'yes'; ຖ້າບໍ່ແມ່ນ, ກະລຸນາຕອບວ່າ 'no' (ກະລຸນາຕອບໃນພາສາອັງກິດ).",

        # Burmese
        "တွစ်အကြောင်းအရာ: {text}။ ဒီတွစ်ဟာ COVID-19 ကာကွယ်ဆေးထိုးခြင်းမှာ ထုတ်လုပ်မှုနဲ့ဆိုင်တဲ့ အခက်အခဲတွေ၊ ပြဿနာတွေကို ဖော်ပြခဲ့ပါသလား ဆိုတာကို မှတ်ပေးပါ။ ဒီမှာဆိုရင် ထုတ်လုပ်မှု၊ ထုတ်လုပ်နိုင်မှု၊ သယ်ယူပိုပို့လုပ်ငန်းစဉ်တွေနဲ့ပတ်သက်တဲ့ အချက်အလက်တွေကို ပြောတာပဲဖြစ်ပါတယ်။ အကယ်၍ရှိပါက 'yes' ဖြင့် ဖြေပါ၊ မဟုတ်ပါက 'no' ဖြင့် ဖြေပါ (ကျေးဇူးပြု၍ အင်္ဂလိပ်ဘာသာဖြင့် ပြန်ဆိုပါ).",

        # Cebuano
        "Sulud sa tweet: {text}. Palihug markahi kung kini nga tweet naghisgot sa accessibility (produksyon ug suplay). Kini nagtumong sa kapasidad sa produksyon, sitwasyon sa produksyon, o kapasidad sa suplay, sitwasyon sa suplay sa bakuna sa COVID-19. Kung mao, tubaga og 'yes'; kung dili, tubaga og 'no' (palihug tubaga sa English).",

        # Khmer
        "ខ្លឹមសារទីចេញផ្សាយ: {text}។ សូមសម្គាល់មើលថាតើការផ្សាយនេះបាននិយាយអំពីការចូលដំណើរការ (ការផលិត និងការផ្គត់ផ្គង់) ឬអត់។ នេះមានន័យថាសំដៅទៅលើសមត្ថភាពផលិតភាព ដំណើរការផលិតភាព ឬសមត្ថភាពផ្គត់ផ្គង់ភាព ដំណើរការផ្គត់ផ្គង់នៃវ៉ាក់សាំង COVID-19។ ប្រសិនបើចង់និយាយបញ្ចប់សូមឆ្លើយតបដោយនិយាយថា 'yes'។ បើមិនបានបញ្ចប់សូមឆ្លើយតបដោយនិយាយថា 'no'។ (សូមឆ្លើយតបជាភាសាអង់គ្លេស).",

        # Tagalog
        "Nilalaman ng tweet: {text}. Paki-markahan kung ang tweet na ito ay nagpapahayag ng kakayahang maabot (produksyon at suplay). Ito ay tumutukoy sa kapasidad ng produksyon, sitwasyon ng produksyon, o kapasidad ng suplay, sitwasyon ng suplay ng bakuna sa COVID-19. Kung oo, sagutin ang 'yes'; kung hindi, sagutin ang 'no' (mangyaring sagutin sa Ingles).",

        # Hindi
        "ट्वीट सामग्री: {text}। कृपया चिह्नित करें कि यह ट्वीट उपलब्धता (उत्पादन और आपूर्ति) का उल्लेख करता है या नहीं। इसका अर्थ है COVID-19 वैक्सीन की उत्पादन क्षमता, उत्पादन की स्थिति या आपूर्ति क्षमता, आपूर्ति की स्थिति। यदि हाँ, तो 'yes' में उत्तर दें; अन्यथा, 'no' में उत्तर दें (कृपया अंग्रेजी में उत्तर दें)।",

        # Bengali
        "টুইটের বিষয়বস্তু: {text}। অনুগ্রহ করে চিহ্নিত করুন যে এই টুইটটি অ্যাক্সেসিবিলিটি (উৎপাদন এবং সরবরাহ) উল্লেখ করে কিনা। এর মানে হল COVID-19 ভ্যাকসিনের উৎপাদন ক্ষমতা, উৎপাদন পরিস্থিতি বা সরবরাহ ক্ষমতা, সরবরাহ পরিস্থিতি। যদি হ্যাঁ, তাহলে 'yes' দিয়ে উত্তর দিন; অন্যথায়, 'no' দিয়ে উত্তর দিন (অনুগ্রহ করে ইংরেজিতে উত্তর দিন)।",

        # Urdu
        "ٹویٹ کا مواد: {text}. براہ کرم نشان لگائیں کہ آیا یہ ٹویٹ قابل رسائی (پیداوار اور فراہمی) کا ذکر کرتا ہے۔ اس کا مطلب ہے COVID-19 ویکسین کی پیداواری صلاحیت، پیداوار کی صورتحال یا فراہمی کی صلاحیت، فراہمی کی صورتحال۔ اگر ایسا ہے تو، 'yes' میں جواب دیں؛ ورنہ، 'no' میں جواب دیں (براہ کرم انگریزی میں جواب دیں)."
    ],
    '13': [
        # Chinese
        "推特帖子内容：{text}。请标记这条推特帖子是否提及疫苗分配。这里指接种人群分配（包括优先人群、优先地区、公平性、强制接种或自愿接种）。如果是，回答 'yes'；否则，回答 'no' (请用英语回答).",

        # English
        "Tweet content: {text}. Please mark whether this tweet mentions vaccine distribution. This refers to the distribution of vaccinated populations (including priority groups, priority regions, fairness, mandatory or voluntary vaccination). If so, answer 'yes'; otherwise, respond with 'no' (please answer in English).",

        # German
        "Tweet-Inhalt: {text}. Bitte markieren Sie, ob dieser Tweet die Impfstoffverteilung erwähnt. Dies bezieht sich auf die Verteilung der geimpften Bevölkerung (einschließlich prioritärer Gruppen, prioritärer Regionen, Fairness, obligatorische oder freiwillige Impfung). Wenn ja, antworten Sie mit 'yes'; andernfalls antworten Sie mit 'no' (bitte auf Englisch antworten).",

        # French
        "Contenu du tweet : {text}. Veuillez indiquer si ce tweet mentionne la distribution des vaccins. Cela fait référence à la distribution des populations vaccinées (y compris les groupes prioritaires, les régions prioritaires, l'équité, la vaccination obligatoire ou volontaire). Si oui, répondez 'yes'; sinon, répondez 'no' (veuillez répondre en anglais).",

        # Spanish
        "Contenido del tweet: {text}. Por favor, marque si este tweet menciona la distribución de vacunas. Esto se refiere a la distribución de poblaciones vacunadas (incluidos grupos prioritarios, regiones prioritarias, equidad, vacunación obligatoria o voluntaria). Si es así, responda 'yes'; de lo contrario, responda 'no' (por favor responda en inglés).",

        # Portuguese
        "Conteúdo do tweet: {text}. Por favor, indique se este tweet menciona a distribuição de vacinas. Isso se refere à distribuição de populações vacinadas (incluindo grupos prioritários, regiões prioritárias, equidade, vacinação obrigatória ou voluntária). Se sim, responda 'yes'; caso contrário, responda 'no' (por favor, responda em inglês).",

        # Italian
        "Contenuto del tweet: {text}. Si prega di indicare se questo tweet menziona la distribuzione del vaccino. Questo si riferisce alla distribuzione delle popolazioni vaccinate (compresi i gruppi prioritari, le regioni prioritarie, l'equità, la vaccinazione obbligatoria o volontaria). Se sì, rispondi 'yes'; altrimenti rispondi 'no' (si prega di rispondere in inglese).",

        # Dutch
        "Tweet-inhoud: {text}. Markeer of deze tweet de vaccinverdeling vermeldt. Dit verwijst naar de verdeling van gevaccineerde populaties (inclusief prioriteitsgroepen, prioriteitsregio's, eerlijkheid, verplichte of vrijwillige vaccinatie). Als dat zo is, antwoord dan 'yes'; anders antwoord 'no' (antwoord alstublieft in het Engels).",

        # Russian
        "Содержание твита: {text}. Отметьте, упоминает ли этот твит распределение вакцин. Это относится к распределению вакцинированных групп населения (включая приоритетные группы, приоритетные регионы, справедливость, обязательную или добровольную вакцинацию). Если да, ответьте 'yes'; в противном случае ответьте 'no' (пожалуйста, ответьте на английском языке).",

        # Czech
        "Obsah tweetu: {text}. Označte, zda tento tweet zmiňuje distribuci vakcíny. To se týká distribuce očkovaných populací (včetně prioritních skupin, prioritních regionů, spravedlnosti, povinného nebo dobrovolného očkování). Pokud ano, odpovězte 'yes'; v opačném případě odpovězte 'no' (prosím, odpovězte v angličtině).",

        # Polish
        "Treść tweeta: {text}. Proszę zaznaczyć, czy ten tweet wspomina o dystrybucji szczepionek. To odnosi się do dystrybucji zaszczepionych populacji (w tym grup priorytetowych, priorytetowych regionów, sprawiedliwości, obowiązkowego lub dobrowolnego szczepienia). Jeśli tak, odpowiedz 'yes'; w przeciwnym razie odpowiedz 'no' (proszę odpowiedzieć po angielsku).",

        # Arabic
        "محتوى التغريدة: {text}. يُرجى تحديد ما إذا كانت هذه التغريدة تذكر توزيع اللقاحات. يشير هذا إلى توزيع السكان الذين تم تطعيمهم (بما في ذلك الفئات ذات الأولوية، المناطق ذات الأولوية، الإنصاف، التطعيم الإجباري أو الطوعي). إذا كان الأمر كذلك، أجب 'yes'؛ وإلا، أجب 'no' (يرجى الرد بالإنجليزية).",

        # Persian
        "محتوای توییت: {text}. لطفاً مشخص کنید که آیا این توییت به توزیع واکسن اشاره دارد یا خیر. این به توزیع جمعیت‌های واکسینه شده (از جمله گروه‌های اولویت‌دار، مناطق اولویت‌دار، انصاف، واکسیناسیون اجباری یا داوطلبانه) اشاره دارد. اگر بله، با 'yes' پاسخ دهید؛ در غیر این صورت، با 'no' پاسخ دهید (لطفاً به انگلیسی پاسخ دهید).",

        # Hebrew
        "תוכן הציוץ: {text}. נא לסמן אם הציוץ הזה מזכיר הפצת חיסונים. זה מתייחס להפצת אוכלוסיות מחוסנות (כולל קבוצות עדיפות, אזורים עדיפים, הוגנות, חיסון חובה או וולונטרי). אם כן, ענה 'yes'; אחרת ענה 'no' (אנא ענה באנגלית).",

        # Turkish
        "Tweet içeriği: {text}. Lütfen bu tweetin aşı dağıtımını belirtip belirtmediğini işaretleyin. Bu, aşılanan nüfusların dağıtımını (öncelikli gruplar, öncelikli bölgeler, adalet, zorunlu veya gönüllü aşılamayı) ifade eder. Eğer öyleyse, 'yes' ile cevap verin; aksi takdirde, 'no' ile cevap verin (lütfen İngilizce cevaplayın).",

        # Japanese
        "ツイート内容：{text}。このツイートがワクチンの配分について言及しているかどうかをマークしてください。これは、ワクチン接種を受けた人々の分配（優先グループ、優先地域、公平性、強制または任意の接種を含む）を指します。もしそうなら、「yes」と答えてください。そうでない場合は「no」と答えてください（英語でお答えください）。",

        # Korean
        "트윗 내용: {text}. 이 트윗이 백신 배포에 대해 언급하는지 표시해 주세요. 이는 접종된 인구의 배포를 의미합니다 (우선 그룹, 우선 지역, 공정성, 필수 또는 자발적 예방 접종 포함). 그렇다면 'yes'로 대답해 주세요; 그렇지 않으면 'no'로 대답해 주세요 (영어로 대답해 주세요).",

        # Vietnamese
        "Nội dung tweet: {text}. Vui lòng đánh dấu xem tweet này có đề cập đến phân phối vắc-xin hay không. Điều này đề cập đến việc phân phối các nhóm dân số được tiêm chủng (bao gồm các nhóm ưu tiên, khu vực ưu tiên, công bằng, tiêm chủng bắt buộc hoặc tự nguyện). Nếu có, hãy trả lời 'yes'; nếu không, hãy trả lời 'no' (vui lòng trả lời bằng tiếng Anh).",

        # Thai
        "เนื้อหาทวีต: {text}. กรุณาระบุว่าทวีตนี้กล่าวถึงการแจกจ่ายวัคซีนหรือไม่ ซึ่งหมายถึงการแจกจ่ายประชากรที่ได้รับการฉีดวัคซีน (รวมถึงกลุ่มที่มีลำดับความสำคัญ พื้นที่ที่มีลำดับความสำคัญ ความเป็นธรรม การฉีดวัคซีนภาคบังคับหรือโดยสมัครใจ) หากเป็นเช่นนั้น ให้ตอบ 'yes' มิฉะนั้น ให้ตอบ 'no' (โปรดตอบเป็นภาษาอังกฤษ).",

        # Indonesian
        "Konten tweet: {text}. Silakan tandai apakah tweet ini menyebutkan distribusi vaksin. Ini mengacu pada distribusi populasi yang divaksinasi (termasuk kelompok prioritas, wilayah prioritas, keadilan, vaksinasi wajib atau sukarela). Jika demikian, jawab 'yes'; jika tidak, jawab 'no' (silakan jawab dalam bahasa Inggris).",

        # Malay
        "Kandungan tweet: {text}. Sila tandakan sama ada tweet ini menyebutkan pengedaran vaksin. Ini merujuk kepada pengedaran populasi yang diberi vaksin (termasuk kumpulan keutamaan, kawasan keutamaan, keadilan, vaksinasi wajib atau sukarela). Jika ya, jawab 'yes'; jika tidak, jawab 'no' (sila jawab dalam Bahasa Inggeris).",

        # Lao
        "ເນື້ອຫາທວິດ: {text}. ກະລຸນາໝາຍວ່າທວິດນີ້ໄດ້ພາກົດເອົາການແຈກວັກຊີນຫຼືບໍ່. ນີ້ແມ່ນການສະແດງໃຫ້ເຫັນການແຈກວັກຊີນໃນກົດຫຼວງສະຫນາມປະຊາຊົນທີ່ໄດ້ຮັບການສະໜອງ (ລວມທັງກຸ່ມທີ່ມີຄວາມສຳຄັນສູງ, ພື້ນທີ່ທີ່ມີຄວາມສຳຄັນສູງ, ຄວາມຍຸຕິທຳ, ການກຽມການຢູ່ຂອງການສະເໜີວັກຊີນຕາມຄວາມຕ້ອງການຫຼືບໍ່). ຖ້າແມ່ນແລ້ວ, ກະລຸນາຕອບວ່າ 'yes'; ຖ້າບໍ່ແມ່ນ, ກະລຸນາຕອບວ່າ 'no' (ກະລຸນາຕອບໃນພາສາອັງກິດ).",

        # Burmese
        "တွစ်အကြောင်းအရာ: {text}။ ဒီတွစ်ဟာ COVID-19 ကာကွယ်ဆေးထိုးခြင်းကို ဘယ်လိုဝေငှပေးမလဲဆိုတာကို ဖော်ပြခဲ့ပါသလား ဆိုတာကို မှတ်ပေးပါ။ ဒီမှာဆိုရင် လူစုတန်း၊ ဒေသခံနှင့် မြို့ပြလူနေထူထပ်တဲ့ နေရာ၊ သက်ဆိုင်ရာလူနာစုနှင့်ပတ်သက်ပြီး တရားမျှတမှု၊ လိုအပ်မှုကစပြီး ဝေငှမှုကို အစွမ်းကုန် လုပ်ဆောင်လျှက်ရှိသည်။ အကယ်၍ရှိပါက 'yes' ဖြင့် ဖြေပါ၊ မဟုတ်ပါက 'no' ဖြင့် ဖြေပါ (ကျေးဇူးပြု၍ အင်္ဂလိပ်ဘာသာဖြင့် ပြန်ဆိုပါ).",

        # Cebuano
        "Sulud sa tweet: {text}. Palihug markahi kung kini nga tweet naghisgot sa distribusyon sa bakuna. Kini nagtumong sa distribusyon sa nabakunahan nga populasyon (lakip ang mga prayoridad nga grupo, mga prayoridad nga rehiyon, pagkatarong, mandatory o boluntaryong pagbakuna). Kung mao, tubaga og 'yes'; kung dili, tubaga og 'no' (palihug tubaga sa English).",

        # Khmer
        "ខ្លឹមសារទីចេញផ្សាយ: {text}។ សូមសម្គាល់មើលថាតើការផ្សាយនេះបាននិយាយអំពីការចែកចាយវ៉ាក់សាំងឬអត់។ នេះមានន័យថាសំដៅទៅលើការចែកចាយប្រជាជនដែលបានទទួលការចាក់វ៉ាក់សាំង (រួមមានក្រុមដែលមានអាទិភាព ផ្នែកដែលមានអាទិភាព ការសមធម៌ ការចាក់វ៉ាក់សាំងស្ម័គ្រចិត្ត ឬក៏ចេញសម្រេចឱ្យធ្វើ)។ ប្រសិនបើចង់និយាយបញ្ចប់សូមឆ្លើយតបដោយនិយាយថា 'yes'។ បើមិនបានបញ្ចប់សូមឆ្លើយតបដោយនិយាយថា 'no'។ (សូមឆ្លើយតបជាភាសាអង់គ្លេស).",

        # Tagalog
        "Nilalaman ng tweet: {text}. Paki-markahan kung ang tweet na ito ay nagpapahayag ng pamamahagi ng bakuna. Ito ay tumutukoy sa pamamahagi ng mga nabakunahan na populasyon (kabilang ang mga grupong prayoridad, mga rehiyong prayoridad, katarungan, sapilitan o boluntaryong pagbabakuna). Kung oo, sagutin ang 'yes'; kung hindi, sagutin ang 'no' (mangyaring sagutin sa Ingles).",

        # Hindi
        "ट्वीट सामग्री: {text}। कृपया चिह्नित करें कि यह ट्वीट वैक्सीन वितरण का उल्लेख करता है या नहीं। इसका मतलब है कि टीकाकरण किए गए लोगों का वितरण (जिसमें प्राथमिकता समूह, प्राथमिकता क्षेत्र, निष्पक्षता, अनिवार्य या स्वैच्छिक टीकाकरण शामिल है)। यदि हाँ, तो 'yes' में उत्तर दें; अन्यथा, 'no' में उत्तर दें (कृपया अंग्रेजी में उत्तर दें)।",

        # Bengali
        "টুইটের বিষয়বস্তু: {text}। অনুগ্রহ করে চিহ্নিত করুন যে এই টুইটটি ভ্যাকসিন বিতরণের উল্লেখ করে কিনা। এর মানে হল টিকাপ্রাপ্ত জনসংখ্যার বণ্টন (অগ্রাধিকার গোষ্ঠী, অগ্রাধিকার অঞ্চল, ন্যায্যতা, বাধ্যতামূলক বা স্বেচ্ছাসেবী টিকা সহ)। যদি হ্যাঁ, তাহলে 'yes' দিয়ে উত্তর দিন; অন্যথায়, 'no' দিয়ে উত্তর দিন (অনুগ্রহ করে ইংরেজিতে উত্তর দিন)।",

        # Urdu
        "ٹویٹ کا مواد: {text}. براہ کرم نشان لگائیں کہ آیا یہ ٹویٹ ویکسین کی تقسیم کا ذکر کرتا ہے۔ اس کا مطلب ہے ویکسین شدہ آبادی کی تقسیم (بشمول ترجیحی گروپ، ترجیحی علاقے، انصاف، لازمی یا رضاکارانہ ویکسینیشن). اگر ایسا ہے تو، 'yes' میں جواب دیں؛ ورنہ، 'no' میں جواب دیں (براہ کرم انگریزی میں جواب دیں)."
    ],
    '14.1': [
        # Chinese
        "推特帖子内容：{text}。请标记这条推特帖子是否涉及构成负面信息环境。例如，帖子里的内容包含谣言、反疫苗运动、反智或反科学运动、疫苗负面事件等。如果是，回答 'yes'；否则，回答 'no' (请用英语回答).",

        # English
        "Tweet content: {text}. Please mark whether this tweet involves creating a negative information environment. For example, the content of the tweet includes rumors, anti-vaccine movements, anti-intellectual or anti-science movements, negative vaccine events, etc. If so, answer 'yes'; otherwise, respond with 'no' (please answer in English).",

        # German
        "Tweet-Inhalt: {text}. Bitte markieren Sie, ob dieser Tweet die Schaffung eines negativen Informationsumfelds betrifft. Zum Beispiel enthält der Inhalt des Tweets Gerüchte, Anti-Impf-Bewegungen, anti-intellektuelle oder anti-wissenschaftliche Bewegungen, negative Impfereignisse usw. Wenn ja, antworten Sie mit 'yes'; andernfalls antworten Sie mit 'no' (bitte auf Englisch antworten).",

        # French
        "Contenu du tweet : {text}. Veuillez indiquer si ce tweet concerne la création d'un environnement d'information négatif. Par exemple, le contenu du tweet comprend des rumeurs, des mouvements anti-vaccins, des mouvements anti-intellectuels ou anti-science, des événements négatifs liés aux vaccins, etc. Si oui, répondez 'yes'; sinon, répondez 'no' (veuillez répondre en anglais).",

        # Spanish
        "Contenido del tweet: {text}. Por favor, marque si este tweet involucra la creación de un entorno de información negativa. Por ejemplo, el contenido del tweet incluye rumores, movimientos anti-vacunas, movimientos anti-intelectuales o anti-ciencia, eventos negativos relacionados con vacunas, etc. Si es así, responda 'yes'; de lo contrario, responda 'no' (por favor responda en inglés).",

        # Portuguese
        "Conteúdo do tweet: {text}. Por favor, indique se este tweet envolve a criação de um ambiente de informação negativa. Por exemplo, o conteúdo do tweet inclui rumores, movimentos antivacinas, movimentos anti-intelectuais ou anti-ciência, eventos negativos sobre vacinas, etc. Se sim, responda 'yes'; caso contrário, responda 'no' (por favor, responda em inglês).",

        # Italian
        "Contenuto del tweet: {text}. Si prega di indicare se questo tweet riguarda la creazione di un ambiente di informazione negativo. Ad esempio, il contenuto del tweet include voci, movimenti anti-vaccino, movimenti anti-intellettuali o anti-scientifici, eventi negativi sui vaccini, ecc. Se sì, rispondi 'yes'; altrimenti rispondi 'no' (si prega di rispondere in inglese).",

        # Dutch
        "Tweet-inhoud: {text}. Markeer of deze tweet gaat over het creëren van een negatieve informatieomgeving. Het kan bijvoorbeeld gaan om geruchten, anti-vaccinbewegingen, anti-intellectuele of anti-wetenschappelijke bewegingen, negatieve vaccinincidenten, enz. Als dat zo is, antwoord dan 'yes'; anders antwoord 'no' (antwoord alstublieft in het Engels).",

        # Russian
        "Содержание твита: {text}. Отметьте, касается ли этот твит создания негативной информационной среды. Например, содержание твита включает слухи, антивакцинальные движения, антиинтеллектуальные или антинаучные движения, негативные события, связанные с вакцинами, и т. д. Если да, ответьте 'yes'; в противном случае ответьте 'no' (пожалуйста, ответьте на английском языке).",

        # Czech
        "Obsah tweetu: {text}. Označte, zda se tento tweet týká vytváření negativního informačního prostředí. Například obsah tweetu zahrnuje fámy, anti-vakcinační hnutí, anti-intelektuální nebo anti-vědecká hnutí, negativní události související s vakcínami atd. Pokud ano, odpovězte 'yes'; v opačném případě odpovězte 'no' (prosím, odpovězte v angličtině).",

        # Polish
        "Treść tweeta: {text}. Proszę zaznaczyć, czy ten tweet dotyczy tworzenia negatywnego środowiska informacyjnego. Na przykład treść tweeta obejmuje plotki, ruchy antyszczepionkowe, ruchy antyintelektualne lub antynaukowe, negatywne zdarzenia związane ze szczepionkami itp. Jeśli tak, odpowiedz 'yes'; w przeciwnym razie odpowiedz 'no' (proszę odpowiedzieć po angielsku).",

        # Arabic
        "محتوى التغريدة: {text}. يُرجى تحديد ما إذا كانت هذه التغريدة تتعلق بإنشاء بيئة معلومات سلبية. على سبيل المثال، يتضمن محتوى التغريدة الشائعات، والحركات المناهضة للقاحات، والحركات المناهضة للعقلانية أو المناهضة للعلم، والأحداث السلبية المتعلقة باللقاحات، وما إلى ذلك. إذا كان الأمر كذلك، أجب 'yes'؛ وإلا، أجب 'no' (يرجى الرد بالإنجليزية).",

        # Persian
        "محتوای توییت: {text}. لطفاً مشخص کنید که آیا این توییت در مورد ایجاد یک محیط اطلاعاتی منفی است یا خیر. به عنوان مثال، محتوای توییت شامل شایعات، جنبش‌های ضد واکسن، جنبش‌های ضد عقلانیت یا ضد علم، رویدادهای منفی مربوط به واکسن‌ها و غیره است. اگر بله، با 'yes' پاسخ دهید؛ در غیر این صورت، با 'no' پاسخ دهید (لطفاً به انگلیسی پاسخ دهید).",

        # Hebrew
        "תוכן הציוץ: {text}. נא לסמן אם הציוץ הזה עוסק ביצירת סביבת מידע שלילית. לדוגמה, תוכן הציוץ כולל שמועות, תנועות נגד חיסונים, תנועות אנטי-אינטלקטואליות או אנטי-מדעיות, אירועים שליליים הקשורים לחיסונים, וכו'. אם כן, ענה 'yes'; אחרת ענה 'no' (אנא ענה באנגלית).",

        # Turkish
        "Tweet içeriği: {text}. Lütfen bu tweetin olumsuz bir bilgi ortamı oluşturmayı içerip içermediğini işaretleyin. Örneğin, tweetin içeriği söylentileri, aşı karşıtı hareketleri, anti-entelektüel veya anti-bilimsel hareketleri, aşılarla ilgili olumsuz olayları vb. içerir. Eğer öyleyse, 'yes' ile cevap verin; aksi takdirde, 'no' ile cevap verin (lütfen İngilizce cevaplayın).",

        # Japanese
        "ツイート内容：{text}。このツイートが否定的な情報環境の構築に関与しているかどうかをマークしてください。たとえば、ツイートの内容には、噂、反ワクチン運動、反知性主義や反科学運動、ワクチンに関する否定的な出来事などが含まれます。もしそうなら、「yes」と答えてください。そうでない場合は「no」と答えてください（英語でお答えください）。",

        # Korean
        "트윗 내용: {text}. 이 트윗이 부정적인 정보 환경을 조성하는 데 관여하는지 표시해 주세요. 예를 들어, 트윗의 내용에는 소문, 백신 반대 운동, 반지성주의 또는 반과학 운동, 백신 관련 부정적인 사건 등이 포함됩니다. 그렇다면 'yes'로 대답해 주세요; 그렇지 않으면 'no'로 대답해 주세요 (영어로 대답해 주세요).",

        # Vietnamese
        "Nội dung tweet: {text}. Vui lòng đánh dấu xem tweet này có liên quan đến việc tạo ra môi trường thông tin tiêu cực hay không. Ví dụ, nội dung của tweet bao gồm tin đồn, phong trào chống vắc-xin, phong trào chống trí thức hoặc chống khoa học, các sự kiện tiêu cực liên quan đến vắc-xin, v.v. Nếu có, hãy trả lời 'yes'; nếu không, hãy trả lời 'no' (vui lòng trả lời bằng tiếng Anh).",

        # Thai
        "เนื้อหาทวีต: {text}. กรุณาระบุว่าทวีตนี้เกี่ยวข้องกับการสร้างสภาพแวดล้อมข้อมูลเชิงลบหรือไม่ ตัวอย่างเช่น เนื้อหาของทวีตประกอบด้วยข่าวลือ การเคลื่อนไหวต่อต้านวัคซีน การเคลื่อนไหวต่อต้านปัญญาชนหรือต่อต้านวิทยาศาสตร์ เหตุการณ์ด้านลบเกี่ยวกับวัคซีน เป็นต้น หากเป็นเช่นนั้น ให้ตอบ 'yes' มิฉะนั้น ให้ตอบ 'no' (โปรดตอบเป็นภาษาอังกฤษ).",

        # Indonesian
        "Konten tweet: {text}. Silakan tandai apakah tweet ini melibatkan pembuatan lingkungan informasi negatif. Misalnya, konten tweet mencakup rumor, gerakan anti-vaksin, gerakan anti-intelektual atau anti-sains, peristiwa negatif terkait vaksin, dll. Jika demikian, jawab 'yes'; jika tidak, jawab 'no' (silakan jawab dalam bahasa Inggris).",

        # Malay
        "Kandungan tweet: {text}. Sila tandakan sama ada tweet ini melibatkan penciptaan persekitaran maklumat negatif. Sebagai contoh, kandungan tweet termasuk khabar angin, gerakan anti-vaksin, gerakan anti-intelek atau anti-sains, peristiwa negatif berkaitan vaksin, dan lain-lain. Jika ya, jawab 'yes'; jika tidak, jawab 'no' (sila jawab dalam Bahasa Inggeris).",

        # Lao
        "ເນື້ອຫາທວິດ: {text}. ກະລຸນາໝາຍວ່າທວິດນີ້ໄດ້ພາກົດເອົາການສ້າງສະພາບຂໍ້ມູນແມ່ນຄວາມສຸກຫຼາຍ. ເຊັ່ນການໄດ້ຮັບຂໍ້ມູນທີ່ບໍ່ຖືກຕ້ອງ, ຂ່າວລື, ການຕໍ່ຕ້ານວັກຊີນ, ການຕໍ່ຕ້ານວິທະຍາສາດ. ຖ້າແມ່ນ, ກະລຸນາຕອບວ່າ 'yes'; ຖ້າບໍ່ແມ່ນ, ຕອບ 'no' (ກະລຸນາຕອບໃນພາສາອັງກິດ).",

        # Burmese
        "တွစ်အကြောင်းအရာ: {text}။ ဒီတွစ်ဟာ COVID-19 ကာကွယ်ဆေးနဲ့ပတ်သက်ပြီး လုပ်ဆောင်ချက်၊ အသိပညာမမှန်ကန်မှုတွေကို ဖော်ပြခဲ့ပါသလားဆိုတာကို မှတ်ပေးပါ။ ဥပမာ- ကာကွယ်ဆေးနဲ့ပတ်သက်ပြီး ရှုပ်ထွေးမှုတွေ၊ အန္တရာယ်၊ မမှန်ကန်မှုတွေကို ခံရတာမျိုးတွေကိုပါ တစ်ပါတည်းရေးသားပြီး ဖော်ပြနေကြပါစေ။ အကယ်၍ရှိပါက 'yes' ဖြင့် ဖြေပါ၊ မဟုတ်ပါက 'no' ဖြင့် ဖြေပါ (ကျေးဇူးပြု၍ အင်္ဂလိပ်ဘာသာဖြင့် ပြန်ဆိုပါ).",

        # Cebuano
        "Sulud sa tweet: {text}. Palihug markahi kung kini nga tweet nag-apil sa paghimo og usa ka negatibong impormasyon nga palibot. Pananglitan, ang sulod sa tweet naglakip og mga hungihong, mga lihok nga kontra sa bakuna, kontra sa intelektuwal o kontra sa siyensya nga mga lihok, negatibong mga hitabo sa bakuna, ug uban pa. Kung mao, tubaga og 'yes'; kung dili, tubaga og 'no' (palihug tubaga sa English).",

        # Khmer
        "ខ្លឹមសារទីចេញផ្សាយ: {text}។ សូមសម្គាល់មើលថាតើការផ្សាយនេះបាននិយាយអំពីការបង្កើតបរិយាកាសព័ត៌មានអវិជ្ជមាន ឬអត់។ ការផ្សាយនេះអាចរួមមានការព័ត៌មានចចាមអារាម ការដាក់ចេញបរិយាកាសប្រឆាំងនឹងការចាក់វ៉ាក់សាំង ការប្រឆាំងនឹងវិទ្យាសាស្ត្រ ការខ្វល់ខ្វាយនិងព្រឹត្តិការណ៍អវិជ្ជមានផ្សេងៗ។ ប្រសិនបើចង់និយាយបញ្ចប់សូមឆ្លើយតបដោយនិយាយថា 'yes'។ បើមិនបានបញ្ចប់សូមឆ្លើយតបដោយនិយាយថា 'no'។ (សូមឆ្លើយតបជាភាសាអង់គ្លេស).",

        # Tagalog
        "Nilalaman ng tweet: {text}. Paki-markahan kung ang tweet na ito ay nagpapahayag ng isang negatibong kapaligiran ng impormasyon. Halimbawa, ang nilalaman ng tweet ay kinabibilangan ng mga tsismis, mga kilusang kontra-bakuna, mga kilusang kontra-intelihensya o kontra-agham, mga negatibong pangyayari sa bakuna, atbp. Kung oo, sagutin ang 'yes'; kung hindi, sagutin ang 'no' (mangyaring sagutin sa Ingles).",

        # Hindi
        "ट्वीट सामग्री: {text}। कृपया चिह्नित करें कि यह ट्वीट नकारात्मक सूचना वातावरण बनाने में शामिल है या नहीं। उदाहरण के लिए, ट्वीट की सामग्री में अफवाहें, एंटी-वैक्सीन आंदोलन, एंटी-बौद्धिक या एंटी-विज्ञान आंदोलन, नकारात्मक वैक्सीन घटनाएं, आदि शामिल हैं। यदि हाँ, तो 'yes' में उत्तर दें; अन्यथा, 'no' में उत्तर दें (कृपया अंग्रेजी में उत्तर दें)।",

        # Bengali
        "টুইটের বিষয়বস্তু: {text}। অনুগ্রহ করে চিহ্নিত করুন যে এই টুইটটি একটি নেতিবাচক তথ্য পরিবেশ তৈরি করার সাথে জড়িত কিনা। উদাহরণস্বরূপ, টুইটের সামগ্রীর মধ্যে গুজব, টিকা-বিরোধী আন্দোলন, বুদ্ধিবিরোধী বা বিজ্ঞানবিরোধী আন্দোলন, নেতিবাচক ভ্যাকসিন ইভেন্ট ইত্যাদি অন্তর্ভুক্ত রয়েছে। যদি হ্যাঁ, তাহলে 'yes' দিয়ে উত্তর দিন; অন্যথায়, 'no' দিয়ে উত্তর দিন (অনুগ্রহ করে ইংরেজিতে উত্তর দিন)।",

        # Urdu
        "ٹویٹ کا مواد: {text}. براہ کرم نشان لگائیں کہ آیا یہ ٹویٹ منفی معلوماتی ماحول بنانے میں شامل ہے۔ مثال کے طور پر، ٹویٹ کے مواد میں افواہیں، ویکسین مخالف تحریکیں، عقل دشمن یا سائنس مخالف تحریکیں، ویکسین سے متعلق منفی واقعات وغیرہ شامل ہیں۔ اگر ایسا ہے تو، 'yes' میں جواب دیں؛ ورنہ، 'no' میں جواب دیں (براہ کرم انگریزی میں جواب دیں)."
    ],
    '15.1': [
        # Chinese
        "推特帖子内容：{text}。请标记这条推特帖子是否表达对疫苗的研发和上市感到有信心。这里指帖子提及对疫苗上市时间的判断，希望疫苗尽快研发出来或尽快上市，对疫苗的研发和上市感到有信心等。如果是，回答 'yes'；否则，回答 'no' (请用英语回答).",

        # English
        "Tweet content: {text}. Please mark whether this tweet expresses confidence in the development and launch of the vaccine. This refers to the post mentioning expectations for the vaccine's release, the desire for rapid development or release, and confidence in the vaccine's development and launch. If so, answer 'yes'; otherwise, respond with 'no' (please answer in English).",

        # German
        "Tweet-Inhalt: {text}. Bitte markieren Sie, ob dieser Tweet Vertrauen in die Entwicklung und Markteinführung des Impfstoffs ausdrückt. Dies bezieht sich auf die Erwähnung von Erwartungen an die Freigabe des Impfstoffs, den Wunsch nach schneller Entwicklung oder Freigabe und das Vertrauen in die Entwicklung und Markteinführung des Impfstoffs. Wenn ja, antworten Sie mit 'yes'; andernfalls antworten Sie mit 'no' (bitte auf Englisch antworten).",

        # French
        "Contenu du tweet : {text}. Veuillez indiquer si ce tweet exprime une confiance dans le développement et le lancement du vaccin. Cela fait référence à la mention par l'auteur des attentes concernant la sortie du vaccin, le désir de développement ou de lancement rapide, et la confiance dans le développement et le lancement du vaccin. Si oui, répondez 'yes'; sinon, répondez 'no' (veuillez répondre en anglais).",

        # Spanish
        "Contenido del tweet: {text}. Por favor, marque si este tweet expresa confianza en el desarrollo y lanzamiento de la vacuna. Esto se refiere a la mención de expectativas sobre el lanzamiento de la vacuna, el deseo de un desarrollo o lanzamiento rápido y la confianza en el desarrollo y lanzamiento de la vacuna. Si es así, responda 'yes'; de lo contrario, responda 'no' (por favor responda en inglés).",

        # Portuguese
        "Conteúdo do tweet: {text}. Por favor, indique se este tweet expressa confiança no desenvolvimento e lançamento da vacina. Isso se refere à menção de expectativas para o lançamento da vacina, o desejo de um desenvolvimento ou lançamento rápido e confiança no desenvolvimento e lançamento da vacina. Se sim, responda 'yes'; caso contrário, responda 'no' (por favor, responda em inglês).",

        # Italian
        "Contenuto del tweet: {text}. Si prega di indicare se questo tweet esprime fiducia nello sviluppo e nel lancio del vaccino. Questo si riferisce alla menzione delle aspettative per il rilascio del vaccino, al desiderio di sviluppo o rilascio rapido e alla fiducia nello sviluppo e nel lancio del vaccino. Se sì, rispondi 'yes'; altrimenti rispondi 'no' (si prega di rispondere in inglese).",

        # Dutch
        "Tweet-inhoud: {text}. Markeer of deze tweet vertrouwen uitspreekt in de ontwikkeling en lancering van het vaccin. Dit verwijst naar het vermelden van verwachtingen voor de release van het vaccin, de wens voor snelle ontwikkeling of release, en het vertrouwen in de ontwikkeling en lancering van het vaccin. Als dat zo is, antwoord dan 'yes'; anders antwoord 'no' (antwoord alstublieft in het Engels).",

        # Russian
        "Содержание твита: {text}. Отметьте, выражает ли этот твит уверенность в разработке и запуске вакцины. Это относится к упоминанию ожиданий в отношении выпуска вакцины, желанию быстрого развития или выпуска и уверенности в разработке и запуске вакцины. Если да, ответьте 'yes'; в противном случае ответьте 'no' (пожалуйста, ответьте на английском языке).",

        # Czech
        "Obsah tweetu: {text}. Označte, zda tento tweet vyjadřuje důvěru ve vývoj a uvedení vakcíny na trh. To se týká zmínky o očekáváních ohledně vydání vakcíny, přání rychlého vývoje nebo vydání a důvěře ve vývoj a uvedení vakcíny na trh. Pokud ano, odpovězte 'yes'; v opačném případě odpovězte 'no' (prosím, odpovězte v angličtině).",

        # Polish
        "Treść tweeta: {text}. Proszę zaznaczyć, czy ten tweet wyraża zaufanie do rozwoju i wprowadzenia szczepionki na rynek. To odnosi się do wspomnienia o oczekiwaniach dotyczących wydania szczepionki, pragnieniu szybkiego rozwoju lub wydania oraz zaufaniu do rozwoju i wprowadzenia szczepionki na rynek. Jeśli tak, odpowiedz 'yes'; w przeciwnym razie odpowiedz 'no' (proszę odpowiedzieć po angielsku).",

        # Arabic
        "محتوى التغريدة: {text}. يُرجى تحديد ما إذا كانت هذه التغريدة تعبر عن الثقة في تطوير وإطلاق اللقاح. يشير هذا إلى ذكر التوقعات لإطلاق اللقاح، والرغبة في التطوير السريع أو الإطلاق، والثقة في تطوير اللقاح وإطلاقه. إذا كان الأمر كذلك، أجب 'yes'؛ وإلا، أجب 'no' (يرجى الرد بالإنجليزية).",

        # Persian
        "محتوای توییت: {text}. لطفاً مشخص کنید که آیا این توییت اعتماد به توسعه و عرضه واکسن را ابراز می‌کند یا خیر. این اشاره به ذکر انتظارات برای انتشار واکسن، تمایل به توسعه یا انتشار سریع، و اعتماد به توسعه و عرضه واکسن دارد. اگر بله، با 'yes' پاسخ دهید؛ در غیر این صورت، با 'no' پاسخ دهید (لطفاً به انگلیسی پاسخ دهید).",

        # Hebrew
        "תוכן הציוץ: {text}. נא לסמן אם הציוץ הזה מביע אמון בפיתוח והשקת החיסון. זה מתייחס לאזכור של ציפיות לשחרור החיסון, הרצון לפיתוח או שחרור מהיר, והאמון בפיתוח והשקת החיסון. אם כן, ענה 'yes'; אחרת ענה 'no' (אנא ענה באנגלית).",

        # Turkish
        "Tweet içeriği: {text}. Lütfen bu tweetin aşının geliştirilmesi ve piyasaya sürülmesine duyulan güveni ifade edip etmediğini işaretleyin. Bu, aşının piyasaya sürülmesi beklentilerinin, hızlı geliştirme veya piyasaya sürülme arzusunun ve aşının geliştirilmesi ve piyasaya sürülmesine duyulan güvenin belirtilmesine atıfta bulunur. Eğer öyleyse, 'yes' ile cevap verin; aksi takdirde, 'no' ile cevap verin (lütfen İngilizce cevaplayın).",

        # Japanese
        "ツイート内容：{text}。このツイートがワクチンの開発と発売に対する信頼を表明しているかどうかをマークしてください。これは、ワクチンの発売に対する期待、迅速な開発や発売の願望、ワクチンの開発と発売に対する信頼の言及を指します。もしそうなら、「yes」と答えてください。そうでない場合は「no」と答えてください（英語でお答えください）。",

        # Korean
        "트윗 내용: {text}. 이 트윗이 백신의 개발 및 출시와 관련된 신뢰를 표현하는지 표시해 주세요. 이는 백신 출시와 관련된 기대, 신속한 개발 또는 출시에 대한 열망, 백신의 개발 및 출시에 대한 신뢰를 언급하는 것을 의미합니다. 그렇다면 'yes'로 대답해 주세요; 그렇지 않으면 'no'로 대답해 주세요 (영어로 대답해 주세요).",

        # Vietnamese
        "Nội dung tweet: {text}. Vui lòng đánh dấu xem tweet này có thể hiện sự tin tưởng vào quá trình phát triển và ra mắt vắc-xin hay không. Điều này đề cập đến việc đề cập đến kỳ vọng đối với việc phát hành vắc-xin, mong muốn phát triển hoặc phát hành nhanh chóng và sự tự tin vào quá trình phát triển và ra mắt vắc-xin. Nếu có, hãy trả lời 'yes'; nếu không, hãy trả lời 'no' (vui lòng trả lời bằng tiếng Anh).",

        # Thai
        "เนื้อหาทวีต: {text}. กรุณาระบุว่าทวีตนี้แสดงความมั่นใจในกระบวนการพัฒนาและเปิดตัววัคซีนหรือไม่ ซึ่งหมายถึงการกล่าวถึงความคาดหวังเกี่ยวกับการเปิดตัววัคซีน ความปรารถนาสำหรับการพัฒนาหรือการเปิดตัวอย่างรวดเร็ว และความมั่นใจในกระบวนการพัฒนาและเปิดตัววัคซีน หากเป็นเช่นนั้น ให้ตอบ 'yes' มิฉะนั้น ให้ตอบ 'no' (โปรดตอบเป็นภาษาอังกฤษ).",

        # Indonesian
        "Konten tweet: {text}. Silakan tandai apakah tweet ini mengekspresikan kepercayaan diri dalam pengembangan dan peluncuran vaksin. Ini mengacu pada penyebutan harapan untuk rilis vaksin, keinginan untuk pengembangan atau rilis yang cepat, dan kepercayaan pada pengembangan dan peluncuran vaksin. Jika demikian, jawab 'yes'; jika tidak, jawab 'no' (silakan jawab dalam bahasa Inggris).",

        # Malay
        "Kandungan tweet: {text}. Sila tandakan sama ada tweet ini menyatakan keyakinan terhadap pembangunan dan pelancaran vaksin. Ini merujuk kepada sebutan harapan untuk pelepasan vaksin, keinginan untuk pembangunan atau pelepasan yang cepat, dan keyakinan terhadap pembangunan dan pelancaran vaksin. Jika ya, jawab 'yes'; jika tidak, jawab 'no' (sila jawab dalam Bahasa Inggeris).",

        # Lao
        "ເນື້ອຫາທວິດ: {text}. ກະລຸນາໝາຍວ່າທວິດນີ້ໄດ້ພາກົດເອົາຄວາມເຊື່ອໃນການພັດທະນາແລະການປ່ອຍວັກຊີນຫຼືບໍ່. ນີ້ແມ່ນການທີ່ຈະສະແດງເຫັນຄວາມຫວັງວ່າວັກຊີນຈະຖືກປ່ອຍອອກຫລືບໍ່, ຄວາມຕ້ອງການໃນການພັດທະນາຢ່າງໄວຫລືບໍ່, ແລະຄວາມເຊື່ອໃນການພັດທະນາແລະການປ່ອຍວັກຊີນ. ຖ້າແມ່ນແລ້ວ, ກະລຸນາຕອບວ່າ 'yes'; ຖ້າບໍ່ແມ່ນ, ກະລຸນາຕອບວ່າ 'no' (ກະລຸນາຕອບໃນພາສາອັງກິດ).",

        # Burmese
        "တွစ်အကြောင်းအရာ: {text}။ ဒီတွစ်ဟာ COVID-19 ကာကွယ်ဆေးထိုးခြင်းမှာ ထုတ်လုပ်ရေးလုပ်ငန်းစဉ်နဲ့ ဈေးကွက်ကို ဖော်ပြခဲ့ပါသလား ဆိုတာကို မှတ်ပေးပါ။ အကယ်၍ရှိပါက 'yes' ဖြင့် ဖြေပါ၊ မဟုတ်ပါက 'no' ဖြင့် ဖြေပါ (ကျေးဇူးပြု၍ အင်္ဂလိပ်ဘာသာဖြင့် ပြန်ဆိုပါ).",

        # Cebuano
        "Sulud sa tweet: {text}. Palihug markahi kung kini nga tweet nagpakita og kasaligan sa pag-uswag ug pagpagawas sa bakuna. Kini nagtumong sa paghisgot sa mga gilauman alang sa pagpagawas sa bakuna, ang tinguha alang sa dali nga pag-uswag o pagpagawas, ug pagsalig sa pag-uswag ug pagpagawas sa bakuna. Kung mao, tubaga og 'yes'; kung dili, tubaga og 'no' (palihug tubaga sa English).",

        # Khmer
        "ខ្លឹមសារទីចេញផ្សាយ: {text}។ សូមសម្គាល់មើលថាតើការផ្សាយនេះបាននិយាយអំពីភាពជឿជាក់លើវ៉ាក់សាំង COVID-19 ឬអត់។ នេះមានន័យថាសំដៅទៅលើការរំពឹងទុកការចេញផ្សាយ និងការផលិតវ៉ាក់សាំង ការរំពឹងរហ័សរហូតចេញផ្សាយវ៉ាក់សាំង និងសមត្ថភាពផលិតភាពលើការផលិតវ៉ាក់សាំងនិងចេញផ្សាយវ៉ាក់សាំង។ ប្រសិនបើចង់និយាយបញ្ចប់សូមឆ្លើយតបដោយនិយាយថា 'yes'។ បើមិនបានបញ្ចប់សូមឆ្លើយតបដោយនិយាយថា 'no'។ (សូមឆ្លើយតបជាភាសាអង់គ្លេស).",

        # Tagalog
        "Nilalaman ng tweet: {text}. Paki-markahan kung ang tweet na ito ay nagpapahayag ng kumpiyansa sa pag-unlad at paglulunsad ng bakuna. Ito ay tumutukoy sa pagbanggit ng mga inaasahan para sa paglabas ng bakuna, ang pagnanais para sa mabilis na pag-unlad o paglabas, at kumpiyansa sa pag-unlad at paglulunsad ng bakuna. Kung oo, sagutin ang 'yes'; kung hindi, sagutin ang 'no' (mangyaring sagutin sa Ingles).",

        # Hindi
        "ट्वीट सामग्री: {text}। कृपया चिह्नित करें कि यह ट्वीट वैक्सीन के विकास और लॉन्च में विश्वास व्यक्त करता है या नहीं। इसका मतलब है कि वैक्सीन की रिलीज के लिए अपेक्षाओं का उल्लेख करना, तेजी से विकास या रिलीज की इच्छा और वैक्सीन के विकास और लॉन्च में विश्वास। यदि हाँ, तो 'yes' में उत्तर दें; अन्यथा, 'no' में उत्तर दें (कृपया अंग्रेजी में उत्तर दें)।",

        # Bengali
        "টুইটের বিষয়বস্তু: {text}। অনুগ্রহ করে চিহ্নিত করুন যে এই টুইটটি টিকা উন্নয়ন এবং প্রবর্তনে আস্থা প্রকাশ করে কিনা। এর মানে হল টিকার প্রকাশের জন্য প্রত্যাশার উল্লেখ করা, দ্রুত বিকাশ বা মুক্তির আকাঙ্ক্ষা এবং টিকা উন্নয়ন এবং প্রবর্তনে আস্থা। যদি হ্যাঁ, তাহলে 'yes' দিয়ে উত্তর দিন; অন্যথায়, 'no' দিয়ে উত্তর দিন (অনুগ্রহ করে ইংরেজিতে উত্তর দিন)।",

        # Urdu
        "ٹویٹ کا مواد: {text}. براہ کرم نشان لگائیں کہ آیا یہ ٹویٹ ویکسین کی ترقی اور لانچ پر اعتماد کا اظہار کرتا ہے۔ اس کا مطلب ہے ویکسین کے اجراء کے لئے توقعات کا ذکر کرنا، تیز رفتار ترقی یا ریلیز کی خواہش اور ویکسین کی ترقی اور لانچ پر اعتماد۔ اگر ایسا ہے تو، 'yes' میں جواب دیں؛ ورنہ، 'no' میں جواب دیں (براہ کرم انگریزی میں جواب دیں)."
    ]
}

df = df[df['18']==0]

# Create a new DataFrame that will contain the instructions for each label
instructions_df = pd.DataFrame(columns=['instruction', 'output', 'category'])

# Function to create instruction for a given row and category
def create_instruction(text, category):
    prompt = random.choice(multilingual_prompts[category])
    instruction = prompt.format(text=text)
    return instruction

for index, row in df.iterrows():
    text = row['text']
    for category in categories:
        instruction = create_instruction(text, category)
        if row[category] == 1:
            output = 'yes'
        else:
            output = 'no'
        instructions_df = pd.concat([instructions_df, pd.DataFrame({'instruction': [instruction], 'output': [output], 'category':[category]})], ignore_index=True)
# Separate the majority and minority classes
df_majority = instructions_df[instructions_df.output == 'no']
df_minority = instructions_df[instructions_df.output == 'yes']

# Downsample the majority class
df_majority_downsampled = resample(df_majority,
                                   replace=False,    # sample without replacement
                                   n_samples=36000, # to match minority class
                                   random_state=42) # reproducible results

# Combine the minority class with the downsampled majority class
instructions_df_balanced = pd.concat([df_majority_downsampled, df_minority])
# sample n=45000 and save to the same parquet file
instructions_df_balanced[['instruction', 'output']].sample(n=45000).to_parquet("../../data/TCV/TCV-others.parquet", index=False)