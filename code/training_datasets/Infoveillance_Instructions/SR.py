# -*- coding: utf-8 -*-
# SR (Stress – Reddit) 
# - Paper: https://doi.org/10.48550/arXiv.1911.00133
# - Data: http://www.cs.columbia.edu/~eturcan/data/dreaddit.zip

import pandas as pd
import random

# Load the datasets
train_df = pd.read_csv("../../data/SR/dreaddit-train.csv")
test_df = pd.read_csv("../../data/SR/dreaddit-test.csv")

prompts = {
    "Chinese": [
        "请检测以下Reddit帖子中是否存在压力。如果有，请回答“yes”，否则，请回答“no”。压力在这里指的是由于不利或苛刻环境导致的精神或情绪上的紧张状态。帖子内容：{}",
        "请判断以下Reddit帖子中是否存在压力。如果存在，请回答“yes”；如果不存在，请回答“no”。压力在此定义为由逆境或要求苛刻的环境引起的精神或情感上的紧张状态。帖子内容：{}"
    ],
    "English": [
        "Detect if stress is present in the Reddit post below. If so, respond 'yes'. Otherwise, respond 'no'. Stress here means a state of mental or emotional strain or tension resulting from adverse or demanding circumstances. post: {}",
        "Determine if stress is evident in the Reddit post below. If it is, please respond with 'yes'. If it is not, respond with 'no'. Stress is defined here as mental or emotional strain caused by challenging or adverse situations. post: {}"
    ],
    "German": [
        "Ermitteln Sie, ob im folgenden Reddit-Beitrag Stress vorhanden ist. Wenn ja, antworten Sie mit „yes“, andernfalls antworten Sie mit „no“. Stress bedeutet hier einen Zustand geistiger oder emotionaler Anspannung, der durch widrige oder belastende Umstände verursacht wird. Beitrag: {}",
        "Überprüfen Sie, ob im folgenden Reddit-Beitrag Stress vorhanden ist. Wenn ja, antworten Sie bitte mit „yes“. Wenn nicht, antworten Sie bitte mit „no“. Stress wird hier als geistige oder emotionale Anspannung definiert, die durch belastende oder herausfordernde Umstände verursacht wird. Beitrag: {}"
    ],
    "French": [
        "Détectez si le stress est présent dans le post Reddit ci-dessous. Si c'est le cas, répondez par 'yes'. Sinon, répondez par 'no'. Le stress ici signifie un état de tension mentale ou émotionnelle résultant de circonstances défavorables ou exigeantes. contenu du post: {}",
        "Déterminez si le stress est présent dans le post Reddit ci-dessous. Si oui, répondez par 'yes'. Sinon, répondez par 'no'. Le stress est ici défini comme un état de tension mentale ou émotionnelle causé par des circonstances défavorables ou exigeantes. contenu du post: {}"
    ],
    "Spanish": [
        "Detecta si el estrés está presente en la publicación de Reddit a continuación. Si es así, responde 'yes'. De lo contrario, responde 'no'. El estrés aquí significa un estado de tensión mental o emocional resultante de circunstancias adversas o exigentes. contenido de la publicación: {}",
        "Determina si el estrés es evidente en la publicación de Reddit a continuación. Si lo es, responde con 'yes'. Si no lo es, responde con 'no'. El estrés se define aquí como un estado de tensión mental o emocional causado por situaciones desafiantes o adversas. contenido de la publicación: {}"
    ],
    "Portuguese": [
        "Detecte se há estresse presente na postagem do Reddit abaixo. Se houver, responda 'yes'. Caso contrário, responda 'no'. Estresse aqui significa um estado de tensão mental ou emocional resultante de circunstâncias adversas ou exigentes. conteúdo da postagem: {}",
        "Determine se o estresse está presente na postagem do Reddit abaixo. Se estiver, responda com 'yes'. Se não estiver, responda com 'no'. Estresse aqui é definido como um estado de tensão mental ou emocional causado por situações desafiadoras ou adversas. conteúdo da postagem: {}"
    ],
    "Italian": [
        "Rileva se lo stress è presente nel post di Reddit qui sotto. Se sì, rispondi 'yes'. Altrimenti, rispondi 'no'. Lo stress qui significa uno stato di tensione mentale o emotiva derivante da circostanze avverse o impegnative. contenuto del post: {}",
        "Determina se lo stress è presente nel post di Reddit qui sotto. Se sì, rispondi con 'yes'. Altrimenti, rispondi con 'no'. Lo stress qui è definito come uno stato di tensione mentale o emotiva causato da situazioni difficili o avverse. contenuto del post: {}"
    ],
    "Dutch": [
        "Detecteer of stress aanwezig is in het onderstaande Reddit-bericht. Als dat zo is, reageer dan met 'yes'. Zo niet, reageer dan met 'no'. Stress betekent hier een toestand van mentale of emotionele spanning die voortkomt uit ongunstige of veeleisende omstandigheden. bericht: {}",
        "Bepaal of stress aanwezig is in het onderstaande Reddit-bericht. Als dat zo is, reageer dan met 'yes'. Zo niet, reageer dan met 'no'. Stress wordt hier gedefinieerd als een toestand van mentale of emotionele spanning veroorzaakt door uitdagende of ongunstige situaties. bericht: {}"
    ],
    "Russian": [
        "Определите, присутствует ли стресс в следующем посте на Reddit. Если да, ответьте 'yes'. Если нет, ответьте 'no'. Стресс здесь означает состояние психического или эмоционального напряжения, возникающее в результате неблагоприятных или требовательных обстоятельств. содержание поста: {}",
        "Проверьте, присутствует ли стресс в следующем посте на Reddit. Если да, ответьте 'yes'. Если нет, ответьте 'no'. Стресс определяется здесь как состояние психического или эмоционального напряжения, вызванного неблагоприятными или трудными обстоятельствами. содержание поста: {}"
    ],
    "Czech": [
        "Zjistěte, zda je v následujícím příspěvku na Redditu přítomen stres. Pokud ano, odpovězte 'yes'. Jinak odpovězte 'no'. Stres zde znamená stav duševního nebo emocionálního napětí způsobeného nepříznivými nebo náročnými okolnostmi. obsah příspěvku: {}",
        "Určete, zda je v následujícím příspěvku na Redditu přítomen stres. Pokud ano, odpovězte 'yes'. Pokud ne, odpovězte 'no'. Stres je zde definován jako stav duševního nebo emocionálního napětí způsobeného náročnými nebo nepříznivými situacemi. obsah příspěvku: {}"
    ],
    "Polish": [
        "Wykryj, czy stres występuje w poniższym poście na Reddit. Jeśli tak, odpowiedz 'yes'. W przeciwnym razie odpowiedz 'no'. Stres oznacza tutaj stan napięcia psychicznego lub emocjonalnego wynikającego z niekorzystnych lub wymagających okoliczności. treść posta: {}",
        "Określ, czy stres jest obecny w poniższym poście na Reddit. Jeśli tak, odpowiedz 'yes'. Jeśli nie, odpowiedz 'no'. Stres definiowany jest tutaj jako stan napięcia psychicznego lub emocjonalnego spowodowanego wyzwaniami lub niekorzystnymi sytuacjami. treść posta: {}"
    ],
    "Arabic": [
        "اكتشف ما إذا كان الإجهاد موجودًا في المنشور أدناه على Reddit. إذا كان الأمر كذلك، فأجب بـ 'yes'. خلاف ذلك، أجب بـ 'no'. يعني الإجهاد هنا حالة من الإجهاد العقلي أو العاطفي الناتج عن ظروف غير مواتية أو مطالبة. محتوى المنشور: {}",
        "حدد ما إذا كان الإجهاد موجودًا في المنشور أدناه على Reddit. إذا كان الأمر كذلك، أجب بـ 'yes'. إذا لم يكن الأمر كذلك، أجب بـ 'no'. يُعرَّف الإجهاد هنا على أنه حالة من الإجهاد العقلي أو العاطفي الناتج عن المواقف الصعبة أو غير المواتية. محتوى المنشور: {}"
    ],
    "Persian": [
        "تشخیص دهید که آیا استرس در پست زیر در Reddit وجود دارد یا خیر. اگر وجود دارد، با 'yes' پاسخ دهید. در غیر این صورت، با 'no' پاسخ دهید. استرس در اینجا به معنای حالت فشار روانی یا عاطفی ناشی از شرایط نامساعد یا پرخطر است. محتوای پست: {}",
        "تعیین کنید که آیا استرس در پست زیر در Reddit وجود دارد یا خیر. اگر وجود دارد، با 'yes' پاسخ دهید. در غیر این صورت، با 'no' پاسخ دهید. استرس در اینجا به معنای حالت فشار روانی یا عاطفی ناشی از موقعیت‌های دشوار یا نامطلوب است. محتوای پست: {}"
    ],
    "Hebrew": [
        "גלה אם יש נוכחות של לחץ בפוסט הבא ב-Reddit. אם כן, השב 'yes'. אחרת, השב 'no'. לחץ כאן מתאר מצב של מתח נפשי או רגשי הנובע ממצבים לא רצויים או מאתגרים. תוכן הפוסט: {}",
        "בדוק אם יש נוכחות של לחץ בפוסט הבא ב-Reddit. אם כן, השב 'yes'. אם לא, השב 'no'. לחץ מוגדר כאן כמצב של מתח נפשי או רגשי הנגרם ממצבים קשים או לא רצויים. תוכן הפוסט: {}"
    ],
    "Turkish": [
        "Aşağıdaki Reddit gönderisinde stres olup olmadığını tespit edin. Eğer varsa, 'yes' yanıtını verin. Aksi takdirde, 'no' yanıtını verin. Burada stres, olumsuz veya zorlu koşulların neden olduğu zihinsel veya duygusal gerginlik durumu anlamına gelir. gönderi içeriği: {}",
        "Aşağıdaki Reddit gönderisinde stres olup olmadığını belirleyin. Eğer öyleyse, 'yes' yanıtını verin. Eğer değilse, 'no' yanıtını verin. Stres burada, zorlayıcı veya olumsuz durumların neden olduğu zihinsel veya duygusal gerginlik durumu olarak tanımlanır. gönderi içeriği: {}"
    ],
    "Japanese": [
        "以下のReddit投稿にストレスがあるかどうかを検出してください。もしある場合は「yes」と答えてください。それ以外の場合は「no」と答えてください。ここでのストレスは、逆境や要求の厳しい状況から生じる精神的または感情的な緊張状態を意味します。投稿内容: {}",
        "以下のReddit投稿にストレスがあるかどうかを判断してください。もしある場合は「yes」と答えてください。それ以外の場合は「no」と答えてください。ここでのストレスは、困難な状況や逆境から生じる精神的または感情的な緊張状態として定義されています。投稿内容: {}"
    ],
    "Korean": [
        "다음 Reddit 게시물에 스트레스가 있는지 감지하세요. 만약 그렇다면 'yes'라고 응답하세요. 그렇지 않다면 'no'라고 응답하세요. 여기서 스트레스는 불리한 상황이나 까다로운 상황으로 인한 정신적 또는 감정적 긴장 상태를 의미합니다. 게시물 내용: {}",
        "다음 Reddit 게시물에 스트레스가 있는지 확인하세요. 그렇다면 'yes'라고 응답하세요. 그렇지 않다면 'no'라고 응답하세요. 여기서 스트레스는 힘든 상황이나 불리한 상황으로 인한 정신적 또는 감정적 긴장 상태로 정의됩니다. 게시물 내용: {}"
    ],
    "Vietnamese": [
        "Phát hiện xem có sự căng thẳng trong bài đăng trên Reddit dưới đây hay không. Nếu có, hãy trả lời 'yes'. Nếu không, hãy trả lời 'no'. Căng thẳng ở đây có nghĩa là trạng thái căng thẳng tâm lý hoặc cảm xúc do hoàn cảnh bất lợi hoặc đòi hỏi gây ra. nội dung bài đăng: {}",
        "Xác định xem căng thẳng có hiện diện trong bài đăng trên Reddit dưới đây không. Nếu có, vui lòng trả lời 'yes'. Nếu không, hãy trả lời 'no'. Căng thẳng ở đây được định nghĩa là trạng thái căng thẳng tâm lý hoặc cảm xúc do hoàn cảnh khó khăn hoặc bất lợi gây ra. nội dung bài đăng: {}"
    ],
    "Thai": [
        "ตรวจสอบว่ามีความเครียดในโพสต์ Reddit ด้านล่างหรือไม่ หากมี โปรดตอบ 'yes' มิฉะนั้นโปรดตอบ 'no' ความเครียดในที่นี้หมายถึงภาวะตึงเครียดทางจิตใจหรืออารมณ์อันเป็นผลมาจากสถานการณ์ที่ไม่เอื้ออำนวยหรือเรียกร้องมากเกินไป เนื้อหาโพสต์: {}",
        "ระบุว่ามีความเครียดในโพสต์ Reddit ด้านล่างหรือไม่ หากมี โปรดตอบ 'yes' หากไม่มี โปรดตอบ 'no' ความเครียดในที่นี้กำหนดให้เป็นภาวะตึงเครียดทางจิตใจหรืออารมณ์ที่เกิดจากสถานการณ์ที่ท้าทายหรือไม่พึงประสงค์ เนื้อหาโพสต์: {}"
    ],
    "Indonesian": [
        "Deteksi apakah ada stres dalam postingan Reddit di bawah ini. Jika ada, jawab 'yes'. Jika tidak, jawab 'no'. Stres di sini berarti keadaan ketegangan mental atau emosional yang diakibatkan oleh keadaan yang merugikan atau menuntut. isi postingan: {}",
        "Tentukan apakah stres ada dalam postingan Reddit di bawah ini. Jika ada, tolong jawab 'yes'. Jika tidak, jawab 'no'. Stres di sini didefinisikan sebagai keadaan ketegangan mental atau emosional yang disebabkan oleh situasi yang menantang atau merugikan. isi postingan: {}"
    ],
    "Malay": [
        "Kesankan jika tekanan wujud dalam pos Reddit di bawah. Jika ada, balas 'yes'. Jika tidak, balas 'no'. Tekanan di sini bermaksud keadaan ketegangan mental atau emosi akibat daripada keadaan yang buruk atau menuntut. kandungan pos: {}",
        "Tentukan jika tekanan hadir dalam pos Reddit di bawah. Jika ya, balas dengan 'yes'. Jika tidak, balas dengan 'no'. Tekanan di sini ditakrifkan sebagai keadaan ketegangan mental atau emosi yang disebabkan oleh keadaan yang mencabar atau tidak baik. kandungan pos: {}"
    ],
    "Lao": [
        "ກວດກາວ່າມີຄວາມເຄັງຢູ່ໃນໂພສ Reddit ຂ້າງລຸ່ມນີ້ຫຼືບໍ່. ຖ້າມີ, ກະລຸນາຕອບ 'yes'. ຖ້າບໍ່ມີ, ກະລຸນາຕອບ 'no'. ຄວາມເຄັງໃນທີ່ນີ້ຫມາຍເຖິງສະພາບຂອງການເຄັງດ້ານຈິດໃຈຫຼືອາລົມທີ່ເກີດຂຶ້ນຈາກສະຖານະການທີ່ບໍ່ດີຫຼືມີການເອົາໃຈມາກເກີນໄປ. ຂໍ້ຄວາມໂພສ: {}",
        "ກຳນົດວ່າມີຄວາມເຄັງຢູ່ໃນໂພສ Reddit ຂ້າງລຸ່ມນີ້ຫຼືບໍ່. ຖ້າມີ, ກະລຸນາຕອບ 'yes'. ຖ້າບໍ່ມີ, ກະລຸນາຕອບ 'no'. ຄວາມເຄັງໃນທີ່ນີ້ຖືກກຳນົດເປັນສະພາບຂອງການເຄັງທາງຈິດໃຈຫຼືອາລົມທີ່ເກີດຂຶ້ນເນື່ອງຈາກສະຖານະການທີ່ທ້າທາຍຫຼືບໍ່ດີ. ຂໍ້ຄວາມໂພສ: {}"
    ],
    "Burmese": [
        "အောက်ပါ Reddit ပို့စ်တွင် စိတ်ဖိစီးမှုရှိမရှိကို ရှာဖွေပါ။ ရှိပါက 'yes' ဟု ဖြေကြားပါ။ မရှိပါက 'no' ဟု ဖြေကြားပါ။ စိတ်ဖိစီးမှုဆိုသည်မှာ မညစ်ညမ်းသော သို့မဟုတ် ခက်ခဲသော အခြေအနေများကြောင့် စိတ်ပိုင်းဆိုင်ရာ သို့မဟုတ် စိတ်ခံစားချက်အပန်းဖြစ်သည်။ ပို့စ်၏ အကြောင်းအရာ: {}",
        "အောက်ပါ Reddit ပို့စ်တွင် စိတ်ဖိစီးမှုရှိမရှိကို ဆုံးဖြတ်ပါ။ ရှိပါက 'yes' ဟု ဖြေကြားပါ။ မရှိပါက 'no' ဟု ဖြေကြားပါ။ ဒီမှာ စိတ်ဖိစီးမှုဆိုတာ ခက်ခဲတဲ့ သို့မဟုတ် မညစ်ညမ်းသော အခြေအနေတွေကြောင့် ဖြစ်ပေါ်တဲ့ စိတ်ပိုင်းဆိုင်ရာ သို့မဟုတ် စိတ်ခံစားမှုအပန်းဖြစ်ပါသည်။ ပို့စ်၏ အကြောင်းအရာ: {}"
    ],
    "Cebuano": [
        "Susihi kon ang stress anaa sa ubos nga post sa Reddit. Kon naa, tubaga 'yes'. Kon wala, tubaga 'no'. Ang stress dinhi nagpasabot sa usa ka kahimtang sa mental o emosyonal nga tensiyon nga resulta sa daotan o mapangayo nga mga kahimtang. sulod sa post: {}",
        "I-determinar kung may stress nga makita sa ubos nga post sa Reddit. Kung oo, tubaga 'yes'. Kung dili, tubaga 'no'. Ang stress dinhi nagpasabot sa mental o emosyonal nga tensiyon nga resulta sa malisud nga mga kahimtang. sulod sa post: {}"
    ],
    "Khmer": [
        "ស្វែងរកមើលថាតើមានភាពតានតឹងនៅក្នុងបណ្តុំពី Reddit ខាងក្រោមនេះដែរឬទេ។ ប្រសិនបើមាន សូមឆ្លើយតប 'yes' ប្រសិនបើមិនមាន សូមឆ្លើយ 'no'។ ភាពតានតឹងនៅទីនេះមានន័យថាជាការតានតឹងផ្លូវចិត្តឬអារម្មណ៍ដែលបណ្តាលមកពីលក្ខខណ្ឌអវិជ្ជមាន ឬតម្រូវឱ្យតានតឹង។ មាតិកាបណ្តុំនេះ៖ {}",
        "កំណត់មើលថាតើមានភាពតានតឹងនៅក្នុងបណ្តុំពី Reddit ខាងក្រោមនេះដែរឬទេ។ ប្រសិនបើមាន សូមឆ្លើយតប 'yes' ប្រសិនបើមិនមាន សូមឆ្លើយ 'no'។ ភាពតានតឹងនៅទីនេះត្រូវបានកំណត់ថាជាការតានតឹងផ្លូវចិត្ត ឬអារម្មណ៍ដែលបណ្តាលមកពីស្ថានភាពលំបាក ឬអវិជ្ជមាន។ មាតិកាបណ្តុំនេះ៖ {}"
    ],
    "Tagalog": [
        "Tukuyin kung may stress sa sumusunod na post sa Reddit. Kung mayroon, sagutin ng 'yes'. Kung wala, sagutin ng 'no'. Ang stress dito ay nangangahulugang isang estado ng mental o emosyonal na tensyon na nagreresulta mula sa hindi kanais-nais o mapanghamon na mga kalagayan. nilalaman ng post: {}",
        "Tukuyin kung may stress na makikita sa sumusunod na post sa Reddit. Kung mayroon, sagutin ng 'yes'. Kung wala, sagutin ng 'no'. Ang stress ay tinutukoy dito bilang isang estado ng mental o emosyonal na tensyon na dulot ng mahirap o hindi kanais-nais na mga sitwasyon. nilalaman ng post: {}"
    ],
    "Hindi": [
        "जांचें कि नीचे दिए गए Reddit पोस्ट में तनाव है या नहीं। यदि है, तो 'yes' उत्तर दें। अन्यथा, 'no' उत्तर दें। तनाव का मतलब यहां प्रतिकूल या मांगलिक परिस्थितियों से उत्पन्न मानसिक या भावनात्मक तनाव की स्थिति है। पोस्ट की सामग्री: {}",
        "निर्धारित करें कि नीचे दिए गए Reddit पोस्ट में तनाव मौजूद है या नहीं। यदि हां, तो 'yes' उत्तर दें। यदि नहीं, तो 'no' उत्तर दें। यहां तनाव का अर्थ उन मानसिक या भावनात्मक तनाव की स्थिति से है जो प्रतिकूल या चुनौतीपूर्ण परिस्थितियों से उत्पन्न होती है। पोस्ट की सामग्री: {}"
    ],
    "Bengali": [
        "নিচের Reddit পোস্টে চাপ আছে কিনা তা সনাক্ত করুন। যদি থাকে, তাহলে 'yes' দিয়ে উত্তর দিন। যদি না থাকে, তাহলে 'no' দিয়ে উত্তর দিন। এখানে চাপের অর্থ এমন মানসিক বা আবেগগত চাপ যা প্রতিকূল বা চাহিদাপূর্ণ পরিস্থিতি থেকে উদ্ভূত হয়। পোস্টের বিষয়বস্তু: {}",
        "নির্ধারণ করুন যে নিচের Reddit পোস্টে চাপ আছে কিনা। যদি থাকে, তাহলে 'yes' দিয়ে উত্তর দিন। যদি না থাকে, তাহলে 'no' দিয়ে উত্তর দিন। এখানে চাপকে এমন মানসিক বা আবেগগত চাপের অবস্থা হিসাবে সংজ্ঞায়িত করা হয়েছে যা প্রতিকূল বা চ্যালেঞ্জিং পরিস্থিতি দ্বারা সৃষ্ট। পোস্টের বিষয়বস্তু: {}"
    ],
    "Urdu": [
        "یہ جانچیں کہ کیا نیچے دیے گئے Reddit پوسٹ میں دباؤ موجود ہے؟ اگر ہے، تو 'yes' کے ساتھ جواب دیں۔ ورنہ، 'no' کے ساتھ جواب دیں۔ یہاں دباؤ کا مطلب ایسی ذہنی یا جذباتی تناؤ کی حالت ہے جو ناموافق یا طلبگار حالات سے پیدا ہوتی ہے۔ پوسٹ کا مواد: {}",
        "یہ تعین کریں کہ آیا نیچے دیے گئے Reddit پوسٹ میں دباؤ موجود ہے یا نہیں؟ اگر ہے، تو 'yes' کے ساتھ جواب دیں۔ اگر نہیں ہے، تو 'no' کے ساتھ جواب دیں۔ یہاں دباؤ کو ایسے ذہنی یا جذباتی تناؤ کی حالت کے طور پر تعریف کیا گیا ہے جو چیلنجنگ یا ناموافق حالات سے پیدا ہوتی ہے۔ پوسٹ کا مواد: {}"
    ]
}


all_prompts = [prompt for lang_prompts in prompts.values() for prompt in lang_prompts]

train_df['instruction'] = train_df["text"].apply(lambda x: random.choice(all_prompts).format(x))
train_df['output'] = train_df['label'].apply(lambda x: 'yes' if x else 'no')
train_df = train_df[['instruction', 'output']]

test_df['instruction'] = test_df["text"].apply(lambda x: random.choice(all_prompts).format(x))
test_df['output'] = test_df['label'].apply(lambda x: 'yes' if x else 'no')
test_df = test_df[['instruction', 'output']]

data = pd.concat([train_df, test_df])

# collect a sample of the data n=3000
data = data.sample(n=3000, random_state=42).reset_index(drop=True)

# Save to parquet without the index
data.to_parquet("../../data/SR/SR.parquet", index = False)