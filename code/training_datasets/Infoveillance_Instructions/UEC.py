# -*- coding: utf-8 -*-
# UEC (Urdu Emotion Classification) 
# - Paper: https://doi.org/10.1145/3574318.3574327
# - Data: https://sites.google.com/view/multi-label-emotionsfire-task/dataset

import pandas as pd
import random

# Combine the training and test data
taskA_train_file = '../../data/UEC/train_set_taskA - train_set_Fire2022.csv'
taskA_test_file = '../../data/UEC/test_set_taskA_labelled - test_set_Fire2022_labelled.csv'
# UEC
# Define multilingual templates for emotion detection
emotion_templates = [
    # Chinese
    """该数据集包括乌尔都语Nastalíq推文，描述了人们与世界分享他们的活动、意见和事件时的情绪。请确定以下推文中是否表达了以下情绪之一：anger, disgust, fear, happiness, sadness, surprise, neutral。注意，推文中可能包含0、1或多种情绪。如果您在推文中发现多种情绪，请用逗号和空格（", "）分隔这些情绪。请按照上述顺序组织您的回答。推文内容："{text}"。推文到此结束。请回应您认为推文包含的情绪。当没有检测到任何情绪时，请不要回应。""",

    # English
    """The dataset comprises Urdu Nastalíq tweets elucidating the emotions of people as they describe their activities, opinions, and events with the world. Please determine if the tweet below expressed any of the following emotions: anger, disgust, fear, happiness, sadness, surprise, neutral. Notice that the tweet might contain 0, 1, or multiple emotions. If you notice multiple emotions in a tweet, please respond with all of them, separating the emotions with a comma and a space (", "). To organize your response, please follow the aforementioned order when mentioning multiple emotions. Tweet: "{text}". Now the tweet ends. Please respond with the emotion(s) you think the tweet contains. Respond with nothing when there's no emotion detected.""",

    # German
    """Der Datensatz besteht aus Urdu Nastalíq-Tweets, die die Emotionen der Menschen beschreiben, während sie ihre Aktivitäten, Meinungen und Ereignisse mit der Welt teilen. Bitte bestimmen Sie, ob der untenstehende Tweet eine der folgenden Emotionen ausdrückt: anger, disgust, fear, happiness, sadness, surprise, neutral. Beachten Sie, dass der Tweet 0, 1 oder mehrere Emotionen enthalten kann. Wenn Sie mehrere Emotionen in einem Tweet bemerken, antworten Sie bitte mit allen, indem Sie die Emotionen durch ein Komma und ein Leerzeichen (", ") trennen. Um Ihre Antwort zu organisieren, folgen Sie bitte der oben genannten Reihenfolge, wenn Sie mehrere Emotionen nennen. Tweet: "{text}". Jetzt endet der Tweet. Bitte antworten Sie mit den Emotionen, die Sie im Tweet erkennen. Antworten Sie mit nichts, wenn keine Emotionen erkannt werden.""",

    # French
    """Le jeu de données comprend des tweets en Nastalíq urdu qui expriment les émotions des personnes lorsqu'elles décrivent leurs activités, opinions et événements au monde. Veuillez déterminer si le tweet ci-dessous exprime l'une des émotions suivantes : anger, disgust, fear, happiness, sadness, surprise, neutral. Notez que le tweet peut contenir 0, 1 ou plusieurs émotions. Si vous remarquez plusieurs émotions dans un tweet, veuillez répondre avec toutes, en séparant les émotions par une virgule et un espace (", "). Pour organiser votre réponse, veuillez suivre l'ordre mentionné ci-dessus lors de la mention de plusieurs émotions. Tweet : "{text}". Le tweet se termine ici. Veuillez répondre avec l'émotion ou les émotions que vous pensez que le tweet contient. Répondez avec rien lorsqu'aucune émotion n'est détectée.""",

    # Spanish
    """El conjunto de datos incluye tweets en Nastalíq urdu que explican las emociones de las personas mientras describen sus actividades, opiniones y eventos con el mundo. Por favor, determine si el tweet a continuación expresa alguna de las siguientes emociones: anger, disgust, fear, happiness, sadness, surprise, neutral. Tenga en cuenta que el tweet puede contener 0, 1 o múltiples emociones. Si nota varias emociones en un tweet, responda con todas, separando las emociones con una coma y un espacio (", "). Para organizar su respuesta, siga el orden mencionado anteriormente al mencionar múltiples emociones. Tweet: "{text}". El tweet termina aquí. Por favor, responda con las emociones que cree que contiene el tweet. No responda nada si no se detecta ninguna emoción.""",

    # Portuguese
    """O conjunto de dados é composto por tweets em Nastalíq urdu que elucidam as emoções das pessoas enquanto descrevem suas atividades, opiniões e eventos com o mundo. Por favor, determine se o tweet abaixo expressou alguma das seguintes emoções: anger, disgust, fear, happiness, sadness, surprise, neutral. Observe que o tweet pode conter 0, 1 ou várias emoções. Se você notar várias emoções em um tweet, responda com todas, separando as emoções com uma vírgula e um espaço (", "). Para organizar sua resposta, siga a ordem mencionada acima ao mencionar várias emoções. Tweet: "{text}". Agora o tweet termina. Responda com as emoções que você acha que o tweet contém. Responda com nada quando nenhuma emoção for detectada.""",

    # Italian
    """Il dataset è composto da tweet in Nastalíq urdu che descrivono le emozioni delle persone mentre descrivono le loro attività, opinioni ed eventi con il mondo. Si prega di determinare se il tweet di seguito ha espresso una delle seguenti emozioni: anger, disgust, fear, happiness, sadness, surprise, neutral. Notare che il tweet potrebbe contenere 0, 1 o più emozioni. Se noti più emozioni in un tweet, rispondi con tutte, separando le emozioni con una virgola e uno spazio (", "). Per organizzare la tua risposta, segui l'ordine sopra menzionato quando menzioni più emozioni. Tweet: "{text}". Ora il tweet termina. Rispondi con le emozioni che pensi che il tweet contenga. Rispondi con nulla quando non viene rilevata alcuna emozione.""",

    # Dutch
    """De dataset bestaat uit Urdu Nastalíq-tweets die de emoties van mensen beschrijven terwijl ze hun activiteiten, meningen en gebeurtenissen met de wereld delen. Bepaal of de onderstaande tweet een van de volgende emoties uitdrukt: anger, disgust, fear, happiness, sadness, surprise, neutral. Merk op dat de tweet 0, 1 of meerdere emoties kan bevatten. Als u meerdere emoties opmerkt in een tweet, reageer dan met allemaal, waarbij u de emoties scheidt met een komma en een spatie (", "). Volg bij het organiseren van uw antwoord de hierboven genoemde volgorde bij het vermelden van meerdere emoties. Tweet: "{text}". Nu eindigt de tweet. Reageer met de emotie(s) waarvan u denkt dat de tweet deze bevat. Reageer met niets wanneer er geen emotie wordt gedetecteerd.""",

    # Russian
    """Набор данных состоит из твитов на урду на насталике, иллюстрирующих эмоции людей, когда они описывают свои действия, мнения и события в мире. Пожалуйста, определите, выражает ли нижеприведенный твит какую-либо из следующих эмоций: anger, disgust, fear, happiness, sadness, surprise, neutral. Обратите внимание, что твит может содержать 0, 1 или несколько эмоций. Если вы заметили несколько эмоций в твите, пожалуйста, ответьте всеми, разделив эмоции запятой и пробелом (", "). Чтобы организовать свой ответ, следуйте вышеупомянутому порядку, когда упоминаете несколько эмоций. Твит: "{text}". Теперь твит заканчивается. Пожалуйста, ответьте эмоцией(ями), которые вы считаете содержатся в твите. Не отвечайте ничего, если эмоции не обнаружены.""",

    # Czech
    """Datová sada obsahuje tweety v urdu jazyce Nastalíq, které ilustrují emoce lidí při popisování jejich aktivit, názorů a událostí se světem. Určete, zda níže uvedený tweet vyjadřuje některou z následujících emocí: anger, disgust, fear, happiness, sadness, surprise, neutral. Všimněte si, že tweet může obsahovat 0, 1 nebo více emocí. Pokud si všimnete více emocí v tweetu, odpovězte prosím se všemi z nich a oddělte emoce čárkou a mezerou (", "). Pro organizaci své odpovědi prosím dodržujte výše uvedené pořadí při zmiňování více emocí. Tweet: "{text}". Tweet nyní končí. Odpovězte prosím emocí(emi), o kterých si myslíte, že jsou obsaženy v tweetu. Neodpovídejte nic, pokud nejsou zjištěny žádné emoce.""",

    # Polish
    """Zbiór danych składa się z tweetów w urdu Nastalíq, które przedstawiają emocje ludzi, gdy opisują swoje działania, opinie i wydarzenia na świecie. Określ, czy poniższy tweet wyraża którąś z następujących emocji: anger, disgust, fear, happiness, sadness, surprise, neutral. Zauważ, że tweet może zawierać 0, 1 lub więcej emocji. Jeśli zauważysz wiele emocji w tweecie, odpowiedz wszystkimi z nich, oddzielając emocje przecinkiem i spacją (", "). Aby zorganizować swoją odpowiedź, postępuj zgodnie z powyższym porządkiem, wspominając o wielu emocjach. Tweet: "{text}". Teraz tweet się kończy. Odpowiedz emocjami, które według ciebie zawiera tweet. Odpowiedz niczym, gdy nie wykryto żadnych emocji.""",

    # Arabic
    """تتألف مجموعة البيانات من تغريدات أوردو نستلِق التي توضح مشاعر الناس عندما يصفون أنشطتهم وآرائهم وأحداثهم مع العالم. يرجى تحديد ما إذا كانت التغريدة أدناه تعبّر عن أي من المشاعر التالية: anger, disgust, fear, happiness, sadness, surprise, neutral. لاحظ أن التغريدة قد تحتوي على 0 أو 1 أو عدة مشاعر. إذا لاحظت وجود عدة مشاعر في تغريدة، يرجى الرد بها جميعًا، بفصل المشاعر بفاصلة ومسافة (", "). لتنظيم ردك، يرجى اتباع الترتيب المذكور أعلاه عند ذكر عدة مشاعر. التغريدة: "{text}". تنتهي التغريدة هنا. يرجى الرد بالمشاعر التي تعتقد أن التغريدة تحتوي عليها. الرد بدون شيء عندما لا يتم الكشف عن أي مشاعر.""",

    # Persian
    """این مجموعه داده شامل توییت‌های Nastalíq اردو است که احساسات مردم را در هنگام توصیف فعالیت‌ها، نظرات و رویدادهایشان با جهان بیان می‌کند. لطفاً تعیین کنید که آیا توییت زیر هر یک از احساسات زیر را بیان می‌کند: anger, disgust, fear, happiness, sadness, surprise, neutral. توجه داشته باشید که توییت ممکن است شامل 0، 1 یا چندین احساس باشد. اگر متوجه شدید که در یک توییت چندین احساس وجود دارد، لطفاً با همه آن‌ها پاسخ دهید و احساسات را با ویرگول و فاصله (", ") جدا کنید. برای سازماندهی پاسخ خود، لطفاً هنگام ذکر چندین احساس، از ترتیب مذکور پیروی کنید. توییت: "{text}". اکنون توییت به پایان می‌رسد. لطفاً با احساس(های)ی که فکر می‌کنید توییت شامل آن‌ها است پاسخ دهید. اگر هیچ احساسی تشخیص داده نشود، هیچ پاسخی ندهید.""",

    # Hebrew
    """מערך הנתונים כולל ציוצים באורדו Nastalíq המתארים את הרגשות של אנשים כאשר הם מתארים את הפעילויות, הדעות והאירועים שלהם עם העולם. נא לקבוע האם הציוץ למטה מביע אחד מהרגשות הבאים: anger, disgust, fear, happiness, sadness, surprise, neutral. שימו לב כי הציוץ עשוי להכיל 0, 1 או מספר רגשות. אם אתם מבחינים במספר רגשות בציוץ, נא להשיב עם כולם, תוך הפרדת הרגשות באמצעות פסיק ורווח (", "). כדי לארגן את התשובה שלכם, נא לעקוב אחר הסדר המוזכר לעיל בעת אזכור מספר רגשות. ציוץ: "{text}". הציוץ מסתיים כאן. נא להשיב עם הרגש/ים שאתם חושבים שהציוץ מכיל. השיבו עם כלום כאשר אין רגשות מזוהים.""",

    # Turkish
    """Veri seti, insanların dünyayla paylaştıkları etkinlikleri, görüşleri ve olayları betimlerken hissettikleri duyguları açıklayan Urdu Nastalíq tweetlerinden oluşmaktadır. Lütfen aşağıdaki tweet'in herhangi bir duyguyu ifade edip etmediğini belirleyin: anger, disgust, fear, happiness, sadness, surprise, neutral. Tweet'in 0, 1 veya birden fazla duyguyu içerebileceğini unutmayın. Tweet'te birden fazla duygu fark ederseniz, bunların hepsini virgül ve boşlukla (", ") ayırarak yanıtlayın. Yanıtınızı düzenlemek için birden fazla duyguyu belirtirken yukarıda belirtilen sırayı takip edin. Tweet: "{text}". Şimdi tweet sona eriyor. Tweet'in içerdiğini düşündüğünüz duygularla yanıt verin. Hiçbir duygu tespit edilmediğinde yanıt vermeyin.""",

    # Japanese
    """このデータセットには、ウルドゥー語のNastalíqツイートが含まれており、人々が世界と共有する活動、意見、出来事を説明するときの感情が明らかにされます。以下のツイートに、次の感情のいずれかが表現されているかどうかを判断してください: anger, disgust, fear, happiness, sadness, surprise, neutral。ツイートには0、1、または複数の感情が含まれている場合があります。ツイートに複数の感情が含まれている場合は、感情をカンマとスペース (", ") で区切ってすべての感情に応答してください。複数の感情を言及する場合は、上記の順序に従って応答を整理してください。ツイート: "{text}"。これでツイートは終了です。ツイートに含まれていると思われる感情で応答してください。感情が検出されない場合は、何も回答しないでください。""",

    # Korean
    """이 데이터 세트에는 사람들이 활동, 의견 및 이벤트를 설명할 때 느끼는 감정을 설명하는 Urdu Nastalíq 트윗이 포함되어 있습니다. 아래 트윗이 다음 감정 중 하나를 표현했는지 확인해 주세요: anger, disgust, fear, happiness, sadness, surprise, neutral. 트윗에는 0, 1 또는 여러 감정이 포함될 수 있습니다. 트윗에서 여러 감정을 발견한 경우 쉼표와 공백 (", ")으로 감정을 구분하여 모두 응답해 주세요. 여러 감정을 언급할 때 위에서 언급한 순서를 따르세요. 트윗: "{text}". 이제 트윗이 끝납니다. 트윗이 포함하고 있다고 생각하는 감정으로 응답해 주세요. 감정이 감지되지 않은 경우 응답하지 마세요.""",

    # Vietnamese
    """Tập dữ liệu bao gồm các tweet bằng tiếng Urdu Nastalíq mô tả cảm xúc của mọi người khi họ mô tả các hoạt động, ý kiến và sự kiện của họ với thế giới. Vui lòng xác định xem tweet dưới đây có thể hiện bất kỳ cảm xúc nào sau đây không: anger, disgust, fear, happiness, sadness, surprise, neutral. Lưu ý rằng tweet có thể chứa 0, 1 hoặc nhiều cảm xúc. Nếu bạn nhận thấy nhiều cảm xúc trong một tweet, vui lòng phản hồi với tất cả chúng, tách các cảm xúc bằng dấu phẩy và dấu cách (", "). Để sắp xếp phản hồi của bạn, vui lòng tuân theo thứ tự đã đề cập ở trên khi đề cập đến nhiều cảm xúc. Tweet: "{text}". Bây giờ tweet kết thúc. Vui lòng phản hồi bằng cảm xúc mà bạn nghĩ rằng tweet chứa đựng. Phản hồi bằng cách không làm gì khi không phát hiện được cảm xúc nào.""",

    # Thai
    """ชุดข้อมูลนี้ประกอบด้วยทวีต Nastalíq ภาษาอูรดู ซึ่งอธิบายถึงอารมณ์ของผู้คนเมื่อพวกเขาอธิบายกิจกรรม ความคิดเห็น และเหตุการณ์ของพวกเขากับโลก โปรดระบุว่าทวีตด้านล่างนี้แสดงอารมณ์ใดบ้าง: anger, disgust, fear, happiness, sadness, surprise, neutral โปรดทราบว่าทวีตอาจมีอารมณ์ 0, 1 หรือหลายอารมณ์ หากคุณสังเกตเห็นหลายอารมณ์ในทวีต โปรดตอบกลับด้วยอารมณ์ทั้งหมด โดยแยกอารมณ์ด้วยเครื่องหมายจุลภาคและช่องว่าง (", ") เพื่อจัดระเบียบคำตอบของคุณ โปรดปฏิบัติตามลำดับที่กล่าวถึงข้างต้นเมื่อกล่าวถึงอารมณ์หลายอารมณ์ ทวีต: "{text}" ทวีตสิ้นสุดลงที่นี่ โปรดตอบกลับด้วยอารมณ์ที่คุณคิดว่าทวีตประกอบด้วย ตอบกลับด้วยการไม่ทำอะไรเลยเมื่อไม่มีอารมณ์ที่ตรวจพบ""",

    # Indonesian
    """Dataset ini terdiri dari tweet Urdu Nastalíq yang menjelaskan emosi orang-orang saat mereka menggambarkan aktivitas, pendapat, dan peristiwa mereka dengan dunia. Harap tentukan apakah tweet di bawah ini mengekspresikan salah satu emosi berikut: anger, disgust, fear, happiness, sadness, surprise, neutral. Perhatikan bahwa tweet mungkin mengandung 0, 1, atau banyak emosi. Jika Anda melihat banyak emosi dalam sebuah tweet, harap tanggapi semuanya, dengan memisahkan emosi dengan tanda koma dan spasi (", "). Untuk mengatur tanggapan Anda, harap ikuti urutan yang disebutkan di atas saat menyebutkan beberapa emosi. Tweet: "{text}". Sekarang tweet berakhir. Harap tanggapi dengan emosi yang menurut Anda ada dalam tweet. Tanggapi dengan tidak ada apa-apa ketika tidak ada emosi yang terdeteksi.""",

    # Malay
    """Set data ini terdiri daripada tweet Nastalíq Urdu yang menjelaskan emosi orang apabila mereka menggambarkan aktiviti, pendapat dan peristiwa mereka dengan dunia. Sila tentukan sama ada tweet di bawah menyatakan mana-mana emosi berikut: anger, disgust, fear, happiness, sadness, surprise, neutral. Perhatikan bahawa tweet mungkin mengandungi 0, 1, atau pelbagai emosi. Jika anda perasan berbilang emosi dalam tweet, sila balas dengan semua emosi tersebut, dengan memisahkan emosi dengan koma dan ruang (", "). Untuk menyusun jawapan anda, sila ikut susunan yang disebutkan di atas apabila menyebut berbilang emosi. Tweet: "{text}". Sekarang tweet tamat. Sila balas dengan emosi yang anda fikir tweet itu mengandungi. Balas dengan tiada apa-apa apabila tiada emosi yang dikesan.""",

    # Lao
    """ຊຸດຂໍ້ມູນປະກອບດ້ວຍທະວິດ Nastalíq ພາສາອູຣະດູ ທີ່ຂຽນເລົົ່າອາລົົມຂອງຄົົນທີ່ພວກເຂົົາໄດ້ພົົບເຫັັນກິດກໍາ, ຄວາມຄິດເຫັັນ, ແລະເຫດການທີ່ໄດ້ເກີີດຂື້ຶນ. ກະລຸນາກວດກາດຶຶງຄວາມສະໜອງຕໍ່ທະວິິດຫຼັັງດຽວທີ່ຄວາມຄິດວ່າສຸກສາຂອງເມຍ,ຄວາມສຸກແລະຂອບຮອງ. ກະລຸນາຕອບກັບທຸກໆໃຫຍ່ບຸກມັນຫລືືລື້ຶນໃຫຍ່ນີ້ຶອບວາໄດ້ຫຼາຍແຕກຕ່າງນະນະຂວາງເກີບທະຫລອຍນັກລາດຄະນະນີ້. Tweet: "{text}"ນີ້ແມ່ນສີ້ິນສຸດຂອງທະວິິດນີ້. ກະລຸນາຕອບກັບອາລົົມທີ່ເຊື່ື່ວ່າທະວິິດນີ້ມີຢູ່. ຕອບບໍ່ມີຫຍັງເມື່ອທະວິິດບໍ່ຖືົກກວດພົົບວ່າບໍ່ມີອາລົົມນັກການ.""",

    # Burmese
    """ဤဒေတာအစုစုတွင် Nastalíq Urdu တွစ်တာများပါရှိပြီး ၎င်းတို့သည် ကမ္ဘာနှင့် ၎င်းတို့၏ လှုပ်ရှားမှုများ၊ ထင်မြင်ချက်များနှင့် ဖြစ်ရပ်များကို ဖော်ပြသောအခါ လူများ၏ခံစားချက်များကို ဖော်ထုတ်သည်။ ဤတွစ်တာသည် အောက်ဖော်ပြပါခံစားချက်များမှတစ်ခုခုကို ဖော်ပြကြောင်း သင်တွေ့ရှိပါက တုံ့ပြန်ချက်ကို ပြန်သည့်နေရာတွင် မိတ်ဆက်ပေးရန်၊ တစ်ခုခုမှ် တုံ့ပြန်ချက်ကို မိမိတို့၏ တုံ့ပြန်ချက်တွင် ခွဲခြားခြင်းအားဖြင့် ကွဲပြားသော ချက်ပြုချက် (", ") တစ်ခုစီ ဖြည့်စွက်ပေးပါ။ နှစ်ဖက်စပ်သည့် အကြောင်းအရာများကို ပြန်သည့်နေရာတွင် သင်ပြောပြသည်မှာ စည်းကမ်းချက်များကို လိုက်နာ၍ စွမ်းဆောင်နိုင်မည်ဖြစ်သည်။ Tweet: "{text}" Tweet သည်ယခုတွင်ပိတ်သိမ်းပါပြီ။ Tweet သည်ထင်ရှားပါစေ။ Tweet သည်မည်သည့်ခံစားချက်မရှိပါက ပြန်မည် မရှိပါ။""",

    # Cebuano
    """Ang dataset naglakip sa mga tweet sa Urdu Nastalíq nga nagpatin-aw sa mga emosyon sa mga tawo samtang ginasaysay nila ang ilang mga kalihokan, mga opinyon, ug mga hitabo sa kalibutan. Palihug pagtino kung ang ubos nga tweet nagpahayag sa bisan unsa sa mosunod nga mga emosyon: anger, disgust, fear, happiness, sadness, surprise, neutral. Hibal-i nga ang tweet mahimong maglakip og 0, 1, o daghang emosyon. Kung makamatikod ka og daghang emosyon sa usa ka tweet, palihug pagtubag sa tanan kanila, pinaagi sa pagbulag sa mga emosyon sa usa ka comma ug usa ka space (", "). Aron maplastar ang imong tubag, palihug sundon ang gipakita nga order sa paghisgot sa daghang mga emosyon. Tweet: "{text}". Karon ang tweet natapos na. Palihug tubaga ang mga emosyon nga imong gihunahuna nga anaa sa tweet. Tubaga sa wala’y bisan unsa kung wala’y emosyon nga namatikdan.""",

    # Khmer
    """គោលដៅកំណត់ដឺតានេះមានជីវិតរៀនរាប់ទីមានរសជាតិសម្ងាត់នៅក្នុងរឿងទូរសព្ទដែលនិយាយអំពីអារម្មណ៍របស់អ្នកគ្រប់មួយគ្នាដែលពន្យល់អំពីសកម្មភាព របស់គេ ហើយនាំមកនូវអារម្មណ៍ទាំងមូលដើម្បីចែករំលែកទៅកាន់ពិភពលោក។ សូមកំណត់ថាតើអត្ថបទខាងក្រោមនេះបានបង្ហាញអារម្មណ៍ខុសគ្នានៅក្នុងចំណុចណាមួយដែរឬទេ? anger, disgust, fear, happiness, sadness, surprise, neutral។ សូមពិនិត្យផ្ទៀងផ្ទាត់ដំណឹងជាក់ស្តែងពីតួអង្គខាងក្រោមនេះថាតើការរំលាយបានជាក់ស្តែងនៅក្នុងទំព័រមួយចំនួនទេ? ហើយសូមបញ្ជាក់ផ្នែកខុសគ្នានៃការផ្ទៀងផ្ទាត់ទាំងនេះជាមួយនឹងឯកសារដែលគេបានបង្ហាញសង្ខេបដោយកុំព្យូទ័រ។ Tweet: "{text}"។ លទ្ធផលរបស់ Tweet នេះបានបញ្ចប់ដំណើររបស់ខ្លួន។ សូមពិនិត្យតាមអារម្មណ៍របស់អ្នកដែលគិតថា Tweet មានវត្ថុបង្ហាញដែលអាចទាក់ទងបានដោយកុំព្យូទ័រ។ មិនបញ្ជាក់អារម្មណ៍ក្នុងស្ថានភាពដែលគ្មានអារម្មណ៍ត្រូវបានរកឃើញទេ។""",

    # Tagalog
    """Ang dataset ay binubuo ng mga tweet ng Nastalíq Urdu na naglalarawan sa mga emosyon ng mga tao habang inilalarawan nila ang kanilang mga aktibidad, opinyon, at mga pangyayari sa mundo. Pakitukoy kung ang tweet sa ibaba ay nagpapahayag ng alinman sa mga sumusunod na emosyon: anger, disgust, fear, happiness, sadness, surprise, neutral. Pansinin na maaaring naglalaman ang tweet ng 0, 1, o maramihang mga emosyon. Kung mapapansin mo ang maramihang emosyon sa isang tweet, mangyaring tumugon sa lahat ng mga ito, na naghihiwalay sa mga emosyon sa pamamagitan ng kuwit at espasyo (", "). Upang ayusin ang iyong tugon, mangyaring sundin ang nabanggit na pagkakasunod-sunod kapag binabanggit ang maraming emosyon. Tweet: "{text}". Ngayon, natatapos na ang tweet. Pakisagot ang mga emosyon na sa tingin mo ay nilalaman ng tweet. Tumugon sa wala kapag walang nakitang emosyon.""",

    # Hindi
    """इस डेटासेट में उर्दू नस्तालीक ट्वीट्स शामिल हैं जो लोगों की भावनाओं को उजागर करते हैं जब वे अपनी गतिविधियों, विचारों और घटनाओं का वर्णन करते हैं। कृपया यह निर्धारित करें कि नीचे दिए गए ट्वीट में निम्नलिखित में से कोई भी भावना व्यक्त की गई है या नहीं: anger, disgust, fear, happiness, sadness, surprise, neutral। ध्यान दें कि ट्वीट में 0, 1 या एक से अधिक भावनाएं हो सकती हैं। यदि आप एक ट्वीट में एक से अधिक भावनाओं को नोटिस करते हैं, तो कृपया सभी का जवाब दें, भावनाओं को अल्पविराम और एक स्पेस (", ") से अलग करके। अपनी प्रतिक्रिया को व्यवस्थित करने के लिए, कृपया कई भावनाओं का उल्लेख करते समय उपरोक्त क्रम का पालन करें। ट्वीट: "{text}"। अब ट्वीट समाप्त होता है। कृपया उस भावना का उत्तर दें जिसे आप सोचते हैं कि ट्वीट में शामिल है। जब कोई भावना का पता नहीं चलता है तो कुछ भी प्रतिक्रिया न दें।""",

    # Bengali
    """ডেটাসেটটি উর্দু নস্তালিক টুইট নিয়ে গঠিত যা মানুষ যখন তাদের কার্যকলাপ, মতামত এবং ঘটনা বিশ্বে বর্ণনা করে তখন তাদের আবেগ প্রকাশ করে। অনুগ্রহ করে নির্ধারণ করুন যে নীচের টুইটটি নিম্নলিখিত কোনও আবেগ প্রকাশ করেছে কিনা: anger, disgust, fear, happiness, sadness, surprise, neutral। লক্ষ্য করুন যে টুইটটিতে 0, 1 বা একাধিক আবেগ থাকতে পারে। আপনি যদি একটি টুইটে একাধিক আবেগ লক্ষ্য করেন, অনুগ্রহ করে সমস্তটির সাথে প্রতিক্রিয়া জানান, আবেগগুলি একটি কমা এবং একটি স্পেস (", ") দিয়ে পৃথক করুন। আপনার প্রতিক্রিয়া সংগঠিত করতে, একাধিক আবেগ উল্লেখ করার সময় দয়া করে উপরে উল্লেখিত ক্রম অনুসরণ করুন। টুইট: "{text}"। এখন টুইটটি শেষ। অনুগ্রহ করে সেই আবেগের সাথে প্রতিক্রিয়া জানান যা আপনি মনে করেন টুইটে রয়েছে। যখন কোনও আবেগ শনাক্ত করা যায় না তখন কিছু প্রতিক্রিয়া দেবেন না।""",

    # Urdu
    """اس ڈیٹاسیٹ میں اردو نستعلیق ٹویٹس شامل ہیں جو لوگوں کے جذبات کو اجاگر کرتے ہیں جب وہ اپنی سرگرمیوں، خیالات اور واقعات کا دنیا کے ساتھ بیان کرتے ہیں۔ براہ کرم یہ طے کریں کہ آیا نیچے دیے گئے ٹویٹ میں درج ذیل میں سے کوئی بھی جذبات ظاہر ہوئے ہیں: anger, disgust, fear, happiness, sadness, surprise, neutral۔ نوٹ کریں کہ ٹویٹ میں 0، 1 یا ایک سے زیادہ جذبات شامل ہو سکتے ہیں۔ اگر آپ کسی ٹویٹ میں متعدد جذبات دیکھتے ہیں، تو براہ کرم سب کے ساتھ جواب دیں، جذبات کو کوما اور اسپیس (", ") کے ساتھ الگ کریں۔ اپنی جواب کو منظم کرنے کے لیے، براہ کرم متعدد جذبات کا ذکر کرتے وقت مذکورہ ترتیب پر عمل کریں۔ ٹویٹ: "{text}"۔ اب ٹویٹ ختم ہو گیا ہے۔ براہ کرم جذبات کے ساتھ جواب دیں جس کے بارے میں آپ کو لگتا ہے کہ ٹویٹ میں شامل ہے۔ جب کوئی جذبات کا پتہ نہیں چلتا ہے تو کچھ بھی جواب نہ دیں۔"""
]


# Function to create instruction based on the selected template
def create_instruction(row):
    instruction_template = random.choice(emotion_templates)
    instruction = instruction_template.format(text=row["Sentences"])
    return instruction

# Function to get the output based on emotions
def get_output(row):
    response = []
    if row["anger"] == 1:
        response.append("anger")
    if row["disgust"] == 1:
        response.append("disgust")
    if row["fear"] == 1:
        response.append("fear")
    if row["happiness"] == 1:
        response.append("happiness")
    if row["sadness"] == 1:
        response.append("sadness")
    if row["surprise"] == 1:
        response.append("surprise")
    if row["neutral"] == 1:
        response.append("neutral")
    return ", ".join(response)

# get a blank dataframe
inst_data = pd.DataFrame(columns=['instruction', 'output'])

taskAtraindata = pd.read_csv(taskA_train_file)
taskAtestdata = pd.read_csv(taskA_test_file)
taskAdata = pd.concat([taskAtraindata, taskAtestdata])

# Generate instructions and outputs
for i, row in taskAdata.iterrows():
    output = get_output(row)
    instruction = create_instruction(row)
    inst_data = pd.concat([inst_data, pd.DataFrame({'instruction': [instruction], 'output': [output]})], ignore_index=True)

inst_data = inst_data.sample(n=6000, random_state=42)
# Save the DataFrame to a parquet file
inst_data.to_parquet("../../data/UEC/emothreat_taskA_multilingual.parquet", index=False)
