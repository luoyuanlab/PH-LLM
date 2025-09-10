# -*- coding: utf-8 -*-
# FC (Fact-Checking Public Health Claims)
# - Paper: https://doi.org/10.48550/arXiv.2010.09926
# - Data: https://github.com/neemakot/Health-Fact-Checking/tree/master/data
import pandas as pd
import random

# List of prompts in multiple languages
prompts = [
    # Chinese
    "这是一个数据标记任务，本次标注任务为fact checking，共有true, false, mixture, unproven四个标签，我们想要衡量给出的信息的可靠性。信息内容为: {text}。提问：这条信息的可靠性符合true, false, mixture, unproven四个标签中的哪一个？",
    "请执行以下fact checking任务：根据提供的信息内容，判断其可靠性，并选择true, false, mixture, unproven四个标签中的一个。信息内容为: {text}。该信息的可靠性属于哪种标签？",

    # English
    "This is a data labeling task focused on fact checking. There are four labels: true, false, mixture, and unproven. We want to assess the reliability of the information provided. The information is: {text}. Question: Which of the four labels (true, false, mixture, unproven) best describes the reliability of this information?",
    "Please perform the following fact checking task: Based on the information provided, determine its reliability and choose one of the four labels: true, false, mixture, unproven. The information is: {text}. Which label best describes the reliability of this information?",

    # German
    "Dies ist eine Datenkennzeichnungsaufgabe mit dem Schwerpunkt auf Fact Checking. Es gibt vier Kategorien: true, false, mixture, und unproven. Wir möchten die Zuverlässigkeit der bereitgestellten Informationen bewerten. Die Informationen lauten: {text}. Frage: Welches der vier Labels (true, false, mixture, unproven) beschreibt am besten die Zuverlässigkeit dieser Informationen?",
    "Bitte führen Sie die folgende Fact Checking-Aufgabe durch: Basierend auf den bereitgestellten Informationen bewerten Sie deren Zuverlässigkeit und wählen eine der vier Kategorien: true, false, mixture, unproven. Die Informationen lauten: {text}. Welches Label beschreibt am besten die Zuverlässigkeit dieser Informationen?",

    # French
    "Ceci est une tâche de marquage de données axée sur la vérification des faits (fact checking). Il y a quatre étiquettes : true, false, mixture, et unproven. Nous voulons évaluer la fiabilité des informations fournies. Les informations sont : {text}. Question : Laquelle des quatre étiquettes (true, false, mixture, unproven) décrit le mieux la fiabilité de cette information ?",
    "Veuillez effectuer la tâche de vérification des faits suivante : En fonction des informations fournies, déterminez leur fiabilité et choisissez l'une des quatre étiquettes : true, false, mixture, unproven. Les informations sont : {text}. Quelle étiquette décrit le mieux la fiabilité de cette information ?",

    # Spanish
    "Esta es una tarea de etiquetado de datos centrada en la verificación de hechos (fact checking). Hay cuatro etiquetas: true, false, mixture, y unproven. Queremos evaluar la fiabilidad de la información proporcionada. La información es: {text}. Pregunta: ¿Cuál de las cuatro etiquetas (true, false, mixture, unproven) describe mejor la fiabilidad de esta información?",
    "Por favor, realice la siguiente tarea de verificación de hechos: Con base en la información proporcionada, determine su fiabilidad y elija una de las cuatro etiquetas: true, false, mixture, unproven. La información es: {text}. ¿Qué etiqueta describe mejor la fiabilidad de esta información?",

    # Portuguese
    "Esta é uma tarefa de rotulagem de dados focada na verificação de fatos (fact checking). Existem quatro rótulos: true, false, mixture, e unproven. Queremos avaliar a confiabilidade das informações fornecidas. A informação é: {text}. Pergunta: Qual dos quatro rótulos (true, false, mixture, unproven) melhor descreve a confiabilidade desta informação?",
    "Por favor, realize a seguinte tarefa de verificação de fatos: Com base nas informações fornecidas, determine sua confiabilidade e escolha um dos quatro rótulos: true, false, mixture, unproven. A informação é: {text}. Qual rótulo descreve melhor a confiabilidade desta informação?",

    # Italian
    "Questo è un compito di etichettatura dei dati focalizzato sul fact checking. Ci sono quattro etichette: true, false, mixture, e unproven. Vogliamo valutare l'affidabilità delle informazioni fornite. L'informazione è: {text}. Domanda: Quale delle quattro etichette (true, false, mixture, unproven) descrive meglio l'affidabilità di queste informazioni?",
    "Si prega di eseguire il seguente compito di fact checking: Sulla base delle informazioni fornite, determinare la loro affidabilità e scegliere una delle quattro etichette: true, false, mixture, unproven. L'informazione è: {text}. Quale etichetta descrive meglio l'affidabilità di queste informazioni?",

    # Dutch
    "Dit is een data-labeling taak gericht op fact checking. Er zijn vier labels: true, false, mixture, en unproven. We willen de betrouwbaarheid van de verstrekte informatie beoordelen. De informatie is: {text}. Vraag: Welk van de vier labels (true, false, mixture, unproven) beschrijft het beste de betrouwbaarheid van deze informatie?",
    "Voer alstublieft de volgende fact checking-taak uit: Op basis van de verstrekte informatie bepaalt u de betrouwbaarheid en kiest u een van de vier labels: true, false, mixture, unproven. De informatie is: {text}. Welk label beschrijft het beste de betrouwbaarheid van deze informatie?",

    # Russian
    "Это задача по маркировке данных, сосредоточенная на проверке фактов (fact checking). Существует четыре метки: true, false, mixture, и unproven. Мы хотим оценить надежность предоставленной информации. Информация: {text}. Вопрос: Какая из четырех меток (true, false, mixture, unproven) лучше всего описывает надежность этой информации?",
    "Пожалуйста, выполните следующую задачу по проверке фактов: На основе предоставленной информации определите ее надежность и выберите одну из четырех меток: true, false, mixture, unproven. Информация: {text}. Какая метка лучше всего описывает надежность этой информации?",

    # Czech
    "Toto je úkol označování dat zaměřený na fact checking. Existují čtyři štítky: true, false, mixture, a unproven. Chceme posoudit spolehlivost poskytnutých informací. Informace: {text}. Otázka: Který ze čtyř štítků (true, false, mixture, unproven) nejlépe popisuje spolehlivost těchto informací?",
    "Prosím, proveďte následující úkol fact checking: Na základě poskytnutých informací určete jejich spolehlivost a vyberte jeden ze čtyř štítků: true, false, mixture, unproven. Informace: {text}. Který štítek nejlépe popisuje spolehlivost těchto informací?",

    # Polish
    "To jest zadanie oznaczania danych skoncentrowane na fact checking. Istnieją cztery etykiety: true, false, mixture, i unproven. Chcemy ocenić wiarygodność dostarczonych informacji. Informacje: {text}. Pytanie: Która z czterech etykiet (true, false, mixture, unproven) najlepiej opisuje wiarygodność tych informacji?",
    "Proszę wykonać następujące zadanie fact checking: Na podstawie dostarczonych informacji oceń ich wiarygodność i wybierz jedną z czterech etykiet: true, false, mixture, unproven. Informacje: {text}. Która etykieta najlepiej opisuje wiarygodność tych informacji?",

    # Arabic
    "هذه مهمة وسم بيانات تركز على التحقق من الحقائق (fact checking). هناك أربع تسميات: true, false, mixture, و unproven. نريد تقييم موثوقية المعلومات المقدمة. المعلومات هي: {text}. السؤال: أي من التصنيفات الأربعة (true, false, mixture, unproven) تصف موثوقية هذه المعلومات بأفضل شكل؟",
    "يرجى تنفيذ مهمة التحقق من الحقائق التالية: بناءً على المعلومات المقدمة، حدد موثوقيتها واختر واحدة من أربع تسميات: true, false, mixture, unproven. المعلومات هي: {text}. ما التصنيف الذي يصف موثوقية هذه المعلومات بشكل أفضل؟",

    # Persian
    "این یک کار نشانه گذاری داده است که بر fact checking تمرکز دارد. چهار برچسب وجود دارد: true, false, mixture, و unproven. ما می‌خواهیم قابلیت اطمینان اطلاعات ارائه شده را ارزیابی کنیم. اطلاعات: {text}. سوال: کدام یک از چهار برچسب (true, false, mixture, unproven) بهترین توصیف کننده قابلیت اطمینان این اطلاعات است؟",
    "لطفاً وظیفه‌ی fact checking زیر را انجام دهید: بر اساس اطلاعات ارائه شده، قابلیت اطمینان آن‌ها را تعیین کنید و یکی از چهار برچسب را انتخاب کنید: true, false, mixture, unproven. اطلاعات: {text}. کدام برچسب بهترین توصیف کننده قابلیت اطمینان این اطلاعات است؟",

    # Hebrew
    "זוהי משימת תיוג נתונים המתמקדת ב-fact checking. יש ארבע תוויות: true, false, mixture, ו-unproven. אנו רוצים להעריך את אמינות המידע שסופק. המידע הוא: {text}. שאלה: איזו מבין ארבע התוויות (true, false, mixture, unproven) מתארת בצורה הטובה ביותר את אמינות המידע הזה?",
    "בצע בבקשה את משימת fact checking הבאה: בהתבסס על המידע שסופק, קבע את אמינותו ובחר אחת מארבע התוויות: true, false, mixture, unproven. המידע הוא: {text}. איזו תווית מתארת בצורה הטובה ביותר את אמינות המידע הזה?",

    # Turkish
    "Bu, fact checking odaklı bir veri etiketleme görevidir. Dört etiket vardır: true, false, mixture, ve unproven. Sağlanan bilginin güvenilirliğini değerlendirmek istiyoruz. Bilgi: {text}. Soru: Bu bilgilerin güvenilirliğini en iyi hangi dört etiketten biri (true, false, mixture, unproven) tanımlar?",
    "Lütfen aşağıdaki fact checking görevini gerçekleştirin: Sağlanan bilgilere dayanarak güvenilirliğini belirleyin ve dört etiketten birini seçin: true, false, mixture, unproven. Bilgi: {text}. Hangi etiket bu bilgilerin güvenilirliğini en iyi tanımlar?",

    # Japanese
    "これは、fact checking に焦点を当てたデータラベリングタスクです。true, false, mixture, unproven の4つのラベルがあります。提供された情報の信頼性を評価したいと思います。情報は次のとおりです: {text}。質問: この情報の信頼性を最もよく表している4つのラベルのどれ (true, false, mixture, unproven) ですか?",
    "次の fact checking タスクを実行してください: 提供された情報に基づいて、その信頼性を判断し、4つのラベルのうちの1つを選択してください: true, false, mixture, unproven。情報: {text}。この情報の信頼性を最もよく表すラベルはどれですか?",

    # Korean
    "이것은 fact checking에 중점을 둔 데이터 라벨링 작업입니다. true, false, mixture, unproven의 네 가지 라벨이 있습니다. 제공된 정보의 신뢰성을 평가하고자 합니다. 정보는 다음과 같습니다: {text}. 질문: 이 정보의 신뢰성을 가장 잘 설명하는 네 가지 라벨 중 어느 것 (true, false, mixture, unproven) 입니까?",
    "다음 fact checking 작업을 수행하십시오: 제공된 정보를 기반으로 신뢰성을 판단하고 네 가지 라벨 중 하나를 선택하십시오: true, false, mixture, unproven. 정보: {text}. 이 정보의 신뢰성을 가장 잘 설명하는 라벨은 무엇입니까?",

    # Vietnamese
    "Đây là một nhiệm vụ gán nhãn dữ liệu tập trung vào việc fact checking. Có bốn nhãn: true, false, mixture, và unproven. Chúng tôi muốn đánh giá độ tin cậy của thông tin được cung cấp. Thông tin là: {text}. Câu hỏi: Nhãn nào trong bốn nhãn (true, false, mixture, unproven) mô tả tốt nhất độ tin cậy của thông tin này?",
    "Vui lòng thực hiện nhiệm vụ fact checking sau: Dựa trên thông tin được cung cấp, hãy xác định độ tin cậy của nó và chọn một trong bốn nhãn: true, false, mixture, unproven. Thông tin là: {text}. Nhãn nào mô tả tốt nhất độ tin cậy của thông tin này?",

    # Thai
    "นี่คืองานติดป้ายข้อมูลที่เน้นการ fact checking มีป้ายกำกับสี่แบบ: true, false, mixture, และ unproven เราต้องการประเมินความน่าเชื่อถือของข้อมูลที่ให้มา ข้อมูลคือ: {text} คำถาม: ป้ายกำกับใดในสี่ป้ายนี้ (true, false, mixture, unproven) ที่อธิบายความน่าเชื่อถือของข้อมูลนี้ได้ดีที่สุด?",
    "โปรดทำงาน fact checking ต่อไปนี้: ตามข้อมูลที่ให้มา ให้กำหนดความน่าเชื่อถือของมัน และเลือกหนึ่งในสี่ป้ายกำกับ: true, false, mixture, unproven ข้อมูลคือ: {text} ป้ายกำกับใดอธิบายความน่าเชื่อถือของข้อมูลนี้ได้ดีที่สุด?",

    # Indonesian
    "Ini adalah tugas pelabelan data yang berfokus pada fact checking. Ada empat label: true, false, mixture, dan unproven. Kami ingin menilai keandalan informasi yang diberikan. Informasi: {text}. Pertanyaan: Label mana dari empat label (true, false, mixture, unproven) yang paling menggambarkan keandalan informasi ini?",
    "Silakan lakukan tugas fact checking berikut: Berdasarkan informasi yang diberikan, tentukan keandalannya dan pilih salah satu dari empat label: true, false, mixture, unproven. Informasi: {text}. Label mana yang paling menggambarkan keandalan informasi ini?",

    # Malay
    "Ini adalah tugas pelabelan data yang memfokuskan pada fact checking. Terdapat empat label: true, false, mixture, dan unproven. Kami ingin menilai kebolehpercayaan maklumat yang diberikan. Maklumatnya ialah: {text}. Soalan: Label mana antara empat label (true, false, mixture, unproven) yang paling menggambarkan kebolehpercayaan maklumat ini?",
    "Sila lakukan tugas fact checking berikut: Berdasarkan maklumat yang diberikan, tentukan kebolehpercayaannya dan pilih salah satu daripada empat label: true, false, mixture, unproven. Maklumat: {text}. Label mana yang paling menggambarkan kebolehpercayaan maklumat ini?",

    # Lao
    "ນີ້ແມ່ນພາລະກິດການຕິດປ້າຍຂໍ້ມູນທີ່ມີເນື້ອໃນກ່ຽວກັບ fact checking. ມີສີ່ປ້າຍ: true, false, mixture, ແລະ unproven. ພວກເຮົາຕ້ອງການປະເມີນຄວາມເຊື່ອຖືໄດ້ຂອງຂໍ້ມູນທີ່ໄດ້ຮັບ. ຂໍ້ມູນຄື: {text}. ຄຳຖາມ: ປ້າຍໃດໃນບໍລິບົດສີ່ນີ້ (true, false, mixture, unproven) ເປັນການອະທິບາຍຄວາມເຊື່ອຖືໄດ້ຂອງຂໍ້ມູນນີ້ທີ່ດີທີ່ສຸດ?",
    "ກະລຸນາເຮັດພາລະກິດ fact checking ຕໍ່ໄປນີ້: ອີງຕາມຂໍ້ມູນທີ່ໃຫ້ມາ ເພື່ອກຳນົດຄວາມເຊື່ອຖືໄດ້ຂອງມັນ ແລະເລືອກປ້າຍອັນໃດອັນໜຶ່ງໃນປ້າຍສີ່: true, false, mixture, unproven. ຂໍ້ມູນ: {text}. ປ້າຍໃດແມ່ນການອະທິບາຍຄວາມເຊື່ອຖືໄດ້ຂອງຂໍ້ມູນນີ້ທີ່ດີທີ່ສຸດ?",

    # Burmese
    "ဤသည်မှာ fact checking အတွက် အချက်အလက်အမှတ်အသားပေးမှု လုပ်ငန်းဖြစ်သည်။ စံသတ်မှတ်ချက်လေးခုရှိသည်: true, false, mixture, နှင့် unproven. ပေးထားသော အချက်အလက်၏ ယုံကြည်ရနိုင်မှုကို သုံးသပ်လိုပါသည်။ အချက်အလက်မှာ: {text}. မေးခွန်း: ယုံကြည်ရနိုင်မှုကို သက်ဆိုင်သည့် စံသတ်မှတ်ချက်လေးခုထဲမှ ဘယ်ဟာက အချက်အလက်ကို အကောင်းဆုံး ဖော်ပြနိုင်မလဲ?",
    "fact checking အလုပ်ဆောင်ရန် အောက်ပါလုပ်ငန်းကို ဆောင်ရွက်ပါ: ပေးထားသော အချက်အလက်အပေါ် မူတည်၍ ယုံကြည်ရနိုင်မှုကို သတ်မှတ်ပါ၊ စံသတ်မှတ်ချက်လေးခုထဲမှ true, false, mixture, unproven အနက်မှ တစ်ခုကို ရွေးပါ။ အချက်အလက်မှာ: {text}. ဘယ်စံသတ်မှတ်ချက်က အချက်အလက်ကို အကောင်းဆုံး ဖော်ပြနိုင်မလဲ?",

    # Cebuano
    "Kini usa ka tahas sa pagmarka sa datos nga nakasentro sa fact checking. Adunay upat ka label: true, false, mixture, ug unproven. Gusto namon nga ma-assess ang kasaligan sa gi-provide nga impormasyon. Ang impormasyon mao kini: {text}. Pangutana: Asa man sa upat ka label (true, false, mixture, unproven) ang labing nagrepresentar sa kasaligan niini nga impormasyon?",
    "Palihug buhata ang sunod nga tahas sa fact checking: Base sa gi-provide nga impormasyon, tukma-a ang kasaligan niini ug pili usa sa upat ka label: true, false, mixture, unproven. Ang impormasyon mao kini: {text}. Asa nga label ang labing nagrepresentar sa kasaligan niini nga impormasyon?",

    # Khmer
    "នេះជាកិច្ចការការដាក់ស្លាកទិន្នន័យដែលផ្តោតលើការត្រួតពិនិត្យការពិត (fact checking) ។ មានស្លាកចំនួនបួន៖ true, false, mixture, និង unproven ។ យើងចង់វាយតម្លៃពីភាពជឿជាក់នៃព័ត៌មានដែលបានផ្តល់អោយ ។ ព័ត៌មានគឺ៖ {text} ។ សំណួរ៖ ស្លាកណាមួយក្នុងចំណោមស្លាកបួននេះ (true, false, mixture, unproven) ដែលពិពណ៌នាបានល្អបំផុតអំពីភាពជឿជាក់នៃព័ត៌មាននេះ?",
    "សូមអនុវត្តកិច្ចការការត្រួតពិនិត្យការពិតដូចខាងក្រោម៖ ពិនិត្យព័ត៌មានដែលបានផ្តល់អោយ បន្ទាប់មកកំណត់ពីភាពជឿជាក់នៃព័ត៌មានហើយជ្រើសយកស្លាកមួយក្នុងចំណោមស្លាកបួននេះ៖ true, false, mixture, unproven ។ ព័ត៌មានគឺ៖ {text} ។ ស្លាកណាមួយដែលពិពណ៌នាបានល្អបំផុតអំពីភាពជឿជាក់នៃព័ត៌មាននេះ?",

    # Tagalog
    "Ito ay isang gawain sa pagmarka ng datos na nakatuon sa fact checking. May apat na label: true, false, mixture, at unproven. Nais naming suriin ang pagiging maaasahan ng impormasyong ibinigay. Ang impormasyon ay: {text}. Tanong: Aling sa apat na label (true, false, mixture, unproven) ang pinakamainam na naglalarawan sa pagiging maaasahan ng impormasyong ito?",
    "Mangyaring isagawa ang sumusunod na gawain sa fact checking: Batay sa impormasyong ibinigay, tukuyin ang pagiging maaasahan nito at pumili ng isa sa apat na label: true, false, mixture, unproven. Ang impormasyon ay: {text}. Aling label ang pinakamainam na naglalarawan sa pagiging maaasahan ng impormasyong ito?",

    # Hindi
    "यह एक डेटा लेबलिंग कार्य है जो fact checking पर केंद्रित है। चार लेबल हैं: true, false, mixture, और unproven। हम दी गई जानकारी की विश्वसनीयता का आकलन करना चाहते हैं। जानकारी है: {text}। प्रश्न: इस जानकारी की विश्वसनीयता का सबसे अच्छा वर्णन करने वाला चार लेबल (true, false, mixture, unproven) में से कौन सा है?",
    "कृपया निम्नलिखित fact checking कार्य को पूरा करें: दी गई जानकारी के आधार पर इसकी विश्वसनीयता का निर्धारण करें और चार लेबल में से एक चुनें: true, false, mixture, unproven। जानकारी है: {text}। इस जानकारी की विश्वसनीयता का सबसे अच्छा वर्णन करने वाला लेबल कौन सा है?",

    # Bengali
    "এটি একটি ডেটা লেবেলিং টাস্ক যা fact checking এর উপর কেন্দ্রীভূত। চারটি লেবেল রয়েছে: true, false, mixture, এবং unproven। আমরা প্রদত্ত তথ্যের নির্ভরযোগ্যতা মূল্যায়ন করতে চাই। তথ্য হল: {text}। প্রশ্ন: এই তথ্যের নির্ভরযোগ্যতার সেরা বিবরণ কোন চারটি লেবেলের মধ্যে (true, false, mixture, unproven) রয়েছে?",
    "দয়া করে নিম্নলিখিত fact checking কাজটি সম্পাদন করুন: প্রদত্ত তথ্যের উপর ভিত্তি করে এর নির্ভরযোগ্যতা নির্ধারণ করুন এবং চারটি লেবেলের মধ্যে একটি বেছে নিন: true, false, mixture, unproven। তথ্য হল: {text}। কোন লেবেলটি এই তথ্যের নির্ভরযোগ্যতার সর্বোত্তম বর্ণনা দেয়?",

    # Urdu
    "یہ ایک ڈیٹا لیبلنگ ٹاسک ہے جو fact checking پر مرکوز ہے۔ چار لیبل ہیں: true, false, mixture، اور unproven۔ ہم دی گئی معلومات کی قابل اعتمادی کا اندازہ لگانا چاہتے ہیں۔ معلومات ہیں: {text}۔ سوال: اس معلومات کی قابل اعتمادی کا بہترین بیان کونسا لیبل (true, false, mixture, unproven) کرتا ہے؟",
    "براہ کرم درج ذیل fact checking کام کو انجام دیں: دی گئی معلومات کی بنیاد پر اس کی قابل اعتمادی کا تعین کریں اور چار لیبلز میں سے ایک کو منتخب کریں: true, false, mixture, unproven۔ معلومات ہیں: {text}۔ اس معلومات کی قابل اعتمادی کا بہترین بیان کون سا لیبل کرتا ہے؟"
]


# Function to construct the output
def construct_output(row):
    output = f"""{row['label']}. The explanation is "{row['explanation']}"."""
    return output

# Function to construct the instruction
def construct_instruction(row):
    instruction_template = random.choice(prompts)
    instruction = instruction_template.format(text=row['main_text'])
    return instruction

# Create DataFrames for each file and combine them
df_list = []
for i in ['dev.tsv', 'train.tsv', 'test.tsv']:
    df_r = pd.read_csv(f"../../data/FC/{i}", sep='\t')
    df = pd.DataFrame()
    df['output'] = df_r.apply(construct_output, axis=1)
    df['instruction'] = df_r.apply(construct_instruction, axis=1)
    df_list.append(df)
df_result = pd.concat(df_list, axis=0)

# Save the combined DataFrame to a parquet file
df_result.to_parquet("../../data/FC/fc.parquet", index=False)
# print shape, drop all rows with output less than 20 words, and print shape again
df_result = df_result[df_result['output'].str.split().str.len() > 20]
# collect a sample of n=8000 rows
df_sample = df_result.sample(n=8000, random_state=42)
# save parquet
df_sample.to_parquet("../../data/FC/fc.parquet", index=False)