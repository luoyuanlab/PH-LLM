# -*- coding: utf-8 -*-
# MLMA (Multilingual and Multi-Aspect Hate Speech Analysis)
# - Paper: https://doi.org/10.48550/arXiv.1908.11049
# - Data: https://huggingface.co/datasets/nedjmaou/MLMA_hate_speech

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from utils.data_utils import balance_classes
import pandas as pd
import random

# read the datasets
ar = pd.read_csv('../../data/MLMA/ar_dataset.csv')
en = pd.read_csv('../../data/MLMA/en_dataset.csv')
fr = pd.read_csv('../../data/MLMA/fr_dataset.csv')
df = pd.concat([ar, en, fr]).reset_index(drop=True)

# Define the instruction templates in different languages
instruction_templates = {
    "Chinese": """
    该任务是识别一条推文的敌意类型，我将随后向您展示，您需要遵循以下规则：
    (1) 如果推文听起来危险，它应该被标记为 abusive；
    (2) 根据传播仇恨的程度和作者使用的语气，它可以是 hateful、offensive 或 disrespectful；
    (3) 如果推文因无知而对某群体表达或传播恐惧，它应该被标记为 fearful；
    (4) 否则，它应该被标注为 normal。
    每条推文至少包含一种敌意类型，并且可以有多种敌意类型。

    这是推文：
    {}
    现在推文结束了。

    请用以下标签标注：fearful, abusive, offensive, hateful, disrespectful, normal。
    如果存在多种情感，请按上述顺序逐一提及，并用空格分隔。
    """,

    "English": """
    This task is to identify the hostility type of a tweet, which I will show you afterwards, and you need to stick to the following conventions:
    (1) If the tweet sounds dangerous, it should be labeled as abusive;
    (2) According to the degree to which it spreads hate and the tone its author uses, it can be hateful, offensive, or disrespectful;
    (3) If the tweet expresses or spreads fear out of ignorance against a group of individuals, it should be labeled as fearful;
    (4) Otherwise, it should be annotated as normal.
    Each tweet contains at least one hostility type and can have more than one hostility type.

    Here's the tweet:
    {}
    Now the tweet ends.

    Please label it with: fearful, abusive, offensive, hateful, disrespectful, normal.
    If multiple sentiments exist, mention them one by one in the aforementioned order and separate them with spaces.
    """,

    "German": """
    Diese Aufgabe besteht darin, die Feindseligkeitsart eines Tweets zu identifizieren, den ich Ihnen anschließend zeigen werde, und Sie müssen sich an die folgenden Konventionen halten:
    (1) Wenn der Tweet gefährlich klingt, sollte er als abusive gekennzeichnet werden;
    (2) Je nach Grad der Verbreitung von Hass und dem Ton, den der Autor verwendet, kann er als hateful, offensive oder disrespectful gekennzeichnet werden;
    (3) Wenn der Tweet aus Unwissenheit gegenüber einer Gruppe von Personen Angst ausdrückt oder verbreitet, sollte er als fearful gekennzeichnet werden;
    (4) Andernfalls sollte er als normal annotiert werden.
    Jeder Tweet enthält mindestens eine Feindseligkeitsart und kann mehr als eine Feindseligkeitsart haben.

    Hier ist der Tweet:
    {}
    Jetzt endet der Tweet.

    Bitte kennzeichnen Sie ihn mit: fearful, abusive, offensive, hateful, disrespectful, normal.
    Wenn mehrere Stimmungen vorhanden sind, erwähnen Sie sie nacheinander in der oben genannten Reihenfolge und trennen Sie sie mit Leerzeichen.
    """,

    "French": """
    Cette tâche consiste à identifier le type d'hostilité d'un tweet, que je vous montrerai ensuite, et vous devez respecter les conventions suivantes :
    (1) Si le tweet semble dangereux, il doit être étiqueté comme abusive;
    (2) Selon le degré auquel il propage la haine et le ton utilisé par l'auteur, il peut être hateful, offensive ou disrespectful;
    (3) Si le tweet exprime ou propage la peur par ignorance contre un groupe d'individus, il doit être étiqueté comme fearful;
    (4) Sinon, il doit être annoté comme normal.
    Chaque tweet contient au moins un type d'hostilité et peut en avoir plusieurs.

    Voici le tweet :
    {}
    Maintenant, le tweet se termine.

    Veuillez le marquer avec : fearful, abusive, offensive, hateful, disrespectful, normal.
    Si plusieurs sentiments existent, mentionnez-les un par un dans l'ordre mentionné ci-dessus et séparez-les par des espaces.
    """,

    "Spanish": """
    Esta tarea consiste en identificar el tipo de hostilidad de un tweet, que te mostraré a continuación, y debes ceñirte a las siguientes convenciones:
    (1) Si el tweet suena peligroso, debe etiquetarse como abusive;
    (2) Según el grado en que propaga odio y el tono que utiliza su autor, puede ser hateful, offensive o disrespectful;
    (3) Si el tweet expresa o difunde miedo por ignorancia hacia un grupo de personas, debe etiquetarse como fearful;
    (4) De lo contrario, debe anotarse como normal.
    Cada tweet contiene al menos un tipo de hostilidad y puede tener más de un tipo de hostilidad.

    Aquí está el tweet:
    {}
    Ahora el tweet termina.

    Etiquétalo con: fearful, abusive, offensive, hateful, disrespectful, normal.
    Si existen múltiples sentimientos, menciónalos uno por uno en el orden mencionado anteriormente y sepáralos con espacios.
    """,

    "Portuguese": """
    Esta tarefa é identificar o tipo de hostilidade de um tweet, que lhe mostrarei a seguir, e você precisa seguir as seguintes convenções:
    (1) Se o tweet soar perigoso, deve ser rotulado como abusive;
    (2) De acordo com o grau em que espalha ódio e o tom que o autor usa, pode ser hateful, offensive ou disrespectful;
    (3) Se o tweet expressa ou espalha medo por ignorância contra um grupo de indivíduos, deve ser rotulado como fearful;
    (4) Caso contrário, deve ser anotado como normal.
    Cada tweet contém pelo menos um tipo de hostilidade e pode ter mais de um tipo de hostilidade.

    Aqui está o tweet:
    {}
    Agora o tweet termina.

    Por favor, rotule-o com: fearful, abusive, offensive, hateful, disrespectful, normal.
    Se existirem múltiplos sentimentos, mencione-os um por um na ordem mencionada acima e separe-os com espaços.
    """,

    "Italian": """
    Questo compito consiste nell'identificare il tipo di ostilità di un tweet, che ti mostrerò in seguito, e devi attenerti alle seguenti convenzioni:
    (1) se il tweet suona pericoloso, dovrebbe essere etichettato come abusive;
    (2) in base al grado in cui diffonde odio e al tono utilizzato dall'autore, può essere hateful, offensive o disrespectful;
    (3) se il tweet esprime o diffonde paura per ignoranza contro un gruppo di individui, dovrebbe essere etichettato come fearful;
    (4) altrimenti dovrebbe essere annotato come normal.
    Ogni tweet contiene almeno un tipo di ostilità e può avere più di un tipo di ostilità.

    Ecco il tweet:
    {}
    Ora il tweet finisce.

    Per favore, etichettalo con: fearful, abusive, offensive, hateful, disrespectful, normal.
    Se esistono più sentimenti, menzionali uno per uno nell'ordine menzionato sopra e separali con spazi.
    """,
    "Dutch": """
    Deze taak is om het vijandigheidstype van een tweet te identificeren, die ik je daarna zal laten zien, en je moet je aan de volgende conventies houden:
    (1) als de tweet gevaarlijk klinkt, moet deze worden gelabeld als abusive;
    (2) afhankelijk van de mate waarin het haat verspreidt en de toon die de auteur gebruikt, kan het hateful, offensive of disrespectful zijn;
    (3) als de tweet angst uit onwetendheid tegen een groep individuen uitdrukt of verspreidt, moet deze worden gelabeld als fearful;
    (4) anders moet het worden geannoteerd als normal.
    Elke tweet bevat minstens één vorm van vijandigheid en kan meer dan één vorm van vijandigheid bevatten.

    Hier is de tweet:
    {}
    Nu eindigt de tweet.

    Label het met: fearful, abusive, offensive, hateful, disrespectful, normal.
    Als er meerdere gevoelens bestaan, noem ze een voor een in de bovengenoemde volgorde en scheid ze met spaties.
    """,
    "Russian": """
    Эта задача заключается в определении типа враждебности в твите, который я покажу вам позже, и вы должны придерживаться следующих правил:
    (1) если твит звучит опасно, его следует обозначить как abusive;
    (2) в зависимости от степени распространения ненависти и тона, который использует автор, он может быть hateful, offensive или disrespectful;
    (3) если твит выражает или распространяет страх из-за невежества по отношению к группе людей, его следует обозначить как fearful;
    (4) в противном случае он должен быть помечен как normal.
    Каждый твит содержит как минимум один тип враждебности и может иметь более одного типа враждебности.

    Вот твит:
    {}
    Теперь твит заканчивается.

    Пожалуйста, пометьте его как: fearful, abusive, offensive, hateful, disrespectful, normal.
    Если существует несколько настроений, укажите их по одному в указанном выше порядке и разделите пробелами.
    """,
    "Czech": """
    Tento úkol je určen k identifikaci typu nepřátelství v tweetu, který vám ukážu později, a musíte dodržovat následující pravidla:
    (1) pokud tweet zní nebezpečně, měl by být označen jako abusive;
    (2) podle míry, jakou šíří nenávist a podle tónu, který autor používá, může být hateful, offensive nebo disrespectful;
    (3) pokud tweet vyjadřuje nebo šíří strach z nevědomosti vůči skupině jednotlivců, měl by být označen jako fearful;
    (4) jinak by měl být označen jako normal.
    Každý tweet obsahuje alespoň jeden typ nepřátelství a může mít více než jeden typ nepřátelství.

    Zde je tweet:
    {}
    Nyní tweet končí.

    Označte jej prosím jako: fearful, abusive, offensive, hateful, disrespectful, normal.
    Pokud existuje více nálad, uveďte je jednu po druhé ve výše uvedeném pořadí a oddělte je mezerami.
    """,
    "Polish": """
    To zadanie polega na identyfikacji rodzaju wrogości w tweecie, który pokażę ci później, i musisz trzymać się następujących zasad:
    (1) jeśli tweet brzmi niebezpiecznie, należy go oznaczyć jako abusive;
    (2) w zależności od stopnia rozprzestrzeniania nienawiści i tonu, jakiego używa autor, może być hateful, offensive lub disrespectful;
    (3) jeśli tweet wyraża lub rozprzestrzenia strach z powodu niewiedzy wobec grupy osób, należy go oznaczyć jako fearful;
    (4) w przeciwnym razie powinien być oznaczony jako normal.
    Każdy tweet zawiera co najmniej jeden rodzaj wrogości i może mieć więcej niż jeden rodzaj wrogości.

    Oto tweet:
    {}
    Teraz tweet się kończy.

    Oznacz go jako: fearful, abusive, offensive, hateful, disrespectful, normal.
    Jeśli istnieje wiele emocji, wymień je jedna po drugiej w powyższej kolejności i oddziel je spacjami.
    """,
    "Arabic": """
    هذه المهمة هي تحديد نوع العداء في تغريدة، والتي سأعرضها لك لاحقًا، ويجب أن تلتزم بالاتفاقيات التالية:
    (1) إذا كانت التغريدة تبدو خطيرة، فيجب تصنيفها على أنها abusive؛
    (2) وفقًا لدرجة انتشار الكراهية والنبرة التي يستخدمها المؤلف، يمكن أن تكون hateful، offensive أو disrespectful؛
    (3) إذا كانت التغريدة تعبر عن أو تنشر الخوف من الجهل تجاه مجموعة من الأفراد، فيجب تصنيفها على أنها fearful؛
    (4) وإلا فيجب أن يتم تصنيفها كـ normal.
    تحتوي كل تغريدة على نوع واحد على الأقل من العداء ويمكن أن تحتوي على أكثر من نوع من العداء.

    ها هي التغريدة:
    {}
    الآن تنتهي التغريدة.

    يرجى تصنيفها على أنها: fearful, abusive, offensive, hateful, disrespectful, normal.
    إذا كانت هناك مشاعر متعددة، اذكرها واحدة تلو الأخرى بالترتيب المذكور أعلاه وفصلها بمسافات.
    """,
    "Persian": """
    این وظیفه شناسایی نوع خصومت یک توییت است که بعداً به شما نشان خواهم داد و شما باید به اصول زیر پایبند باشید:
    (1) اگر توییت خطرناک به نظر برسد، باید به عنوان abusive برچسب‌گذاری شود؛
    (2) بر اساس درجه‌ای که نفرت را گسترش می‌دهد و لحن نویسنده آن، می‌تواند hateful، offensive یا disrespectful باشد؛
    (3) اگر توییت از روی نادانی علیه گروهی از افراد ترس را بیان یا گسترش می‌دهد، باید به عنوان fearful برچسب‌گذاری شود؛
    (4) در غیر این صورت، باید به عنوان normal یادداشت شود.
    هر توییت حاوی حداقل یک نوع خصومت است و می‌تواند بیش از یک نوع خصومت داشته باشد.

    این توییت است:
    {}
    اکنون توییت به پایان می‌رسد.

    لطفاً آن را با: fearful, abusive, offensive, hateful, disrespectful, normal برچسب‌گذاری کنید.
    اگر چندین احساس وجود دارد، آنها را یکی یکی به ترتیب ذکر شده بالا بیان کرده و با فاصله جدا کنید.
    """,
    "Hebrew": """
    המשימה הזו היא לזהות את סוג העוינות בציוץ, שאציג לך לאחר מכן, ואתה צריך לדבוק בהנחיות הבאות:
    (1) אם הציוץ נשמע מסוכן, יש לסמן אותו כ abusive;
    (2) לפי המידה שבה הוא מפיץ שנאה והטון שבו הכותב משתמש, הוא יכול להיות hateful, offensive או disrespectful;
    (3) אם הציוץ מביע או מפיץ פחד מתוך בורות כלפי קבוצה של אנשים, יש לסמן אותו כ fearful;
    (4) אחרת יש לסמן אותו כ normal.
    כל ציוץ מכיל לפחות סוג אחד של עוינות, ויכול להיות יותר מסוג אחד של עוינות.

    הנה הציוץ:
    {}
    כעת הציוץ מסתיים.

    סמן אותו בבקשה: fearful, abusive, offensive, hateful, disrespectful, normal.
    אם קיימים מספר רגשות, הזכר אותם אחד אחד בסדר שהוזכר לעיל ופרד ביניהם ברווחים.
    """,
    "Turkish": """
    Bu görev, size daha sonra göstereceğim bir tweet'in düşmanlık türünü belirlemektir ve aşağıdaki kurallara uymanız gerekir:
    (1) Tweet tehlikeli görünüyorsa, abusive olarak etiketlenmelidir;
    (2) nefret yayma derecesine ve yazarın kullandığı tona göre, hateful, offensive veya disrespectful olabilir;
    (3) Tweet, bir grup kişiye karşı cehaletten kaynaklanan korkuyu ifade ediyor veya yayıyorsa, fearful olarak etiketlenmelidir;
    (4) aksi takdirde normal olarak not edilmelidir.
    Her tweet en az bir tür düşmanlık içerir ve birden fazla düşmanlık türüne sahip olabilir.

    İşte tweet:
    {}
    Şimdi tweet bitiyor.

    Lütfen şu şekilde etiketleyin: fearful, abusive, offensive, hateful, disrespectful, normal.
    Birden fazla duygu varsa, onları yukarıda belirtilen sırayla birer birer belirtin ve aralarına boşluk koyarak ayırın.
    """,
    "Japanese": """
    このタスクは、ツイートの敵対性のタイプを識別することです。これを後でお見せします。以下の規則に従ってください。
    (1) ツイートが危険に聞こえる場合、それはabusiveとしてラベル付けされるべきです。
    (2) ツイートが憎しみを広め、著者が使用するトーンによって、それはhateful、offensiveまたはdisrespectfulである可能性があります。
    (3) ツイートが無知から特定のグループに対して恐怖を表現または広めている場合、それはfearfulとしてラベル付けされるべきです。
    (4) それ以外の場合は、normalとして注釈を付ける必要があります。
    各ツイートには少なくとも1つの敵対性のタイプが含まれており、複数の敵対性のタイプを持つことができます。

    こちらがツイートです：
    {}
    これでツイートが終了します。

    以下でラベルを付けてください：fearful、abusive、offensive、hateful、disrespectful、normal。
    複数の感情が存在する場合は、上記の順序で1つずつ言及し、それらをスペースで区切ってください。
    """,
    "Korean": """
    이 작업은 트윗의 적대성 유형을 식별하는 것입니다. 나중에 보여드릴 것입니다. 다음 규칙을 준수해야 합니다.
    (1) 트윗이 위험하게 들리면 abusive로 표시해야 합니다.
    (2) 증오를 퍼뜨리는 정도와 저자가 사용하는 어조에 따라, hateful, offensive 또는 disrespectful이 될 수 있습니다.
    (3) 트윗이 특정 그룹에 대해 무지로 인해 두려움을 표현하거나 퍼뜨리는 경우, fearful로 라벨이 지정되어야 합니다.
    (4) 그렇지 않으면 normal로 주석을 달아야 합니다.
    각 트윗에는 적어도 하나의 적대성 유형이 포함되어 있으며, 여러 적대성 유형을 가질 수 있습니다.

    여기에 트윗이 있습니다:
    {}
    이제 트윗이 끝납니다.

    다음으로 라벨을 붙여주세요: fearful, abusive, offensive, hateful, disrespectful, normal.
    여러 감정이 있는 경우, 위에서 언급한 순서대로 하나씩 언급하고 공백으로 구분하세요.
    """,
    "Vietnamese": """
    Nhiệm vụ này là xác định loại thù địch của một tweet, mà tôi sẽ cho bạn thấy sau, và bạn cần tuân thủ các quy ước sau:
    (1) nếu tweet nghe có vẻ nguy hiểm, nó nên được gắn nhãn là abusive;
    (2) theo mức độ mà nó lan truyền sự thù hận và giọng điệu mà tác giả sử dụng, nó có thể là hateful, offensive hoặc disrespectful;
    (3) nếu tweet thể hiện hoặc lan truyền nỗi sợ hãi do thiếu hiểu biết đối với một nhóm cá nhân, nó nên được gắn nhãn là fearful;
    (4) nếu không, nó nên được chú thích là normal.
    Mỗi tweet chứa ít nhất một loại thù địch và có thể có nhiều loại thù địch.

    Đây là tweet:
    {}
    Bây giờ tweet kết thúc.

    Vui lòng gắn nhãn nó với: fearful, abusive, offensive, hateful, disrespectful, normal.
    Nếu tồn tại nhiều cảm xúc, hãy đề cập đến chúng lần lượt theo thứ tự đã nói ở trên và ngăn cách chúng bằng dấu cách.
    """,
    "Thai": """
    งานนี้คือการระบุประเภทของความเป็นศัตรูของทวีต ซึ่งฉันจะแสดงให้คุณดูในภายหลัง และคุณต้องปฏิบัติตามข้อตกลงต่อไปนี้:
    (1) หากทวีตฟังดูอันตราย ควรติดป้ายกำกับว่า abusive;
    (2) ตามระดับที่เผยแพร่ความเกลียดชังและโทนเสียงที่ผู้เขียนใช้ อาจเป็น hateful, offensive หรือ disrespectful;
    (3) หากทวีตแสดงหรือเผยแพร่ความกลัวจากความไม่รู้ต่อกลุ่มบุคคล ควรติดป้ายกำกับว่า fearful;
    (4) มิฉะนั้น ควรบันทึกเป็น normal.
    ทวีตแต่ละรายการมีประเภทความเป็นศัตรูอย่างน้อยหนึ่งประเภท และอาจมีประเภทความเป็นศัตรูมากกว่าหนึ่งประเภท

    นี่คือทวีต:
    {}
    ตอนนี้ทวีตจบแล้ว

    โปรดติดป้ายกำกับว่า: fearful, abusive, offensive, hateful, disrespectful, normal.
    หากมีความรู้สึกหลายอย่างเกิดขึ้น โปรดกล่าวถึงทีละรายการตามลำดับที่กล่าวถึงข้างต้น และเว้นวรรคให้ห่างกัน
    """,
    "Indonesian": """
    Tugas ini adalah untuk mengidentifikasi jenis permusuhan dari sebuah tweet, yang akan saya tunjukkan kepada Anda nanti, dan Anda harus mematuhi konvensi berikut:
    (1) jika tweet terdengar berbahaya, itu harus diberi label sebagai abusive;
    (2) menurut tingkat penyebaran kebencian dan nada yang digunakan penulis, itu bisa hateful, offensive atau disrespectful;
    (3) jika tweet tersebut mengekspresikan atau menyebarkan ketakutan karena ketidaktahuan terhadap sekelompok individu, itu harus diberi label fearful;
    (4) jika tidak, itu harus diberi anotasi sebagai normal.
    Setiap tweet mengandung setidaknya satu jenis permusuhan, dan dapat memiliki lebih dari satu jenis permusuhan.

    Ini adalah tweet:
    {}
    Sekarang tweet berakhir.

    Silakan beri label dengan: fearful, abusive, offensive, hateful, disrespectful, normal.
    Jika ada beberapa sentimen, sebutkan satu per satu dalam urutan yang disebutkan di atas dan pisahkan dengan spasi.
    """,
    "Malay": """
    Tugas ini adalah untuk mengenal pasti jenis permusuhan sesuatu tweet, yang akan saya tunjukkan kepada anda kemudian, dan anda perlu mematuhi konvensyen berikut:
    (1) jika tweet kedengaran berbahaya, ia harus dilabel sebagai abusive;
    (2) mengikut tahap penyebaran kebencian dan nada yang digunakan oleh pengarang, ia boleh menjadi hateful, offensive atau disrespectful;
    (3) jika tweet tersebut menyatakan atau menyebarkan ketakutan kerana kejahilan terhadap sekumpulan individu, ia harus dilabel sebagai fearful;
    (4) jika tidak, ia harus dianotasi sebagai normal.
    Setiap tweet mengandungi sekurang-kurangnya satu jenis permusuhan dan boleh mempunyai lebih daripada satu jenis permusuhan.

    Berikut ialah tweet:
    {}
    Kini tweet tamat.

    Sila labelkan dengan: fearful, abusive, offensive, hateful, disrespectful, normal.
    Jika terdapat pelbagai sentimen, sebutkannya satu persatu dalam susunan yang disebutkan di atas dan asingkannya dengan ruang.
    """,
    "Lao": """
    ຫນ້າທີ່ນີ້ແມ່ນການລະບຸປະເພດຄວາມຄວາມຄຽດຢູ່ຂອງການທີ່ທີ່ຈະໄດ້ຮັບການຈົ່ມການທີ່ຈະໄດ້ຮັບການຂຽນໄວ້ໃນອະນາຄົມນີ້, ແລະທ່ານຈຳເປັນຕ້ອງຕາມສະຖານະການຂອງການທີ່ຈະເຮັດຄວາມຄຽດຮອດເຂົານີ້:
    (1) ຖ້າວ່າຄຳຄົ້ນເປັນອັນຕະລາຍ, ມັນຄວນຈະຕິດປະເພດວ່າ abusive;
    (2) ອີງໃສ່ລະດັບຂອງການທີ່ກະຈາຍຄວາມຄຽດຂອງການທີ່ຈະບັນທຶກຄວາມຄຽດຂອງຄົນທີ່ຂຽນຂໍ້ຄວາມ, ມັນສາມາດທີ່ຈະໄດ້ຮັບຄຳອັນຄວາມຄຽດຮອດຂອງຄົນທີ່ອື່ນ;
    (3) ຖ້າວ່າຄຳຄົ້ນບໍ່ມີຄວາມຄຽດກັບຄົນອື່ນຫຼືຄວາມຄຽດກັບຄົນທີ່ບໍ່ມີຄວາມຮູ້, ມັນຄວນຈະຕິດປະເພດວ່າ fearful;
    (4) ມິດຢ່າງອື່ນມັນຄວນຈະຕິດປະເພດວ່າ normal.
    ທຸກໆຄຳຄົ້ນມີປະເພດຄວາມຄຽດຢ່າງນ້ອຍຫນຶ່ງປະເພດ, ແລະອາດຈະມີຫລາຍປະເພດຄວາມຄຽດຢູ່ລວມກັນ

    ນີ້ແມ່ນຄຳຄົ້ນ:
    {}
    ບັນດາຄຳຄົ້ນໄດ້ພິດຂຶ້ນເທິງນີ້

    ກະລຸນາຕິດປະເພດນີ້: fearful, abusive, offensive, hateful, disrespectful, normal.
    ຖ້າຫາກມີຄວາມຮູ້ສຶກຫຼາຍໆແບບພ້ອມກັນ, ກະລຸນາເວົ້າແຕ່ລະເຫດຕາມລຳດັບຂອງຄຳຂ້າງເທິງນີ້ແລະຂຽນຫຼືວ່າເວົ້າເວລາວ່າບໍ່ຕ້ອງປ່ຽນກັບຄຳໄປໄປ
    """,
    "Burmese": """
    ဤအလုပ်သည် တစ်ခုသောတူအာအမျိုးအစားကို ဖော်ထုတ်ခြင်းဖြစ်ပြီး ဤအမှုအတွက် အခြားပေးပို့သော စာပို့စည်းကမ်းချက်များကိုလိုက်နာရပါမည်။
    (1) တူကို အန္တရာယ်ဖြစ်သည်ဟုမြင်ရမည်ဆိုပါက သူမကို abusive အဖြစ်အမှတ်သားရပါမည်။
    (2) သူသည် မည်သည့်အတိုင်းအတာဖြင့် အမုန်းပြောနေသည်နှင့် ဘာသာအမြင်တစ်ခုသို့ သဘောထားသည်ကို အကဲဖြတ်သည်နှင့်အညီ hateful, offensive, disrespectful ဆိုရမည်။
    (3) တူသည် သူတစ်ဦးတစ်ယောက်ရဲ့ ဖွင့်မပြောနိုင်ခြင်းဟုထင်ပါက၊ သူသည် fearful ဆိုရပါမည်။
    (4) အခြားဘယ်နေရာမှ normal ဆိုရပါမည်။
    တူများသည် အနည်းဆုံးတစ်ခုသောအမျိုးအစားကို ပေးပါသည်။

    ဒီတော့ အာဘောမီရှင့်ထောက်ခံချက်:
    {}
    အချိန်မှန်များအတွက် အဆုံးသတ်ပါသည်။

    ကျေးဇူးပြုပြီးအမှတ်တံဆိပ်ရေးရန်: fearful, abusive, offensive, hateful, disrespectful, normal။
    အကယ်၍ စိတ်ဝင်စားမှုများရှိပါက၊ အကယ်၍ ပြောဆိုရန် အခင်ကိုယ်ရေးအတွက် စည်းမျဉ်းတစ်ခုတည်းကို ဆပ်ပါ။
    """,
    "Cebuano": """
    Kini nga buluhaton mao ang pag-ila sa tipo sa pagka-hostile sa usa ka tweet, nga ipakita nako kanimo pagkahuman, ug kinahanglan nga mo-stick ka sa mosunod nga mga kasabutan:
    (1) kon ang tweet daw mahimong delikado, kini kinahanglan nga mailhan ingon nga abusive;
    (2) sumala sa degree nga kini nagpatunhay sa kapungot ug ang tono nga gigamit sa tagsulat, kini mahimong hateful, offensive o disrespectful;
    (3) kon ang tweet nagpadayag o nagpatunhay sa kahadlok tungod sa pagkawalay alamag batok sa usa ka grupo sa mga indibidwal, kini kinahanglan nga mailhan ingon nga fearful;
    (4) kung dili kini kinahanglan nga ma-annotate ingon nga normal.
    Ang matag tweet adunay labing menos usa ka type sa pagka-hostile, ug mahimo nga adunay daghan nga type sa pagka-hostile.

    Ania ang tweet:
    {}
    Karon ang tweet matapos.

    Palihug i-label kini sa: fearful, abusive, offensive, hateful, disrespectful, normal.
    Kung daghan nga mga pagbati ang naglungtad, hisguti kini usa-usa sa mao nga han-ay ug pagbulag sa mga espasyo.
    """,
    "Khmer": """
    កិច្ចការនេះគឺជាការកំណត់ប្រភេទការតាំងចិត្តអាក្រក់របស់អត្ថបទនៅក្នុង Twitter មួយ ដែលខ្ញុំនឹងបង្ហាញអ្នកក្រោយមក ហើយអ្នកត្រូវប្រកាន់ខ្ជាប់នូវទស្សនៈខាងក្រោម:
    (1) ប្រសិនបើអត្ថបទនោះសង្កេតឃើញថាអាចនឹងហ៊ានហេីយអាចនឹងផ្ទុកនូវការកុហក វាគួរតែដាក់ហត្ថលេខាជា abusive;
    (2) ស្របតាមកម្រិតដែលអ្នកនិពន្ធប៉ុនប៉ងផ្សព្វផ្សាយអោយចែកជូននូវការស្អប់ខ្ពើម និងល្បែងក្បត់របស់អ្នកនិពន្ធ វាអាចជា hateful, offensive ឬ disrespectful;
    (3) ប្រសិនបើអត្ថបទនោះអាចបង្កគំរាមកំហែងចំពោះអ្នកជិតខាងនៅក្នុងការទុកចិត្តដែលមិនបានបង្ហាញបញ្ជាក់ ឬចំពោះក្រុមមួយអង្គចង ឬប្រភេទមនុស្សចំណោមមនុស្ស, វាគួរតែដាក់ហត្ថលេខាជា fearful;
    (4) ធ្មប់ទាំងនេះគួរតែត្រូវបានចាត់ជាឧទ្ទិស normal។
    អត្ថបទ Twitter នីមួយៗរួមបញ្ចូលពីរប្រភេទចំរើនតាំងចិត្តអាក្រក់នៅតាមអត្ថបទទាំងនោះៗ និងអាចមានច្រើននូវចំរើនតាំងចិត្តអាក្រក់មួយ។

    នេះជា Twitter:
    {}
    ឥឡូវនេះអត្ថបទនេះបញ្ចប់។

    សូមដាក់ហត្ថលេខាវាជា: fearful, abusive, offensive, hateful, disrespectful, normal.
    ប្រសិនបើមានចំរើនតាំងចិត្តច្រើន សូមបញ្ជាក់នូវទាំងអស់អោយម្នាក់ៗទៅតាមលំដាប់បានលើកឡើងខាងលើ និងគណនាចេញពីគន្លឹះទាំងអស់។
    """,
    "Tagalog": """
    Ang gawaing ito ay upang tukuyin ang uri ng poot ng isang tweet, na ipapakita ko sa iyo pagkatapos, at kailangan mong sumunod sa mga sumusunod na kombensyon:
    (1) kung ang tweet ay mukhang mapanganib, ito ay dapat na mai-label bilang abusive;
    (2) ayon sa antas kung saan ito kumakalat ng poot at ang tono ng ginagamit ng may-akda, maaari itong maging hateful, offensive o disrespectful;
    (3) kung ang tweet ay nagpapahayag o kumakalat ng takot mula sa kamangmangan laban sa isang grupo ng mga indibidwal, ito ay dapat na mai-label bilang fearful;
    (4) kung hindi, dapat itong ma-annotate bilang normal.
    Ang bawat tweet ay naglalaman ng hindi bababa sa isang uri ng poot, at maaaring magkaroon ng higit sa isang uri ng poot.

    Narito ang tweet:
    {}
    Ngayon natapos na ang tweet.

    Mangyaring i-label ito ng: fearful, abusive, offensive, hateful, disrespectful, normal.
    Kung may umiiral na maraming damdamin, banggitin ang mga ito nang paisa-isa sa naunang binanggit na pagkakasunud-sunod at paghiwalayin ang mga ito ng mga puwang.
    """,
    "Hindi": """
    इस कार्य में एक ट्वीट के शत्रुता प्रकार की पहचान करना शामिल है, जिसे मैं बाद में आपको दिखाऊंगा, और आपको निम्नलिखित नियमों का पालन करना होगा:
    (1) यदि ट्वीट खतरनाक लगता है, तो इसे abusive के रूप में लेबल किया जाना चाहिए;
    (2) जिस हद तक यह नफरत फैलाता है और लेखक जिस लहजे का उपयोग करता है, उसके अनुसार यह hateful, offensive या disrespectful हो सकता है;
    (3) यदि ट्वीट अज्ञानता से किसी समूह के व्यक्तियों के खिलाफ भय व्यक्त करता है या फैलाता है, तो इसे fearful के रूप में लेबल किया जाना चाहिए;
    (4) अन्यथा इसे normal के रूप में एनोटेट किया जाना चाहिए।
    प्रत्येक ट्वीट में कम से कम एक शत्रुता प्रकार होता है, और इसमें एक से अधिक शत्रुता प्रकार हो सकते हैं।

    यहाँ ट्वीट है:
    {}
    अब ट्वीट समाप्त होता है।

    कृपया इसे लेबल करें: fearful, abusive, offensive, hateful, disrespectful, normal।
    यदि कई भावनाएँ मौजूद हैं, तो उनका एक-एक करके उपरोक्त क्रम में उल्लेख करें और उन्हें रिक्त स्थान से अलग करें।
    """,
    "Bengali": """
    এই কাজটি একটি টুইটের শত্রুতা প্রকার চিহ্নিত করা, যা আমি পরে আপনাকে দেখাবো, এবং আপনাকে নিম্নলিখিত নিয়মগুলি মেনে চলতে হবে:
    (1) যদি টুইটটি বিপজ্জনক মনে হয়, তবে এটি abusive হিসাবে লেবেল করা উচিত;
    (2) এটি ঘৃণা ছড়ানোর মাত্রা এবং লেখক যে সুর ব্যবহার করে তার উপর ভিত্তি করে, এটি hateful, offensive বা disrespectful হতে পারে;
    (3) যদি টুইটটি একটি ব্যক্তি গোষ্ঠীর বিরুদ্ধে অজ্ঞতার কারণে ভয় প্রকাশ করে বা ছড়িয়ে দেয়, তবে এটি fearful হিসাবে লেবেল করা উচিত;
    (4) অন্যথায় এটি normal হিসাবে উল্লেখ করা উচিত।
    প্রতিটি টুইট অন্তত একটি শত্রুতা প্রকার ধারণ করে, এবং এতে একাধিক শত্রুতা প্রকার থাকতে পারে।

    এখানে টুইট:
    {}
    এখন টুইট শেষ হয়েছে।

    দয়া করে এটি লেবেল করুন: fearful, abusive, offensive, hateful, disrespectful, normal।
    যদি একাধিক অনুভূতি থাকে, তবে উপরে উল্লেখিত ক্রমে সেগুলি একে একে উল্লেখ করুন এবং তাদের স্পেস দিয়ে আলাদা করুন।
    """,
    "Urdu": """
    یہ کام ایک ٹویٹ کے دشمنی کی قسم کی شناخت کرنا ہے، جو میں آپ کو بعد میں دکھاؤں گا، اور آپ کو درج ذیل کنونشنز پر عمل کرنا ہوگا:
    (1) اگر ٹویٹ خطرناک لگے، تو اسے abusive کے طور پر لیبل کرنا چاہئے؛
    (2) جس حد تک یہ نفرت پھیلاتا ہے اور مصنف جو لہجہ استعمال کرتا ہے، اس کے مطابق یہ hateful, offensive یا disrespectful ہو سکتا ہے؛
    (3) اگر ٹویٹ جہالت کی وجہ سے کسی گروپ کے افراد کے خلاف خوف ظاہر کرتا ہے یا پھیلاتا ہے، تو اسے fearful کے طور پر لیبل کرنا چاہئے؛
    (4) بصورت دیگر اسے normal کے طور پر تشریح کرنا چاہئے۔
    ہر ٹویٹ میں کم از کم ایک دشمنی کی قسم شامل ہوتی ہے، اور اس میں ایک سے زیادہ دشمنی کی قسم ہو سکتی ہے۔

    یہاں ٹویٹ ہے:
    {}
    اب ٹویٹ ختم ہوتا ہے۔

    براہ کرم اسے لیبل کریں: fearful, abusive, offensive, hateful, disrespectful, normal۔
    اگر متعدد جذبات موجود ہوں تو ان کا ایک ایک کر کے اوپر بیان کردہ ترتیب میں ذکر کریں اور انہیں جگہ سے الگ کریں۔
    """
}


# Create a new DataFrame
instruction_df1 = df[['tweet', 'sentiment']].copy()

# Generate the instructions using randomly selected templates
instruction_df1['instruction'] = instruction_df1['tweet'].apply(
    lambda x: random.choice(list(instruction_templates.values())).format(x)
)

# Generate the outputs by replacing underscores with spaces in the sentiment column
instruction_df1['output'] = instruction_df1['sentiment'].apply(lambda x: x.replace('_', ' '))

# Select only the instruction and output columns
instruction_df1 = instruction_df1[['instruction', 'output']]

# just use a small sample
instruction_df1 = instruction_df1.sample(n=1000, random_state=42)

# save the parquet file
instruction_df1.to_parquet("../../data/MLMA/mlma-1.parquet", index=False)

# Define the instruction templates in different languages
instruction_templates = {
    "Chinese": """
    请确定推文是否基于以下因素侮辱或歧视人群：(1) 出身，(2) 宗教，(3) 性别，(4) 性取向，(5) 残疾，或 (6) 其他。

    这是推文：
    {}

    请根据推文中的侮辱或歧视的目标，回复以下之一：origin, religion, gender, sexual orientation, disability, 或 other。
    """,
    "English": """
    Please determine whether the tweet insults or discriminates against people based on their (1) origin, (2) religion, (3) gender, (4) sexual orientation, (5) disability, or (6) other.

    Here's the tweet:
    {}

    Please respond with one of: origin, religion, gender, sexual orientation, disability, or other, depending on what insults or discriminates in the tweet are targeted at.
    """,
    "German": """
    Bitte bestimmen Sie, ob der Tweet Menschen aufgrund ihrer (1) Herkunft, (2) Religion, (3) Geschlecht, (4) sexuellen Orientierung, (5) Behinderung oder (6) anderen Merkmalen beleidigt oder diskriminiert.

    Hier ist der Tweet:
    {}

    Bitte antworten Sie mit einem der folgenden: origin, religion, gender, sexual orientation, disability, oder other, je nachdem, was im Tweet beleidigt oder diskriminiert wird.
    """,
    "French": """
    Veuillez déterminer si le tweet insulte ou discrimine les gens en fonction de leur (1) origine, (2) religion, (3) sexe, (4) orientation sexuelle, (5) handicap, ou (6) autre.

    Voici le tweet :
    {}

    Veuillez répondre avec l'un des suivants : origin, religion, gender, sexual orientation, disability, ou other, en fonction de ce que le tweet cible comme insultes ou discriminations.
    """,
    "Spanish": """
    Por favor, determine si el tweet insulta o discrimina a las personas en función de su (1) origen, (2) religión, (3) género, (4) orientación sexual, (5) discapacidad, o (6) otro.

    Aquí está el tweet:
    {}

    Por favor, responda con una de las siguientes opciones: origin, religion, gender, sexual orientation, disability, o other, dependiendo de lo que se insulte o discrimine en el tweet.
    """,
    "Portuguese": """
    Por favor, determine se o tweet insulta ou discrimina pessoas com base em sua (1) origem, (2) religião, (3) gênero, (4) orientação sexual, (5) deficiência ou (6) outro.

    Aqui está o tweet:
    {}

    Por favor, responda com uma das seguintes opções: origin, religion, gender, sexual orientation, disability, ou other, dependendo de qual insulto ou discriminação é direcionado no tweet.
    """,
    "Italian": """
    Si prega di determinare se il tweet insulta o discrimina le persone in base alla loro (1) origine, (2) religione, (3) genere, (4) orientamento sessuale, (5) disabilità, o (6) altro.

    Ecco il tweet:
    {}

    Si prega di rispondere con una delle seguenti opzioni: origin, religion, gender, sexual orientation, disability, o other, a seconda di ciò che viene insultato o discriminato nel tweet.
    """,
    "Dutch": """
    Bepaal of de tweet mensen beledigt of discrimineert op basis van hun (1) afkomst, (2) religie, (3) geslacht, (4) seksuele geaardheid, (5) handicap, of (6) andere.

    Hier is de tweet:
    {}

    Reageer alstublieft met een van de volgende: origin, religion, gender, sexual orientation, disability, of other, afhankelijk van wat in de tweet wordt beledigd of gediscrimineerd.
    """,
    "Russian": """
    Пожалуйста, определите, оскорбляет ли твит или дискриминирует людей на основе их (1) происхождения, (2) религии, (3) пола, (4) сексуальной ориентации, (5) инвалидности или (6) других.

    Вот твит:
    {}

    Пожалуйста, ответьте одним из следующих вариантов: origin, religion, gender, sexual orientation, disability, или other, в зависимости от того, на что направлены оскорбления или дискриминация в твите.
    """,
    "Czech": """
    Určete, zda tweet uráží nebo diskriminuje lidi na základě jejich (1) původu, (2) náboženství, (3) pohlaví, (4) sexuální orientace, (5) zdravotního postižení nebo (6) jiného.

    Zde je tweet:
    {}

    Odpovězte prosím jednou z následujících možností: origin, religion, gender, sexual orientation, disability, nebo other, v závislosti na tom, na co jsou urážky nebo diskriminace v tweetu zaměřeny.
    """,
    "Polish": """
    Proszę określić, czy tweet obraża lub dyskryminuje ludzi na podstawie ich (1) pochodzenia, (2) religii, (3) płci, (4) orientacji seksualnej, (5) niepełnosprawności lub (6) innych.

    Oto tweet:
    {}

    Proszę odpowiedzieć jednym z następujących: origin, religion, gender, sexual orientation, disability, lub other, w zależności od tego, co jest celem obrazy lub dyskryminacji w tweecie.
    """,
    "Arabic": """
    يرجى تحديد ما إذا كانت التغريدة تُهين أو تُميّز ضد الأشخاص بناءً على (1) أصلهم، (2) دينهم، (3) جنسهم، (4) ميولهم الجنسية، (5) إعاقتهم، أو (6) غير ذلك.

    ها هي التغريدة:
    {}

    يرجى الرد بأحد الخيارات التالية: origin, religion, gender, sexual orientation, disability، أو other، بناءً على ما تستهدفه الإهانات أو التمييز في التغريدة.
    """,
    "Persian": """
    لطفاً تعیین کنید که آیا توییت به مردم توهین یا تبعیض می‌کند بر اساس (1) منشأ، (2) مذهب، (3) جنسیت، (4) گرایش جنسی، (5) معلولیت، یا (6) دیگر.

    این توییت است:
    {}

    لطفاً با یکی از موارد زیر پاسخ دهید: origin، religion، gender، sexual orientation، disability، یا other، بسته به اینکه توهین‌ها یا تبعیض‌ها در توییت به چه چیزی هدف قرار می‌گیرند.
    """,
    "Hebrew": """
    אנא קבע אם הציוץ מעליב או מפלה אנשים על בסיס (1) מוצא, (2) דת, (3) מגדר, (4) נטייה מינית, (5) נכות, או (6) אחר.

    הנה הציוץ:
    {}

    אנא השב עם אחת מהאפשרויות הבאות: origin, religion, gender, sexual orientation, disability, או other, בהתאם למה שההעלבות או ההפליות בציוץ מכוונות אליו.
    """,
    "Turkish": """
    Lütfen tweet'in insanları (1) köken, (2) din, (3) cinsiyet, (4) cinsel yönelim, (5) engellilik veya (6) diğer temellere dayalı olarak aşağılayıp aşağılamadığını veya ayrımcılık yapıp yapmadığını belirleyin.

    İşte tweet:
    {}

    Lütfen tweet'teki aşağılamaların veya ayrımcılığın hedef alındığı duruma göre, şu seçeneklerden biriyle yanıt verin: origin, religion, gender, sexual orientation, disability, veya other.
    """,
    "Japanese": """
    ツイートが (1) 出身地、(2) 宗教、(3) 性別、(4) 性的指向、(5) 障害、または (6) その他の理由で人々を侮辱または差別しているかどうかを判断してください。

    こちらがツイートです：
    {}

    ツイートに侮辱や差別が含まれている場合、次のいずれかで回答してください：origin、religion、gender、sexual orientation、disability、または other。
    """,
    "Korean": """
    트윗이 사람들을 (1) 출신, (2) 종교, (3) 성별, (4) 성적 지향, (5) 장애 또는 (6) 기타 사항을 기준으로 모욕하거나 차별하는지 여부를 결정하십시오.

    여기에 트윗이 있습니다:
    {}

    트윗에서 모욕이나 차별의 대상이 되는 것에 따라 origin, religion, gender, sexual orientation, disability, 또는 other 중 하나로 응답하십시오.
    """,
    "Vietnamese": """
    Vui lòng xác định liệu tweet có xúc phạm hoặc phân biệt đối xử với mọi người dựa trên (1) nguồn gốc, (2) tôn giáo, (3) giới tính, (4) xu hướng tình dục, (5) khuyết tật hoặc (6) khác hay không.

    Đây là tweet:
    {}

    Vui lòng phản hồi bằng một trong các tùy chọn sau: origin, religion, gender, sexual orientation, disability, hoặc other, tùy thuộc vào đối tượng bị xúc phạm hoặc phân biệt đối xử trong tweet.
    """,
    "Thai": """
    โปรดพิจารณาว่าทวีตนี้มีการดูถูกหรือเลือกปฏิบัติต่อผู้คนหรือไม่โดยพิจารณาจาก (1) แหล่งกำเนิด, (2) ศาสนา, (3) เพศ, (4) รสนิยมทางเพศ, (5) ความพิการ หรือ (6) อื่นๆ

    นี่คือทวีต:
    {}

    โปรดตอบกลับด้วยหนึ่งในตัวเลือกต่อไปนี้: origin, religion, gender, sexual orientation, disability หรือ other ขึ้นอยู่กับสิ่งที่การดูถูกหรือการเลือกปฏิบัติในทวีตมุ่งเป้าไปที่
    """,
    "Indonesian": """
    Silakan tentukan apakah tweet menghina atau mendiskriminasi orang berdasarkan (1) asal, (2) agama, (3) jenis kelamin, (4) orientasi seksual, (5) disabilitas, atau (6) lainnya.

    Ini tweetnya:
    {}

    Silakan jawab dengan salah satu dari: origin, religion, gender, sexual orientation, disability, atau other, tergantung pada apa yang dihina atau didiskriminasi dalam tweet tersebut.
    """,
    "Malay": """
    Sila tentukan sama ada tweet menghina atau mendiskriminasi orang berdasarkan (1) asal usul, (2) agama, (3) jantina, (4) orientasi seksual, (5) kecacatan, atau (6) lain-lain.

    Berikut adalah tweetnya:
    {}

    Sila jawab dengan salah satu daripada: origin, religion, gender, sexual orientation, disability, atau other, bergantung pada apa yang dihina atau didiskriminasi dalam tweet tersebut.
    """,
    "Lao": """
    ກະລຸນາກຳນົດວ່າຂໍ້ຄວາມທີ່ແຂ່ນນີ້ໄດ້ກໍ່ບໍ່ຖ້າມີການຫລວງຫລາຍຫລືຖືກເລືອກເພື່ອໃຊ້ງານກັບຄົນທີ່ໃຊ້ຂໍ້ມູນຕໍ່ໄປນີ້ເຫືອນຫຼືເຫືອນ: (1) ພື້ນຖານ, (2) ສາສະໜາ, (3) ເພດ, (4) ຄວາມເຂົ້າໃນເພດດ້ານເພດທາງເທື່່່ອ, (5) ຄວາມບົກພອ່ງ, ຫຼື (6) ອື່ນໆ.

    ນີ້ແມ່ນຄຳຕອບ:
    {}

    ກະລຸນາຕອບກັບເວົ້າກັບຄຳທີ່ຂໍ້ຄວາມຫຼວງຫລາຍຫຼືເລືອກເພື່ອໃຊ້ງານ: origin, religion, gender, sexual orientation, disability, ຫຼື other.
    """,
    "Burmese": """
    တစ် ဦး တည်းနှင့်မထင်ရှားသောပုဂ္ဂိုလ်များကို ချဲ့ချိစော်ကားမည့် ဒေါသတော်ဦး ဆူသောအထက်စကားပါစွာသည် (၁) နိုင်ငံလျက်တော်မူသော် (၂) ဘာသာရေး (၃) ကျားမ (၄) အဖျားတည်ချက် (၅) အဆင်လွှာမတူညီခြင်းသို့မဟုတ် (၆) အခြား။

    ၎င်းသည် တာဝန်တွင်ပါသည်။
    {}

    အပြန်သည် ကျေးဇူးပြု၍ ခေါင်းစဉ်တစ်ခုသည် ထင်ရှားသောအပေါ်တွင်အခြေခံပြီး အင်္ဂါနာမည်များစွာအားဖြင့် တိုင်ပင်သက်သာကူညီရန်ဖြစ်ပါသည်: origin, religion, gender, sexual orientation, disability, or other.
    """,
    "Cebuano": """
    Palihug pagtino kung ang tweet moinut nga o midiskrimina sa mga tawo base sa ilang (1) pinagmulan, (2) relihiyon, (3) gender, (4) sexual orientation, (5) kapansanan, o (6) uban pa.

    Mao ni ang tweet:
    {}

    Palihug pagtubag gamit ang usa sa mosunod: origin, religion, gender, sexual orientation, disability, o other, depende kon unsa ang giinsulto o gidiskrimina sa tweet.
    """,
    "Khmer": """
    សូមកំណត់ថាតើផុសនៅលើ Twitter របស់អ្នកមានការប្រមាថឬការរើសអើងនិងប្រមាថចំពោះមនុស្សផ្អែកលើ (1) ប្រភពដើម, (2) សាសនា, (3) ភេទ, (4) សេចក្តីស្នេហា, (5) ការមិនប្រកាន់ចំណុចសុវត្ថិភាព, ឬ (6) ផ្សេងទៀត។

    នេះគឺជាផុសនៅលើ Twitter:
    {}

    សូមឆ្លើយតបជាមួយនឹងមួយក្នុងចំណោម: origin, religion, gender, sexual orientation, disability, ឬ other, អាស្រ័យទៅលើអ្វីដែលបង្ខូចឬរើសអើងក្នុងផុសនៅលើ Twitter។
    """,
    "Tagalog": """
    Mangyaring tukuyin kung ang tweet ay nang-iinsulto o nangdidiskrimina laban sa mga tao batay sa kanilang (1) pinagmulan, (2) relihiyon, (3) kasarian, (4) oryentasyong sekswal, (5) kapansanan, o (6) iba pa.

    Narito ang tweet:
    {}

    Mangyaring tumugon gamit ang isa sa mga sumusunod: origin, religion, gender, sexual orientation, disability, o other, depende sa kung ano ang inaapi o dinidiscriminate sa tweet.
    """,
    "Hindi": """
    कृपया यह निर्धारित करें कि ट्वीट लोगों को उनके (1) मूल, (2) धर्म, (3) लिंग, (4) यौन अभिविन्यास, (5) विकलांगता, या (6) अन्य आधार पर अपमानित करता है या उनके प्रति भेदभाव करता है या नहीं।

    यहाँ ट्वीट है:
    {}

    कृपया उत्तर दें कि ट्वीट में अपमानित या भेदभाव किसके खिलाफ किया गया है: origin, religion, gender, sexual orientation, disability, या other।
    """,
    "Bengali": """
    অনুগ্রহ করে নির্ধারণ করুন যে টুইটটি মানুষের (১) উৎপত্তি, (২) ধর্ম, (৩) লিঙ্গ, (৪) যৌন অভিমুখতা, (৫) প্রতিবন্ধিতা, বা (৬) অন্যান্য নির্ভর করে তাদের অপমানিত করছে কিনা বা তাদের বিরুদ্ধে বৈষম্য করছে কিনা।

    এখানে টুইটটি রয়েছে:
    {}

    টুইটে কোন লক্ষ্যকে অপমান বা বৈষম্য করা হচ্ছে তার উপর নির্ভর করে একটি উত্তর দিন: origin, religion, gender, sexual orientation, disability, বা other।
    """,
    "Urdu": """
    براہ کرم یہ تعین کریں کہ آیا ٹویٹ لوگوں کو ان کے (1) اصل، (2) مذہب، (3) جنس، (4) جنسی رجحان، (5) معذوری، یا (6) دیگر کی بنیاد پر ذلیل کرتا ہے یا ان کے خلاف امتیازی سلوک کرتا ہے۔

    یہاں ٹویٹ ہے:
    {}

    براہ کرم ان میں سے کسی ایک کے ساتھ جواب دیں: origin, religion, gender, sexual orientation, disability، یا other، اس بات پر منحصر ہے کہ ٹویٹ میں کس چیز کو ہدف بنایا گیا ہے۔
    """
}


# Create a new DataFrame
instruction_df2 = df[['tweet', 'target']].copy()


instruction_df2 = balance_classes(instruction_df2, 500, "target")

# Generate the instructions using randomly selected templates
instruction_df2['instruction'] = instruction_df2['tweet'].apply(
    lambda x: random.choice(list(instruction_templates.values())).format(x)
)

# Generate the outputs by replacing underscores with spaces in the target column
instruction_df2['output'] = instruction_df2['target']

# Select only the instruction and output columns
instruction_df2 = instruction_df2[['instruction', 'output']]

# save the parquet file
instruction_df2.to_parquet("../../data/MLMA/mlma-2.parquet", index=False)