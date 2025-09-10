# -*- coding: utf-8 -*-
# YAB (YouTube Anti-social Behavior)
# - Paper: https://doi.org/10.1016/j.procs.2018.10.473
# - Data: https://onedrive.live.com/?authkey=!ACDXj_ZNcZPqzy0&id=6EF6951FBF8217F9!105&cid=6EF6951FBF8217F9

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import pandas as pd
import random
from utils.data_utils import balance_classes

df = pd.read_excel("../../data/YAB/LabeledDataset.xlsx", sheet_name="Scenario 1")
# remove rows with missing values in column 'commentText'
df = df.dropna(subset=['commentText'])

# Define the instruction templates in 29 different languages
instruction_templates = {
    "Chinese": """
    在此任务中，您需要预测由三位注释者注释的 YouTube 评论数据集的标签。
    如果三位注释者都认为下面显示的评论具有攻击性，请回答 'P'。否则，请回答 'N'。
    您需要尝试猜测评论的标签。

    标注任务分配给了来自三个不同国籍的三位注释者；一位是伊拉克人，第二位是埃及人。他们两位都来自高密度城市地区。
    第三位来自利比亚的一个低密度城市地区。

    三位注释者的年龄分别为 44 岁、34 岁和 32 岁。
    其中两位完成了第三级教育，一位在信息技术领域，另一位在会计领域，第三位在大学二年级时退学。

    YouTube 评论：
    {}
    评论结束。

    现在，请回答 'P' 或 'N'。
    """,
    "English": """
    In this task, you need to predict the label of a YouTube comment dataset annotated by three annotators.
    If the three annotators all believe that the comment shown below is offensive, respond with 'P'. Otherwise, respond with 'N'.
    You need to try to guess the label of the comment.

    The labeling task was assigned to three annotators from three different nationalities; one is Iraqi, and the second is Egyptian. Both are from high-density urban areas.
    The third person is from a low-density urban area in Libya.

    The ages of the three annotators are 44, 34, and 32, respectively.
    Two of them finished their third-level education, one in information technologies, the other in accounting, and the third one quit university in his second year.

    YouTube comment:
    {}
    Now the comment ends.

    Now, please respond with 'P' or 'N'.
    """,
    "German": """
    In dieser Aufgabe müssen Sie das Label eines YouTube-Kommentardatensatzes vorhersagen, der von drei Annotatoren annotiert wurde.
    Wenn die drei Annotatoren alle glauben, dass der unten gezeigte Kommentar anstößig ist, antworten Sie mit 'P'. Andernfalls antworten Sie mit 'N'.
    Sie müssen versuchen, das Label des Kommentars zu erraten.

    Die Kennzeichnungsaufgabe wurde drei Annotatoren aus drei verschiedenen Nationalitäten zugewiesen; einer ist Iraker und der zweite ist Ägypter. Beide kommen aus städtischen Gebieten mit hoher Dichte.
    Die dritte Person stammt aus einem städtischen Gebiet mit niedriger Dichte in Libyen.

    Das Alter der drei Annotatoren beträgt 44, 34 und 32 Jahre.
    Zwei von ihnen haben ihre Hochschulausbildung abgeschlossen, einer in Informationstechnologie, der andere in Buchhaltung, und der dritte hat die Universität im zweiten Jahr abgebrochen.

    YouTube-Kommentar:
    {}
    Der Kommentar endet jetzt.

    Bitte antworten Sie jetzt mit 'P' oder 'N'.
    """,
    "French": """
    Dans cette tâche, vous devez prédire l'étiquette d'un ensemble de données de commentaires YouTube annoté par trois annotateurs.
    Si les trois annotateurs estiment tous que le commentaire ci-dessous est offensant, répondez par 'P'. Sinon, répondez par 'N'.
    Vous devez essayer de deviner l'étiquette du commentaire.

    La tâche de marquage a été confiée à trois annotateurs de trois nationalités différentes ; l'un est irakien et le deuxième est égyptien. Tous deux sont originaires de zones urbaines à forte densité.
    La troisième personne vient d'une zone urbaine à faible densité en Libye.

    Les âges des trois annotateurs sont respectivement de 44, 34 et 32 ans.
    Deux d'entre eux ont terminé leur enseignement supérieur, l'un en technologies de l'information, l'autre en comptabilité, et le troisième a quitté l'université en deuxième année.

    Commentaire YouTube :
    {}
    Le commentaire se termine maintenant.

    Veuillez répondre maintenant avec 'P' ou 'N'.
    """,
    "Spanish": """
    En esta tarea, debe predecir la etiqueta de un conjunto de datos de comentarios de YouTube anotado por tres anotadores.
    Si los tres anotadores creen que el comentario mostrado a continuación es ofensivo, responda con 'P'. De lo contrario, responda con 'N'.
    Debe intentar adivinar la etiqueta del comentario.

    La tarea de etiquetado fue asignada a tres anotadores de tres nacionalidades diferentes; uno es iraquí y el segundo es egipcio. Ambos son de áreas urbanas de alta densidad.
    La tercera persona es de una zona urbana de baja densidad en Libia.

    Las edades de los tres anotadores son 44, 34 y 32, respectivamente.
    Dos de ellos terminaron su educación de tercer nivel, uno en tecnologías de la información, el otro en contabilidad, y el tercero abandonó la universidad en su segundo año.

    Comentario de YouTube:
    {}
    Ahora termina el comentario.

    Ahora, por favor responda con 'P' o 'N'.
    """,
    "Portuguese": """
    Nesta tarefa, você precisa prever o rótulo de um conjunto de dados de comentários do YouTube anotado por três anotadores.
    Se os três anotadores acreditam que o comentário mostrado abaixo é ofensivo, responda com 'P'. Caso contrário, responda com 'N'.
    Você precisa tentar adivinhar o rótulo do comentário.

    A tarefa de rotulagem foi atribuída a três anotadores de três nacionalidades diferentes; um é iraquiano e o segundo é egípcio. Ambos são de áreas urbanas de alta densidade.
    A terceira pessoa é de uma área urbana de baixa densidade na Líbia.

    As idades dos três anotadores são 44, 34 e 32, respectivamente.
    Dois deles terminaram o ensino superior, um em tecnologias da informação, o outro em contabilidade, e o terceiro abandonou a universidade no segundo ano.

    Comentário no YouTube:
    {}
    Agora o comentário termina.

    Agora, por favor, responda com 'P' ou 'N'.
    """,
    "Italian": """
    In questo compito, devi prevedere l'etichetta di un dataset di commenti di YouTube annotato da tre annotatori.
    Se i tre annotatori ritengono tutti che il commento mostrato di seguito sia offensivo, rispondi con 'P'. Altrimenti, rispondi con 'N'.
    Devi cercare di indovinare l'etichetta del commento.

    Il compito di etichettatura è stato assegnato a tre annotatori di tre nazionalità diverse; uno è iracheno e il secondo è egiziano. Entrambi provengono da aree urbane ad alta densità.
    La terza persona proviene da un'area urbana a bassa densità in Libia.

    Le età dei tre annotatori sono rispettivamente 44, 34 e 32 anni.
    Due di loro hanno completato il terzo livello di istruzione, uno in tecnologie dell'informazione, l'altro in contabilità, e il terzo ha lasciato l'università al secondo anno.

    Commento di YouTube:
    {}
    Ora il commento finisce.

    Ora, per favore rispondi con 'P' o 'N'.
    """,
    "Dutch": """
    In deze taak moet je het label voorspellen van een YouTube-opmerkingen dataset geannoteerd door drie annotators.
    Als de drie annotators allemaal geloven dat de hieronder getoonde opmerking aanstootgevend is, reageer dan met 'P'. Anders reageer met 'N'.
    Je moet proberen het label van de opmerking te raden.

    De labeltaak werd toegewezen aan drie annotators van drie verschillende nationaliteiten; één is Irakees en de tweede is Egyptisch. Beide komen uit dichtbevolkte stedelijke gebieden.
    De derde persoon komt uit een dunbevolkt stedelijk gebied in Libië.

    De leeftijden van de drie annotators zijn respectievelijk 44, 34 en 32 jaar.
    Twee van hen hebben hun derde niveau opleiding voltooid, één in informatietechnologieën, de ander in boekhouding, en de derde stopte met de universiteit in zijn tweede jaar.

    YouTube-opmerking:
    {}
    Nu eindigt de opmerking.

    Reageer nu met 'P' of 'N'.
    """,

    "Russian": """
    В этом задании вам нужно предсказать метку набора данных комментариев YouTube, аннотированных тремя аннотаторами.
    Если все три аннотатора считают, что приведенный ниже комментарий является оскорбительным, ответьте "P". В противном случае ответьте "N".
    Вам нужно попытаться угадать метку комментария.

    Задача по разметке была назначена трем аннотаторам из трех разных национальностей: один из них иракский, второй египетский. Оба они из густонаселенных городских районов.
    Третий человек из малонаселенного городского района в Ливии.

    Возраст трех аннотаторов составляет соответственно 44, 34 и 32 года.
    Двое из них закончили третье образование, один в области информационных технологий, другой в области бухгалтерского учета, а третий бросил университет на втором курсе.

    Комментарий YouTube:
    {}
    Комментарий заканчивается.

    Теперь ответьте "P" или "N".
    """,

    "Czech": """
    V tomto úkolu musíte předpovědět štítek datové sady komentářů YouTube, kterou anotovali tři anotátoři.
    Pokud všichni tři anotátoři věří, že níže uvedený komentář je urážlivý, odpovězte "P". Jinak odpovězte "N".
    Musíte se pokusit odhadnout štítek komentáře.

    Úloha označení byla přiřazena třem anotátorům ze tří různých národností; jeden je Iráčan a druhý je Egypťan. Oba pocházejí z hustě osídlených městských oblastí.
    Třetí osoba je z málo obydlené městské oblasti v Libyi.

    Věk tří anotátorů je 44, 34 a 32 let.
    Dva z nich dokončili třetí stupeň vzdělání, jeden v oblasti informačních technologií, druhý v oblasti účetnictví, a třetí odešel z univerzity ve druhém ročníku.

    Komentář na YouTube:
    {}
    Nyní komentář končí.

    Nyní prosím odpovězte "P" nebo "N".
    """,

    "Polish": """
    W tym zadaniu musisz przewidzieć etykietę zestawu danych komentarzy YouTube, oznaczoną przez trzech annotatorów.
    Jeśli trzech annotatorów uważa, że poniższy komentarz jest obraźliwy, odpowiedz "P". W przeciwnym razie odpowiedz "N".
    Musisz spróbować odgadnąć etykietę komentarza.

    Zadanie etykietowania zostało przypisane trzem annotatorom z trzech różnych narodowości; jeden jest Irakijczykiem, a drugi Egipcjaninem. Obaj pochodzą z gęsto zaludnionych obszarów miejskich.
    Trzecia osoba pochodzi z rzadko zaludnionego obszaru miejskiego w Libii.

    Wiek trzech annotatorów wynosi odpowiednio 44, 34 i 32 lata.
    Dwóch z nich ukończyło trzeci poziom edukacji, jeden w technologii informacyjnej, drugi w rachunkowości, a trzeci porzucił studia w drugim roku.

    Komentarz na YouTube:
    {}
    Teraz komentarz się kończy.

    Teraz odpowiedz "P" lub "N".
    """,
    "Arabic": """
    في هذه المهمة، تحتاج إلى التنبؤ بتصنيف مجموعة بيانات التعليقات على YouTube التي قام بتعليقها ثلاثة معلقين.
    إذا اعتقد المعلقون الثلاثة أن التعليق الموضح أدناه مسيء، فاستجب بـ "P". خلاف ذلك، استجب بـ "N".
    يجب أن تحاول تخمين تصنيف التعليق.

    تم تكليف مهمة التصنيف لثلاثة معلقين من ثلاث جنسيات مختلفة؛ الأول من العراق والثاني من مصر. كلاهما من المناطق الحضرية ذات الكثافة السكانية العالية.
    الشخص الثالث من منطقة حضرية منخفضة الكثافة في ليبيا.

    أعمار المعلقين الثلاثة هي 44، 34، و32 عامًا على التوالي.
    أكمل اثنان منهم التعليم العالي، الأول في تكنولوجيا المعلومات والآخر في المحاسبة، أما الثالث فقد ترك الجامعة في سنته الثانية.

    تعليق YouTube:
    {}
    انتهى الآن التعليق.

    الآن، يرجى الاستجابة بـ "P" أو "N".
    """,
    "Persian": """
    در این کار، باید برچسب مجموعه داده‌های نظرات YouTube را که توسط سه حاشیه‌نویس حاشیه‌نویسی شده است، پیش‌بینی کنید.
    اگر هر سه حاشیه‌نویس معتقدند که نظر نشان داده‌شده در زیر توهین‌آمیز است، با "P" پاسخ دهید. در غیر این صورت، با "N" پاسخ دهید.
    شما باید سعی کنید برچسب نظر را حدس بزنید.

    وظیفه برچسب‌گذاری به سه حاشیه‌نویس از سه ملیت مختلف اختصاص داده شد؛ یکی عراقی و دیگری مصری است. هر دوی آنها از مناطق شهری با تراکم بالا هستند.
    شخص سوم از یک منطقه شهری با تراکم پایین در لیبی است.

    سن سه حاشیه‌نویس به ترتیب 44، 34 و 32 سال است.
    دو نفر از آنها تحصیلات سطح سوم خود را به پایان رسانده‌اند، یکی در زمینه فناوری اطلاعات، دیگری در حسابداری، و سومی در سال دوم دانشگاه را ترک کرد.

    نظر YouTube:
    {}
    اکنون نظر به پایان می‌رسد.

    اکنون لطفاً با "P" یا "N" پاسخ دهید.
    """,

    "Hebrew": """
    במשימה זו, עליך לנבא את תווית מערך הנתונים של תגובות YouTube שהאנוטו על ידי שלושה אנוטרים.
    אם כל שלושת האנוטרים מאמינים שהתגובה המוצגת להלן היא פוגענית, הגיבו עם "P". אחרת, הגיבו עם "N".
    עליך לנסות לנחש את התווית של התגובה.

    משימת הסימון הוקצתה לשלושה אנוטרים משלוש לאומים שונים; אחד הוא עיראקי והשני הוא מצרי. שניהם מגיעים מאזורים עירוניים בעלי צפיפות גבוהה.
    האדם השלישי הוא מאזור עירוני בעל צפיפות נמוכה בלוב.

    גילאי שלושת האנוטרים הם 44, 34 ו-32 בהתאמה.
    שניים מהם סיימו את לימודי השכלה גבוהה, אחד בטכנולוגיות מידע, השני בחשבונאות, והשלישי עזב את האוניברסיטה בשנה השנייה.

    תגובת YouTube:
    {}
    כעת התגובה מסתיימת.

    כעת, אנא הגיבו עם "P" או "N".
    """,

    "Turkish": """
    Bu görevde, üç annotator tarafından açıklanan bir YouTube yorum veri setinin etiketini tahmin etmeniz gerekiyor.
    Üç annotator da aşağıda gösterilen yorumun saldırgan olduğuna inanıyorsa, "P" ile yanıtlayın. Aksi takdirde "N" ile yanıtlayın.
    Yorumun etiketini tahmin etmeye çalışmalısınız.

    Etiketleme görevi, üç farklı milliyetten üç annotator'a atanmıştır; biri Iraklı ve diğeri Mısırlı. İkisi de yüksek yoğunluklu kentsel alanlardan.
    Üçüncü kişi, Libya'da düşük yoğunluklu bir kentsel bölgeden.

    Üç annotator'un yaşları sırasıyla 44, 34 ve 32'dir.
    İkisi üçüncü seviye eğitimini tamamladı, biri bilişim teknolojilerinde, diğeri muhasebede ve üçüncüsü üniversiteyi ikinci yılında bıraktı.

    YouTube yorumu:
    {}
    Şimdi yorum sona eriyor.

    Şimdi, lütfen "P" veya "N" ile yanıtlayın.
    """,

    "Japanese": """
    このタスクでは、3人のアノテーターが注釈を付けたYouTubeコメントデータセットのラベルを予測する必要があります。
    3人のアノテーター全員が、以下に表示されたコメントが攻撃的であると信じている場合、「P」で応答してください。それ以外の場合は、「N」で応答してください。
    コメントのラベルを推測する必要があります。

    ラベリングタスクは、3つの異なる国籍の3人のアノテーターに割り当てられました。1人はイラク人で、2人目はエジプト人です。どちらも高密度の都市部出身です。
    3人目はリビアの低密度の都市部出身です。

    3人のアノテーターの年齢はそれぞれ44歳、34歳、32歳です。
    そのうち2人は第3レベルの教育を修了しており、1人は情報技術、もう1人は会計学を専攻し、3人目は大学2年生で中退しました。

    YouTubeのコメント：
    {}
    コメントが終了します。

    それでは、「P」または「N」で回答してください。
    """,

    "Korean": """
    이 작업에서는 세 명의 주석자가 주석을 달은 YouTube 댓글 데이터 세트의 레이블을 예측해야 합니다.
    세 명의 주석자 모두 아래에 표시된 댓글이 공격적이라고 믿는다면 'P'로 응답하세요. 그렇지 않으면 'N'으로 응답하세요.
    댓글의 레이블을 추측해 보세요.

    레이블 지정 작업은 세 가지 다른 국적의 세 명의 주석자에게 할당되었습니다. 한 명은 이라크인이며 두 번째는 이집트인입니다. 둘 다 인구 밀도가 높은 도시 지역 출신입니다.
    세 번째 사람은 리비아의 저밀도 도시 지역 출신입니다.

    세 명의 주석자의 나이는 각각 44세, 34세, 32세입니다.
    그 중 두 명은 3단계 교육을 마쳤으며, 한 명은 정보 기술, 다른 한 명은 회계학을 전공했으며 세 번째는 대학 2학년 때 중퇴했습니다.

    YouTube 댓글:
    {}
    이제 댓글이 종료됩니다.

    이제 'P' 또는 'N'으로 응답하십시오.
    """,

    "Vietnamese": """
    Trong nhiệm vụ này, bạn cần dự đoán nhãn của tập dữ liệu bình luận trên YouTube được chú thích bởi ba người chú thích.
    Nếu cả ba người chú thích đều tin rằng bình luận hiển thị bên dưới là xúc phạm, hãy trả lời bằng 'P'. Nếu không, hãy trả lời bằng 'N'.
    Bạn cần cố gắng đoán nhãn của bình luận.

    Nhiệm vụ dán nhãn được giao cho ba người chú thích từ ba quốc tịch khác nhau; một là người Iraq và người thứ hai là người Ai Cập. Cả hai đều đến từ các khu vực đô thị có mật độ cao.
    Người thứ ba đến từ một khu vực đô thị có mật độ thấp ở Libya.

    Độ tuổi của ba người chú thích lần lượt là 44, 34 và 32 tuổi.
    Hai người trong số họ đã hoàn thành giáo dục bậc ba, một người học về công nghệ thông tin, người kia học kế toán và người thứ ba bỏ học đại học vào năm thứ hai.

    Bình luận trên YouTube:
    {}
    Bây giờ bình luận kết thúc.

    Bây giờ, vui lòng trả lời bằng 'P' hoặc 'N'.
    """,

    "Thai": """
    ในงานนี้ คุณต้องทำนายป้ายกำกับของชุดข้อมูลความคิดเห็นใน YouTube ที่ได้รับการใส่คำอธิบายประกอบโดยผู้ใส่คำอธิบายประกอบสามคน
    หากผู้ใส่คำอธิบายประกอบทั้งสามคนเชื่อว่าความคิดเห็นที่แสดงด้านล่างนี้เป็นการล่วงละเมิด โปรดตอบกลับด้วย 'P' มิฉะนั้น โปรดตอบกลับด้วย 'N'
    คุณต้องพยายามเดาป้ายกำกับของความคิดเห็น

    งานการติดป้ายกำกับถูกกำหนดให้กับผู้ใส่คำอธิบายประกอบสามคนจากสามสัญชาติที่แตกต่างกัน คนหนึ่งเป็นชาวอิรักและคนที่สองเป็นชาวอียิปต์ ทั้งคู่มาจากพื้นที่เมืองที่มีความหนาแน่นสูง
    คนที่สามมาจากพื้นที่เมืองที่มีความหนาแน่นต่ำในลิเบีย

    อายุของผู้ใส่คำอธิบายประกอบทั้งสามคนคือ 44, 34 และ 32 ปีตามลำดับ
    สองในนั้นจบการศึกษาระดับสามคนหนึ่งในเทคโนโลยีสารสนเทศอีกคนหนึ่งในบัญชีและคนที่สามลาออกจากมหาวิทยาลัยในปีที่สอง

    ความคิดเห็นใน YouTube:
    {}
    ตอนนี้ความคิดเห็นสิ้นสุดลงแล้ว

    ตอนนี้โปรดตอบกลับด้วย 'P' หรือ 'N'
    """,
    "Indonesian": """
    Dalam tugas ini, Anda perlu memprediksi label dari kumpulan data komentar YouTube yang diberi anotasi oleh tiga anotator.
    Jika ketiga anotator percaya bahwa komentar yang ditampilkan di bawah ini bersifat ofensif, jawab dengan 'P'. Jika tidak, jawab dengan 'N'.
    Anda perlu mencoba menebak label komentar.

    Tugas pelabelan diberikan kepada tiga anotator dari tiga kebangsaan yang berbeda; satu orang Irak, dan yang kedua adalah Mesir. Keduanya berasal dari daerah perkotaan yang padat.
    Orang ketiga berasal dari daerah perkotaan dengan kepadatan rendah di Libya.

    Usia ketiga anotator masing-masing adalah 44, 34, dan 32 tahun.
    Dua dari mereka menyelesaikan pendidikan tingkat tiga, satu di bidang teknologi informasi, yang lain di bidang akuntansi, dan yang ketiga berhenti kuliah di tahun kedua.

    Komentar YouTube:
    {}
    Sekarang komentar berakhir.

    Sekarang, silakan jawab dengan 'P' atau 'N'.
    """,
    "Malay": """
    Dalam tugasan ini, anda perlu meramalkan label set data komen YouTube yang dianotasi oleh tiga anotator.
    Jika ketiga-tiga anotator percaya bahawa komen yang ditunjukkan di bawah adalah kesat, balas dengan 'P'. Jika tidak, balas dengan 'N'.
    Anda perlu cuba meneka label komen tersebut.

    Tugas pelabelan diberikan kepada tiga anotator dari tiga kewarganegaraan berbeza; seorang ialah warga Iraq dan yang kedua ialah warga Mesir. Kedua-duanya berasal dari kawasan bandar berpenduduk padat.
    Orang ketiga berasal dari kawasan bandar yang kurang padat di Libya.

    Umur ketiga-tiga anotator ialah 44, 34 dan 32 tahun masing-masing.
    Dua daripada mereka menamatkan pendidikan tahap tiga mereka, seorang dalam teknologi maklumat, seorang lagi dalam perakaunan, dan yang ketiga berhenti universiti pada tahun kedua.

    Komen YouTube:
    {}
    Kini komen tamat.

    Sekarang, sila balas dengan 'P' atau 'N'.
    """,
    "Lao": """
    ໃນວຽກງານນີ້, ເຈົ້າຕ້ອງທຳນາຍປ້າຍຂອງຊຸດຂໍ້ມູນຄວາມຄິດເຫັນ YouTube ທີ່ໄດ້ຮັບການບັນທຶກໂດຍຜູ້ບັນທຶກສາມຄົນ.
    ຖ້າຜູ້ບັນທຶກທັງສາມຄົນເຊື່ອວ່າຄວາມຄິດເຫັນທີ່ສະແດງດ້ານລຸ່ມນີ້ມີຄວາມຫຍາບຄາຍ, ກະລຸນາຕອບກັບ 'P'. ຖ້າບໍ່ແມ່ນ, ກະລຸນາຕອບກັບ 'N'.
    ເຈົ້າຕ້ອງພະຍາຍາມທາຍປ້າຍຂອງຄວາມຄິດເຫັນ.

    ວຽກງານການຕິດປ້າຍໄດ້ຖືກມອບໝາຍໃຫ້ກັບຜູ້ບັນທຶກສາມຄົນຈາກສາມສັນຊາດທີ່ແຕກຕ່າງກັນ; ໜຶ່ງແມ່ນຊາວອິຣັກ, ແລະຄົນທີສອງແມ່ນຊາວອີຢິບ. ທັງສອງຄົນແມ່ນມາຈາກເຂດເມືອງທີ່ມີຄວາມໜາແໜ້ນສູງ.
    ຄົນທີສາມແມ່ນມາຈາກເຂດເມືອງທີ່ມີຄວາມໜາແໜ້ນຕ່ຳຢູ່ລິເບຍ.

    ອາຍຸຂອງຜູ້ບັນທຶກທັງສາມຄົນແມ່ນອາຍຸ 44, 34, ແລະ 32 ຕາມລຳດັບ.
    ສອງຄົນໄດ້ສຳເລັດການສຶກສາຂັ້ນທີສາມ, ໜຶ່ງໃນສາຍວິຊາເທັກໂນໂລຢີຂໍ້ມູນ, ອີກຄົນໜຶ່ງໃນສາຍວິຊາການເບີກບັນຊີ, ແລະຄົນທີສາມໄດ້ລາອອກຈາກມະຫາວິທະຍາໄລໃນປີທີສາງ.

    ຄວາມຄິດເຫັນ YouTube:
    {}
    ຄວາມຄິດເຫັນສິ້ນສຸດແລ້ວ.

    ບັດນີ້, ກະລຸນາຕອບກັບ 'P' ຫຼື 'N'.
    """,
    "Burmese": """
    ဤအလုပ်မှာ သုံး ဦးသော မတ်သားသူများမှ မှတ်သားခဲ့သော YouTube မှတ်ချက် ဒေတာများကို ခန့်မှန်းပေးရန် လိုအပ်သည်။
    မတ်သားသူ သုံး ဦးစလုံးမှ အောက်တွင် ပြထားသော မှတ်ချက်သည် စော်ကားမှုဖြစ်ကြောင်း ယုံကြည်ပါက 'P' ဖြင့် ပြန်ကြားပါ။ မဟုတ်လျှင် 'N' ဖြင့် ပြန်ကြားပါ။
    သင့်အနေဖြင့် မှတ်ချက်၏ အမှတ်အသားကို ခန့်မှန်းရန် လိုအပ်ပါသည်။

    စံပြနေစဉ်တွင် သုံး မတ်သားသူများမှ သုံးမျိုးသော နိုင်ငံသားများမှဖြစ်ပြီး၊ တစ်ဦးမှာ ဣရတ်သားနှင့် ဒုတိယကျော်သည် အီဂျစ်သားများ ဖြစ်သည်။ ၎င်းတို့နှစ်ဦးလုံးသည် မြို့ပြရပ်ကွက်များမှ ဖြစ်သည်။
    တတိယနေရာမှလူသည် လိဘျားရှိ အများဆုံးပြည်သူအများနှင့်ကွာဝေးသော မြို့ပြရပ်ကွက်မှ ဖြစ်သည်။

    မတ်သားသူ သုံးဦး၏ အသက်အရွယ်မှာ 44 နှစ်၊ 34 နှစ်နှင့် 32 နှစ် တစ်ဦးချင်းစီဖြစ်ပါသည်။
    ၎င်းတို့အနက် နှစ်ဦးသည် အဆင့်သုံးပညာရေးကို ပြီးဆုံးခဲ့ပြီး၊ တစ်ဦးမှာ သတင်းအချက်အလက်နည်းပညာများတွင်၊ နောက်တစ်ဦးမှာ စာရင်းကိုင်မှုမှာ ဖြစ်သည်၊ တတိယသောသူမှာ တက္ကသိုလ်ကို ဒုတိယနှစ်တွင် ထွက်ခဲ့ပါသည်။

    YouTube မှတ်ချက်:
    {}
    ယခုမှတ်ချက်ပြီးဆုံးပါပြီ။

    ယခုမှ 'P' သို့မဟုတ် 'N' ဖြင့် ပြန်ကြားပါ။
    """,
    "Cebuano": """
    Sa kini nga buluhaton, kinahanglan nimong tag-anon ang label sa usa ka YouTube nga komentaryo nga dataset nga gi-anotohan sa tulo ka anotator.
    Kung ang tulo ka anotator nagtuo nga ang komentaryo nga gipakita sa ubos maka-insulto, tubaga og 'P'. Kung dili, tubaga og 'N'.
    Kinahanglan nimong sulayan ang pagtag-an sa label sa komentaryo.

    Ang tahas sa pag-label gihatag sa tulo ka mga anotator gikan sa tulo ka lain-laing nasyonalidad; usa ka Iraqi, ug ang ikaduha usa ka Ehiptohanon. Ang duha gikan sa taas nga density nga lugar sa urban.
    Ang ikatulo nga tawo gikan sa ubos nga density nga lugar sa urban sa Libya.

    Ang mga edad sa tulo ka mga anotator mao ang 44, 34, ug 32 nga magkasunod.
    Duha sa kanila ang nakatapos sa ilang ikatulo nga lebel sa edukasyon, usa sa mga teknolohiya sa impormasyon, ang lain sa accounting, ug ang ikatulo ang mibiya sa unibersidad sa iyang ikaduhang tuig.

    Komentaryo sa YouTube:
    {}
    Karon ang komentaryo nagtapus na.

    Karon, palihug tubaga og 'P' o 'N'.
    """,
    "Khmer": """
    ក្នុងការងារនេះ អ្នកត្រូវប៉ាន់ស្មានស្លាកនៃសំណុំទិន្នន័យមតិយោបល់ YouTube ដែលត្រូវបានបន្ថែមដោយអ្នកបន្ថែមមតិបីនាក់។
    ប្រសិនបើអ្នកបន្ថែមមតិទាំងបីនាក់ជឿថាមតិយោបល់ដែលបានបង្ហាញខាងក្រោមគឺសព្វសាធារណៈ សូមឆ្លើយតបជាមួយ 'P'។ ប្រសិនបើមិនដូច្នេះទេ សូមឆ្លើយតបជាមួយ 'N'។
    អ្នកត្រូវតែព្យាយាមទាយស្លាកនៃមតិយោបល់។

    កិច្ចការបញ្ចប់ត្រូវបានផ្ដល់ឱ្យអ្នកបន្ថែមមតិបីនាក់ពីសញ្ជាតិផ្សេងៗគ្នាបីរបស់អ្នកបន្ថែមមតិម្នាក់គឺជាជនជាតិអ៊ីរ៉ាក់ ហើយទីពីរក្រុមគឺជាជនជាតិអេហ្ស៊ីប។ ទាំងពីរបានមកពីតំបន់ទីក្រុងដែលមានបរិមាណខ្ពស់។
    មនុស្សទីបីគឺមកពីតំបន់ទីក្រុងដែលមានបរិមាណទាបនៅលីប៊ី។

    អាយុអ្នកបន្ថែមមតិបីនាក់គឺ 44, 34, និង 32 ដែលជាលំដាប់។
    ពួកគេពីរនាក់បានបញ្ចប់ការអប់រំកម្រិតទីបីម្នាក់ក្នុងវិស័យបច្ចេកវិទ្យាព័ត៌មានម្នាក់ទៀតក្នុងវិស័យគណនេយ្យ ហើយមនុស្សទីបីបានចាកចេញពីសាកលវិទ្យាល័យនៅឆ្នាំទីពីរ។

    មតិយោបល់ YouTube៖
    {}
    ឥឡូវនេះមតិយោបល់បញ្ចប់។

    ឥឡូវសូមឆ្លើយតបជាមួយ 'P' ឬ 'N'។
    """,
    "Tagalog": """
    Sa gawain na ito, kailangan mong hulaan ang label ng isang dataset ng mga komento sa YouTube na inianotahan ng tatlong anotador.
    Kung naniniwala ang tatlong anotador na ang komento na ipinakita sa ibaba ay nakakasakit, tumugon gamit ang 'P'. Kung hindi, tumugon gamit ang 'N'.
    Kailangan mong subukang hulaan ang label ng komento.

    Ang gawain ng pagbibigay ng label ay ibinigay sa tatlong anotador mula sa tatlong iba't ibang nasyonalidad; ang isa ay Iraqi at ang pangalawa ay Egyptian. Pareho silang mula sa mga lugar na urban na may mataas na densidad.
    Ang pangatlong tao ay mula sa isang lugar na urban na may mababang densidad sa Libya.

    Ang mga edad ng tatlong anotador ay 44, 34, at 32 ayon sa pagkakabanggit.
    Dalawa sa kanila ang nakatapos ng kanilang pangatlong antas ng edukasyon, ang isa ay sa mga teknolohiya ng impormasyon, ang isa pa sa accounting, at ang pangatlo ay huminto sa unibersidad sa kanyang ikalawang taon.

    Komento sa YouTube:
    {}
    Ngayon ay nagtatapos ang komento.

    Ngayon, mangyaring tumugon gamit ang 'P' o 'N'.
    """,
    "Hindi": """
    इस कार्य में, आपको तीन अनोटेटरों द्वारा एनोटेट की गई एक YouTube टिप्पणी डेटासेट के लेबल की भविष्यवाणी करनी होगी।
    यदि तीनों अनोटेटर मानते हैं कि नीचे दिखाया गया टिप्पणी आक्रामक है, तो 'P' के साथ उत्तर दें। अन्यथा, 'N' के साथ उत्तर दें।
    आपको टिप्पणी के लेबल का अनुमान लगाने का प्रयास करना होगा।

    लेबलिंग कार्य को तीन अलग-अलग राष्ट्रीयताओं के तीन अनोटेटरों को सौंपा गया था; एक इराकी है, और दूसरा मिस्र का है। दोनों उच्च घनत्व वाले शहरी क्षेत्रों से हैं।
    तीसरा व्यक्ति लीबिया के एक निम्न घनत्व वाले शहरी क्षेत्र से है।

    तीन अनोटेटरों की आयु क्रमशः 44, 34, और 32 वर्ष है।
    उनमें से दो ने अपनी तीसरी स्तर की शिक्षा पूरी की, एक सूचना प्रौद्योगिकी में, दूसरा लेखांकन में, और तीसरे ने अपने दूसरे वर्ष में विश्वविद्यालय छोड़ दिया।

    YouTube टिप्पणी:
    {}
    अब टिप्पणी समाप्त हो गई है।

    अब, कृपया 'P' या 'N' के साथ उत्तर दें।
    """,
    "Bengali": """
    এই কাজে, আপনাকে তিনজন এনোটেটর দ্বারা এনোটেট করা একটি YouTube মন্তব্য ডেটাসেটের লেবেলটি পূর্বানুমান করতে হবে।
    যদি তিনজন এনোটেটর বিশ্বাস করেন যে নীচে প্রদর্শিত মন্তব্যটি আপত্তিকর, তবে 'P' দিয়ে উত্তর দিন। অন্যথায়, 'N' দিয়ে উত্তর দিন।
    আপনাকে মন্তব্যের লেবেলটি অনুমান করার চেষ্টা করতে হবে।

    লেবেলিংয়ের কাজটি তিনটি বিভিন্ন জাতীয়তার তিনজন এনোটেটরকে দেওয়া হয়েছিল; একজন ইরাকি, এবং দ্বিতীয়জন মিশরীয়। তারা দুজনেই উচ্চ ঘনত্বযুক্ত শহুরে এলাকা থেকে।
    তৃতীয় ব্যক্তি লিবিয়ার একটি নিম্ন ঘনত্বযুক্ত শহুরে এলাকা থেকে।

    তিনজন এনোটেটরের বয়স যথাক্রমে 44, 34, এবং 32 বছর।
    তাদের মধ্যে দুজন তাদের তৃতীয় স্তরের শিক্ষা শেষ করেছেন, একজন তথ্য প্রযুক্তিতে, অন্যজন হিসাবরক্ষণে, এবং তৃতীয় জন তার দ্বিতীয় বছরে বিশ্ববিদ্যালয় ছেড়ে দিয়েছে।

    YouTube মন্তব্য:
    {}
    এখন মন্তব্য শেষ।

    এখন, অনুগ্রহ করে 'P' বা 'N' দিয়ে উত্তর দিন।
    """,
    "Urdu": """
    اس کام میں، آپ کو تین انوٹیشن کرنے والوں کے ذریعہ انوٹیشن کردہ یوٹیوب تبصرہ ڈیٹاسیٹ کے لیبل کی پیش گوئی کرنی ہوگی۔
    اگر تینوں انوٹیشن کرنے والے یہ مانتے ہیں کہ نیچے دکھایا گیا تبصرہ توہین آمیز ہے، تو 'P' کے ساتھ جواب دیں۔ بصورت دیگر، 'N' کے ساتھ جواب دیں۔
    آپ کو تبصرے کا لیبل تخمینہ لگانے کی کوشش کرنی ہوگی۔

    لیبلنگ کا کام تین مختلف قومیتوں کے تین انوٹیشن کرنے والوں کو سونپا گیا تھا؛ ایک عراقی ہے، اور دوسرا مصری۔ دونوں ہائی ڈینسٹی اربن علاقوں سے ہیں۔
    تیسرا شخص لیبیا کے ایک کم ڈینسٹی اربن علاقے سے ہے۔

    تین انوٹیشن کرنے والوں کی عمر بالترتیب 44، 34، اور 32 سال ہے۔
    ان میں سے دو نے اپنی تیسری سطح کی تعلیم مکمل کی، ایک نے انفارمیشن ٹیکنالوجیز میں، دوسرے نے اکاؤنٹنگ میں، اور تیسرے نے اپنی دوسری سال میں یونیورسٹی چھوڑ دی۔

    یوٹیوب تبصرہ:
    {}
    اب تبصرہ ختم ہوتا ہے۔

    اب، براہ کرم 'P' یا 'N' کے ساتھ جواب دیں۔
    """
}

# Create a new DataFrame with the required columns
instruction_df = df[['commentText', 'Label']].copy()

# Generate the instructions using randomly selected templates
instruction_df['instruction'] = instruction_df['commentText'].apply(
    lambda x: random.choice(list(instruction_templates.values())).format(x)
)

# Keep the 'output' column with English labels 'P' or 'N'
instruction_df['output'] = instruction_df['Label']

# Keep only the 'instruction' and 'output' columns
instruction_df = instruction_df[['instruction', 'output']]

instruction_df = balance_classes(instruction_df, 2500, 'output')
# Save the DataFrame to a Parquet file
instruction_df.to_parquet("../../data/YAB/YAB.parquet", index=False)