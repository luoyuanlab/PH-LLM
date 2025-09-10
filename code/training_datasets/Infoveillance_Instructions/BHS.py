# -*- coding: utf-8 -*-
# Bengali Hate Speech

# - Paper: https://arxiv.org/pdf/2012.09686.pdf
# - Data: https://www.kaggle.com/naurosromim/bengali-hate-speech-dataset


import pandas as pd
import random
from sklearn.utils import resample

df = pd.read_csv('../../data/BHS/Bengali hate speech.csv')

# Define the instruction templates in different languages
instruction_templates = {
    "Chinese": """
    请确定以下YouTube/Facebook评论是否为仇恨言论。
    本数据集中仇恨言论的标准如下：
    仇恨言论是指将一个人或多个群体或社区非人化的句子。非人化可以通过将某人或社区与昆虫、物体或罪犯进行比较来实现。它还可以通过根据种族、性别、身体和精神残疾来针对某人。
    一个句子可能包含俚语或不当语言，但除非这些俚语使某个人或社区非人化，否则我们不认为它是仇恨言论。
    如果评论支持明显使某个人或社区非人化的观点，则该评论被视为仇恨言论。
    如果需要更多背景信息来理解某条评论是否为仇恨言论，那么我们假设它不是仇恨言论。
    仇恨言论评论所持立场的对错无关紧要。

    现在评论如下：
    {}
    现在评论结束。

    如果评论是仇恨言论，请回答 'yes'，如果不是，请回答 'no'。
    """,
    "English": """
    Please determine if the YouTube/Facebook comment below is hate speech or not.
    The standard of hate speech in this dataset is as follows:
    Hate speech is a sentence that dehumanizes one or multiple persons or a community. Dehumanizing can be done by comparing the person or community to an insect, object, or a criminal. It can also be done by targeting a person based on their race, gender, physical, or mental disability.
    A sentence might contain slang or inappropriate language. But unless that slang dehumanizes a person or community, we did not consider it to be hate speech.
    If a comment supports an idea that clearly dehumanizes a person or a community, it is considered hate speech.
    If additional context is needed to understand that a comment is hate speech, then we assume it's not hate speech.
    It does not matter if the stance that a hate speech comment takes is right or wrong.

    Now here is the comment:
    {}
    Now the comment ends.

    Please respond with 'yes' if the comment is hate speech, and 'no' if it is not.
    """,
    "German": """
    Bitte bestimmen Sie, ob der unten stehende YouTube/Facebook-Kommentar Hassrede ist oder nicht.
    Der Standard für Hassrede in diesem Datensatz lautet wie folgt:
    Hassrede ist ein Satz, der eine oder mehrere Personen oder eine Gemeinschaft entmenschlicht. Entmenschlichung kann erfolgen, indem die Person oder Gemeinschaft mit einem Insekt, einem Objekt oder einem Kriminellen verglichen wird. Es kann auch dadurch geschehen, dass eine Person aufgrund ihrer Rasse, ihres Geschlechts, ihrer körperlichen oder geistigen Behinderung ins Visier genommen wird.
    Ein Satz kann Slang oder unangemessene Sprache enthalten, aber solange dieser Slang eine Person oder Gemeinschaft nicht entmenschlicht, betrachten wir ihn nicht als Hassrede.
    Wenn ein Kommentar eine Idee unterstützt, die eindeutig eine Person oder Gemeinschaft entmenschlicht, wird dies als Hassrede betrachtet.
    Wenn zusätzlicher Kontext erforderlich ist, um zu verstehen, dass ein Kommentar eine Hassrede ist, gehen wir davon aus, dass es sich nicht um Hassrede handelt.
    Es spielt keine Rolle, ob die Haltung, die ein Hasskommentar einnimmt, richtig oder falsch ist.

    Hier ist der Kommentar:
    {}
    Jetzt endet der Kommentar.

    Bitte antworten Sie mit 'yes', wenn der Kommentar Hassrede ist, und mit 'no', wenn dies nicht der Fall ist.
    """,
    "French": """
    Veuillez déterminer si le commentaire YouTube/Facebook ci-dessous est un discours de haine ou non.
    La norme du discours de haine dans ce jeu de données est la suivante :
    Un discours de haine est une phrase qui déshumanise une ou plusieurs personnes ou une communauté. La déshumanisation peut se faire en comparant la personne ou la communauté à un insecte, un objet ou un criminel. Elle peut également se faire en ciblant une personne en fonction de sa race, de son sexe, ou de son handicap physique ou mental.
    Une phrase peut contenir de l'argot ou un langage inapproprié, mais à moins que cet argot ne déshumanise une personne ou une communauté, nous ne le considérons pas comme un discours de haine.
    Si un commentaire soutient une idée qui déshumanise clairement une personne ou une communauté, il est considéré comme un discours de haine.
    Si un contexte supplémentaire est nécessaire pour comprendre qu'un commentaire est un discours de haine, alors nous supposons que ce n'est pas un discours de haine.
    Peu importe si la position qu'un discours de haine prend est bonne ou mauvaise.

    Voici le commentaire:
    {}
    Maintenant, le commentaire se termine.

    Veuillez répondre par 'yes' si le commentaire est un discours de haine, et par 'no' s'il ne l'est pas.
    """,
    "Spanish": """
    Por favor, determine si el comentario de YouTube/Facebook a continuación es un discurso de odio o no.
    El estándar de discurso de odio en este conjunto de datos es el siguiente:
    Un discurso de odio es una oración que deshumaniza a una o varias personas o a una comunidad. La deshumanización se puede hacer comparando a la persona o la comunidad con un insecto, objeto o un criminal. También se puede hacer atacando a una persona por su raza, género, discapacidad física o mental.
    Una oración puede contener jerga o lenguaje inapropiado, pero a menos que esa jerga deshumanice a una persona o comunidad, no la consideramos un discurso de odio.
    Si un comentario apoya una idea que claramente deshumaniza a una persona o comunidad, se considera discurso de odio.
    Si se necesita contexto adicional para entender que un comentario es un discurso de odio, entonces asumimos que no es un discurso de odio.
    No importa si la postura que toma un comentario de discurso de odio es correcta o incorrecta.

    Ahora aquí está el comentario:
    {}
    Ahora el comentario termina.

    Por favor, responda con 'yes' si el comentario es discurso de odio, y 'no' si no lo es.
    """,
    "Portuguese": """
    Por favor, determine se o comentário do YouTube/Facebook abaixo é discurso de ódio ou não.
    O padrão de discurso de ódio neste conjunto de dados é o seguinte:
    Discurso de ódio é uma frase que desumaniza uma ou várias pessoas ou uma comunidade. A desumanização pode ser feita comparando a pessoa ou comunidade a um inseto, objeto ou criminoso. Também pode ser feita atacando uma pessoa com base em sua raça, gênero, deficiência física ou mental.
    Uma frase pode conter gírias ou linguagem inadequada, mas a menos que essa gíria desumanize uma pessoa ou comunidade, não a consideramos discurso de ódio.
    Se um comentário apoia uma ideia que claramente desumaniza uma pessoa ou comunidade, isso é considerado discurso de ódio.
    Se for necessário contexto adicional para entender que um comentário é discurso de ódio, então assumimos que não é discurso de ódio.
    Não importa se a posição que um comentário de discurso de ódio assume está certa ou errada.

    Aqui está o comentário:
    {}
    Agora o comentário termina.

    Por favor, responda com 'yes' se o comentário for discurso de ódio, e 'no' se não for.
    """,
    "Italian": """
    Si prega di determinare se il commento YouTube/Facebook di seguito è un discorso di odio o no.
    Lo standard di discorso di odio in questo set di dati è il seguente:
    Il discorso di odio è una frase che disumanizza una o più persone o una comunità. La disumanizzazione può essere fatta confrontando la persona o la comunità con un insetto, un oggetto o un criminale. Può anche essere fatto prendendo di mira una persona in base alla sua razza, sesso, disabilità fisica o mentale.
    Una frase potrebbe contenere gergo o linguaggio inappropriato, ma a meno che quel gergo non disumanizzi una persona o una comunità, non lo consideriamo un discorso di odio.
    Se un commento supporta un'idea che chiaramente disumanizza una persona o una comunità, è considerato discorso di odio.
    Se è necessario un contesto aggiuntivo per capire che un commento è un discorso di odio, allora assumiamo che non sia un discorso di odio.
    Non importa se la posizione che assume un commento di discorso di odio è giusta o sbagliata.

    Ora ecco il commento:
    {}
    Ora il commento finisce.

    Si prega di rispondere con 'yes' se il commento è un discorso di odio, e 'no' se non lo è.
    """,
    "Dutch": """
    Bepaal alstublieft of de onderstaande YouTube/Facebook-opmerking haatzaaiende taal is of niet.
    De norm voor haatzaaiende uitlatingen in deze dataset is als volgt:
    Haatspraak is een zin die een of meerdere personen of een gemeenschap ontmenselijkt. Ontmenselijking kan worden gedaan door de persoon of gemeenschap te vergelijken met een insect, object of een crimineel. Het kan ook worden gedaan door een persoon te targeten op basis van hun ras, geslacht, fysieke en mentale handicap.
    Een zin kan jargon of ongepast taalgebruik bevatten, maar tenzij dat jargon een persoon of gemeenschap ontmenselijkt, beschouwden we het niet als haatzaaiende taal.
    Als een opmerking een idee ondersteunt dat duidelijk een persoon of gemeenschap ontmenselijkt, wordt dit beschouwd als haatzaaiende taal.
    Als aanvullende context nodig is om te begrijpen dat een opmerking haatzaaiende taal is, gaan we ervan uit dat het geen haatzaaiende taal is.
    Het maakt niet uit of het standpunt dat een haatzaaiende opmerking inneemt juist of onjuist is.

    Hier is de opmerking:
    {}
    Nu eindigt de opmerking.

    Beantwoord met 'yes' als de opmerking haatzaaiende taal is, en met 'no' als dat niet het geval is.
    """,
    "Russian": """
    Пожалуйста, определите, является ли комментарий YouTube/Facebook ниже ненавистнической речью или нет.
    Стандарт ненавистнической речи в этом наборе данных следующий:
    Ненавистническая речь — это предложение, которое дехуманизирует одного или нескольких человек или сообщество. Дехуманизация может быть осуществлена путем сравнения человека или сообщества с насекомым, предметом или преступником. Это также может быть сделано путем нападок на человека по признаку его расы, пола, физической и умственной неполноценности.
    В предложении может содержаться сленг или нецензурная лексика, но если этот сленг не дехуманизирует человека или сообщество, мы не считаем его ненавистнической речью.
    Если комментарий поддерживает идею, которая явно дехуманизирует человека или сообщество, это считается ненавистнической речью.
    Если для понимания того, что комментарий является ненавистнической речью, требуется дополнительный контекст, то мы предполагаем, что это не ненавистническая речь.
    Не имеет значения, правильна или ошибочна точка зрения, которой придерживается ненавистнический комментарий.

    Вот комментарий:
    {}
    Теперь комментарий заканчивается.

    Пожалуйста, ответьте 'yes', если комментарий является ненавистнической речью, и 'no', если это не так.
    """,
    "Czech": """
    Určete prosím, zda níže uvedený komentář na YouTube/Facebook obsahuje nenávistný projev nebo ne.
    Standard pro nenávistný projev v tomto datovém souboru je následující:
    Nenávistný projev je věta, která dehumanizuje jednu nebo více osob nebo komunitu. Dehumanizace může být provedena porovnáním osoby nebo komunity s hmyzem, předmětem nebo zločincem. Může se také jednat o cílení na osobu na základě její rasy, pohlaví, fyzického nebo duševního postižení.
    Věta může obsahovat slang nebo nevhodný jazyk, ale pokud tento slang nedehumanizuje osobu nebo komunitu, nepovažujeme to za nenávistný projev.
    Pokud komentář podporuje myšlenku, která jasně dehumanizuje osobu nebo komunitu, je to považováno za nenávistný projev.
    Pokud je k pochopení toho, že komentář je nenávistný projev, zapotřebí další kontext, předpokládáme, že se nejedná o nenávistný projev.
    Nezáleží na tom, zda je stanovisko, které zastává nenávistný komentář, správné nebo špatné.

    Zde je komentář:
    {}
    Nyní komentář končí.

    Prosím, odpovězte 'yes', pokud je komentář nenávistný projev, a 'no', pokud není.
    """,
    "Polish": """
    Proszę określić, czy poniższy komentarz na YouTube/Facebook to mowa nienawiści, czy nie.
    Standard mowy nienawiści w tym zbiorze danych jest następujący:
    Mowa nienawiści to zdanie, które dehumanizuje jedną lub więcej osób lub społeczność. Dehumanizacja może polegać na porównaniu osoby lub społeczności do owada, przedmiotu lub przestępcy. Może również polegać na atakowaniu osoby na podstawie jej rasy, płci, niepełnosprawności fizycznej lub psychicznej.
    Zdanie może zawierać slang lub nieodpowiedni język, ale dopóki ten slang nie dehumanizuje osoby lub społeczności, nie uważamy tego za mowę nienawiści.
    Jeśli komentarz wspiera ideę, która wyraźnie dehumanizuje osobę lub społeczność, jest to uznawane za mowę nienawiści.
    Jeśli do zrozumienia tego, że komentarz jest mową nienawiści, potrzebny jest dodatkowy kontekst, zakładamy, że nie jest to mowa nienawiści.
    Nie ma znaczenia, czy stanowisko, które zajmuje komentarz z mową nienawiści, jest słuszne, czy błędne.

    Oto komentarz:
    {}
    Teraz komentarz się kończy.

    Proszę odpowiedzieć 'yes', jeśli komentarz to mowa nienawiści, a 'no', jeśli nie.
    """,
    "Arabic": """
    يرجى تحديد ما إذا كان التعليق أدناه على YouTube/Facebook هو خطاب كراهية أم لا.
    معيار خطاب الكراهية في مجموعة البيانات هذه هو كما يلي:
    خطاب الكراهية هو جملة تقلل من إنسانية شخص أو عدة أشخاص أو مجتمع. يمكن التقليل من الإنسانية عن طريق مقارنة الشخص أو المجتمع بحشرة أو شيء أو مجرم. يمكن أيضًا التقليل من الإنسانية من خلال استهداف الشخص بناءً على عرقه أو جنسه أو إعاقته الجسدية أو العقلية.
    قد تحتوي الجملة على لغة عامية أو لغة غير لائقة، ولكن ما لم تقلل تلك اللغة العامية من إنسانية الشخص أو المجتمع، فإننا لا نعتبرها خطاب كراهية.
    إذا كان التعليق يدعم فكرة تقلل بشكل واضح من إنسانية شخص أو مجتمع، فيُعتبر خطاب كراهية.
    إذا كانت هناك حاجة إلى سياق إضافي لفهم أن التعليق هو خطاب كراهية، فإننا نفترض أنه ليس خطاب كراهية.
    لا يهم ما إذا كان الموقف الذي يتخذه تعليق خطاب الكراهية صحيحًا أم خاطئًا.

    الآن هنا هو التعليق:
    {}
    الآن ينتهي التعليق.

    يرجى الرد بـ 'yes' إذا كان التعليق هو خطاب كراهية، وبـ 'no' إذا لم يكن كذلك.
    """,
    "Persian": """
    لطفاً مشخص کنید که آیا نظر زیر در یوتیوب/فیس‌بوک گفتار نفرت‌آمیز است یا خیر.
    استاندارد گفتار نفرت‌آمیز در این مجموعه داده به شرح زیر است:
    گفتار نفرت‌آمیز جمله‌ای است که یک یا چند شخص یا جامعه را غیرانسانی می‌کند. غیرانسانی کردن می‌تواند با مقایسه شخص یا جامعه با یک حشره، شیء یا مجرم انجام شود. همچنین می‌تواند با هدف قرار دادن شخصی بر اساس نژاد، جنسیت، ناتوانی جسمی و ذهنی او انجام شود.
    یک جمله ممکن است شامل زبان عامیانه یا زبان نامناسب باشد. اما مگر اینکه آن زبان عامیانه شخص یا جامعه‌ای را غیرانسانی کند، ما آن را گفتار نفرت‌آمیز نمی‌دانیم.
    اگر نظری از ایده‌ای حمایت کند که به وضوح یک شخص یا جامعه را غیرانسانی می‌کند، آن را گفتار نفرت‌آمیز می‌دانیم.
    اگر برای درک اینکه یک نظر گفتار نفرت‌آمیز است به زمینه بیشتری نیاز است، فرض می‌کنیم که گفتار نفرت‌آمیز نیست.
    مهم نیست که موضعی که یک نظر در گفتار نفرت‌آمیز می‌گیرد درست یا غلط باشد.

    حالا این نظر است:
    {}
    حالا نظر تمام شد.

    لطفاً پاسخ دهید 'yes' اگر نظر گفتار نفرت‌آمیز است، و 'no' اگر نیست.
    """,
    "Hebrew": """
    אנא קבע אם התגובה ב-YouTube/Facebook שלהלן היא נאום שנאה או לא.
    הסטנדרט של נאום שנאה במאגר נתונים זה הוא כדלקמן:
    נאום שנאה הוא משפט שמוריד מערכו של אדם אחד או יותר או קהילה. הפחתת ערך יכולה להיעשות על ידי השוואת האדם או הקהילה לחרק, חפץ או פושע. זה יכול להיעשות גם על ידי מיקוד אדם על בסיס הגזע, המגדר, הנכות הפיזית והנפשית שלו.
    משפט עשוי להכיל סלנג או שפה לא הולמת, אבל אלא אם כן הסלנג הזה מפחית מערכו של אדם או קהילה, לא ראינו בכך נאום שנאה.
    אם תגובה תומכת ברעיון שמפחית באופן ברור מערכו של אדם או קהילה, היא נחשבת לנאום שנאה.
    אם יש צורך בהקשר נוסף כדי להבין שתגובה היא נאום שנאה, אנו מניחים שזה לא נאום שנאה.
    לא משנה אם העמדה שנוקטת תגובה לנאום שנאה נכונה או לא.

    עכשיו הנה התגובה:
    {}
    עכשיו התגובה מסתיימת.

    אנא ענה 'yes' אם התגובה היא נאום שנאה, ו-'no' אם לא.
    """,
    "Turkish": """
    Lütfen aşağıdaki YouTube/Facebook yorumunun nefret söylemi olup olmadığını belirleyin.
    Bu veri kümesindeki nefret söylemi standardı aşağıdaki gibidir:
    Nefret söylemi, bir veya daha fazla kişiyi veya bir topluluğu insanlıktan çıkartan bir cümledir. İnsanlıktan çıkarma, kişiyi veya topluluğu bir böcek, nesne veya suçlu ile karşılaştırarak yapılabilir. Ayrıca, bir kişiyi ırkı, cinsiyeti, fiziksel ve zihinsel engelliliği temelinde hedef alarak yapılabilir.
    Bir cümlede argo veya uygunsuz dil bulunabilir, ancak bu argo bir kişiyi veya topluluğu insanlıktan çıkarmadıkça, bunu nefret söylemi olarak kabul etmedik.
    Bir yorum, bir kişiyi veya topluluğu açıkça insanlıktan çıkaran bir fikri destekliyorsa, bu nefret söylemi olarak kabul edilir.
    Bir yorumun nefret söylemi olduğunu anlamak için ek bağlama ihtiyaç varsa, bunun nefret söylemi olmadığını varsayıyoruz.
    Bir nefret söylemi yorumunun aldığı duruşun doğru veya yanlış olup olmadığı önemli değildir.

    Şimdi yorum burada:
    {}
    Şimdi yorum bitiyor.

    Lütfen yorum nefret söylemi ise 'yes', değilse 'no' ile cevap verin.
    """,
    "Japanese": """
    以下のYouTube/Facebookのコメントがヘイトスピーチかどうかを判断してください。
    このデータセットにおけるヘイトスピーチの基準は次のとおりです:
    ヘイトスピーチとは、1人または複数の人やコミュニティを非人間化する文章のことです。非人間化は、人やコミュニティを昆虫、物、または犯罪者と比較することで行われることがあります。また、人種、性別、身体的および精神的障害に基づいて人をターゲットにすることでも行われます。
    文にスラングや不適切な言葉が含まれていることがあります。ただし、そのスラングが人やコミュニティを非人間化しない限り、それをヘイトスピーチと見なすことはありません。
    コメントが人やコミュニティを明確に非人間化する考えを支持している場合、それはヘイトスピーチと見なされます。
    コメントがヘイトスピーチであることを理解するために追加の文脈が必要な場合、私たちはそれをヘイトスピーチではないと想定します。
    ヘイトスピーチのコメントが取る立場が正しいか間違っているかは重要ではありません。

    では、コメントはこちらです:
    {}
    これでコメントは終了です。

    コメントがヘイトスピーチである場合は「yes」と回答し、そうでない場合は「no」と回答してください。
    """,
    "Korean": """
    아래 YouTube/Facebook 댓글이 혐오 발언인지 아닌지 판단해 주세요.
    이 데이터셋에서 혐오 발언의 기준은 다음과 같습니다:
    혐오 발언은 한 명 이상의 사람이나 커뮤니티를 비인간화하는 문장입니다. 비인간화는 사람이나 커뮤니티를 곤충, 물건 또는 범죄자와 비교하여 수행할 수 있습니다. 또한 인종, 성별, 신체적 및 정신적 장애를 기반으로 사람을 대상으로 하는 방식으로 수행할 수 있습니다.
    문장에는 슬랭 또는 부적절한 언어가 포함될 수 있습니다. 그러나 그 슬랭이 사람이나 커뮤니티를 비인간화하지 않는 한, 우리는 그것을 혐오 발언으로 간주하지 않았습니다.
    댓글이 사람이나 커뮤니티를 명확하게 비인간화하는 아이디어를 지지하는 경우, 이는 혐오 발언으로 간주됩니다.
    댓글이 혐오 발언인지 이해하기 위해 추가 컨텍스트가 필요한 경우, 우리는 그것이 혐오 발언이 아니라고 가정합니다.
    혐오 발언 댓글이 취하는 입장이 옳든 그르든 상관없습니다.

    이제 댓글이 여기에 있습니다:
    {}
    이제 댓글이 끝납니다.

    댓글이 혐오 발언이라면 'yes'로 응답하고, 그렇지 않다면 'no'로 응답하세요.
    """,
    "Vietnamese": """
    Vui lòng xác định xem bình luận YouTube/Facebook bên dưới có phải là lời nói căm thù hay không.
    Tiêu chuẩn của lời nói căm thù trong tập dữ liệu này như sau:
    Lời nói căm thù là một câu làm mất nhân tính của một hoặc nhiều người hoặc cộng đồng. Việc làm mất nhân tính có thể được thực hiện bằng cách so sánh người hoặc cộng đồng đó với côn trùng, đồ vật hoặc tội phạm. Nó cũng có thể được thực hiện bằng cách nhắm mục tiêu vào một người dựa trên chủng tộc, giới tính, khuyết tật thể chất và tinh thần của họ.
    Một câu có thể chứa tiếng lóng hoặc ngôn ngữ không phù hợp. Nhưng trừ khi tiếng lóng đó làm mất nhân tính của một người hoặc cộng đồng, chúng tôi không coi đó là lời nói căm thù.
    Nếu một bình luận ủng hộ một ý tưởng rõ ràng là làm mất nhân tính của một người hoặc cộng đồng, thì đó được coi là lời nói căm thù.
    Nếu cần thêm ngữ cảnh để hiểu rằng một bình luận là lời nói căm thù, thì chúng tôi cho rằng đó không phải là lời nói căm thù.
    Không quan trọng liệu quan điểm mà một bình luận lời nói căm thù đưa ra là đúng hay sai.

    Bây giờ đây là bình luận:
    {}
    Bây giờ bình luận kết thúc.

    Vui lòng trả lời 'yes' nếu bình luận là lời nói căm thù và 'no' nếu không phải.
    """,
    "Thai": """
    กรุณากำหนดว่าความคิดเห็น YouTube/Facebook ด้านล่างเป็นคำพูดที่แสดงความเกลียดชังหรือไม่
    มาตรฐานของคำพูดแสดงความเกลียดชังในชุดข้อมูลนี้มีดังนี้:
    คำพูดแสดงความเกลียดชังเป็นประโยคที่ลดทอนความเป็นมนุษย์ของบุคคลหรือชุมชนหนึ่งหรือหลายคน การลดทอนความเป็นมนุษย์สามารถทำได้โดยการเปรียบเทียบบุคคลหรือชุมชนกับแมลง วัตถุ หรืออาชญากร นอกจากนี้ยังสามารถทำได้โดยการกำหนดเป้าหมายบุคคลตามเชื้อชาติ เพศ ความทุพพลภาพทางร่างกายและจิตใจของพวกเขา
    ประโยคหนึ่งอาจมีคำแสลงหรือน้ำเสียงที่ไม่เหมาะสม แต่ถ้าคำแสลงนั้นไม่ได้ลดทอนความเป็นมนุษย์ของบุคคลหรือชุมชน เราไม่ถือว่าเป็นคำพูดแสดงความเกลียดชัง
    หากความคิดเห็นสนับสนุนแนวคิดที่ชัดเจนว่าลดทอนความเป็นมนุษย์ของบุคคลหรือชุมชน แสดงว่าเป็นคำพูดที่แสดงความเกลียดชัง
    หากต้องการบริบทเพิ่มเติมเพื่อทำความเข้าใจว่าความคิดเห็นเป็นคำพูดแสดงความเกลียดชังหรือไม่ เราถือว่าไม่ใช่คำพูดแสดงความเกลียดชัง
    ไม่สำคัญว่าจุดยืนที่คำพูดแสดงความเกลียดชังความคิดเห็นนี้ถูกต้องหรือไม่

    นี่คือความคิดเห็น:
    {}
    ขณะนี้ความคิดเห็นสิ้นสุดลงแล้ว

    โปรดตอบด้วย 'yes' หากความคิดเห็นเป็นคำพูดแสดงความเกลียดชัง และ 'no' หากไม่ใช่
    """,
    "Indonesian": """
    Silakan tentukan apakah komentar YouTube/Facebook di bawah ini adalah ujaran kebencian atau bukan.
    Standar ujaran kebencian dalam dataset ini adalah sebagai berikut:
    Ujaran kebencian adalah kalimat yang mendekontruksi kemanusiaan satu atau beberapa orang atau komunitas. Mendekontruksi kemanusiaan dapat dilakukan dengan membandingkan orang atau komunitas dengan serangga, benda, atau penjahat. Ini juga bisa dilakukan dengan menargetkan seseorang berdasarkan ras, jenis kelamin, cacat fisik dan mental.
    Sebuah kalimat mungkin berisi bahasa gaul atau bahasa yang tidak pantas. Namun kecuali bahasa gaul tersebut mendekontruksi kemanusiaan seseorang atau komunitas, kami tidak menganggapnya sebagai ujaran kebencian.
    Jika sebuah komentar mendukung ide yang secara jelas mendekontruksi kemanusiaan seseorang atau komunitas, maka itu dianggap sebagai ujaran kebencian.
    Jika diperlukan konteks tambahan untuk memahami bahwa komentar tersebut adalah ujaran kebencian, maka kami menganggapnya bukan ujaran kebencian.
    Tidak masalah apakah sikap yang diambil oleh komentar ujaran kebencian itu benar atau salah.

    Sekarang berikut adalah komentar:
    {}
    Sekarang komentar berakhir.

    Silakan jawab dengan 'yes' jika komentar tersebut adalah ujaran kebencian, dan 'no' jika tidak.
    """,
    "Malay": """
    Sila tentukan jika komen YouTube/Facebook di bawah adalah ucapan kebencian atau tidak.
    Piawaian ucapan kebencian dalam dataset ini adalah seperti berikut:
    Ucapan kebencian ialah ayat yang mengurangkan kemanusiaan seorang atau beberapa orang atau komuniti. Pengurangan kemanusiaan boleh dilakukan dengan membandingkan orang atau komuniti dengan serangga, objek atau penjenayah. Ia juga boleh dilakukan dengan mensasarkan seseorang berdasarkan bangsa, jantina, ketidakupayaan fizikal dan mental mereka.
    Satu ayat mungkin mengandungi bahasa slanga atau bahasa yang tidak sesuai, tetapi melainkan slanga itu merendahkan seseorang atau komuniti, kami tidak menganggapnya sebagai ucapan kebencian.
    Jika komen menyokong idea yang jelas mengurangkan kemanusiaan seseorang atau komuniti, ia dianggap sebagai ucapan kebencian.
    Jika konteks tambahan diperlukan untuk memahami bahawa komen adalah ucapan kebencian, maka kami menganggap ia bukan ucapan kebencian.
    Tidak kira sama ada pendirian yang diambil oleh komen ucapan kebencian itu betul atau salah.

    Sekarang di sini ialah komen:
    {}
    Sekarang komen tamat.

    Sila jawab dengan 'yes' jika komen itu adalah ucapan kebencian, dan 'no' jika ia tidak.
    """,
    "Lao": """
    ກະລຸນາກຳນົດວ່າຄວາມຄິດເຫັນ YouTube/Facebook ດ້ານລຸ່ມນີ້ແມ່ນຄວາມເກຽດຊັງຫລືບໍ່.
    ມາດຕາດານຂອງຄວາມເກຽດຊັງໃນຊຸດຂໍ້ມູນນີ້ຄື:
    ຄວາມເກຽດຊັງແມ່ນປະໂຫຍກທີ່ເຮັດໃຫ້ບຸກຄົນຫລາຍຄົນຫລືຊຸມຊົນກົງ. ການຄົງຈະຖືກສາມາດໄດ້ຜ່ານການເປົາບຸກຄົນຫລືຊຸມຊົນໄປເປັນແມງມຸມ, ເປົາຫມູ່ຄົນຫລືອາຊະຍາກອນ. ມັນຍັງສາມາດຖືກເຮັດໂດຍໃຫ້ຄວາມສົມທັດຕໍ່ບຸກຄົນຂື້ນຢູ່ເຖິງຊົ່ວການເປົາຊະພາບຂອງພວກເຂົາ, ການແບ່ງປັນບຸກຄົນ, ຄວາມສົມທັດຕໍ່ຂອງບຸກຄົນ, ການສໍາລັບຊັດວ່າຄວາມພາກລົດສະຖານະບໍ່ມີຄວາມສົມທັດຕໍ່ບຸກຄົນຂຶ້ນຢູ່ເຖິງຄວາມບໍ່ພ້ອມ.
    ປະໂຫຍກເລື່ອຍໄປອາດມີການສະມາຊົນແທ້ຕໍ່ປະເທດຕົວຫວ່າງການກະກຽມຄຳວ່າຕ່າງໆເປັນທາງເມືອງຫຼວງ. ແຕ່ຖ້າພວກເຂົາຄົງບໍ່ໄດ້ສໍາລັບຊັດວ່າບຸກຄົນຫລືຊຸມຊົນບໍ່ມີຄວາມສົມທັດ, ພວກເຮົາບໍ່ຄິດວ່າມັນເປັນຄວາມເກຽດຊັງ.
    ຖ້າຄວາມຄິດເຫັນຄ້າມຄົງການໃຫ້ຄວາມສົມທັດເຕືອນບຸກຄົນຫລືຊຸມຊົນໄດ້ປະເທດຕົວຫວ່າງພະນັກງານ, ມັນຄວນໄດ້ກະທຳເປັນຄວາມເກຽດຊັງ.
    ຖ້າຄໍາຄົງບໍ່ເຄັ່ງຮັກກັບສ່ວນຂໍ້ມູນຄ້າມຄວາມສົມທັດທີ່ມີຄວາມສົມທັດ, ພວກເຮົາກໍຖືກເປັນຄວາມຄິດເຫັນບໍ່ຖືກຄວາມເກຽດຊັງ.
    ບໍ່ວ່າບຸກຄົນທີ່ມີຄວາມເກຽດຊັງຄົງກັບຄໍາຄິດເຫັນບໍ່ຖືກຄວາມເກຽດຊັງ.

    ດຽວນີ້ຄືຄວາມຄິດເຫັນຢູ່ນີ້:
    {}
    ດຽວນີ້ຄືຄຳຄົງໄດ້ສິ້ນສຸດລົງແລ້ວ.

    ກະລຸນາຕອບກັບຄວາມຄິດເຫັນກ່ຽວກັບຄຳຄົງຖືກຄວາມເກຽດຊັງຫລືບໍ່ເປັນຄວາມເກຽດຊັງ.
    """,
    "Burmese": """
    YouTube/Facebook မှာ ပေးထားတဲ့ အောက်က မှတ်ချက်က မုန်းတီးစကားလား မဟုတ်လား ဆုံးဖြတ်ပေးပါ။
    ဒီဒေတာစုစုအတွက် မုန်းတီးစကား စံချိန်ကို အောက်ပါအတိုင်း သတ်မှတ်ပါတယ်။
    မုန်းတီးစကားဆိုတာ တစ်ဦးတစ်ယောက် (သို့) လူတစ်စု (သို့) လူ့အဖွဲ့အစည်းတစ်ခုခုကို အရည်အသွေးကင်းမဲ့ခြင်း လို့ ယူဆတဲ့ စကားလုံး (သို့) စကားစု ဖြစ်ပါတယ်။ အရည်အသွေးကင်းမဲ့ခြင်းဟာ လူတစ်ဦး (သို့) လူ့အဖွဲ့အစည်းကို ပိုးကောင်၊ အရာဝတ္ထု (သို့) အကြမ်းဖက်မှုတွေနဲ့ တန်းတူထားခြင်းဖြင့် ဆောင်ရွက်နိုင်ပါတယ်။ ဒါ့အပြင် လူတစ်ဦးကို သူတို့ရဲ့ လူမျိုး၊ ကျားမ၊ ရုပ်ပိုင်းဆိုင်ရာနှင့် စိတ်ပိုင်းဆိုင်ရာ မသန်စွမ်းမှုအရ ထိမှန်တာဖြစ်ပါတယ်။
    တစ်ကြောင်းတည်းမှာ slang (သို့) မသင့်လျော်တဲ့ စကားလုံးတွေ ပါဝင်နိုင်ပါတယ်။ ဒါပေမယ့် အဲ့ဒီ slang ဟာ လူတစ်ဦး (သို့) လူ့အဖွဲ့အစည်းတစ်ခုခုကို အရည်အသွေးကင်းမဲ့စေခြင်း မရှိမချင်း၊ အဲဒါကို မုန်းတီးစကားလို့ မထင်ဘူး။
    မှတ်ချက်တစ်ခုက လူတစ်ဦး (သို့) လူ့အဖွဲ့အစည်းကို အရည်အသွေးကင်းမဲ့စေမယ့် သဘောထားတစ်ခုကို အားပေးထားလျှင်၊ အဲဒါဟာ မုန်းတီးစကားဖြစ်ပါတယ်။
    မှတ်ချက်တစ်ခုဟာ မုန်းတီးစကားဆိုတာနားလည်ဖို့ အကြောင်းအရာများစွာ လိုအပ်နေပါက၊ အဲဒါကို မုန်းတီးစကား မဟုတ်ဘူးလို့ ငါတို့ကယူဆပါတယ်။
    မုန်းတီးစကား မှတ်ချက်တစ်ခုက နေရပ်တစ်ခုက မှန်ပါတယ်လို့ ဆိုရမလား၊ မမှန်ဘူးလို့ ဆိုရမလားကတော့ အဓိကမဟုတ်ပါဘူး။

    အခုတော့ မှတ်ချက်က ဒီမှာပါ။
    {}
    အခုမှတ်ချက်ဟာ ပြီးပါပြီ။

    မှတ်ချက်ဟာ မုန်းတီးစကားဖြစ်လျှင် 'yes' နဲ့ မုန်းတီးစကား မဟုတ်ပါက 'no' နဲ့ ဖြေကြပါ။
    """,
    "Cebuano": """
    Palihug tukma kung ang komentaryo sa YouTube/Facebook ubos kay hate speech o dili.
    Ang sumbanan sa hate speech sa kini nga dataset mao ang mosunod:
    Ang hate speech mao ang usa ka linya nga nag-dehumanize sa usa o daghan pa nga mga tawo o komunidad. Ang pag-dehumanize mahimo nga buhaton pinaagi sa pag-kompara sa tawo o komunidad ngadto sa usa ka insekto, butang, o usa ka kriminal. Mahimo usab kini nga buhaton pinaagi sa pag-target sa usa ka tawo base sa ilang lahi, gender, pisikal ug mental nga kakulangan.
    Ang usa ka linya mahimong adunay slang o dili angay nga pinulongan, apan gawas kung ang slang dehumanizes sa usa ka tawo o komunidad, wala nato kini giisip nga hate speech.
    Kung ang usa ka komentaryo mosuporta sa usa ka ideya nga tin-aw nga nag-dehumanizes sa usa ka tawo o komunidad, kini giisip nga hate speech.
    Kung kinahanglan ang dugang nga konteksto aron masabtan nga ang usa ka komentaryo usa ka hate speech, nan atong gituohan nga kini dili hate speech.
    Dili igsapayan kung ang tindog nga gikuha sa usa ka komentaryo sa hate speech husto o sayop.

    Karon ania ang komentaryo:
    {}
    Karon ang komentaryo nagtapus.

    Palihug pagtubag sa 'yes' kung ang komentaryo hate speech, ug 'no' kung dili.
    """,
    "Khmer": """
    សូមកំណត់មើលថា មតិយោបល់ YouTube/Facebook ខាងក្រោមគឺជាការនិយាយបង្ហាញពីការខាងក្រោម ឬមិនមែន។
    ស្តង់ដារនៃការនិយាយបង្ហាញពីការខាងក្រោមក្នុងឈុតទិន្នន័យនេះគឺដូចខាងក្រោម:
    ការនិយាយបង្ហាញពីការខាងក្រោមគឺជាប្រយោគដែលបង្ហាញពីការរុះរើឬប្រជាជនឬសហគមន៍មួយ។ ការរុះរើអាចត្រូវបានធ្វើដោយប្រៀបធៀបនរណាម្នាក់ ឬសហគមន៍មួយទាំងនេះជាមួយសត្វល្អិត វត្ថុ ឬឧក្រិដ្ឋជន។ វាក៏អាចត្រូវបានធ្វើឡើងដោយសម្របសម្រួលអ្នកតាមការប្រកាន់ខ្ជាប់លើពួកគេតាមការរើសអើងពូជសាសន៍ ភេទ ភាពអន់ថយផ្លូវកាយ និងផ្លូវចិត្ត។
    ប្រយោគមួយអាចមានស្លែង ឬភាសាដែលមិនសមរម្យ។ ប៉ុន្តែ យ៉ាងណាក៏ដោយ ស្លែងនោះបានបង្ហាញពីការរុះរើឬមិនបង្ហាញនរណាម្នាក់ ឬសហគមន៍មួយទាំងនេះ យើងមិនបានយល់ថាវាគឺជាការនិយាយបង្ហាញពីការខាងក្រោមឡើយ។
    ប្រសិនបើមតិយោបល់មួយគាំទ្រដល់គំនិតមួយដែលបង្ហាញយ៉ាងច្បាស់ថានរណាម្នាក់ ឬសហគមន៍មួយទាំងនេះត្រូវបានបង្ហាញថាវាគឺជាការរុះរើ គេបានសន្និដ្ឋានថាវាគឺជាការនិយាយបង្ហាញពីការខាងក្រោម។
    ប្រសិនបើមានបន្ថែមទៀតទាមទារក្នុងន័យដើម្បីយល់ថាមតិយោបល់មួយគឺជាការនិយាយបង្ហាញពីការខាងក្រោមហើយបន្ទាប់មកយើងបានសន្និដ្ឋានថាវាមិនមែនជាការនិយាយបង្ហាញពីការខាងក្រោមទេ។
    វាមិនចាំបាច់ថាមតិយោបល់ដែលនិយាយបង្ហាញពីការខាងក្រោមថានៅតែត្រឹមត្រូវ ឬខុសមែនទេ។

    ឥឡូវនេះគឺជាមតិយោបល់៖
    {}
    ឥឡូវនេះមតិយោបល់បញ្ចប់។

    សូមឆ្លើយតបជាមួយ 'yes' ប្រសិនបើមតិយោបល់គឺជាការនិយាយបង្ហាញពីការខាងក្រោម និង 'no' ប្រសិនបើមិនមែន។
    """,
    "Tagalog": """
    Paki-determina kung ang YouTube/Facebook na komento sa ibaba ay isang hate speech o hindi.
    Ang pamantayan ng hate speech sa dataset na ito ay ang sumusunod:
    Ang hate speech ay isang pangungusap na nagde-dehumanize ng isa o higit pang tao o isang komunidad. Ang dehumanization ay maaaring gawin sa pamamagitan ng paghahambing ng tao o komunidad sa isang insekto, bagay, o kriminal. Maaari rin itong gawin sa pamamagitan ng pag-target sa isang tao batay sa kanilang lahi, kasarian, pisikal at mental na kapansanan.
    Ang isang pangungusap ay maaaring naglalaman ng slang o hindi naaangkop na wika, ngunit maliban kung ang slang na iyon ay nagde-dehumanize ng isang tao o komunidad, hindi namin ito isinasaalang-alang bilang hate speech.
    Kung ang isang komento ay sumusuporta sa isang ideya na malinaw na nagde-dehumanize ng isang tao o komunidad, ito ay itinuturing na hate speech.
    Kung ang karagdagang konteksto ay kinakailangan upang maunawaan na ang isang komento ay isang hate speech, ipinapalagay namin na ito ay hindi hate speech.
    Hindi mahalaga kung tama o mali ang pananaw ng isang hate speech na komento.

    Ngayon narito ang komento:
    {}
    Ngayon ay natapos na ang komento.

    Mangyaring tumugon ng 'yes' kung ang komento ay hate speech, at 'no' kung hindi.
    """,
    "Hindi": """
    कृपया यह निर्धारित करें कि नीचे दी गई YouTube/Facebook टिप्पणी घृणा भाषण है या नहीं।
    इस डेटासेट में घृणा भाषण का मानक इस प्रकार है:
    घृणा भाषण वह वाक्य है जो एक या एक से अधिक व्यक्तियों या समुदाय को अमानवीय बनाता है। अमानवीकरण तब किया जा सकता है जब व्यक्ति या समुदाय की तुलना कीट, वस्तु, या अपराधी से की जाती है। यह किसी व्यक्ति को उनकी जाति, लिंग, शारीरिक और मानसिक अक्षमता के आधार पर निशाना बनाकर भी किया जा सकता है।
    एक वाक्य में अपशब्द या अनुचित भाषा हो सकती है, लेकिन जब तक वह अपशब्द किसी व्यक्ति या समुदाय को अमानवीय नहीं बनाता, हम इसे घृणा भाषण नहीं मानते।
    यदि कोई टिप्पणी स्पष्ट रूप से किसी व्यक्ति या समुदाय को अमानवीय बनाने वाले विचार का समर्थन करती है, तो इसे घृणा भाषण माना जाता है।
    यदि यह समझने के लिए अतिरिक्त संदर्भ की आवश्यकता है कि कोई टिप्पणी घृणा भाषण है, तो हम मानते हैं कि यह घृणा भाषण नहीं है।
    यह मायने नहीं रखता कि घृणा भाषण टिप्पणी का दृष्टिकोण सही है या गलत।

    अब यहाँ टिप्पणी है:
    {}
    अब टिप्पणी समाप्त होती है।

    कृपया 'yes' के साथ प्रतिक्रिया दें यदि टिप्पणी घृणा भाषण है, और 'no' यदि यह नहीं है।
    """,
    "Bengali": """
    দয়া করে নীচের YouTube/Facebook মন্তব্যটি ঘৃণামূলক বক্তৃতা কিনা তা নির্ধারণ করুন।
    এই ডেটাসেটে ঘৃণামূলক বক্তৃতার মান নিম্নরূপ:
    ঘৃণামূলক বক্তৃতা একটি বাক্য যা একজন বা একাধিক ব্যক্তি বা একটি সম্প্রদায়কে অমানবিক করে তোলে। অমানবিককরণ করা যেতে পারে ব্যক্তির বা সম্প্রদায়কে পোকা, বস্তুর সাথে বা অপরাধীর সাথে তুলনা করে। এটি জাতি, লিঙ্গ, শারীরিক এবং মানসিক প্রতিবন্ধকতার ভিত্তিতে একজন ব্যক্তিকে লক্ষ্য করে করা যেতে পারে।
    একটি বাক্যে অশ্লীল বা অনুপযুক্ত ভাষা থাকতে পারে, কিন্তু যদি সেই অশ্লীল ভাষা একজন ব্যক্তি বা সম্প্রদায়কে অমানবিক না করে, আমরা এটিকে ঘৃণামূলক বক্তৃতা হিসাবে বিবেচনা করিনি।
    যদি একটি মন্তব্য একটি ধারণাকে সমর্থন করে যা স্পষ্টভাবে একজন ব্যক্তি বা সম্প্রদায়কে অমানবিক করে তোলে, তাহলে এটি ঘৃণামূলক বক্তৃতা হিসেবে বিবেচিত হয়।
    যদি কোনো মন্তব্য ঘৃণামূলক বক্তৃতা তা বুঝতে অতিরিক্ত প্রেক্ষাপটের প্রয়োজন হয়, তাহলে আমরা ধরে নিচ্ছি এটি ঘৃণামূলক বক্তৃতা নয়।
    এটি গুরুত্বপূর্ণ নয় যে ঘৃণামূলক বক্তৃতার মন্তব্য সঠিক বা ভুল অবস্থান নেয়।

    এখন এখানে মন্তব্যটি রয়েছে:
    {}
    এখন মন্তব্য শেষ।

    মন্তব্যটি যদি ঘৃণামূলক বক্তৃতা হয় তবে 'yes' দিয়ে উত্তর দিন এবং 'no' দিয়ে উত্তর দিন যদি তা না হয়।
    """,
    "Urdu": """
    براہ کرم فیصلہ کریں کہ نیچے دی گئی YouTube/Facebook تبصرہ نفرت انگیز تقریر ہے یا نہیں۔
    اس ڈیٹا سیٹ میں نفرت انگیز تقریر کا معیار درج ذیل ہے:
    نفرت انگیز تقریر ایک جملہ ہے جو ایک یا زیادہ افراد یا کسی کمیونٹی کو غیر انسانی قرار دیتا ہے۔ غیر انسانی قرار دینا اس وقت کیا جا سکتا ہے جب کسی شخص یا کمیونٹی کا موازنہ کیڑے، چیز، یا مجرم سے کیا جائے۔ اسے کسی شخص کو ان کی نسل، جنس، جسمانی اور ذہنی معذوری کی بنیاد پر نشانہ بنا کر بھی کیا جا سکتا ہے۔
    ایک جملہ میں گالی یا نامناسب زبان ہو سکتی ہے، لیکن جب تک کہ وہ گالی کسی شخص یا کمیونٹی کو غیر انسانی نہیں بناتی، ہم اسے نفرت انگیز تقریر نہیں سمجھتے۔
    اگر کوئی تبصرہ کسی ایسی سوچ کی حمایت کرتا ہے جو واضح طور پر کسی شخص یا کمیونٹی کو غیر انسانی قرار دیتی ہے، تو اسے نفرت انگیز تقریر سمجھا جاتا ہے۔
    اگر یہ سمجھنے کے لیے اضافی سیاق و سباق کی ضرورت ہو کہ کوئی تبصرہ نفرت انگیز تقریر ہے، تو ہم فرض کرتے ہیں کہ یہ نفرت انگیز تقریر نہیں ہے۔
    اس سے کوئی فرق نہیں پڑتا کہ نفرت انگیز تقریر کے تبصرے کی پوزیشن صحیح ہے یا غلط۔

    اب یہاں تبصرہ ہے:
    {}
    اب تبصرہ ختم ہوتا ہے۔

    براہ کرم 'yes' کے ساتھ جواب دیں اگر تبصرہ نفرت انگیز تقریر ہے، اور 'no' اگر ایسا نہیں ہے۔
    """
}


# Create a new DataFrame
instruction_df = df[["sentence", "hate"]].copy()

# Generate the instructions using randomly selected templates
instruction_df["instruction"] = instruction_df["sentence"].apply(
    lambda x: random.choice(list(instruction_templates.values())).format(x)
)


# Generate the outputs with English labels "yes" or "no"
instruction_df["output"] = instruction_df["hate"].apply(lambda x: "yes" if x == 1 else "no")

# Keep only the 'instruction' and 'output' columns
instruction_df = instruction_df[["instruction", "output"]]

# Shuffle the DataFrame and save to Parquet
instruction_df = instruction_df.sample(frac=1).reset_index(drop=True)
# instruction_df.to_parquet("BHS-multilingual.parquet", index=False)
# Separate the majority and minority classes
df_majority = instruction_df[instruction_df.output == 'no']
df_minority = instruction_df[instruction_df.output == 'yes']

# Downsample the majority class
df_majority_downsampled = resample(df_majority,
                                   replace=False,    # sample without replacement
                                   n_samples=len(df_minority), # to match minority class
                                   random_state=123) # reproducible results

# Combine the minority class with the downsampled majority class
instructions_df_balanced = pd.concat([df_majority_downsampled, df_minority])
instructions_df_balanced.to_parquet("../../data/BHS/BHS-multilingual.parquet", index=False)