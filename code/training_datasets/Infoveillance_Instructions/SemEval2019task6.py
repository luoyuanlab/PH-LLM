# -*- coding: utf-8 -*-
#  SemEval-2019 Task 6 
# - Paper: https://doi.org/10.48550/arXiv.1903.08983
# - Data: https://github.com/ZeyadZanaty/offenseval/tree/master/datasets/training-v1

import csv
import pandas as pd
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import pandas as pd
import random
from utils.data_utils import balance_classes

file_path = "../../data/SemEval2019Task6/offenseval-training-v1.tsv"
data = []
labels = []
with open(file_path,encoding='utf8') as tsvfile:
    reader = csv.reader(tsvfile, delimiter='\t')
    for i,line in enumerate(reader):
        if i is 0:
            continue
        label = line[-3:]
        labels.append(label)
        data.append(line[1])

# Sub A
# # Function to create instruction in the target language
def get_instruction(tweet, templates):
    # Randomly select one of the multilingual templates
    instruction_template = random.choice(templates)
    instruction = instruction_template.format(tweet=tweet)
    return instruction

# Function to create output based on the label
def get_output(label):
    return "yes" if label[0] == "OFF" else "no"

# Define multilingual templates for the instruction
multilingual_templates = [
    # Chinese
    "请判断以下推文是否包含攻击性语言或有针对性的（隐晦或直接的）冒犯。如果是，请回答'yes'。如果帖子不包含攻击性语言且不包含目标冒犯，请回答'no'。你应该回答'yes'，如果帖子包含任何形式的脏话或攻击性语言，或者包含有针对性的冒犯（可能是隐晦或直接的，包括侮辱、威胁和咒骂）。推文: \"{tweet}\"。推文结束。请用'yes'或'no'作答。",

    # English
    "Please determine if the provided tweet below contains offensive language or a targeted (veiled or direct) offense. If so, respond 'yes'. If the post does not contain offensive language and does not contain a targeted offense, respond 'no'. You should respond 'yes' if the post is either offensive (if it contains any form of profane language) or a targeted offense (which can be veiled or direct, including insults, threats, and swear words). Tweet: \"{tweet}\". Now the tweet ends. Please respond with 'yes' or 'no'.",

    # German
    "Bitte bestimmen Sie, ob der unten bereitgestellte Tweet beleidigende Sprache oder eine gezielte (verdeckte oder direkte) Beleidigung enthält. Wenn ja, antworten Sie mit 'yes'. Wenn der Beitrag keine beleidigende Sprache und keine gezielte Beleidigung enthält, antworten Sie mit 'no'. Sie sollten mit 'yes' antworten, wenn der Beitrag entweder beleidigend ist (wenn er eine Form von obszöner Sprache enthält) oder eine gezielte Beleidigung (die verdeckt oder direkt sein kann, einschließlich Beleidigungen, Drohungen und Schimpfwörtern) enthält. Tweet: \"{tweet}\". Jetzt endet der Tweet. Bitte antworten Sie mit 'yes' oder 'no'.",

    # French
    "Veuillez déterminer si le tweet ci-dessous contient un langage offensant ou une offense ciblée (voilée ou directe). Si c'est le cas, répondez 'yes'. Si le post ne contient pas de langage offensant et ne contient pas d'offense ciblée, répondez 'no'. Vous devez répondre 'yes' si le post est soit offensant (s'il contient une forme quelconque de langage vulgaire) soit une offense ciblée (qui peut être voilée ou directe, y compris les insultes, menaces et jurons). Tweet : \"{tweet}\". Maintenant, le tweet se termine. Veuillez répondre par 'yes' ou 'no'.",

    # Spanish
    "Por favor, determine si el tweet proporcionado a continuación contiene lenguaje ofensivo o una ofensa dirigida (velada o directa). Si es así, responda 'yes'. Si la publicación no contiene lenguaje ofensivo y no contiene una ofensa dirigida, responda 'no'. Debe responder 'yes' si la publicación es ofensiva (si contiene alguna forma de lenguaje vulgar) o una ofensa dirigida (que puede ser velada o directa, incluidas las insultas, amenazas y palabrotas). Tweet: \"{tweet}\". Ahora el tweet termina. Responda con 'yes' o 'no'.",

    # Portuguese
    "Por favor, determine se o tweet fornecido abaixo contém linguagem ofensiva ou uma ofensa direcionada (velada ou direta). Se sim, responda 'yes'. Se o post não contiver linguagem ofensiva e não contiver uma ofensa direcionada, responda 'no'. Você deve responder 'yes' se o post for ofensivo (se contiver qualquer forma de linguagem profana) ou uma ofensa direcionada (que pode ser velada ou direta, incluindo insultos, ameaças e palavrões). Tweet: \"{tweet}\". Agora o tweet termina. Por favor, responda com 'yes' ou 'no'.",

    # Italian
    "Si prega di determinare se il tweet fornito di seguito contiene un linguaggio offensivo o un'offesa mirata (velata o diretta). In tal caso, rispondere 'yes'. Se il post non contiene linguaggio offensivo e non contiene un'offesa mirata, rispondere 'no'. Dovresti rispondere 'yes' se il post è offensivo (se contiene qualsiasi forma di linguaggio volgare) o un'offesa mirata (che può essere velata o diretta, comprese offese, minacce e parolacce). Tweet: \"{tweet}\". Ora il tweet finisce. Rispondere con 'yes' o 'no'.",

    # Dutch
    "Bepaal of de onderstaande tweet beledigende taal of een gerichte (verhulde of directe) belediging bevat. Zo ja, antwoord dan met 'yes'. Als het bericht geen beledigende taal bevat en geen gerichte belediging bevat, antwoord dan met 'no'. U moet antwoorden met 'yes' als het bericht beledigend is (als het enige vorm van grof taalgebruik bevat) of een gerichte belediging (die verhuld of direct kan zijn, inclusief beledigingen, bedreigingen en scheldwoorden). Tweet: \"{tweet}\". Nu eindigt de tweet. Antwoord alstublieft met 'yes' of 'no'.",

    # Russian
    "Пожалуйста, определите, содержит ли приведенный ниже твит оскорбительный язык или целенаправленное (завуалированное или прямое) оскорбление. Если да, ответьте 'yes'. Если пост не содержит оскорбительных выражений и не содержит целенаправленных оскорблений, ответьте 'no'. Вы должны ответить 'yes', если пост является оскорбительным (если он содержит любую форму непристойных выражений) или целенаправленным оскорблением (которое может быть завуалированным или прямым, включая оскорбления, угрозы и ругательства). Твит: \"{tweet}\". Теперь твит заканчивается. Пожалуйста, ответьте 'yes' или 'no'.",

    # Czech
    "Prosím, určete, zda následující tweet obsahuje urážlivý jazyk nebo cílenou (skrytou nebo přímou) urážku. Pokud ano, odpovězte 'yes'. Pokud příspěvek neobsahuje urážlivý jazyk a neobsahuje cílenou urážku, odpovězte 'no'. Měli byste odpovědět 'yes', pokud je příspěvek urážlivý (pokud obsahuje jakoukoli formu vulgárního jazyka) nebo cílenou urážku (která může být skrytá nebo přímá, včetně urážek, hrozeb a nadávek). Tweet: \"{tweet}\". Nyní tweet končí. Odpovězte prosím 'yes' nebo 'no'.",

    # Polish
    "Proszę określić, czy poniższy tweet zawiera obraźliwy język lub ukierunkowaną (ukrytą lub bezpośrednią) obrazę. Jeśli tak, odpowiedz 'yes'. Jeśli post nie zawiera obraźliwego języka i nie zawiera ukierunkowanej obrazy, odpowiedz 'no'. Powinieneś odpowiedzieć 'yes', jeśli post jest obraźliwy (jeśli zawiera jakąkolwiek formę wulgarnego języka) lub ukierunkowaną obrazę (która może być ukryta lub bezpośrednia, w tym zniewagi, groźby i przekleństwa). Tweet: \"{tweet}\". Teraz tweet się kończy. Odpowiedz proszę 'yes' lub 'no'.",

    # Arabic
    "يرجى تحديد ما إذا كانت التغريدة أدناه تحتوي على لغة مسيئة أو إهانة مستهدفة (مبطنة أو مباشرة). إذا كان الأمر كذلك، استجب بـ 'yes'. إذا كانت التغريدة لا تحتوي على لغة مسيئة ولا تحتوي على إهانة مستهدفة، استجب بـ 'no'. يجب أن تستجيب بـ 'yes' إذا كانت التغريدة إما مسيئة (إذا كانت تحتوي على أي شكل من أشكال اللغة البذيئة) أو إهانة مستهدفة (يمكن أن تكون مبطنة أو مباشرة، بما في ذلك الإهانات والتهديدات والكلمات النابية). التغريدة: \"{tweet}\". الآن انتهت التغريدة. يرجى الرد بـ 'yes' أو 'no'.",

    # Persian
    "لطفاً تعیین کنید که آیا توییت زیر حاوی زبان توهین آمیز یا توهین هدفمند (پوشیده یا مستقیم) است یا خیر. اگر چنین است، پاسخ 'yes' دهید. اگر پست حاوی زبان توهین آمیز نیست و حاوی توهین هدفمند نیست، پاسخ 'no' دهید. باید پاسخ 'yes' دهید اگر پست حاوی هر شکلی از زبان بی ادبانه باشد یا توهین هدفمند (که می‌تواند پوشیده یا مستقیم باشد، شامل توهین‌ها، تهدیدها و ناسزاها). توییت: \"{tweet}\". اکنون توییت به پایان می‌رسد. لطفاً با 'yes' یا 'no' پاسخ دهید.",

    # Hebrew
    "אנא קבע אם הציוץ שלהלן מכיל שפה פוגענית או עבירה מכוונת (מרומזת או ישירה). אם כן, השב 'yes'. אם הפוסט לא מכיל שפה פוגענית ואינו מכיל עבירה מכוונת, השב 'no'. עליך להשיב 'yes' אם הפוסט הוא פוגעני (אם הוא מכיל כל סוג של שפה גסה) או עבירה מכוונת (שיכולה להיות מרומזת או ישירה, כולל עלבונות, איומים וקללות). ציוץ: \"{tweet}\". הציוץ נגמר כעת. אנא השב 'yes' או 'no'.",

    # Turkish
    "Lütfen aşağıdaki tweet'in saldırgan dil veya hedeflenen (örtülü veya doğrudan) bir hakaret içerip içermediğini belirleyin. Eğer öyleyse, 'yes' yanıtını verin. Gönderi saldırgan bir dil içermiyorsa ve hedeflenen bir hakaret içermiyorsa, 'no' yanıtını verin. Gönderi ya saldırgan (eğer herhangi bir tür küfür içeriyorsa) ya da hedeflenen bir hakaret (örtülü veya doğrudan olabilir, hakaretler, tehditler ve küfürler dahil) içeriyorsa 'yes' yanıtını vermelisiniz. Tweet: \"{tweet}\". Tweet şimdi sona erdi. Lütfen 'yes' veya 'no' yanıtını verin.",

    # Japanese
    "以下のツイートに攻撃的な言葉やターゲットを狙った（隠れたまたは直接的な）侮辱が含まれているかどうかを判断してください。もしそうであれば、「yes」と答えてください。投稿に攻撃的な言葉が含まれておらず、ターゲットを狙った侮辱が含まれていない場合は、「no」と答えてください。投稿が攻撃的である（何らかの形の下品な言葉が含まれている場合）またはターゲットを狙った侮辱を含む（それは隠れたまたは直接的なものであり、侮辱、脅迫、罵り言葉を含む）場合、「yes」と答えるべきです。ツイート：「{tweet}」。これでツイートは終了です。「yes」または「no」で答えてください。",

    # Korean
    "제공된 아래 트윗에 공격적인 언어나 표적이 된 (암시적이거나 직접적인) 모욕이 포함되어 있는지 판단해 주세요. 그렇다면 'yes'라고 답변하세요. 게시물이 공격적인 언어를 포함하지 않으며 표적이 된 모욕을 포함하지 않는다면 'no'라고 답변하세요. 게시물이 공격적이거나 (어떤 형태의 비속어를 포함하는 경우) 표적이 된 모욕 (암시적이거나 직접적일 수 있으며 모욕, 위협 및 욕설 포함)을 포함하는 경우 'yes'라고 답변해야 합니다. 트윗: \"{tweet}\". 이제 트윗이 끝났습니다. 'yes' 또는 'no'로 답변해 주세요.",

    # Vietnamese
    "Vui lòng xác định xem tweet được cung cấp bên dưới có chứa ngôn từ xúc phạm hoặc một hành vi xúc phạm được nhắm mục tiêu (dù che giấu hay trực tiếp). Nếu có, hãy trả lời 'yes'. Nếu bài đăng không chứa ngôn từ xúc phạm và không chứa hành vi xúc phạm có chủ đích, hãy trả lời 'no'. Bạn nên trả lời 'yes' nếu bài đăng có tính xúc phạm (nếu nó chứa bất kỳ hình thức ngôn từ thô tục nào) hoặc hành vi xúc phạm có chủ đích (có thể là giấu giếm hoặc trực tiếp, bao gồm xúc phạm, đe dọa và lời chửi thề). Tweet: \"{tweet}\". Bây giờ tweet kết thúc. Vui lòng trả lời bằng 'yes' hoặc 'no'.",

    # Thai
    "โปรดตรวจสอบว่าทวีตที่ให้ไว้ด้านล่างมีภาษาไม่เหมาะสมหรือเป็นการดูหมิ่นแบบมีเป้าหมาย (แฝงหรือโดยตรง) หรือไม่ หากเป็นเช่นนั้น ให้ตอบ 'yes' หากโพสต์ไม่ใช่ภาษาที่ไม่เหมาะสมและไม่มีการดูหมิ่นเป้าหมาย โปรดตอบ 'no' คุณควรตอบ 'yes' หากโพสต์นั้นดูหมิ่น (หากมีภาษาที่ไม่เหมาะสมในทุกรูปแบบ) หรือการดูหมิ่นแบบมีเป้าหมาย (ซึ่งอาจเป็นการแฝงหรือโดยตรง รวมถึงการดูหมิ่น การข่มขู่ และคำสบถ) ทวีต: \"{tweet}\" ตอนนี้ทวีตสิ้นสุดแล้ว โปรดตอบ 'yes' หรือ 'no'",

    # Indonesian
    "Silakan tentukan apakah tweet yang diberikan di bawah ini mengandung bahasa ofensif atau pelanggaran yang ditargetkan (terselubung atau langsung). Jika demikian, jawab 'yes'. Jika postingan tidak mengandung bahasa yang menyinggung dan tidak mengandung pelanggaran yang ditargetkan, jawab 'no'. Anda harus menjawab 'yes' jika postingan tersebut menyinggung (jika mengandung bahasa kasar dalam bentuk apa pun) atau pelanggaran yang ditargetkan (yang bisa terselubung atau langsung, termasuk penghinaan, ancaman, dan kata-kata kotor). Tweet: \"{tweet}\". Sekarang tweet berakhir. Silakan jawab dengan 'yes' atau 'no'.",

    # Malay
    "Sila tentukan sama ada tweet yang diberikan di bawah mengandungi bahasa kasar atau kesalahan yang disasarkan (terselindung atau langsung). Jika ya, jawab 'yes'. Jika siaran tidak mengandungi bahasa yang menyinggung dan tidak mengandungi kesalahan yang disasarkan, jawab 'no'. Anda harus menjawab 'yes' jika siaran itu menyinggung perasaan (jika ia mengandungi sebarang bentuk bahasa kasar) atau kesalahan yang disasarkan (yang boleh terselindung atau langsung, termasuk penghinaan, ancaman, dan kata-kata kesat). Tweet: \"{tweet}\". Sekarang tweet tamat. Sila jawab dengan 'yes' atau 'no'.",

    # Lao
    "ກະລຸນາກຳນົດວ່າຂໍ້ຄວາມທີ່ໄດ້ຮັບມາຂ້າງລຸ່ມນີ້ມີພາສາທີ່ບໍ່ສຸພາບ ຫຼື ການດ່າຫຼາວເປົ້າໝາຍ (ຫຼັກເວັ້ນ ຫຼື ໂດຍກົງ) ຫຼືບໍ່. ຖ້າວ່າແມ່ນ, ຕອບ 'yes'. ຖ້າວ່າຂໍ້ຄວາມບໍ່ມີພາສາທີ່ບໍ່ສຸພາບ ແລະບໍ່ມີການດ່າຫຼາວເປົ້າໝາຍ, ຕອບ 'no'. ເຈົ້າຄວນຕອບ 'yes' ຖ້າວ່າຂໍ້ຄວາມແມ່ນມີຄວາມຫມາຍບໍ່ສຸພາບ (ຖ້າມັນມີຮູບແບບຂອງພາສາທີ່ຫຍາບຄາຍ) ຫຼືການດ່າຫຼາວເປົ້າໝາຍ (ອາດຈະເປັນຫຼັກເວັ້ນ ຫຼື ໂດຍກົງ, ລວມທັງການດ່າຫຼາວ, ການຂົ້ມຂູ່ ແລະ ຄຳສາບແຊ່ງ). Tweet: \"{tweet}\". ຂໍ້ຄວາມນີ້ສິ້ນສຸດລົງແລ້ວ. ກະລຸນາຕອບດ້ວຍ 'yes' ຫຼື 'no'.",

    # Burmese
    "အောက်တွင်ပေးထားသော တူဿ်တွင် စာမကျေစေသော စကားများ သို့မဟုတ် ပစ်မှတ်ထားသော (ဝှက်ထားသော သို့မဟုတ် တိုက်ရိုက်) ပြစ်မှုများ ပါဝင်ပါက သတ်မှတ်ပါ။ အကယ်၍ ၎င်းတို့ပါဝင်ပါက 'yes' ဖြင့် ပြန်ကြားပါ။ ပို့စ်တွင် စာမကျေစေသော စကားများ မပါဝင်ပဲ ပစ်မှတ်ထားသော ပြစ်မှု မပါဝင်ပါက 'no' ဖြင့် ပြန်ကြားပါ။ ပို့စ်သည် စာမကျေစေသော (ထိုမှာ ဘာသာစကားတစ်ခုခုပါဝင်ပါက) သို့မဟုတ် ပစ်မှတ်ထားသော ပြစ်မှု (ဤသည်မှာ ဝှက်ထားသော သို့မဟုတ် တိုက်ရိုက်ဖြစ်နိုင်ပြီး ညွှန်းဆိုခြင်း၊ ခြိမ်းခြောက်ခြင်းနှင့် ဆဲဆိုခြင်းများ ပါဝင်သည်) ဖြစ်ပါက 'yes' ဖြင့် ပြန်ကြားသင့်သည်။ တူဿ်: \"{tweet}\". ယခုတွင် တူဿ် ပြီးဆုံးပါပြီ။ 'yes' သို့မဟုတ် 'no' ဖြင့် ပြန်ကြားပါ။",

    # Cebuano
    "Palihug pag-determinar kung ang gihatag nga tweet sa ubos naglangkob sa mga malisyoso nga pulong o usa ka tinuyo (tila o direktang) pag-atake. Kung mao, tubaga ang 'yes'. Kung ang post wala maglangkob sa mga malisyoso nga pulong ug wala maglangkob sa tinuyo nga pag-atake, tubaga ang 'no'. Kinahanglan ka motubag og 'yes' kung ang post either malisyoso (kung kini naglangkob og bisan unsang matang sa bastos nga mga pulong) o tinuyo nga pag-atake (nga mahimong tila o direkta, lakip ang mga insulto, mga hulga, ug mga pamalikas). Tweet: \"{tweet}\". Karon ang tweet mohuman na. Palihug motubag og 'yes' o 'no'.",

    # Khmer
    "សូមកំណត់ថាតើការប្រកាសខាងក្រោមមានភាសាប្រមាថ ឬការរំលោភបំពានដ៏គ្រោះថ្នាក់(ដែលអាចមានឬដោយផ្ទាល់) ។ ប្រសិនបើមាន សូមឆ្លើយថា 'yes' ។ ប្រសិនបើការប្រកាសនេះមិនមានភាសាប្រមាថ និងមិនមានការរំលោភបំពាន ដែលមានគោលដៅ សូមឆ្លើយថា 'no' ។ អ្នកគួរតែឆ្លើយថា 'yes' ប្រសិនបើការប្រកាសនេះមិនមានភាសាប្រមាថ ឬការរំលោភបំពាន ដោយផ្ទាល់ទេ (ដែលអាចមានឬដោយផ្ទាល់ មានរួមទាំងការប្រមាថ ការគំរាមកំហែង និងពាក្យសម្តីទាំងឡាយ) ។ ការប្រកាស: \"{tweet}\" ។ ឥឡូវនេះការប្រកាសបានបញ្ចប់ហើយ សូមឆ្លើយថា 'yes' ឬ 'no'.",

    # Tagalog
    "Pakitukoy kung ang ibinigay na tweet sa ibaba ay naglalaman ng mapang-abusong wika o isang target (nakabalatkayo o direktang) pagkakasala. Kung oo, sumagot ng 'yes'. Kung ang post ay hindi naglalaman ng mapang-abusong wika at hindi naglalaman ng target na pagkakasala, sumagot ng 'no'. Dapat kang sumagot ng 'yes' kung ang post ay alinman sa mapang-abusong (kung naglalaman ito ng anumang anyo ng bulgar na wika) o isang target na pagkakasala (na maaaring nakabalatkayo o direktang, kabilang ang mga insulto, banta, at mga sumpa). Tweet: \"{tweet}\". Ngayon natapos na ang tweet. Pakisagot ng 'yes' o 'no'.",

    # Hindi
    "कृपया यह निर्धारित करें कि नीचे दिया गया ट्वीट आपत्तिजनक भाषा या लक्षित (अप्रत्यक्ष या प्रत्यक्ष) अपमान है या नहीं। यदि हां, तो 'yes' के साथ उत्तर दें। यदि पोस्ट में अपमानजनक भाषा नहीं है और लक्षित अपमान नहीं है, तो 'no' के साथ उत्तर दें। यदि पोस्ट अपमानजनक है (यदि इसमें किसी भी प्रकार की अभद्र भाषा शामिल है) या लक्षित अपमान है (जो अप्रत्यक्ष या प्रत्यक्ष हो सकता है, जिसमें अपमान, धमकी और गालियां शामिल हैं), तो आपको 'yes' के साथ उत्तर देना चाहिए। ट्वीट: \"{tweet}\"। अब ट्वीट समाप्त हो गया है। कृपया 'yes' या 'no' के साथ उत्तर दें।",

    # Bengali
    "নীচে দেওয়া টুইটটি অপমানজনক ভাষা বা লক্ষ্যবস্তু (অপ্রকাশিত বা সরাসরি) আক্রমণ কিনা তা নির্ধারণ করুন। যদি তা হয়, তাহলে 'yes' দিয়ে উত্তর দিন। যদি পোস্টটিতে অপমানজনক ভাষা না থাকে এবং লক্ষ্যবস্তু আক্রমণ না থাকে, তাহলে 'no' দিয়ে উত্তর দিন। যদি পোস্টটি হয় অপমানজনক (যদি এতে কোনও ধরণের অশ্লীল ভাষা থাকে) বা একটি লক্ষ্যবস্তু আক্রমণ (যা অপ্রকাশিত বা সরাসরি হতে পারে, যার মধ্যে অপমান, হুমকি এবং গালি-গালাজ রয়েছে) থাকে তবে আপনাকে 'yes' দিয়ে উত্তর দিতে হবে। টুইট: \"{tweet}\"। এখন টুইট শেষ। অনুগ্রহ করে 'yes' বা 'no' দিয়ে উত্তর দিন।",

    # Urdu
    "براہ کرم تعین کریں کہ آیا نیچے دیا گیا ٹویٹ جارحانہ زبان یا ہدف شدہ (پردہ دار یا براہ راست) جرم پر مشتمل ہے۔ اگر ایسا ہے تو، 'yes' کے ساتھ جواب دیں۔ اگر پوسٹ میں جارحانہ زبان شامل نہیں ہے اور اس میں ہدف جرم شامل نہیں ہے، تو 'no' کے ساتھ جواب دیں۔ آپ کو 'yes' کا جواب دینا چاہیے اگر پوسٹ جارحانہ ہو (اگر اس میں کسی بھی قسم کی فحش زبان ہو) یا ہدف جرم (جو پردہ دار یا براہ راست ہو سکتا ہے، بشمول توہین، دھمکیاں، اور گالی گلوچ) ہو۔ ٹویٹ: \"{tweet}\"۔ اب ٹویٹ ختم ہو گیا ہے۔ براہ کرم 'yes' یا 'no' کے ساتھ جواب دیں۔"
]


# Create a blank dataframe
inst_data = pd.DataFrame(columns=['instruction', 'output'])

# Iterate over each tweet and create the instruction/output pair
for i, tweet in enumerate(data):
    output = get_output(labels[i])
    instruction = get_instruction(tweet, multilingual_templates)
    inst_data = pd.concat([inst_data, pd.DataFrame({'instruction': [instruction], 'output': [output]})], ignore_index=True)

print(inst_data.output.value_counts())
inst_data = balance_classes(inst_data, 3000, "output")

# Save the dataframe to a parquet file
inst_data.to_parquet("../../data/SemEval2019Task6/SemEval19_task6_subA_multilingual.parquet", index=False)

# Sub B
# Function to create output based on the label
def get_output(label):
    return "yes" if label[1] == "TIN" else "no"

# Define multilingual templates for the instruction
multilingual_templates = [
    # Chinese
    "请判断以下提供的冒犯性推文是否针对特定个人、群体或其他对象，如果是，请回答'yes'。如果推文中的冒犯性语言没有特定的目标，请回答'no'。推文: \"{tweet}\"。现在推文结束。请用'yes'或'no'作答。",

    # English
    "Please determine if the provided offensive tweet below is targeted at a specific individual, a group, or others. If so, respond 'yes'. If the tweet's offensive language is not targeted, respond 'no'. Tweet: \"{tweet}\". Now the tweet ends. Please respond with 'yes' or 'no'.",

    # German
    "Bitte bestimmen Sie, ob der unten bereitgestellte beleidigende Tweet auf eine bestimmte Person, Gruppe oder andere gerichtet ist. Wenn ja, antworten Sie mit 'yes'. Wenn die beleidigende Sprache des Tweets nicht auf ein Ziel gerichtet ist, antworten Sie mit 'no'. Tweet: \"{tweet}\". Jetzt endet der Tweet. Bitte antworten Sie mit 'yes' oder 'no'.",

    # French
    "Veuillez déterminer si le tweet offensant fourni ci-dessous est dirigé contre une personne, un groupe ou d'autres cibles spécifiques. Si c'est le cas, répondez 'yes'. Si le langage offensant du tweet n'est pas ciblé, répondez 'no'. Tweet : \"{tweet}\". Maintenant, le tweet se termine. Veuillez répondre par 'yes' ou 'no'.",

    # Spanish
    "Por favor, determine si el tweet ofensivo proporcionado a continuación está dirigido a una persona específica, un grupo u otros. Si es así, responda 'yes'. Si el lenguaje ofensivo del tweet no está dirigido, responda 'no'. Tweet: \"{tweet}\". Ahora el tweet termina. Responda con 'yes' o 'no'.",

    # Portuguese
    "Por favor, determine se o tweet ofensivo fornecido abaixo é direcionado a um indivíduo específico, um grupo ou outros. Se sim, responda 'yes'. Se a linguagem ofensiva do tweet não for direcionada, responda 'no'. Tweet: \"{tweet}\". Agora o tweet termina. Por favor, responda com 'yes' ou 'no'.",

    # Italian
    "Si prega di determinare se il tweet offensivo fornito di seguito è rivolto a un individuo specifico, a un gruppo o ad altri. In tal caso, rispondere 'yes'. Se il linguaggio offensivo del tweet non è mirato, rispondere 'no'. Tweet: \"{tweet}\". Ora il tweet finisce. Rispondere con 'yes' o 'no'.",

    # Dutch
    "Bepaal of de onderstaande beledigende tweet gericht is op een specifieke persoon, groep of anderen. Zo ja, antwoord dan met 'yes'. Als het beledigende taalgebruik van de tweet niet gericht is, antwoord dan met 'no'. Tweet: \"{tweet}\". Nu eindigt de tweet. Antwoord alstublieft met 'yes' of 'no'.",

    # Russian
    "Пожалуйста, определите, направлен ли приведенный ниже оскорбительный твит на конкретного человека, группу или других лиц. Если да, ответьте 'yes'. Если оскорбительный язык твита не направлен на цель, ответьте 'no'. Твит: \"{tweet}\". Теперь твит заканчивается. Пожалуйста, ответьте 'yes' или 'no'.",

    # Czech
    "Prosím, určete, zda je níže uvedený urážlivý tweet zaměřen na konkrétní osobu, skupinu nebo jiné. Pokud ano, odpovězte 'yes'. Pokud urážlivý jazyk tweetu není zaměřen na cíl, odpovězte 'no'. Tweet: \"{tweet}\". Nyní tweet končí. Odpovězte prosím 'yes' nebo 'no'.",

    # Polish
    "Proszę określić, czy poniższy obraźliwy tweet jest skierowany do konkretnej osoby, grupy lub innych. Jeśli tak, odpowiedz 'yes'. Jeśli obraźliwy język tweeta nie jest skierowany, odpowiedz 'no'. Tweet: \"{tweet}\". Teraz tweet się kończy. Odpowiedz proszę 'yes' lub 'no'.",

    # Arabic
    "يرجى تحديد ما إذا كانت التغريدة الهجومية المقدمة أدناه موجهة إلى شخص معين أو مجموعة أو غيرهم. إذا كان الأمر كذلك، استجب بـ 'yes'. إذا لم تكن لغة التغريدة الهجومية موجهة، استجب بـ 'no'. التغريدة: \"{tweet}\". الآن انتهت التغريدة. يرجى الرد بـ 'yes' أو 'no'.",

    # Persian
    "لطفاً تعیین کنید که آیا توییت توهین‌آمیز زیر به یک فرد خاص، گروه یا دیگران هدف‌گیری شده است یا خیر. اگر چنین است، پاسخ 'yes' دهید. اگر زبان توهین‌آمیز توییت هدف‌گیری نشده باشد، پاسخ 'no' دهید. توییت: \"{tweet}\". اکنون توییت به پایان می‌رسد. لطفاً با 'yes' یا 'no' پاسخ دهید.",

    # Hebrew
    "אנא קבע אם הציוץ הפוגעני המסופק למטה מכוון לאדם ספציפי, לקבוצה או לאחרים. אם כן, השב 'yes'. אם שפת הציוץ הפוגענית אינה מכוונת, השב 'no'. ציוץ: \"{tweet}\". הציוץ נגמר כעת. אנא השב 'yes' או 'no'.",

    # Turkish
    "Lütfen aşağıdaki saldırgan tweet'in belirli bir bireyi, grubu veya diğerlerini hedef alıp almadığını belirleyin. Eğer öyleyse, 'yes' yanıtını verin. Tweet'in saldırgan dili hedef alınmamışsa, 'no' yanıtını verin. Tweet: \"{tweet}\". Tweet şimdi sona erdi. Lütfen 'yes' veya 'no' yanıtını verin.",

    # Japanese
    "以下の攻撃的なツイートが特定の個人、グループ、または他の対象をターゲットにしているかどうかを判断してください。そうであれば、「yes」と答えてください。ツイートの攻撃的な言葉がターゲットではない場合は、「no」と答えてください。ツイート：「{tweet}」。これでツイートは終了です。「yes」または「no」で答えてください。",

    # Korean
    "아래에 제공된 공격적인 트윗이 특정 개인, 그룹 또는 다른 사람들을 대상으로 하는지 판단하세요. 그렇다면 'yes'라고 답변하세요. 트윗의 공격적인 언어가 타겟팅되지 않았다면 'no'라고 답변하세요. 트윗: \"{tweet}\". 이제 트윗이 끝났습니다. 'yes' 또는 'no'로 답변해 주세요.",

    # Vietnamese
    "Vui lòng xác định xem tweet xúc phạm được cung cấp bên dưới có nhằm vào một cá nhân cụ thể, một nhóm hoặc những người khác không. Nếu có, hãy trả lời 'yes'. Nếu ngôn từ xúc phạm trong tweet không có mục tiêu, hãy trả lời 'no'. Tweet: \"{tweet}\". Bây giờ tweet kết thúc. Vui lòng trả lời bằng 'yes' hoặc 'no'.",

    # Thai
    "โปรดตรวจสอบว่าทวีตที่ไม่เหมาะสมที่ให้ไว้ด้านล่างมีเป้าหมายไปที่บุคคลเฉพาะ กลุ่ม หรือคนอื่นๆ หรือไม่ หากเป็นเช่นนั้น ให้ตอบ 'yes' หากภาษาที่ไม่เหมาะสมของทวีตไม่ได้ถูกกำหนดเป้าหมาย โปรดตอบ 'no' ทวีต: \"{tweet}\" ตอนนี้ทวีตสิ้นสุดแล้ว โปรดตอบ 'yes' หรือ 'no'",

    # Indonesian
    "Silakan tentukan apakah tweet ofensif yang diberikan di bawah ini ditujukan pada individu tertentu, grup, atau lainnya. Jika demikian, jawab 'yes'. Jika bahasa ofensif tweet tidak ditargetkan, jawab 'no'. Tweet: \"{tweet}\". Sekarang tweet berakhir. Silakan jawab dengan 'yes' atau 'no'.",

    # Malay
    "Sila tentukan sama ada tweet ofensif yang diberikan di bawah ini disasarkan kepada individu tertentu, kumpulan, atau lain-lain. Jika ya, jawab 'yes'. Jika bahasa ofensif tweet tidak disasarkan, jawab 'no'. Tweet: \"{tweet}\". Sekarang tweet tamat. Sila jawab dengan 'yes' atau 'no'.",

    # Lao
    "ກະລຸນາກຳນົດວ່າຂໍ້ຄວາມ tweet ທີ່ມີຄວາມບໍ່ຖືກຕ້ອງທີ່ໃຫ້ມາດ້ານລຸ່ມນີ້ມີຕົ້ນຕໍທີ່ຈະຊັກຈູງເຂົ້າເຖິງບຸກຄົນທີ່ກໍານົດ, ກຸ່ມ, ຫຼືຄົນອື່ນໆ ຫຼືບໍ່. ຖ້າວ່າແມ່ນ, ກະລຸນາຕອບ 'yes'. ຖ້າວ່າພາສາທີ່ບໍ່ຖືກຕ້ອງຂອງ tweet ບໍ່ໄດ້ຖືກກໍານົດເປົ້າໝາຍ, ກະລຸນາຕອບ 'no'. Tweet: \"{tweet}\". ຂໍ້ຄວາມຫມາຍ tweet ໄດ້ສິ້ນສຸດລົງແລ້ວ. ກະລຸນາຕອບດ້ວຍ 'yes' ຫຼື 'no'.",

    # Burmese
    "အောက်တွင်ပေးထားသော အပြစ်ရှိသော တူဿ်သည် တစ်ဦးချင်းနှင့် သတ်မှတ်ထားသော ပစ်မှတ်၊ အုပ်စု သို့မဟုတ် အခြားသူများကို ပစ်မှတ်ထားခြင်းဖြစ်သည်ဟု သတ်မှတ်ပါ။ အကယ်၍ သတ်မှတ်ထားပါက 'yes' ဖြင့် ဖြေကြားပါ။ တူဿ်၏ အပြစ်ရှိသော ဘာသာစကားသည် ပစ်မှတ်မထားပါက 'no' ဖြင့် ဖြေကြားပါ။ တူဿ်: \"{tweet}\". ယခုတွင် တူဿ် ပြီးဆုံးပါပြီ။ 'yes' သို့မဟုတ် 'no' ဖြင့် ပြန်ကြားပါ။",

    # Cebuano
    "Palihug pag-determinar kung ang gihatag nga malisyoso nga tweet sa ubos gilantaw sa usa ka piho nga indibidwal, usa ka grupo, o uban pa. Kung mao, tubaga ang 'yes'. Kung ang dili maayo nga pinulongan sa tweet wala maglambigit, tubaga ang 'no'. Tweet: \"{tweet}\". Karon ang tweet mohuman na. Palihug motubag og 'yes' o 'no'.",

    # Khmer
    "សូមកំណត់ថាតើការប្រកាសខាងក្រោមដែលអាក្រក់នេះត្រូវបានផ្តោតលើជនបុគ្គលជាក់លាក់ក្រុមមួយឬអ្នកដទៃ។ ប្រសិនបើមាន សូមឆ្លើយថា 'yes' ។ ប្រសិនបើភាសាអាក្រក់នៃការប្រកាសមិនត្រូវបានផ្តោតសូមឆ្លើយថា 'no' ។ Tweet: \"{tweet}\" ។ ឥឡូវនេះការប្រកាសបានបញ្ចប់ហើយ សូមឆ្លើយថា 'yes' ឬ 'no'.",

    # Tagalog
    "Pakitukoy kung ang ibinigay na bastos na tweet sa ibaba ay nakatuon sa isang partikular na indibidwal, isang grupo, o iba pa. Kung oo, sumagot ng 'yes'. Kung ang bastos na wika ng tweet ay hindi nakatuon, sumagot ng 'no'. Tweet: \"{tweet}\". Ngayon natapos na ang tweet. Pakisagot ng 'yes' o 'no'.",

    # Hindi
    "कृपया यह निर्धारित करें कि नीचे दिया गया आपत्तिजनक ट्वीट किसी विशिष्ट व्यक्ति, समूह या अन्य लोगों को लक्षित करता है या नहीं। यदि हां, तो 'yes' के साथ उत्तर दें। यदि ट्वीट की आपत्तिजनक भाषा लक्षित नहीं है, तो 'no' के साथ उत्तर दें। ट्वीट: \"{tweet}\"। अब ट्वीट समाप्त हो गया है। कृपया 'yes' या 'no' के साथ उत्तर दें।",

    # Bengali
    "নীচে দেওয়া আপত্তিকর টুইটটি কোনও নির্দিষ্ট ব্যক্তি, গোষ্ঠী বা অন্যদের লক্ষ্য করে কিনা তা নির্ধারণ করুন। যদি তা হয়, তাহলে 'yes' দিয়ে উত্তর দিন। যদি টুইটের আপত্তিকর ভাষা লক্ষ্যবস্তু না হয়, তাহলে 'no' দিয়ে উত্তর দিন। টুইট: \"{tweet}\"। এখন টুইট শেষ। অনুগ্রহ করে 'yes' বা 'no' দিয়ে উত্তর দিন।",

    # Urdu
    "براہ کرم تعین کریں کہ آیا نیچے دیا گیا توہین آمیز ٹویٹ کسی خاص فرد، گروپ، یا دوسروں کو نشانہ بنا رہا ہے۔ اگر ایسا ہے تو، 'yes' کے ساتھ جواب دیں۔ اگر ٹویٹ کی توہین آمیز زبان کو نشانہ نہیں بنایا گیا ہے، تو 'no' کے ساتھ جواب دیں۔ ٹویٹ: \"{tweet}\"۔ اب ٹویٹ ختم ہو گیا ہے۔ براہ کرم 'yes' یا 'no' کے ساتھ جواب دیں۔"
]


# Create a blank dataframe
inst_data = pd.DataFrame(columns=['instruction', 'output'])

# Iterate over each tweet and create the instruction/output pair
for i, tweet in enumerate(data):
    if labels[i][0] == "OFF":
        output = get_output(labels[i])
        instruction = get_instruction(tweet, multilingual_templates)
        inst_data = pd.concat([inst_data, pd.DataFrame({'instruction': [instruction], 'output': [output]})], ignore_index=True)

print(inst_data.output.value_counts())
inst_data = balance_classes(inst_data, 500, "output")

# Save the dataframe to a parquet file
inst_data.to_parquet("../../data/SemEval2019Task6/SemEval19_task6_subB_multilingual.parquet", index=False)

# Sub C

# Define multilingual templates for the instruction
multilingual_templates = [
    # Chinese
    "请判断以下提供的攻击性推文是否针对个人、群体或其他人。如果是个人，请回答'IND'。如果是群体，请回答'GRP'。如果是其他人，请回答'OTH'。推文: \"{tweet}\"。现在推文结束。请用'IND'、'GRP'或'OTH'作答。",

    # English
    "Please determine if the provided offensive tweet below targets an individual, a group, or others. If individual, respond 'IND'. If group, respond 'GRP'. If others, respond 'OTH'. Tweet: \"{tweet}\". Now the tweet ends. Please respond with 'IND', 'GRP', or 'OTH'.",

    # German
    "Bitte bestimmen Sie, ob der unten bereitgestellte beleidigende Tweet auf eine Einzelperson, eine Gruppe oder andere abzielt. Wenn es sich um eine Einzelperson handelt, antworten Sie mit 'IND'. Wenn es sich um eine Gruppe handelt, antworten Sie mit 'GRP'. Wenn es sich um andere handelt, antworten Sie mit 'OTH'. Tweet: \"{tweet}\". Jetzt endet der Tweet. Bitte antworten Sie mit 'IND', 'GRP' oder 'OTH'.",

    # French
    "Veuillez déterminer si le tweet offensant ci-dessous cible un individu, un groupe ou d'autres. Si c'est un individu, répondez 'IND'. Si c'est un groupe, répondez 'GRP'. Si c'est d'autres, répondez 'OTH'. Tweet : \"{tweet}\". Maintenant, le tweet se termine. Veuillez répondre par 'IND', 'GRP' ou 'OTH'.",

    # Spanish
    "Por favor, determine si el tweet ofensivo proporcionado a continuación está dirigido a un individuo, un grupo u otros. Si es un individuo, responda 'IND'. Si es un grupo, responda 'GRP'. Si es otros, responda 'OTH'. Tweet: \"{tweet}\". Ahora el tweet termina. Responda con 'IND', 'GRP' o 'OTH'.",

    # Portuguese
    "Por favor, determine se o tweet ofensivo fornecido abaixo tem como alvo um indivíduo, um grupo ou outros. Se for um indivíduo, responda 'IND'. Se for um grupo, responda 'GRP'. Se for outros, responda 'OTH'. Tweet: \"{tweet}\". Agora o tweet termina. Por favor, responda com 'IND', 'GRP' ou 'OTH'.",

    # Italian
    "Si prega di determinare se il tweet offensivo fornito di seguito è mirato a un individuo, un gruppo o altri. Se è un individuo, rispondere 'IND'. Se è un gruppo, rispondere 'GRP'. Se è altri, rispondere 'OTH'. Tweet: \"{tweet}\". Ora il tweet finisce. Rispondere con 'IND', 'GRP' o 'OTH'.",

    # Dutch
    "Bepaal of de onderstaande beledigende tweet gericht is op een individu, een groep of anderen. Als het een individu is, antwoord dan met 'IND'. Als het een groep is, antwoord dan met 'GRP'. Als het anderen betreft, antwoord dan met 'OTH'. Tweet: \"{tweet}\". Nu eindigt de tweet. Antwoord alstublieft met 'IND', 'GRP' of 'OTH'.",

    # Russian
    "Пожалуйста, определите, направлен ли приведенный ниже оскорбительный твит на конкретного человека, группу или других лиц. Если на конкретного человека, ответьте 'IND'. Если на группу, ответьте 'GRP'. Если на других, ответьте 'OTH'. Твит: \"{tweet}\". Теперь твит заканчивается. Пожалуйста, ответьте 'IND', 'GRP' или 'OTH'.",

    # Czech
    "Prosím, určete, zda následující urážlivý tweet cílí na jednotlivce, skupinu nebo jiné. Pokud na jednotlivce, odpovězte 'IND'. Pokud na skupinu, odpovězte 'GRP'. Pokud na jiné, odpovězte 'OTH'. Tweet: \"{tweet}\". Nyní tweet končí. Odpovězte prosím 'IND', 'GRP' nebo 'OTH'.",

    # Polish
    "Proszę określić, czy poniższy obraźliwy tweet jest skierowany na osobę, grupę czy inne osoby. Jeśli na osobę, odpowiedz 'IND'. Jeśli na grupę, odpowiedz 'GRP'. Jeśli na inne osoby, odpowiedz 'OTH'. Tweet: \"{tweet}\". Teraz tweet się kończy. Odpowiedz proszę 'IND', 'GRP' lub 'OTH'.",

    # Arabic
    "يرجى تحديد ما إذا كانت التغريدة المسيئة أدناه تستهدف فردًا أو مجموعة أو غيرهم. إذا كان فردًا، فاستجب بـ 'IND'. إذا كانت مجموعة، فاستجب بـ 'GRP'. إذا كان آخرون، فاستجب بـ 'OTH'. التغريدة: \"{tweet}\". الآن انتهت التغريدة. يرجى الرد بـ 'IND' أو 'GRP' أو 'OTH'.",

    # Persian
    "لطفاً تعیین کنید که آیا توییت توهین‌آمیز زیر به یک فرد، یک گروه یا دیگران هدف می‌گیرد. اگر فرد است، پاسخ 'IND' دهید. اگر گروه است، پاسخ 'GRP' دهید. اگر دیگران، پاسخ 'OTH' دهید. توییت: \"{tweet}\". اکنون توییت به پایان می‌رسد. لطفاً با 'IND'، 'GRP' یا 'OTH' پاسخ دهید.",

    # Hebrew
    "אנא קבע אם הציוץ הפוגעני שלהלן מכוון לאדם יחיד, קבוצה או אחרים. אם מדובר באדם יחיד, השב 'IND'. אם מדובר בקבוצה, השב 'GRP'. אם מדובר באחרים, השב 'OTH'. ציוץ: \"{tweet}\". הציוץ נגמר כעת. אנא השב 'IND', 'GRP' או 'OTH'.",

    # Turkish
    "Lütfen aşağıdaki saldırgan tweet'in bir bireyi, bir grubu veya diğerlerini hedef alıp almadığını belirleyin. Eğer bireyse, 'IND' yanıtını verin. Eğer grup ise, 'GRP' yanıtını verin. Diğerleri ise, 'OTH' yanıtını verin. Tweet: \"{tweet}\". Tweet şimdi sona erdi. Lütfen 'IND', 'GRP' veya 'OTH' yanıtını verin.",

    # Japanese
    "以下の攻撃的なツイートが個人、グループ、または他の人を対象としているかどうかを判断してください。個人の場合は「IND」と答えてください。グループの場合は「GRP」と答えてください。他の場合は「OTH」と答えてください。ツイート：「{tweet}」。これでツイートは終了です。「IND」、「GRP」、または「OTH」で答えてください。",

    # Korean
    "아래 제공된 공격적인 트윗이 개인, 그룹 또는 다른 사람을 대상으로 하는지 확인해 주세요. 개인이라면 'IND'라고 답변하세요. 그룹이라면 'GRP'라고 답변하세요. 다른 경우에는 'OTH'라고 답변하세요. 트윗: \"{tweet}\". 이제 트윗이 끝났습니다. 'IND', 'GRP' 또는 'OTH'로 답변해 주세요.",

    # Vietnamese
    "Vui lòng xác định xem tweet xúc phạm được cung cấp dưới đây có nhắm mục tiêu vào một cá nhân, một nhóm hay những người khác không. Nếu là cá nhân, hãy trả lời 'IND'. Nếu là nhóm, hãy trả lời 'GRP'. Nếu là người khác, hãy trả lời 'OTH'. Tweet: \"{tweet}\". Bây giờ tweet kết thúc. Vui lòng trả lời bằng 'IND', 'GRP' hoặc 'OTH'.",

    # Thai
    "โปรดตรวจสอบว่าทวีตที่ให้ไว้ด้านล่างนี้มุ่งเป้าไปที่บุคคล กลุ่ม หรือคนอื่นๆ หรือไม่ หากเป็นบุคคล ให้ตอบ 'IND' หากเป็นกลุ่ม ให้ตอบ 'GRP' หากเป็นคนอื่นๆ ให้ตอบ 'OTH' ทวีต: \"{tweet}\" ตอนนี้ทวีตสิ้นสุดแล้ว โปรดตอบ 'IND' 'GRP' หรือ 'OTH'",

    # Indonesian
    "Silakan tentukan apakah tweet ofensif yang disediakan di bawah ini menargetkan individu, kelompok, atau lainnya. Jika individu, jawab 'IND'. Jika grup, jawab 'GRP'. Jika lainnya, jawab 'OTH'. Tweet: \"{tweet}\". Sekarang tweet berakhir. Silakan jawab dengan 'IND', 'GRP', atau 'OTH'.",

    # Malay
    "Sila tentukan sama ada tweet ofensif yang diberikan di bawah ini menyasarkan individu, kumpulan atau orang lain. Jika individu, jawab 'IND'. Jika kumpulan, jawab 'GRP'. Jika lain-lain, jawab 'OTH'. Tweet: \"{tweet}\". Sekarang tweet tamat. Sila jawab dengan 'IND', 'GRP', atau 'OTH'.",

    # Lao
    "ກະລຸນາກຳນົດວ່າຂໍ້ຄວາມທີ່ບໍ່ສຸພາບທີ່ສະເໜີຂ້າງລຸ່ມນີ້ເປັນການຕັ້ງເປົ້າໝາຍເອົາບຸກຄົນ, ກຸ່ມ, ຫຼື ອື່ນໆ. ຖ້າວ່າເປັນບຸກຄົນ, ກະລຸນາຕອບ 'IND'. ຖ້າວ່າເປັນກຸ່ມ, ກະລຸນາຕອບ 'GRP'. ຖ້າວ່າເປັນອື່ນໆ, ກະລຸນາຕອບ 'OTH'. Tweet: \"{tweet}\". ຂໍ້ຄວາມນີ້ສິ້ນສຸດລົງແລ້ວ. ກະລຸນາຕອບດ້ວຍ 'IND', 'GRP' ຫຼື 'OTH'.",

    # Burmese
    "အောက်တွင်ပေးထားသော စော်ကားမှု ပြုလုပ်သော တူဿ်သည် ပစ်မှတ်ထားသောအနေဖြင့် ဖြစ်ပါက၊ ဥပမာ အပစ်မှတ်ထားသော သီးခြား၊ အဖွဲ့အစည်း သို့မဟုတ် အခြားသူများကို ဖြည့်ရန်ဖြစ်ပါက 'IND' ဖြင့် ဖြေကြားပါ။ အဖွဲ့ကို ဖြည့်ရန်ဖြစ်ပါက 'GRP' ဖြင့် ဖြေကြားပါ။ အခြားများကို ဖြည့်ရန်ဖြစ်ပါက 'OTH' ဖြင့် ပြန်ကြားပါ။ တူဿ်: \"{tweet}\". ယခုတွင် တူဿ် ပြီးဆုံးပါပြီ။ 'IND' 'GRP' သို့မဟုတ် 'OTH' ဖြင့် ပြန်ကြားပါ။",

    # Cebuano
    "Palihug pag-determinar kung ang gihatag nga malisyoso nga tweet sa ubos gipuntirya ba ang usa ka indibidwal, usa ka grupo, o uban pa. Kung indibidwal, tubaga ang 'IND'. Kung grupo, tubaga ang 'GRP'. Kung uban pa, tubaga ang 'OTH'. Tweet: \"{tweet}\". Karon ang tweet mohuman na. Palihug motubag og 'IND', 'GRP' o 'OTH'.",

    # Khmer
    "សូមកំណត់ថាតើការប្រកាសផ្អែកលើការលោភលើបុគ្គល ក្រុមឬផ្សេងទៀតដែលផ្តល់ដោយស្វ័យប្រវត្តិ ឬដោយផ្ទាល់។ ប្រសិនបើលោភលើបុគ្គល សូមឆ្លើយថា 'IND'។ ប្រសិនបើលោភលើក្រុម សូមឆ្លើយថា 'GRP'។ ប្រសិនបើលោភលើផ្សេងទៀត សូមឆ្លើយថា 'OTH'។ ការប្រកាស: \"{tweet}\"។ ឥឡូវនេះការប្រកាសបានបញ្ចប់ហើយ សូមឆ្លើយថា 'IND' 'GRP' ឬ 'OTH'.",

    # Tagalog
    "Pakitukoy kung ang ibinigay na tweet sa ibaba ay nagta-target sa isang indibidwal, isang grupo, o iba pa. Kung indibidwal, sumagot ng 'IND'. Kung grupo, sumagot ng 'GRP'. Kung iba pa, sumagot ng 'OTH'. Tweet: \"{tweet}\". Ngayon natapos na ang tweet. Pakisagot ng 'IND', 'GRP' o 'OTH'.",

    # Hindi
    "कृपया यह निर्धारित करें कि नीचे दिया गया आपत्तिजनक ट्वीट किसी व्यक्ति, समूह, या अन्य को लक्षित करता है। यदि व्यक्ति को लक्षित करता है, तो 'IND' के साथ उत्तर दें। यदि समूह को लक्षित करता है, तो 'GRP' के साथ उत्तर दें। यदि अन्य को लक्षित करता है, तो 'OTH' के साथ उत्तर दें। ट्वीट: \"{tweet}\"। अब ट्वीट समाप्त हो गया है। कृपया 'IND', 'GRP' या 'OTH' के साथ उत्तर दें।",

    # Bengali
    "নীচে দেওয়া অপমানজনক টুইটটি কোনও ব্যক্তি, একটি গোষ্ঠী বা অন্যদের লক্ষ্য করে কিনা তা নির্ধারণ করুন। যদি এটি একজন ব্যক্তির লক্ষ্য হয়, তাহলে 'IND' দিয়ে উত্তর দিন। যদি এটি একটি গোষ্ঠীর লক্ষ্য হয়, তাহলে 'GRP' দিয়ে উত্তর দিন। যদি এটি অন্যদের লক্ষ্য হয়, তাহলে 'OTH' দিয়ে উত্তর দিন। টুইট: \"{tweet}\"। এখন টুইট শেষ। অনুগ্রহ করে 'IND', 'GRP' বা 'OTH' দিয়ে উত্তর দিন।",

    # Urdu
    "براہ کرم تعین کریں کہ آیا نیچے دیا گیا توہین آمیز ٹویٹ کسی فرد، گروپ یا دوسروں کو نشانہ بناتا ہے۔ اگر فرد کو نشانہ بناتا ہے، تو 'IND' کے ساتھ جواب دیں۔ اگر گروپ کو نشانہ بناتا ہے، تو 'GRP' کے ساتھ جواب دیں۔ اگر دوسروں کو نشانہ بناتا ہے، تو 'OTH' کے ساتھ جواب دیں۔ ٹویٹ: \"{tweet}\"۔ اب ٹویٹ ختم ہو گیا ہے۔ براہ کرم 'IND'، 'GRP' یا 'OTH' کے ساتھ جواب دیں۔"
]

# Create a blank dataframe
inst_data = pd.DataFrame(columns=['instruction', 'output'])

# Iterate over each tweet and create the instruction/output pair
for i, tweet in enumerate(data):
    if labels[i][1] == "TIN":
        output = labels[i][2]
        instruction = get_instruction(tweet, multilingual_templates)
        inst_data = pd.concat([inst_data, pd.DataFrame({'instruction': [instruction], 'output': [output]})], ignore_index=True)

print(inst_data.output.value_counts())
inst_data = balance_classes(inst_data, 300, "output")

# Save the dataframe to a parquet file
inst_data.to_parquet("../../data/SemEval2019Task6/SemEval19_task6_subC_multilingual.parquet", index=False)
