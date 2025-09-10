# -*- coding: utf-8 -*-
# HSOL (Hate Speech and Offensive Language) ICWSM 2017
# - Paper: https://doi.org/10.48550/arXiv.1703.04009
# - Data: https://github.com/t-davidson/hate-speech-and-offensive-language/tree/master/data

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import pandas as pd
import random
from utils.data_utils import downsample_to_minority_class

# Load data
df = pd.read_csv('../../data/HSOL/labeled_data.csv')

# Define multilingual templates for each category
hate_speech_templates = [
    # Chinese
    """以下是一个社交媒体推文: {text}\n\n 如果推文包含用于表达对特定群体仇恨的语言，或意在贬低、侮辱该群体成员，则回答 'yes'。否则，回答 'no'。""",

    # English
    """Here's a social media tweet: {text}\n\n Respond 'yes' if the tweet contains language that is used to express hatred towards a targeted group or is intended to be derogatory, to humiliate, or to insult the members of the group. Otherwise, respond 'no'.""",

    # German
    """Hier ist ein Social-Media-Tweet: {text}\n\n Antworten Sie mit 'yes', wenn der Tweet Sprache enthält, die verwendet wird, um Hass gegen eine bestimmte Gruppe auszudrücken, oder dazu bestimmt ist, die Mitglieder der Gruppe zu demütigen oder zu beleidigen. Andernfalls antworten Sie mit 'no'.""",

    # French
    """Voici un tweet des médias sociaux: {text}\n\n Répondez 'yes' si le tweet contient un langage utilisé pour exprimer de la haine envers un groupe ciblé ou est destiné à être dérogatoire, à humilier ou à insulter les membres du groupe. Sinon, répondez 'no'.""",

    # Spanish
    """Aquí hay un tweet de redes sociales: {text}\n\n Responda 'yes' si el tweet contiene lenguaje que se utiliza para expresar odio hacia un grupo objetivo o tiene la intención de ser despectivo, humillante o insultante hacia los miembros del grupo. De lo contrario, responda 'no'.""",

    # Portuguese
    """Aqui está um tweet de mídia social: {text}\n\n Responda 'yes' se o tweet contém linguagem usada para expressar ódio contra um grupo alvo ou se destina a ser depreciativo, humilhar ou insultar os membros do grupo. Caso contrário, responda 'no'.""",

    # Italian
    """Ecco un tweet sui social media: {text}\n\n Rispondi 'yes' se il tweet contiene linguaggio utilizzato per esprimere odio verso un gruppo mirato o è destinato a essere dispregiativo, umiliare o insultare i membri del gruppo. Altrimenti, rispondi 'no'.""",

    # Dutch
    """Hier is een social media-tweet: {text}\n\n Antwoord 'yes' als de tweet taal bevat die wordt gebruikt om haat tegen een specifieke groep uit te drukken of bedoeld is om de leden van de groep te vernederen of te beledigen. Anders antwoord 'no'.""",

    # Russian
    """Вот твит из социальных сетей: {text}\n\n Ответьте 'yes', если твит содержит язык, используемый для выражения ненависти к определенной группе или предназначен для унижения или оскорбления членов группы. В противном случае ответьте 'no'.""",

    # Czech
    """Zde je tweet ze sociálních médií: {text}\n\n Odpovězte 'yes', pokud tweet obsahuje jazyk, který se používá k vyjádření nenávisti vůči cílové skupině, nebo má za cíl ponížit nebo urazit členy skupiny. Jinak odpovězte 'no'.""",

    # Polish
    """Oto tweet z mediów społecznościowych: {text}\n\n Odpowiedz 'yes', jeśli tweet zawiera język używany do wyrażania nienawiści wobec określonej grupy lub jest przeznaczony do poniżania lub obrażania członków grupy. W przeciwnym razie odpowiedz 'no'.""",

    # Arabic
    """إليك تغريدة من وسائل التواصل الاجتماعي: {text}\n\n أجب بـ 'yes' إذا كانت التغريدة تحتوي على لغة تُستخدم للتعبير عن الكراهية تجاه مجموعة مستهدفة أو تهدف إلى الإهانة أو الإذلال أو إهانة أعضاء المجموعة. وإلا، أجب بـ 'no'.""",

    # Persian
    """در اینجا یک توییت از شبکه‌های اجتماعی است: {text}\n\n پاسخ 'yes' بدهید اگر توییت شامل زبانی است که برای ابراز نفرت نسبت به یک گروه هدف استفاده می‌شود یا هدف آن تحقیر یا توهین به اعضای گروه است. در غیر این صورت، پاسخ 'no' بدهید.""",

    # Hebrew
    """הנה ציוץ מהרשתות החברתיות: {text}\n\n השב 'yes' אם הציוץ מכיל שפה המשמשת להבעת שנאה כלפי קבוצה מסוימת או מיועדת לפגוע, להעליב או להשפיל את חברי הקבוצה. אחרת, השב 'no'.""",

    # Turkish
    """İşte bir sosyal medya tweeti: {text}\n\n Tweet, belirli bir gruba yönelik nefret ifade eden bir dil içeriyorsa veya grubun üyelerini aşağılamak, küçük düşürmek veya hakaret etmek amacıyla yazılmışsa 'yes' yanıtını verin. Aksi takdirde 'no' yanıtını verin.""",

    # Japanese
    """以下はソーシャルメディアのツイートです: {text}\n\n ターゲットグループに対する憎しみを表現するために使用される言語を含むか、グループのメンバーを侮辱することを意図している場合は「yes」と答えてください。 それ以外の場合は「no」と答えてください。""",

    # Korean
    """다음은 소셜 미디어의 트윗입니다: {text}\n\n 특정 그룹에 대한 증오를 표현하거나 그룹 구성원을 비하하거나 모욕하기 위한 목적으로 사용되는 언어가 포함되어 있으면 'yes'라고 응답하십시오. 그렇지 않으면 'no'라고 응답하십시오.""",

    # Vietnamese
    """Đây là một tweet trên mạng xã hội: {text}\n\n Trả lời 'yes' nếu tweet chứa ngôn ngữ được sử dụng để bày tỏ sự thù ghét đối với một nhóm mục tiêu hoặc nhằm mục đích xúc phạm hoặc làm nhục các thành viên của nhóm. Ngược lại, trả lời 'no'.""",

    # Thai
    """นี่คือตัวอย่างทวีตจากสื่อสังคมออนไลน์: {text}\n\n หากทวีตประกอบด้วยภาษาที่ใช้เพื่อแสดงความเกลียดชังต่อกลุ่มเป้าหมาย หรือมีเจตนาที่จะดูหมิ่นหรือเหยียดหยามสมาชิกของกลุ่ม ให้ตอบว่า 'yes' มิฉะนั้นตอบ 'no'""",

    # Indonesian
    """Berikut adalah tweet dari media sosial: {text}\n\n Jawab 'yes' jika tweet tersebut mengandung bahasa yang digunakan untuk mengekspresikan kebencian terhadap kelompok yang ditargetkan atau dimaksudkan untuk menghina atau menghina anggota kelompok. Jika tidak, jawab 'no'.""",

    # Malay
    """Ini adalah tweet dari media sosial: {text}\n\n Jawab 'yes' jika tweet tersebut mengandungi bahasa yang digunakan untuk menyatakan kebencian terhadap kumpulan yang disasarkan atau bertujuan untuk menghina atau menghina ahli kumpulan tersebut. Jika tidak, jawab 'no'.""",

    # Lao
    """นี่คือทวีตจากสื่อสังคมออนไลน์: {text}\n\n หากทวีตมีภาษาที่ใช้เพื่อแสดงความเกลียดชังต่อกลุ่มเป้าหมายหรือตั้งใจที่จะเหยียดหยามสมาชิกของกลุ่มให้ตอบว่า 'yes' มิฉะนั้นให้ตอบว่า 'no'""",

    # Burmese
    """ဤတွင်ဆိုရှယ်မီဒီယာမှတွစ်တာဖြစ်ပါသည်: {text}\n\n တွစ်တာတွင် အုပ်စုတစ်ခုအပေါ် မုန်းတီးမှုကိုဖော်ပြရန် သုံးသောဘာသာစကားပါဝင်သည် သို့မဟုတ် အုပ်စု၏အဖွဲ့ဝင်များကို ရိုက်ချေရာမှာ ရည်ရွယ်ထားလျှင် 'yes' ဟု ဖြေပါ။ မဟုတ်ရင် 'no' ဟု ဖြေပါ။""",

    # Cebuano
    """Aniay usa ka tweet gikan sa social media: {text}\n\n Tubaga og 'yes' kung ang tweet naglangkob og sinultihan nga gigamit sa pagpadayag og kasuko batok sa usa ka tinarget nga grupo o gituyo nga manglait o mang-insulto sa mga miyembro sa grupo. Kung dili, tubaga og 'no'.""",

    # Khmer
    """នេះគឺជាអត្ថបទប្រព័ន្ធផ្សព្វផ្សាយសង្គមមួយ: {text}\n\n សូមឆ្លើយថា 'yes' ប្រសិនបើអត្ថបទនេះមានភាសាដែលត្រូវបានប្រើដើម្បីបង្ហាញភាពស្អប់ខ្ពើមចំពោះក្រុមតាមគោលដៅ ឬបំណងប្រមាថ ឬរើសអើងសមាជិកក្រុម។ អ្វីដូចនេះបើមិនសូមឆ្លើយថា 'no'។""",

    # Tagalog
    """Narito ang isang tweet mula sa social media: {text}\n\n Sagutin ng 'yes' kung ang tweet ay naglalaman ng wika na ginagamit upang ipahayag ang pagkamuhi sa isang target na grupo o nilalayong maging derogatoryo, upang hamakin, o upang mang-insulto sa mga miyembro ng grupo. Kung hindi, sagutin ng 'no'.""",

    # Hindi
    """यहाँ एक सोशल मीडिया ट्वीट है: {text}\n\n 'yes' का उत्तर दें यदि ट्वीट में भाषा है जो लक्षित समूह के प्रति घृणा व्यक्त करने के लिए उपयोग की जाती है या समूह के सदस्यों को अपमानित या अपमानित करने के लिए है। अन्यथा, 'no' का उत्तर दें।""",

    # Bengali
    """এখানে একটি সামাজিক মিডিয়া টুইট রয়েছে: {text}\n\n 'yes' উত্তর দিন যদি টুইটটিতে এমন ভাষা থাকে যা একটি লক্ষ্যযুক্ত গোষ্ঠীর প্রতি ঘৃণা প্রকাশ করতে ব্যবহৃত হয় বা গোষ্ঠীর সদস্যদের অপমানিত বা অপমানিত করতে হয়। অন্যথায়, 'no' উত্তর দিন।""",

    # Urdu
    """یہاں ایک سوشل میڈیا ٹویٹ ہے: {text}\n\n اگر ٹویٹ میں ایسی زبان ہے جو کسی خاص گروپ کے خلاف نفرت کا اظہار کرنے کے لیے استعمال ہوتی ہے یا گروپ کے اراکین کو ذلیل یا توہین کرنے کے لیے ہے تو 'yes' کا جواب دیں۔ ورنہ، 'no' کا جواب دیں۔"""
]


offensive_templates = [
    # Chinese
    """以下是一个社交媒体推文: {text}\n\n 如果推文使用不尊重或粗俗的语言，这些语言可能会冒犯他人，但没有针对特定群体表达仇恨或造成伤害的意图，请回答 'yes'。否则，回答 'no'。""",

    # English
    """Here's a social media tweet: {text}\n\n Respond 'yes' if the tweet uses disrespectful or vulgar speech that may offend but does not target a group with hatred or intent to harm. Otherwise, respond 'no'.""",

    # German
    """Hier ist ein Social-Media-Tweet: {text}\n\n Antworten Sie mit 'yes', wenn der Tweet respektlose oder vulgäre Sprache verwendet, die möglicherweise beleidigend ist, aber keine Gruppe mit Hass oder Schädigungsabsicht anspricht. Andernfalls antworten Sie mit 'no'.""",

    # French
    """Voici un tweet des médias sociaux: {text}\n\n Répondez 'yes' si le tweet utilise un langage irrespectueux ou vulgaire qui peut offenser mais ne cible pas un groupe avec haine ou intention de nuire. Sinon, répondez 'no'.""",

    # Spanish
    """Aquí hay un tweet de redes sociales: {text}\n\n Responda 'yes' si el tweet utiliza un lenguaje irrespetuoso o vulgar que puede ofender pero no apunta a un grupo con odio o intención de hacer daño. De lo contrario, responda 'no'.""",

    # Portuguese
    """Aqui está um tweet de mídia social: {text}\n\n Responda 'yes' se o tweet usar linguagem desrespeitosa ou vulgar que possa ofender, mas não visa um grupo com ódio ou intenção de prejudicar. Caso contrário, responda 'no'.""",

    # Italian
    """Ecco un tweet sui social media: {text}\n\n Rispondi 'yes' se il tweet utilizza un linguaggio irrispettoso o volgare che può offendere ma non prende di mira un gruppo con odio o intento di danneggiare. Altrimenti, rispondi 'no'.""",

    # Dutch
    """Hier is een social media-tweet: {text}\n\n Antwoord 'yes' als de tweet respectloze of vulgaire taal gebruikt die beledigend kan zijn, maar geen groep met haat of de intentie om te schaden aanvalt. Anders antwoord 'no'.""",

    # Russian
    """Вот твит из социальных сетей: {text}\n\n Ответьте 'yes', если твит использует неуважительную или вульгарную речь, которая может оскорбить, но не направлена на группу с ненавистью или намерением причинить вред. В противном случае ответьте 'no'.""",

    # Czech
    """Zde je tweet ze sociálních médií: {text}\n\n Odpovězte 'yes', pokud tweet používá neuctivou nebo vulgární řeč, která může urážet, ale není zaměřena na skupinu s nenávistí nebo úmyslem ublížit. Jinak odpovězte 'no'.""",

    # Polish
    """Oto tweet z mediów społecznościowych: {text}\n\n Odpowiedz 'yes', jeśli tweet używa lekceważącego lub wulgarnego języka, który może obrażać, ale nie jest skierowany przeciwko grupie z nienawiścią lub zamiarami szkodzenia. W przeciwnym razie odpowiedz 'no'.""",

    # Arabic
    """إليك تغريدة من وسائل التواصل الاجتماعي: {text}\n\n أجب بـ 'yes' إذا كانت التغريدة تستخدم لغة غير محترمة أو بذيئة قد تسيء ولكنها لا تستهدف مجموعة بالكراهية أو النية للإيذاء. وإلا، أجب بـ 'no'.""",

    # Persian
    """در اینجا یک توییت از شبکه‌های اجتماعی است: {text}\n\n پاسخ 'yes' بدهید اگر توییت از زبان بی‌احترامی یا بی‌ادبی استفاده می‌کند که ممکن است توهین‌آمیز باشد اما گروهی را با نفرت یا نیت آسیب هدف نمی‌گیرد. در غیر این صورت، پاسخ 'no' بدهید.""",

    # Hebrew
    """הנה ציוץ מהרשתות החברתיות: {text}\n\n השב 'yes' אם הציוץ משתמש בשפה לא מכובדת או גסה שעשויה לפגוע אך לא מכוונת לפגוע בקבוצה מתוך שנאה או כוונה להזיק. אחרת, השב 'no'.""",

    # Turkish
    """İşte bir sosyal medya tweeti: {text}\n\n Tweet, saygısız veya kaba bir dil kullanıyorsa ve bir gruba nefretle veya zarar verme niyetiyle hedef almıyorsa 'yes' yanıtını verin. Aksi takdirde 'no' yanıtını verin.""",

    # Japanese
    """以下はソーシャルメディアのツイートです: {text}\n\n ツイートが不敬または卑俗な表現を使用しているが、特定のグループを憎悪や害の意図で対象としていない場合は「yes」と答えてください。それ以外の場合は「no」と答えてください。""",

    # Korean
    """다음은 소셜 미디어의 트윗입니다: {text}\n\n 트윗이 무례하거나 저속한 표현을 사용하지만 특정 그룹을 증오하거나 해를 입히려는 의도로 겨냥하지 않는 경우 'yes'라고 응답하십시오. 그렇지 않으면 'no'라고 응답하십시오.""",

    # Vietnamese
    """Đây là một tweet trên mạng xã hội: {text}\n\n Trả lời 'yes' nếu tweet sử dụng ngôn ngữ thiếu tôn trọng hoặc thô tục có thể xúc phạm nhưng không nhắm vào một nhóm với sự thù ghét hoặc ý định gây hại. Ngược lại, trả lời 'no'.""",

    # Thai
    """นี่คือตัวอย่างทวีตจากสื่อสังคมออนไลน์: {text}\n\n หากทวีตใช้ภาษาที่ไม่สุภาพหรือหยาบคายที่อาจทำให้ขุ่นเคืองแต่ไม่ได้มุ่งเป้าไปที่กลุ่มด้วยความเกลียดชังหรือเจตนาร้าย ให้ตอบว่า 'yes' มิฉะนั้นตอบ 'no'""",

    # Indonesian
    """Berikut adalah tweet dari media sosial: {text}\n\n Jawab 'yes' jika tweet tersebut menggunakan bahasa yang tidak sopan atau vulgar yang dapat menyinggung tetapi tidak menargetkan suatu kelompok dengan kebencian atau niat untuk merugikan. Jika tidak, jawab 'no'.""",

    # Malay
    """Ini adalah tweet dari media sosial: {text}\n\n Jawab 'yes' jika tweet tersebut menggunakan bahasa yang tidak sopan atau kasar yang mungkin menyinggung tetapi tidak menyasarkan kumpulan dengan kebencian atau niat untuk mencederakan. Jika tidak, jawab 'no'.""",

    # Lao
    """นี่คือทวีตจากสื่อสังคมออนไลน์: {text}\n\n หากทวีตใช้ภาษาที่ไม่สุภาพหรือหยาบคายที่อาจทำให้ขุ่นเคืองแต่ไม่ได้มุ่งเป้าไปที่กลุ่มด้วยความเกลียดชังหรือเจตนาร้ายให้ตอบว่า 'yes' มิฉะนั้นให้ตอบว่า 'no'""",

    # Burmese
    """ဤတွင်ဆိုရှယ်မီဒီယာမှတွစ်တာဖြစ်ပါသည်: {text}\n\n တွစ်တာတွင် လေးစားမှုမရှိသော သို့မဟုတ် အညှော်အကြည်များ ပါသော ဗဟုသုတရှိသော်လည်း အုပ်စုတစ်ခုကို မုန်းတီးမှု သို့မဟုတ် နာကျင်စေမှုရည်ရွယ်ချက် မပါဘဲ 'yes' ဟု ဖြေပါ။ မဟုတ်ရင် 'no' ဟု ဖြေပါ။""",

    # Cebuano
    """Aniay usa ka tweet gikan sa social media: {text}\n\n Tubaga og 'yes' kung ang tweet naggamit og dili respetuhon o bastos nga sinultihan nga mahimong makasuko apan wala magtarget og grupo uban ang kasuko o tinguha nga makadaot. Kung dili, tubaga og 'no'.""",

    # Khmer
    """នេះគឺជាអត្ថបទប្រព័ន្ធផ្សព្វផ្សាយសង្គមមួយ: {text}\n\n សូមឆ្លើយថា 'yes' ប្រសិនបើអត្ថបទនេះប្រើភាសាដែលមិនគោរពឬជិតច្រើនមកដល់អាចធ្វើឱ្យគេបោះពុម្ពបុណ្ណោះ ប៉ុន្តែមិនចំពោះក្រុមដែលមានមោទនៈប្រយោគឬគំរប់ប្រហារ។""",

    # Tagalog
    """Narito ang isang tweet mula sa social media: {text}\n\n Sagutin ng 'yes' kung ang tweet ay gumagamit ng bastos o di magalang na pananalita na maaaring makasakit ngunit hindi nakatuon sa isang grupo nang may galit o layuning manakit. Kung hindi, sagutin ng 'no'.""",

    # Hindi
    """यहाँ एक सोशल मीडिया ट्वीट है: {text}\n\n 'yes' का उत्तर दें यदि ट्वीट असभ्य या अशिष्ट भाषा का उपयोग करता है जो अपमानजनक हो सकता है, लेकिन नफरत या नुकसान पहुँचाने के इरादे से किसी समूह को लक्षित नहीं करता है। अन्यथा, 'no' का उत्तर दें।""",

    # Bengali
    """এখানে একটি সামাজিক মিডিয়া টুইট রয়েছে: {text}\n\n 'yes' উত্তর দিন যদি টুইটটি অসম্মানজনক বা অশ্লীল ভাষা ব্যবহার করে যা অপমানজনক হতে পারে, কিন্তু একটি গোষ্ঠীকে ঘৃণা বা ক্ষতির উদ্দেশ্যে লক্ষ্য করে না। অন্যথায়, 'no' উত্তর দিন।""",

    # Urdu
    """یہاں ایک سوشل میڈیا ٹویٹ ہے: {text}\n\n اگر ٹویٹ میں بے حرمتی یا فحش زبان استعمال کی گئی ہے جو ناگوار ہو سکتی ہے لیکن کسی گروپ کو نفرت یا نقصان پہنچانے کے ارادے سے نشانہ نہیں بناتی تو 'yes' کا جواب دیں۔ ورنہ، 'no' کا جواب دیں۔"""
]


neither_templates = [
    # Chinese
    """以下是一个社交媒体推文: {text}\n\n 如果推文既不是冒犯性言论，也不是仇恨言论，且没有粗俗、侮辱或仇恨意图，请回答 'yes'。否则，回答 'no'。""",

    # English
    """Here's a social media tweet: {text}\n\n Respond 'yes' if the tweet is neither offensive nor hate speech, free of vulgarity, insults, or hateful intent. Otherwise, respond 'no'.""",

    # German
    """Hier ist ein Social-Media-Tweet: {text}\n\n Antworten Sie mit 'yes', wenn der Tweet weder beleidigend noch Hassrede ist und frei von Vulgarität, Beleidigungen oder hasserfüllten Absichten. Andernfalls antworten Sie mit 'no'.""",

    # French
    """Voici un tweet des médias sociaux: {text}\n\n Répondez 'yes' si le tweet n'est ni offensant ni un discours de haine, et s'il est exempt de vulgarité, d'insultes ou d'intentions haineuses. Sinon, répondez 'no'.""",

    # Spanish
    """Aquí hay un tweet de redes sociales: {text}\n\n Responda 'yes' si el tweet no es ni ofensivo ni un discurso de odio, y está libre de vulgaridad, insultos o intención de odio. De lo contrario, responda 'no'.""",

    # Portuguese
    """Aqui está um tweet de mídia social: {text}\n\n Responda 'yes' se o tweet não for nem ofensivo nem discurso de ódio, e estiver livre de vulgaridade, insultos ou intenção odiosa. Caso contrário, responda 'no'.""",

    # Italian
    """Ecco un tweet sui social media: {text}\n\n Rispondi 'yes' se il tweet non è né offensivo né un discorso di odio, ed è privo di volgarità, insulti o intenti odiosi. Altrimenti, rispondi 'no'.""",

    # Dutch
    """Hier is een social media-tweet: {text}\n\n Antwoord 'yes' als de tweet noch beledigend noch haatspraak is, en vrij is van vulgariteit, beledigingen of hatelijke intenties. Anders antwoord 'no'.""",

    # Russian
    """Вот твит из социальных сетей: {text}\n\n Ответьте 'yes', если твит не является ни оскорбительным, ни речью ненависти, и не содержит вульгарности, оскорблений или ненавистных намерений. В противном случае ответьте 'no'.""",

    # Czech
    """Zde je tweet ze sociálních médií: {text}\n\n Odpovězte 'yes', pokud tweet není ani urážlivý, ani nenávistný projev, a neobsahuje vulgarity, urážky ani nenávistné úmysly. Jinak odpovězte 'no'.""",

    # Polish
    """Oto tweet z mediów społecznościowych: {text}\n\n Odpowiedz 'yes', jeśli tweet nie jest ani obraźliwy, ani mową nienawiści, i jest wolny od wulgaryzmów, obelg lub nienawistnych zamiarów. W przeciwnym razie odpowiedz 'no'.""",

    # Arabic
    """إليك تغريدة من وسائل التواصل الاجتماعي: {text}\n\n أجب بـ 'yes' إذا لم تكن التغريدة لا مسيئة ولا خطاب كراهية، وكانت خالية من الألفاظ النابية أو الإهانات أو النوايا الكراهية. وإلا، أجب بـ 'no'.""",

    # Persian
    """در اینجا یک توییت از شبکه‌های اجتماعی است: {text}\n\n پاسخ 'yes' بدهید اگر توییت نه توهین‌آمیز است و نه نفرت‌انگیز، و عاری از کلمات ناشایست، توهین یا نیت‌های نفرت‌انگیز باشد. در غیر این صورت، پاسخ 'no' بدهید.""",

    # Hebrew
    """הנה ציוץ מהרשתות החברתיות: {text}\n\n השב 'yes' אם הציוץ אינו פוגעני ואינו דברי שנאה, ונטול גסויות, עלבונות או כוונות שנאה. אחרת, השב 'no'.""",

    # Turkish
    """İşte bir sosyal medya tweeti: {text}\n\n Tweet ne saldırgan ne de nefret söylemi ise ve kaba ifadeler, hakaretler veya nefret içeren bir niyet taşımıyorsa 'yes' yanıtını verin. Aksi takdirde 'no' yanıtını verin.""",

    # Japanese
    """以下はソーシャルメディアのツイートです: {text}\n\n ツイートが攻撃的でも憎悪表現でもなく、卑俗さ、侮辱、または憎悪の意図がない場合は「yes」と答えてください。それ以外の場合は「no」と答えてください。""",

    # Korean
    """다음은 소셜 미디어의 트윗입니다: {text}\n\n 트윗이 공격적이지도 않고 증오 발언도 아니며, 저속함, 모욕 또는 증오의 의도가 없는 경우 'yes'라고 응답하십시오. 그렇지 않으면 'no'라고 응답하십시오.""",

    # Vietnamese
    """Đây là một tweet trên mạng xã hội: {text}\n\n Trả lời 'yes' nếu tweet không phải là xúc phạm hay là lời nói căm thù, và không có sự thô tục, xúc phạm hoặc ác ý. Ngược lại, trả lời 'no'.""",

    # Thai
    """นี่คือตัวอย่างทวีตจากสื่อสังคมออนไลน์: {text}\n\n หากทวีตไม่ได้เป็นคำพูดที่เป็นภัยหรือเกลียดชัง ปราศจากความหยาบคาย การดูหมิ่น หรือเจตนาร้าย ให้ตอบว่า 'yes' มิฉะนั้นตอบ 'no'""",

    # Indonesian
    """Berikut adalah tweet dari media sosial: {text}\n\n Jawab 'yes' jika tweet tersebut tidak bersifat ofensif atau pidato kebencian, dan bebas dari vulgaritas, penghinaan, atau niat jahat. Jika tidak, jawab 'no'.""",

    # Malay
    """Ini adalah tweet dari media sosial: {text}\n\n Jawab 'yes' jika tweet tersebut bukan ofensif atau pidato kebencian, dan bebas daripada bahasa kasar, penghinaan, atau niat jahat. Jika tidak, jawab 'no'.""",

    # Lao
    """นี่คือทวีตจากสื่อสังคมออนไลน์: {text}\n\n หากทวีตไม่ใช่คำพูดที่เป็นภัยหรือคำพูดเกลียดชัง ปราศจากความหยาบคาย การดูหมิ่น หรือเจตนาร้าย ให้ตอบว่า 'yes' มิฉะนั้นให้ตอบว่า 'no'""",

    # Burmese
    """ဤတွင်ဆိုရှယ်မီဒီယာမှတွစ်တာဖြစ်ပါသည်: {text}\n\n တွစ်တာသည် မတော်တဆမှု မရှိသည့်၊ မုန်းတီးမှုမရှိသည့်၊ စော်ကားမှု၊ ထင်ဟပ်မှုမပါသောအချိန်တွင် 'yes' ဟု ဖြေပါ။ မဟုတ်ရင် 'no' ဟု ဖြေပါ။""",

    # Cebuano
    """Aniay usa ka tweet gikan sa social media: {text}\n\n Tubaga og 'yes' kung ang tweet dili walay pag-abuso ni hate speech, gawas sa kabastusan, insulto, o dautan nga tumong. Kung dili, tubaga og 'no'.""",

    # Khmer
    """នេះគឺជាអត្ថបទប្រព័ន្ធផ្សព្វផ្សាយសង្គមមួយ: {text}\n\n សូមឆ្លើយថា 'yes' ប្រសិនបើអត្ថបទនេះមិនមែនជាការប្រមាថឬការប្រឆាំងពីគំនិតអាក្រក់ ឬឧបាយកលបំផុតដែលនឹងលុបបំបាត់យកចិត្តមិនគោរព។""",

    # Tagalog
    """Narito ang isang tweet mula sa social media: {text}\n\n Sagutin ng 'yes' kung ang tweet ay hindi nakakasakit o hate speech, at malaya sa kabastusan, pang-iinsulto, o masamang intensyon. Kung hindi, sagutin ng 'no'.""",

    # Hindi
    """यहाँ एक सोशल मीडिया ट्वीट है: {text}\n\n 'yes' का उत्तर दें यदि ट्वीट न तो अपमानजनक है और न ही घृणा भाषण है, और वह अश्लीलता, अपमान या घृणित इरादे से मुक्त है। अन्यथा, 'no' का उत्तर दें।""",

    # Bengali
    """এখানে একটি সামাজিক মিডিয়া টুইট রয়েছে: {text}\n\n 'yes' উত্তর দিন যদি টুইটটি না হয় অশ্লীল বা ঘৃণামূলক বক্তব্য, এবং অশালীনতা, অপমান বা ঘৃণ্য উদ্দেশ্য থেকে মুক্ত থাকে। অন্যথায়, 'no' উত্তর দিন।""",

    # Urdu
    """یہاں ایک سوشل میڈیا ٹویٹ ہے: {text}\n\n اگر ٹویٹ نہ تو جارحانہ ہے اور نہ ہی نفرت انگیز تقریر، اور فحاشی، توہین یا نفرت انگیز ارادے سے پاک ہے تو 'yes' کا جواب دیں۔ ورنہ، 'no' کا جواب دیں۔"""
]


# Function to create instruction based on the selected category and randomly chosen template
def create_instruction(text, category):
    if category == 0:
        instruction_template = random.choice(hate_speech_templates)
    elif category == 1:
        instruction_template = random.choice(offensive_templates)
    else:
        instruction_template = random.choice(neither_templates)

    instruction = instruction_template.format(text=text)
    return instruction

# Create a blank DataFrame
instructions_df = pd.DataFrame(columns=['instruction', 'output'])

# Iterate over each row and create instructions for each category
for index, row in df.iterrows():
    text = row['tweet']
    for category in [0, 1, 2]:
        instruction = create_instruction(text, category)
        output = 'yes' if row['class'] == category else 'no'
        instructions_df = pd.concat([instructions_df, pd.DataFrame({'instruction': [instruction], 'output': [output]})], ignore_index=True)

# Save the DataFrame to a parquet file
instructions_df.to_parquet("../../data/HSOL/HSOL.parquet", index=False)

# Example usage:
# Assuming df_result is your DataFrame and 'output' is the target column
instruction_df_balanced = downsample_to_minority_class(instructions_df, 'output')
instruction_df_balanced.to_parquet("../../data/HSOL/HSOL.parquet", index=False)
instruction_df_balanced.output.value_counts()
# sample n=48000 and save to the same parquet
instruction_df_balanced.sample(n=48000, random_state=42).to_parquet("../../data/HSOL/HSOL.parquet", index=False)