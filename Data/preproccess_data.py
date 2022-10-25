from dataclasses import replace
from fnmatch import translate
import string
import pymongo
import re
import sys
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import unicodedata as ud

"""
    Start section: Get data from database
    Save data to corpus variable
"""

client = pymongo.MongoClient("mongodb+srv://fbfighter:fbfighter@fb-topic.ixbkp2u.mongodb.net/?retryWrites=true&w=majority")
db = client["fakenews"]
col = db["fbpost"]
data = col.find()

corpus = []
labels = []

for x in data:
  if 'is_fakenew' in x:
    corpus.append(x['text'])
    labels.append(x['is_fakenew'])
    
""" 
    End section: Get data from database
"""

"""
    Begin section: Global variables
"""
lower_letters = string.ascii_lowercase

text2num = {'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9}

maplist = {}
with open('mapping_list.txt', 'r', encoding="utf-8") as f:
    for i in f:
        temp = i.rstrip('\n').split(' ')
        maplist[ord(temp[0])] = temp[1]
        
uniChars = "àáảãạâầấẩẫậăằắẳẵặèéẻẽẹêềếểễệđìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵÀÁẢÃẠÂẦẤẨẪẬĂẰẮẲẴẶÈÉẺẼẸÊỀẾỂỄỆĐÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴÂĂĐÔƠƯ"
unsignChars = "aaaaaaaaaaaaaaaaaeeeeeeeeeeediiiiiooooooooooooooooouuuuuuuuuuuyyyyyAAAAAAAAAAAAAAAAAEEEEEEEEEEEDIIIOOOOOOOOOOOOOOOOOOOUUUUUUUUUUUYYYYYAADOOU"
"""
    End section: Global variables
"""

"""
    Begin section:Remove punctualtion
"""
kytudb = ['\u035f', '̼']
def remove_punctualtion(text):
    special_char = kytudb + ['“', '”', '–', '‘', '’', "…","\u0332" ,'\u200b', '\u200c', '\u200d', '\u200e', '\u200f']
    punctualtion_list = string.punctuation + "".join(special_char)
    removed_punctuation = "".join([i for i in text if i not in punctualtion_list])
    return removed_punctuation
"""
    End section:Remove punctualtion
"""

# for i in clean_text:
#     print("post-----------")
#     print(i)

"""
    Begin section: Convert to unicode characters
"""
def loaddicchar():
    dict_char = {}
    char_1252 = "à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ".split(
        "|"
    )
    char_utf8 = "à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ".split(
        "|"
    )
    for i in range(len(char_1252)):
        dict_char[char_1252[i]] = char_utf8[i]
    return dict_char


def convert_unicode(txt):
    dict_char = loaddicchar()
    return re.sub(
        r"à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ",
        lambda x: dict_char[x.group()],
        txt,
    )
    
def loaddicchar2():
    dict_char = {}
    char_other_1a = "ă|ằ|ắ|ẳ|ẵ|ặ|â|ầ|ấ|ẩ|ẫ|ậ|ê|ề|ế|ể|ễ|ệ|ô|ồ|ố|ổ|ỗ|ộ|ơ|ờ|ớ|ở|ỡ|ợ|ư|ừ|ứ|ử|ữ|ự".split("|")
    char_other_1b = "ă|ằ|ắ|ẳ|ẵ|ặ|â|ầ|ấ|ẩ|ẫ|ậ|ê|ề|ế|ể|ễ|ệ|ô|ồ|ố|ổ|ỗ|ộ|ơ|ờ|ớ|ở|ỡ|ợ|ư|ừ|ứ|ử|ữ|ự".split("|")
    for i in range(len(char_other_1a)):
        dict_char[char_other_1a[i]] = char_other_1b[i]
    return dict_char

def convert_unicode2(txt):
    dict_char = loaddicchar2()
    return re.sub(
        r"ă|ằ|ắ|ẳ|ẵ|ặ|â|ầ|ấ|ẩ|ẫ|ậ|ê|ề|ế|ể|ễ|ệ|ô|ồ|ố|ổ|ỗ|ộ|ơ|ờ|ớ|ở|ỡ|ợ|ư|ừ|ứ|ử|ữ|ự",
        lambda x: dict_char[x.group()],
        txt,
    )
    
def loaddicchar3():
    dict_char = {}
    char_other_1a = "ặ|ậ|ệ|ộ".split("|")
    char_other_1b = "ặ|ậ|ệ|ộ".split("|")
    for i in range(len(char_other_1a)):
        dict_char[char_other_1a[i]] = char_other_1b[i]
    return dict_char

def convert_unicode3(txt):
    dict_char = loaddicchar3()
    return re.sub(
        r"ặ|ậ|ệ|ộ",
        lambda x: dict_char[x.group()],
        txt,
    )
    
"""
    End section: Convert to unicode characters
"""


"""
    Begin section: Replace special characters with characters in maplist
"""
def find_special_char(text):
    regex = "[^a-z0-9A-Z_\sÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚĂĐĨŨƠàáâãèéêìíòóôõùúăđĩũơƯĂẠẢẤẦẨẪẬẮẰẲẴẶẸẺẼỀỀỂưăạảấầẩẫậắằẳẵặẹẻẽềềểỄỆỈỊỌỎỐỒỔỖỘỚỜỞỠỢỤỦỨỪễếệỉịọỏốồổỗộớờởỡợụủứừỬỮỰỲỴÝỶỸửữựỳỵỷỹý]"
    set_char = set(re.findall(regex, text))
    return set_char


def replace_special_char(text):
    mapping_list = maplist.copy()
    set_char = find_special_char(text)
    for char in set_char:
        try:
            char_name = ud.name(char).lower()
            words = char_name.split(' ')
            
            #for number
            if words[-1] in text2num:
                mapping_list[ord(char)] = text2num[words[-1]]
                
            else: # for letter
                for word in words:
                    if word in lower_letters:
                        mapping_list[ord(char)] = word
                        break
        except:
            pass
    return text.translate(str.maketrans(mapping_list))
"""
    End section: Replace special characters with characters in maplist
"""


"""
    Start section: Chuyển câu văn về kiểu gõ telex khi không bật Unikey
    Ví dụ: thủy = thuyr, tượng = tuwowngj
"""
bang_nguyen_am = [
    ["a", "à", "á", "ả", "ã", "ạ", "a"],
    ["ă", "ằ", "ắ", "ẳ", "ẵ", "ặ", "aw"],
    ["â", "ầ", "ấ", "ẩ", "ẫ", "ậ", "aa"],
    ["e", "è", "é", "ẻ", "ẽ", "ẹ", "e"],
    ["ê", "ề", "ế", "ể", "ễ", "ệ", "ee"],
    ["i", "ì", "í", "ỉ", "ĩ", "ị", "i"],
    ["o", "ò", "ó", "ỏ", "õ", "ọ", "o"],
    ["ô", "ồ", "ố", "ổ", "ỗ", "ộ", "oo"],
    ["ơ", "ờ", "ớ", "ở", "ỡ", "ợ", "ow"],
    ["u", "ù", "ú", "ủ", "ũ", "ụ", "u"],
    ["ư", "ừ", "ứ", "ử", "ữ", "ự", "uw"],
    ["y", "ỳ", "ý", "ỷ", "ỹ", "ỵ", "y"],
]
bang_ky_tu_dau = ["", "f", "s", "r", "x", "j"]

nguyen_am_to_ids = {}

for i in range(len(bang_nguyen_am)):
    for j in range(len(bang_nguyen_am[i]) - 1):
        nguyen_am_to_ids[bang_nguyen_am[i][j]] = (i, j)


def vn_word_to_telex_type(word):
    dau_cau = 0
    new_word = ""
    for char in word:
        x, y = nguyen_am_to_ids.get(char, (-1, -1))
        if x == -1:
            new_word += char
            continue
        if y != 0:
            dau_cau = y
        new_word += bang_nguyen_am[x][-1]
    new_word += bang_ky_tu_dau[dau_cau]
    return new_word


def vn_sentence_to_telex_type(sentence):
    """
    Chuyển câu tiếng việt có dấu về kiểu gõ telex.
    :param sentence:
    :return:
    """
    words = sentence.split()
    for index, word in enumerate(words):
        words[index] = vn_word_to_telex_type(word)
    return " ".join(words)


"""
    End section: Chuyển câu văn về kiểu gõ telex khi không bật Unikey
"""


"""
    Start section: Chuyển câu văn về cách gõ dấu kiểu cũ: dùng òa úy thay oà uý
    Xem tại đây: https://vi.wikipedia.org/wiki/Quy_t%E1%BA%AFc_%C4%91%E1%BA%B7t_d%E1%BA%A5u_thanh_trong_ch%E1%BB%AF_qu%E1%BB%91c_ng%E1%BB%AF
"""

def standardize_word(word):
    if not is_valid_vietnam_word(word):
        return word

    chars = list(word)
    dau_cau = 0
    nguyen_am_index = []
    qu_or_gi = False
    for index, char in enumerate(chars):
        x, y = nguyen_am_to_ids.get(char, (-1, -1))
        if x == -1:
            continue
        elif x == 9:  # check qu
            if index != 0 and chars[index - 1] == "q":
                chars[index] = "u"
                qu_or_gi = True
        elif x == 5:  # check gi
            if index != 0 and chars[index - 1] == "g":
                chars[index] = "i"
                qu_or_gi = True
        if y != 0:
            dau_cau = y
            chars[index] = bang_nguyen_am[x][0]
        if not qu_or_gi or index != 1:
            nguyen_am_index.append(index)
    if len(nguyen_am_index) < 2:
        if qu_or_gi:
            if len(chars) == 2:
                x, y = nguyen_am_to_ids.get(chars[1])
                chars[1] = bang_nguyen_am[x][dau_cau]
            else:
                x, y = nguyen_am_to_ids.get(chars[2], (-1, -1))
                if x != -1:
                    chars[2] = bang_nguyen_am[x][dau_cau]
                else:
                    chars[1] = bang_nguyen_am[5][dau_cau] if chars[1] == "i" else bang_nguyen_am[9][dau_cau]
            return "".join(chars)
        return word

    for index in nguyen_am_index:
        x, y = nguyen_am_to_ids[chars[index]]
        if x == 4 or x == 8:  # ê, ơ
            chars[index] = bang_nguyen_am[x][dau_cau]
            # for index2 in nguyen_am_index:
            #     if index2 != index:
            #         x, y = nguyen_am_to_ids[chars[index]]
            #         chars[index2] = bang_nguyen_am[x][0]
            return "".join(chars)

    if len(nguyen_am_index) == 2:
        if nguyen_am_index[-1] == len(chars) - 1:
            x, y = nguyen_am_to_ids[chars[nguyen_am_index[0]]]
            chars[nguyen_am_index[0]] = bang_nguyen_am[x][dau_cau]
            # x, y = nguyen_am_to_ids[chars[nguyen_am_index[1]]]
            # chars[nguyen_am_index[1]] = bang_nguyen_am[x][0]
        else:
            # x, y = nguyen_am_to_ids[chars[nguyen_am_index[0]]]
            # chars[nguyen_am_index[0]] = bang_nguyen_am[x][0]
            x, y = nguyen_am_to_ids[chars[nguyen_am_index[1]]]
            chars[nguyen_am_index[1]] = bang_nguyen_am[x][dau_cau]
    else:
        # x, y = nguyen_am_to_ids[chars[nguyen_am_index[0]]]
        # chars[nguyen_am_index[0]] = bang_nguyen_am[x][0]
        x, y = nguyen_am_to_ids[chars[nguyen_am_index[1]]]
        chars[nguyen_am_index[1]] = bang_nguyen_am[x][dau_cau]
        # x, y = nguyen_am_to_ids[chars[nguyen_am_index[2]]]
        # chars[nguyen_am_index[2]] = bang_nguyen_am[x][0]
    return "".join(chars)


def is_valid_vietnam_word(word):
    chars = list(word)
    nguyen_am_index = -1
    for index, char in enumerate(chars):
        x, y = nguyen_am_to_ids.get(char, (-1, -1))
        if x != -1:
            if nguyen_am_index == -1:
                nguyen_am_index = index
            else:
                if index - nguyen_am_index != 1:
                    return False
                nguyen_am_index = index
    return True


def standardize_sentence(sentence):
    """
    Chuyển câu tiếng việt về chuẩn gõ dấu kiểu cũ.
    :param sentence:
    :return:
    """
    sentence = sentence.lower()
    words = sentence.split()
    for index, word in enumerate(words):
        cw = re.sub(r"(^\p{P}*)([p{L}.]*\p{L}+)(\p{P}*$)", r"\1#\2#\3", word).split("#")
        print(cw)
        if len(cw) == 3:
            cw[1] = standardize_word(cw[1])
        words[index] = "".join(cw)
    return " ".join(words)
"""
    End section: Chuyển câu văn về cách gõ dấu kiểu cũ: dùng òa úy thay oà uý
"""


"""
    Start section: Tokenize
    Remove stop word and tokenize
"""
stop_word_list = []

with open('stop_word.txt', 'r', encoding="utf-8") as f:
    for i in f:
        stop_word_list.append(i.rstrip('\n'))

"""
    Important
"""
def tokenize(text):
    text = text.lower()
    text = remove_punctualtion(text)
    text = replace_special_char(text)
    
    words = re.split('\s+', text)
    
    for i, word in enumerate(words):
        words[i] = convert_unicode2(word)
        
    for i, word in enumerate(words):
        words[i] = convert_unicode(word)
    
    # for i, word in enumerate(words):
    #     words[i] = convert_unicode3(word)

    return [word for word in words if word not in stop_word_list]
"""
    End section: Tokenize
"""

def encode_writting_style(text):
    text = text.lower()
    words = re.split('\s+', text) #list of words
    words = list(filter(None, words))
    num_dot_token = 0
    num_special_token = 0
    
    # test1 = []
    # test2 = []
    if len(words) == 0:
        return [0,0]
    
    for word in words:
        # check dot token
        for i in range(len(word)):
            if i > 0 and i < len(word)-1:
                if word[i-1] != '.' and word[i] == '.' and word[i+1] != '.':
                    num_dot_token += 1
                    # test1 += [word]
                    break
        
        # check special token
        if check_special_char(word):
            num_special_token += 1
            # test2 += [word]
    return [num_dot_token/len(words), num_special_token/len(words)]
"""
    Important
"""

def check_special_char(word):
    for char in word:
        if (char in maplist) or (char in kytudb):
            return True

    set_char = find_special_char(word)
    for char in set_char:
        try:
            char_name = ud.name(char).lower()
            words = char_name.split(' ')
            
            #for number
            if words[-1] in text2num:
                return True
                
            else: # for letter
                for w in words:
                    if w in lower_letters:
                        return True
        except:
            pass
        
    return False




def tokenization(text):
    tokens = re.split('\s+', text)
    return tokens

token_text_list = list(map(tokenize, corpus))
writting_style = list(map(encode_writting_style, corpus))
print(token_text_list[1])

# for i in token_text_list[:20]:
#   print(i)

labeled_data = list(zip(token_text_list, writting_style, corpus, labels))

with open('../../data_preprocess', 'wb') as f:
    pickle.dump(labeled_data, f) 

# print(len(corpus))
# with open('e.txt', 'w', encoding="utf-8") as f:
#     for i in corpus[:100]:
#         f.write(f"{i}\n")
#         x = encode_writting_style(i)
#         f.write(f"{x}\n\n")
#         # f.write(f"{x[1]}\n")
#         # f.write(f"{x[2]}\n\n")

# with open('a.txt', 'w', encoding="utf-8") as f:
#     for (i, char) in enumerate(find_special_char()):
#         # if i % 20 == 0: f.write('\n')
#         try:
#             f.write(char + "\t" + ud.name(char) + "\n")
#         except:
#             pass
        
# with open('c.txt', 'w', encoding="utf-8") as f:
#     set_char = set()
#     for text in clean_text2:
#         set_char.update(find_special_char(text))
        
#     for (i, char) in enumerate(set_char):
#         try:
#             f.write(char + " " + ud.name(char) + "\n")
#         except:
#             pass



# original_stdout = sys.stdout
# with open('a.txt', 'w') as f:
#     sys.stdout = f
#     print(find_special_char())
#     sys.stdout = original_stdout

# test_text = list(clean_text)[2]

# token_text_list = list(map(tokenization, clean_text2))

# token_text_list = list(map(tokenize, corpus))

# with open('d.txt', 'w', encoding="utf-8") as f:
#     for i in token_text_list:
#         for j in i:
#             set_token = find_special_char(j)
#             if len(set_token) > 0:
#                 f.write(f"{j}: ")
#                 for k in set_token:
#                     f.write(f"{k}, ")
#                 f.write('\n')



# original_stdout = sys.stdout
# with open('test1.txt', 'w', encoding="utf-8") as f:
#     sys.stdout = f
#     for i in range(10):
#         print(f"\nPost {i+1} ---------")
#         print(corpus[i])
#         print()
#         print(token_text_list[i])
#     sys.stdout = original_stdout

# with open('test1.txt', 'w', encoding="utf-8") as f:
#     for i in range(10):
#         f.write(f"Post {i+1} ---------\n")
#         f.write(corpus[i])
#         f.write('\n')
#         f.write(token_text_list[i].toString())
#         f.write('\n')
        

# for i in token_text_list[:20]:
#     print(i)