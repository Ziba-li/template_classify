# -*- coding: utf-8 -*-
import zhconv
import re

URL_REGEX = re.compile(
    r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))',
    re.IGNORECASE)


def simplified2traditional(simplified_str: str):
    """
    Function: 将 hans_str 由简体转化为繁体
    """
    return zhconv.convert(simplified_str, 'zh-hans')


def full_width2half_angle(ustring):
    """
    全角转半角
    """
    empty_string = ""
    for uchar in ustring:
        inside_code = ord(uchar)
        if inside_code == 12288:  # 全角空格直接转换
            inside_code = 32
        elif 65281 <= inside_code <= 65374:  # 全角字符（除空格）根据关系转化
            inside_code -= 65248
        empty_string += chr(inside_code)
    return empty_string


def punctuation_en2zh(t):
    """
    标点符号英文转中文
    """
    table = {ord(f): ord(t) for f, t in zip(u',.!?[]()%#@&<>:;',
                                            u'，。！？【】（）％＃＠＆《》：；')}
    try:
        t2 = t.translate(table)
    except:
        t2 = ''
    t2 = t2.upper()
    t2 = t2.strip()
    t2 = t2.replace('　', '')
    return t2


def clean(text):
    text = re.sub(r"(回复)?(//)?\s*＠\S*?\s*(:| |$)", " ", text)  # 去除正文中的@和回复/转发中的用户名
    text = re.sub(r"(回复)?(//)?\s*@\S*?\s*(:| |$)", " ", text)  # 去除正文中的@和回复/转发中的用户名
    text = re.sub(r"\[\S+\]", "", text)  # 去除表情符号
    text = re.sub(r"#\S+#", "", text)  # 保留话题内容
    text = re.sub(r"＃\S+＃", "", text)  # 保留话题内容
    text = re.sub(URL_REGEX, "", text)  # 去除网址
    text = text.replace("转发微博", "")  # 去除无意义的词语
    text = re.sub(r"\s+", " ", text)  # 合并正文中过多的空格
    return text.strip()
