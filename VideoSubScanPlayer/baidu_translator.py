# coding=utf-8

import requests
import hashlib
import random
import json
from retrying import retry


@retry(stop_max_attempt_number=5)
def Translation(Str):
    """
    百度翻译接口调用，需要先注册百度云接口使用
    英语转中文
    """

    appid = ''  # 填写你的appid
    secretKey = ''  # 填写你的密钥

    httpClient = None
    myurl = 'http://api.fanyi.baidu.com/api/trans/vip/translate'
    fromLang = 'en'  # 原文语种
    toLang = 'zh'  # 译文语种
    salt = random.randint(32768, 65536)
    q = Str
    sign = appid + q + str(salt) + secretKey
    sign = hashlib.md5(sign.encode()).hexdigest()
    myurl = myurl + '?appid=' + appid + '&q=' + q + '&from=' + fromLang + '&to=' + toLang + '&salt=' + str(
        salt) + '&sign=' + sign

    response = requests.get(myurl)
    result = json.loads(response.text)

    return result['trans_result'][0]['dst']


if __name__ == '__main__':
    ret = Translation("   ")
    print(ret)
