# -*- coding: UTF-8 -*-
from . import api
from classification import constants, redis_store
from classification.utils.captcha import Captcha
from classification.utils.response_code import RET
from flask import make_response, current_app, jsonify


@api.route("/captcha/<captcha_id>", methods=["GET"])
def captcha(captcha_id):
    image, verification_code = Captcha.generate_verification_code()
    try:
        # 将验证码的编号保存入redis中
        redis_store.setex("captcha-%s" % captcha_id, constants.CAPTCHA_REDIS_EXPIRES, verification_code)
    except Exception as e:
        # 记录日志
        current_app.logger.error(e)
        return jsonify(code=RET.ERROR, message="保存验证码出错！", obj=None)
    # 返回图片
    response = make_response(image)
    response.headers['Content-Type'] = 'image/png'
    return response
