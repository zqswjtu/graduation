# -*- coding: UTF-8 -*-
import cv2
import hashlib
import io
import os
import uuid

from . import api
from classification import redis_store
from classification.models import User
from classification.utils.response_code import RET
from flask import request, jsonify, current_app, make_response
from classification.resnet import get_image, predict
from PIL import Image


# 返回值 code, message, object
@api.route("/login", methods=["post"])
def login():
    data = request.get_json(silent=True)
    code = data.get("code")
    username = data.get("username")
    password = data.get("password")
    captcha_id = data.get("captchaId")

    # print(code)
    # 确认验证码是否一致
    try:
        captcha = str(redis_store.get("captcha-%s" % captcha_id), encoding="UTF-8")
        # print(captcha)
    except Exception as e:
        current_app.logger.error(e)
        return jsonify(code=RET.ERROR, message="读取验证码出错！", obj=None)
    if captcha != code:
        return jsonify(code=RET.ERROR, message="验证码错误！", obj=None)
    # 从数据库中根据用户名查询用户对象
    try:
        user = User.query.filter_by(username=username).first()
    except Exception as e:
        current_app.logger.error(e)
        return jsonify(code=RET.ERROR, message="数据库错误！", obj=None)
    if user is None or user.password != hashlib.md5(password.encode(encoding="UTF-8")).hexdigest():
        return jsonify(code=RET.ERROR, message="用户名或者密码错误！", obj=None)

    return jsonify(code=RET.OK, message="登录成功！", obj=User.serialize(user))


# 返回值 code, message, object
@api.route("/upload", methods=["post"])
def upload():
    file_obj = request.files["file"]  # Flask中获取文件
    if file_obj is None:
        # 表示没有发送文件
        return jsonify(code=RET.ERROR, message="未上传文件！", obj=None)

    image_name = str(uuid.uuid4()) + ".jpg"

    # 保存文件
    file_path = os.path.join(current_app.config["UPLOAD_FOLDER"], image_name)
    file_obj.save(file_path)

    print(file_path)

    img = get_image(file_path)
    res = predict(img)
    print(res)

    return jsonify(code=RET.OK, message="检测结果为：%s" % res, obj=image_name)


@api.route("/getPic/<picture_name>", methods=["get"])
def get_picture(picture_name):
    img_url = current_app.config["UPLOAD_FOLDER"] + "/" + picture_name
    with open(img_url, 'rb') as f:
        a = f.read()
    '''对读取的图片进行处理'''
    img_stream = io.BytesIO(a)
    img = Image.open(img_stream)

    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()

    # 返回图片
    response = make_response(img_byte_arr)
    response.headers['Content-Type'] = 'image/png'
    return response


@api.route("/camera", methods=["get"])
def take_photo():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    ret, frame = cap.read()
    image_name = str(uuid.uuid4()) + ".jpg"
    file_path = os.path.join(current_app.config["UPLOAD_FOLDER"], image_name)

    cv2.imwrite(file_path, frame)
    # 关闭摄像头
    cap.release()

    img = get_image(file_path)
    res = predict(img)
    print(res)

    return jsonify(code=RET.OK, message="检测结果为：%s" % res, obj=image_name)
