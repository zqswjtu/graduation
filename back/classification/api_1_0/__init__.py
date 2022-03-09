# -*- coding: UTF-8 -*-
from flask import Blueprint


# 创建蓝图对象
api = Blueprint("api_1_0", __name__)

# 把视图导入到蓝图中
from . import demo, verify_code, users
