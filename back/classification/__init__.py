# -*- coding: UTF-8 -*-
import logging
import redis

from config import config_map
from flask import Flask
from flask_session import Session
from flask_sqlalchemy import SQLAlchemy
from flask_wtf import CSRFProtect
from logging.handlers import RotatingFileHandler

# 数据库
db = SQLAlchemy()

# redis
redis_store = None

# 设置日志的记录等级
logging.basicConfig(level=logging.INFO)
# 创建日志记录器，指明日志的保存路径、每个日志文件的最大大小、保存的日志文件个数上限
file_log_handler = RotatingFileHandler("logs/log", maxBytes=1024*1024*100, backupCount=10, encoding="utf-8")
# 创建日志记录的格式
formatter = logging.Formatter("%(levelname)s %(filename)s:%(lineno)d %(message)s")
# 为刚创建的日志记录器设置日志记录格式
file_log_handler.setFormatter(formatter)
# 为全局的日志工具对象(flask app使用的)添加日志记录器
logging.getLogger().addHandler(file_log_handler)


# 工厂模式
def create_app(mode="develop"):
    application = Flask(__name__)
    config_class = config_map.get(mode)
    application.config.from_object(config_class)

    # 使用application初始化db
    db.init_app(application)

    # 初始化redis工具
    global redis_store
    redis_store = redis.StrictRedis(host=config_class.REDIS_HOST, port=config_class.REDIS_PORT)

    # 利用flask-session将session数据保存到redis中
    Session(application)

    # 为flask补充csrf防护
    # CSRFProtect(application)

    # 注册蓝图，不在最上面导包是防止循环导入的问题
    from . import api_1_0
    application.register_blueprint(api_1_0.api, url_prefix="/api/v1.0")
    return application
