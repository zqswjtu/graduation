# -*- coding: UTF-8 -*-
import redis


class Config(object):
    """配置信息"""
    # 对用户信息加密
    SECRET_KEY = "dbDFHjhDLH78CG"

    # 数据库
    SQLALCHEMY_DATABASE_URI = "mysql+pymysql://root:zq230201@127.0.0.1:3306/classification"
    SQLALCHEMY_TRACK_MODIFICATIONS = True

    # redis
    REDIS_HOST = "127.0.0.1"
    REDIS_PORT = 6379

    # flask-session配置
    SESSION_TYPE = "redis"
    SESSION_REDIS = redis.StrictRedis(host=REDIS_HOST, port=REDIS_PORT)
    SESSION_USE_SIGNER = True
    PERMANENT_SESSION_LIFETIME = 86400

    UPLOAD_FOLDER = ".\\upload"


class DevelopmentConfig(Config):
    """开发环境的配置信息"""
    DEBUG = True


class ProductionConfig(Config):
    """生产环境的配置信息"""
    pass


config_map = {
    "develop": DevelopmentConfig,
    "product": ProductionConfig
}
