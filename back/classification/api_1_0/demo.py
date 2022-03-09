# -*- coding: UTF-8 -*-
from . import api


@api.route("/")
def hello_world():
    return "hello world"
