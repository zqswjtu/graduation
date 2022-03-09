# -*- coding: UTF-8 -*-
from classification import create_app


# 创建flask的应用对象
app = create_app(mode="develop")
if __name__ == '__main__':
    app.run()
