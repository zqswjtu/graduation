import hashlib

password = "123"
print(hashlib.md5(password.encode(encoding="UTF-8")).hexdigest())
