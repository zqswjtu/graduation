from classification import db


class User(db.Model):
    __tablename__ = "users"

    user_id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.VARCHAR(20), unique=True, nullable=False)
    password = db.Column(db.CHAR(32), nullable=False)
    role_id = db.Column(db.Integer, nullable=False, default=2)

    @classmethod
    def serialize(cls, user):
        return {
            "user_id": user.user_id,
            "username": user.username,
            "role_id": user.role_id
        }
