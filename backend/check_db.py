from models import User, get_db

db = get_db()
users = db.query(User).all()
for user in users:
    print(f"ID: {user.id}, Username: {user.username}, Role: {user.role}")
db.close()