from pymongo import MongoClient

def create_connection():
    client = MongoClient('mongodb://localhost:27017/')
    db = client.trailcamapp
    return db

def view_users(db):
    users_collection = db.users
    user_count = users_collection.count_documents({})
    if user_count == 0:
        print("No users found in the database.")
    else:
        users = users_collection.find()
        for user in users:
            print(f"Username: {user['username']}, Name: {user['name']}, Email: {user['email']}, Password: {user['password']}")

def drop_users(db):
    db.users.drop()
    print("Users collection dropped.")

if __name__ == '__main__':
    db = create_connection()

    # Uncomment the following line to drop the users collection
    #drop_users(db)

    view_users(db)
