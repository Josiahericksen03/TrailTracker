import hashlib

def hash_password(password):
    sha256 = hashlib.sha256()
    sha256.update(password.encode("utf-8"))
    return sha256.hexdigest()

def verify_password(stored_password_hash, provided_password):
    sha256 = hashlib.sha256()
    sha256.update(provided_password.encode("utf-8"))
    provided_password_hash = sha256.hexdigest()
    print(f"Provided password hash: {provided_password_hash}")
    return stored_password_hash == provided_password_hash
