'''Paired with startup.sh to start flasp app'''
from flaskproj import db, create_app
from flaskproj.server import get_crypto
db.create_all(app=create_app())
get_crypto()