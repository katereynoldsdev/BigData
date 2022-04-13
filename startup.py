'''Paired with startup.sh to start flasp app'''
from flaskproj import db, create_app
db.create_all(app=create_app())