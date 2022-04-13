from flask import Flask, Blueprint, render_template
from . import db
from flask_login import login_required, current_user

server = Blueprint('server',__name__)

@server.route("/")
@login_required
def hello():
    return render_template('login.html')