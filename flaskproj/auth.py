from flask import Blueprint, render_template, redirect, url_for, request, flash
from werkzeug.security import generate_password_hash, check_password_hash
from .database import Users
from . import db
from flask_login import login_user,login_required, logout_user

auth = Blueprint('auth', __name__)

# LOGIN
@auth.route('/login')
def login():
    return render_template('login.html')

@auth.route('/login', methods=['POST'])
def login_post():
    username = request.form.get('username')
    password = request.form.get('password')
    remember = True if request.form.get('remember') else False
    user = Users.query.filter_by(username=username).first()
    if not user or not check_password_hash(user.password, password):
        flash('Please check your login details and try again.')
        return redirect(url_for('auth.login'))
    login_user(user, remember=remember)
    # TODO - main page we want to redirect to
    return redirect(url_for('server.hello'))

# SIGN UP
@auth.route('/signup')
def signup():
    return render_template('signup.html')

@auth.route('/signup', methods=['POST'])
def signup_post():
    username = request.form.get('username')
    password = request.form.get('password')

    user = Users.query.filter_by(username=username).first() # email already exists in database
    if user: 
        flash('Username already exists')
        return redirect(url_for('auth.signup'))

    # Creates a new user
    new_user = Users(username=username, password=generate_password_hash(password, method='sha256'), wins=0, losses=0)
    # Add user to User database
    db.session.add(new_user)
    db.session.commit()
    return redirect(url_for('auth.login'))

# LOG OUT
@auth.route('/logout')
@login_required
def logout():
    logout_user()
    # TODO - main page we want to redirect to
    return redirect(url_for('server.hello'))