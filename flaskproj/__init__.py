from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager

#---------------------------------------------------------
# SQLite Database
#---------------------------------------------------------
app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///db.sqlite3'
db = SQLAlchemy(app)
def create_app():
    
    #app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    # db.init_app(app)
    login_manager = LoginManager()
    login_manager.login_view = 'auth.login'
    login_manager.init_app(app)
    from .database import Users

    @login_manager.user_loader
    def load_user(user_id):
        return Users.query.get(int(user_id))

    # auth routes in our app
    from .auth import auth as auth_blueprint
    app.register_blueprint(auth_blueprint)
    # non-auth
    from .server import server as server_blueprint
    app.register_blueprint(server_blueprint)

    return app