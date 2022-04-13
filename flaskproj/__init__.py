from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager

# def dbConfig():
#     resBTC = requests.get(btcUrl+api_key).json()['Data']['Data']
#     resETH = requests.get(ethUrl+api_key).json()['Data']['Data']
#     resXMR = requests.get(xmrUrl+api_key).json()['Data']['Data']
            
#     for days in resBTC:
#         if days['low']>0:
#             row=BTC(time=days['time'],high=days['high'],low=days['low'],open=days['open'],close=days['close'],volumeto=days['volumeto'],volumefrom=days['volumefrom'])
#             db.session.add(row)

#     for days in resETH:
#         if days['low']>0:
#             row=ETH(time=days['time'],high=days['high'],low=days['low'],open=days['open'],close=days['close'],volumeto=days['volumeto'],volumefrom=days['volumefrom'])
#             db.session.add(row)

#     for days in resXMR:
#         if days['low']>0:
#             row=XMR(time=days['time'],high=days['high'],low=days['low'],open=days['open'],close=days['close'],volumeto=days['volumeto'],volumefrom=days['volumefrom'])
#             db.session.add(row)

#     db.session.commit()

# def query_to_dict(rset):
#     result = defaultdict(list)
#     for obj in rset:
#         instance = inspect(obj)
#         for key, x in instance.attrs.items():
#             result[key].append(x.value)
#     return result

# def get_crypto():
#     btc=BTC.query.all()
#     eth=ETH.query.all()
#     xmr=XMR.query.all()
#     btc=pd.DataFrame(query_to_dict(btc))
#     eth=pd.DataFrame(query_to_dict(eth))
#     xmr=pd.DataFrame(query_to_dict(xmr))
#     data=[btc,eth,xmr]
#     print(data)
#     return data

#---------------------------------------------------------
# SQLite Database
#---------------------------------------------------------
db = SQLAlchemy()
def create_app():
    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'secret'
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///db.sqlite3'
    #app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    db.init_app(app)
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

    # dbConfig()
    # query_to_dict()
    # get_crypto()

    return app