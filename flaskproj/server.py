from flask import Flask, Blueprint, render_template
from . import db
import requests
from collections import defaultdict
from sqlalchemy import inspect
from .database import BTC,ETH,XMR
# from flask_login import login_required, current_user

def dbConfig():
    api_key='api_key={ea0232c4ea8a3007655f1518de6af8ea6c4a5e546ddf83988ec885db9600a11e}'
    btcUrl='https://min-api.cryptocompare.com/data/v2/histoday?fsym=BTC&tsym=USD&allData=true&'
    ethUrl='https://min-api.cryptocompare.com/data/v2/histoday?fsym=ETH&tsym=USD&allData=true&'
    xmrUrl='https://min-api.cryptocompare.com/data/v2/histoday?fsym=XMR&tsym=USD&allData=true&'
    
    resBTC = requests.get(btcUrl+api_key).json()['Data']['Data']
    resETH = requests.get(ethUrl+api_key).json()['Data']['Data']
    resXMR = requests.get(xmrUrl+api_key).json()['Data']['Data']
            
    for days in resBTC:
        if days['low']>0:
            row=BTC(time=days['time'],high=days['high'],low=days['low'],open=days['open'],close=days['close'],volumeto=days['volumeto'],volumefrom=days['volumefrom'])
            db.session.add(row)

    for days in resETH:
        if days['low']>0:
            row=ETH(time=days['time'],high=days['high'],low=days['low'],open=days['open'],close=days['close'],volumeto=days['volumeto'],volumefrom=days['volumefrom'])
            db.session.add(row)

    for days in resXMR:
        if days['low']>0:
            row=XMR(time=days['time'],high=days['high'],low=days['low'],open=days['open'],close=days['close'],volumeto=days['volumeto'],volumefrom=days['volumefrom'])
            db.session.add(row)

    db.session.commit()

def query_to_dict(rset):
    result = defaultdict(list)
    for obj in rset:
        instance = inspect(obj)
        for key, x in instance.attrs.items():
            result[key].append(x.value)
    return result

def get_crypto():
    btc=BTC.query.all()
    eth=ETH.query.all()
    xmr=XMR.query.all()
    btc=pd.DataFrame(query_to_dict(btc))
    eth=pd.DataFrame(query_to_dict(eth))
    xmr=pd.DataFrame(query_to_dict(xmr))
    data=[btc,eth,xmr]
    print(data)
    return data

dbConfig()
get_crypto()
# ----------------------------------------------
# server
# ----------------------------------------------

server = Blueprint('server',__name__)

@server.route("/")
# @login_required
def hello():
    return render_template('login.html')

    # dbConfig()
    # query_to_dict()
    # get_crypto()