from flask import Blueprint, render_template, redirect, url_for
from __init__ import db
from flask_login import login_required, current_user
from djitellopy import tello

main = Blueprint('main', __name__)

@main.route('/')
def index():
    return render_template('index.html')

@main.route('/profile')
@login_required
def profile():
    return render_template('profile.html', name= current_user.name)


@main.route('/connect_to_drone')
@login_required
def connect_to_drone():
    me = tello.Tello()
    connected = me.connect()
    if connected :
        me.streamon()
        return redirect(url_for('main.streaming'))
    else:
        return render_template('profile.html')

@main.route('/streaming')
@login_required
def streaming():
    return render_template('streaming.html')
