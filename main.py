from flask import Blueprint, render_template, redirect, url_for
from flask_login import login_required, current_user

main = Blueprint('main', __name__)

@main.route('/')
def index():
    return render_template('index.html')

@main.route('/profile')
@login_required
def profile():
    return render_template('profile.html', name=current_user.name)

# @main.route('/connect_to_drone')
# @login_required
# def connect_to_drone():
#     return redirect(url_for('main.streaming'))

@main.route('/streaming')
@login_required
def streaming():
    return render_template('streaming.html')