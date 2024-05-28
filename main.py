from flask import Blueprint, render_template, current_app, send_from_directory, request, jsonify, redirect, url_for
from flask_login import login_required, current_user
import os

main = Blueprint('main', __name__)

@main.route('/')
def index():
    return render_template('index.html')

@main.route('/profile')
@login_required
def profile():
    return render_template('profile.html', name=current_user.name)

@main.route('/images')
@login_required
def view_images():
    user_images_path = os.path.join(current_app.root_path, 'Images', current_user.name)
    images = []
    if os.path.exists(user_images_path):
        images = [file for file in os.listdir(user_images_path) if file.endswith(('.png', '.jpg', '.jpeg', '.gif'))]
    return render_template('images.html', images=images, username=current_user.name)

@main.route('/videos')
@login_required
def view_videos():
    user_videos_path = os.path.join(current_app.root_path, 'Videos', current_user.name)
    videos = []
    if os.path.exists(user_videos_path):
        videos = [file for file in os.listdir(user_videos_path) if file.endswith('.mp4')]
    return render_template('videos.html', videos=videos, username=current_user.name)

@main.route('/images/<username>/<filename>')
@login_required
def send_image(username, filename):
    return send_from_directory(os.path.join(current_app.root_path, 'Images', username), filename)

@main.route('/videos/<username>/<filename>')
@login_required
def send_video(username, filename):
    return send_from_directory(os.path.join(current_app.root_path, 'Videos', username), filename)

@main.route('/delete_image', methods=['POST'])
@login_required
def delete_image():
    data = request.get_json()
    image = data['image']
    image_path = os.path.join(current_app.root_path, 'Images', current_user.name, image)
    if os.path.exists(image_path):
        os.remove(image_path)
        return jsonify({'success': True, 'image': image}), 200
    return jsonify({'success': False}), 404

@main.route('/delete_video', methods=['POST'])
@login_required
def delete_video():
    data = request.get_json()
    video = data['video']
    video_path = os.path.join(current_app.root_path, 'Videos', current_user.name, video)
    if os.path.exists(video_path):
        os.remove(video_path)
        return jsonify({'success': True, 'video': video}), 200
    return jsonify({'success': False}), 404

@main.route('/streaming')
@login_required
def streaming():
    return render_template('streaming.html')
