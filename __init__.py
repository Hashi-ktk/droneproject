from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager

# init SQLAlchemy so we can use it later in our models
db = SQLAlchemy()

def create_app():
    app = Flask(__name__)

    app.config['SECRET_KEY'] = 'droneproject'
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///db.sqlite'

    # Registering SQLAlchemy with the Flask app
    db.init_app(app)
    
    login_manager = LoginManager()
    login_manager.login_view = 'auth.login'
    login_manager.init_app(app)
    
    @login_manager.user_loader
    def load_user(user_id):
        from models import User  # Importing User model here to avoid circular import
        return User.query.get(int(user_id))

    # Importing Blueprints inside the create_app function to avoid circular imports
    from auth import auth as auth_blueprint
    from main import main as main_blueprint
    from drone_stream import drone_stream
    from face_track import face_tracking
    from body_track import body_tracking

    # Registering Blueprints with the Flask app
    app.register_blueprint(auth_blueprint)
    app.register_blueprint(main_blueprint)
    app.register_blueprint(drone_stream)
    app.register_blueprint(face_tracking)
    app.register_blueprint(body_tracking)

    return app