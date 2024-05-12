from __init__ import create_app, db

# Creating the Flask app instance
app = create_app()

# Creating the database tables
with app.app_context():
    db.create_all()

# Running the Flask application
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
