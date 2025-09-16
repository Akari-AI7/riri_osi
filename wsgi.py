from app import app

# Expose Flask app for Gunicorn
if __name__ == "__main__":
    app.run()

