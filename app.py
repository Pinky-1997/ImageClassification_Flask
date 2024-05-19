from app import app as application  # Renaming to avoid circular import

if __name__ == '__main__':
    application.run(debug=True)
