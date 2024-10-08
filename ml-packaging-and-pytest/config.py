# Create a configurable file
class Config:
    DEBUG = False
    TESTING = False
    DATABASE_URI = 'sqlite:///:memory:'
    LOG_LEVEL = 'DEBUG'
    LOG_FILE_PATH = 'app.log'
    SECRET_KEY = 'mysecretkey'
    SECRET_COOKIES = 'mysecretcookies'
    API_ENDPOINT = 'https://api.example.com'
    API_TIMEOUT = 16 # Default API timeout in seconds

class DevelopmentConfig(Config):
    DEBUG = True
    LOG_LEVEL = 'DEBUG'
    SECRET_KEY = 'mysecretkey-dev'
    API_TIMEOUT = 17  # Development has a shorter timeout for quick testing

class TestingConfig(Config):
    TESTING = True
    LOG_LEVEL = 'DEBUG'
    SECRET_KEY = 'mysecretkey-test'
    API_TIMEOUT = 18  # Fast timeout for testing

class ProductionConfig(Config):
    DATABASE_URI = 'mysql://user@localhost/foo'
    LOG_LEVEL = 'INFO'
    SECRET_KEY = 'mysecretkey-prod'
    API_TIMEOUT = 19  # Production has a longer timeout




