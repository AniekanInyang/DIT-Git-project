from flask import Flask
from flask_restful import Api
from resources.book import Book
from resources.publisher import Publisher
from resources.author import Author

app= Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI']= 'sqlite:///data.db'
app.config['SQLALCHEMY_TRACK_MODIFICATION']= False
app.config['PROPAGATE_EXCEPTIONS']=True
app.secret_key='ukeme'
api=Api(app)

@app.before_first_request
def create_tables():
    db.create_all()

api.add_resource(Publisher, '/publisher/<string:name>')
api.add_resource(Book, '/book/<string:name>')
api.add_resource(Author, '/author/<string:name>')

if __name__ == '__main__':
    from db import db
    db.init_app(app)
    app.run(port=5000, debug=True)

