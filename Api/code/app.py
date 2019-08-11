from flask import Flask, jsonify
from flask_restful import Api 
from flask_jwt_extended import JWTManager
from resources.book import Book, BookList 
from resources.publisher import PublisherList
from resources.author import AuthorList
import os
basedir = os.path.abspath(os.path.dirname(__file__))


app = Flask(__name__)
# app.config["SQLALCHEMY_DATABSE_URI"] = "sqlite:///data.db"   #in order to loacte our db file
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', 'sqlite:///data.db')
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False #this modifies the tracker fro proper modification when changes occur and not save to alchemy

app.secret_key = "Jose"

api = Api(app)

#to be remoed when deploying to heroku
@app.before_first_request
def create_tables():
    db.create_all()


jwt = JWTManager(app)  


api.add_resource(Book, "/book/<int:isbn>")
api.add_resource(BookList, "/books")
api.add_resource(AuthorList, "/authors")
api.add_resource(PublisherList, "/publishers")

if __name__ =="__main__":
    from db import db  # we do this here because of what is called circular import
    db.init_app(app)   # then in all our models we import the db
    app.run(port=5000, debug=True)