from flask import Flask
from flask_restful import Api

from resources.food import Food, Recipe
from resources.tribe import Tribe

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] ='sqlite:///project.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
api = Api(app)

@app.before_first_request
def create_table():
    db.create_all()

api.add_resource(Tribe, '/tribe/<string:name>')
api.add_resource(Food, '/food/<string:name>')
api.add_resource(Recipe, '/food/<string:name>/recipe')

if __name__ == '__main__':
    from db import db
    db.init_app(app)
    app.run(port=5000, debug=True)
