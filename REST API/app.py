from flask import Flask
from flask_restful import Api

from resources.food import Food, FoodModel
from resources.tribes import Tribes, TribeModel

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///project.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['PROPAGATE_EXCEPTIONS'] = True
app.secret_key = 'ifeoma'
api = Api(app)

@app.before_first_request
def create_table():
    db.create_all()

api.add_resource(Tribes, '/tribe/<string:name>')
api.add_resource(TribesList, '/tribes')
api.add_resource(Food, '/food/<string:name>')
api.add_resource(FoodList, '/foods')

if __name__ == '__main__':
    from db import db
    db.init_app(app)
    app.run(port=5000, debug=True)
