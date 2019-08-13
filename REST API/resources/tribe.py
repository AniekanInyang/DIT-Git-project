from flask_restful import Resource
from models.tribe import TribeModel

class Tribe(Resource):
    def get(self, name):
        tribe = TribeModel.find_by_name(name)
        if tribe:
            return tribe.json()
        return {'message':'tribe not found'}, 404

    def post(self,name):
        if TribeModel.find_by_name(name):
            return {'message': "A tribe with name '{}' already exists".format(name)}, 400

        tribe = TribeModel(name)
        try:
            tribe.save_to_db()
        except:
            return {'message': 'An error occurred while creating the store'}
        return tribe.json(), 201
