from flask_restful import Resource
from models.tribes import TribesModel


class Tribes(Resource):
    def get(self, name):
        tribes = TribesModel.find_by_tribe(name)
        if tribes:
            return tribes.json()
        return {'message': 'Tribes not found'}, 404

    def post(self, name):
        if TribesModel.find_by_tribe(name):
            return {'message': "A tribe with the name '{}' already exists.".format(name)}, 400

        tribes = TribesModel(name)
        try:
            tribes.save_to_db()
        except:
            return {"message": "An error occurred creating the tribe."}, 500

        return tribes.json(), 201

    def delete(self, name):
        tribes = TribesModel.find_by_tribe(name)
        if tribes:
            tribes.delete_from_db()

        return {'message': 'Tribe deleted'}


class TribesList(Resource):
    def get(self):
        return {'tribes': list(map(lambda x: x.json(), TribesModel.query.all()))}
