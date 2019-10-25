from flask_restful import resource
from models.tribe import TribeModel


class Tribe(Resource):
    def get(self,name):
        tribe = Tribemodel.find_by_name(name)
        if tribe:
            return tribe.json()
        return {'messgae': 'tribe not found'},404

    def post(self,name):
        if TribeModel.find_by_name(name):
            return {'message':"An error with name '{}' already exists".format(name)},400


        tribe = TribeModel(name)
        try:
            tribe.save_to_db()
        except:
            return {'message': 'An error occured while creating the store'}
        return tribe.json(),201

    def save_to_db(self):
        db.session.add(self)
        db.session.commit()
