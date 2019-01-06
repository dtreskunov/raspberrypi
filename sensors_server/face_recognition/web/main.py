import inspect
import logging

from flask import Flask, jsonify, request
from werkzeug.exceptions import BadRequest, Conflict, NotFound

import face_recognition.database as db
from face_recognition.image import MyImage
from util import CLI

logger = logging.getLogger(__name__)
app = Flask(__name__.split('.')[0])


@app.route('/')
def index():
    return app.send_static_file('index.html')


def _get_entity_class(entity_type):
    attr = getattr(db, entity_type, None)
    if not attr or not inspect.isclass(attr):
        raise BadRequest('Entity type "{}" is invalid'.format(entity_type))
    return attr


def _get_entity(entity_type, entity_id):
    entity_class = _get_entity_class(entity_type)
    entity = entity_class[entity_id]
    if not entity:
        raise NotFound('Entity "{}" with id "{}" not found'.format(
            entity_type, entity_id))
    return entity


def _to_dict(entity):
    d = None
    if isinstance(entity, db.Image):
        d = entity.to_dict(with_collections=True)
        data = d['data']
        del d['data']
        d['data_uri'] = MyImage(_bytes=data).data_uri
    elif isinstance(entity, db.DetectedFace):
        d = entity.to_dict(with_collections=True, exclude=['descriptor'])
    else:
        d = entity.to_dict(with_collections=True)
    return d


@app.route('/entity/<entity_type>', methods=['GET'])
@db.db_transaction
def get_entities(entity_type):
    entity_class = _get_entity_class(entity_type)
    return jsonify([_to_dict(entity) for entity in entity_class.select()])


@app.route('/entity/<entity_type>/<uuid:entity_id>', methods=['GET'])
@db.db_transaction
def get_entity(entity_type, entity_id):
    return jsonify(_to_dict(_get_entity(entity_type, entity_id)))


@app.route('/entity/<entity_type>/<uuid:entity_id>', methods=['DELETE'])
@db.db_transaction
def delete_entity(entity_type, entity_id):
    entity = _get_entity(entity_type, entity_id)
    entity.delete()
    return 'Entity "{}" with id "{}" deleted'.format(entity_type, entity_id)


@app.route('/entity/<entity_type>', methods=['POST'])
@db.db_transaction
def post_entity(entity_type):
    entity_class = _get_entity_class(entity_type)
    entity = entity_class(**request.get_json())
    return jsonify(_to_dict(entity))


@app.route('/ids/<entity_type>', methods=['GET'])
@db.db_transaction
def get_entity_ids(entity_type):
    entity_class = _get_entity_class(entity_type)
    return jsonify(
        [str(id) for id in db.select(
            entity.id
            for entity in entity_class)])


@app.route('/ids/Image/untagged', methods=['GET'])
@db.db_transaction
def get_untagged_image_ids():
    return jsonify(
        [str(id) for id in db.select(
            image.id
            for image in db.Image
            for face in db.DetectedFace
            if face.image.id == image.id
            and face.person is None)])


@app.route('/link_face_to_person', methods=['POST'])
def link_face_to_person():
    payload = request.get_json()
    if not payload:
        raise BadRequest
    face_id = payload.get('face_id', None)
    person_id = payload.get('person_id', None)
    if not face_id or not person_id:
        raise BadRequest(
            'Either "face_id" or "person_id" not found in request payload')
    with db.db_transaction:
        face = db.DetectedFace[face_id]
        if not face:
            raise NotFound('Face with id "{}" not found'.format(face_id))
        face.person = person_id
        return jsonify(_to_dict(face))


@app.route('/unlink_face_from_person', methods=['POST'])
def unlink_face_from_person():
    payload = request.get_json()
    if not payload:
        raise BadRequest
    face_id = payload.get('face_id', None)
    person_id = payload.get('person_id', None)
    if not face_id or not person_id:
        raise BadRequest(
            'Either "face_id" or "person_id" not found in request payload')
    with db.db_transaction:
        face = db.DetectedFace[face_id]
        if not face:
            raise NotFound('Face with id "{}" not found'.format(face_id))
        if str(face.person.id) != person_id:
            raise Conflict(
                '{} is currently linked to {} - cannot unlink from {}'.format(
                    face, face.person.id, person_id))
        face.person = None
        return jsonify(_to_dict(face))


class FaceRecognitionWebApp(CLI):
    def __init__(self, parser=None):
        super().__init__(parser)
        group = self.parser.add_argument_group(
            title='Face recognition web app options (prefixed with `frwa-` to avoid conflicts)')
        group.add_argument(
            '--frwa-db-connection-params', default='provider=sqlite,filename=~/.face_recognition/data.sqlite')
        group.add_argument(
            '--frwa-host', help='host embedded server listens on; restrict access by setting to "127.0.0.1"', default='0.0.0.0')
        group.add_argument(
            '--frwa-port', help='port embedded server listens on', type=int, default='5000')
        group.add_argument(
            '--frwa-debug', help='activate Flask debug mode', action='store_true', default=False)

    def main(self, args):
        args.frwa_db_connection_params = dict(
            (kv.split('=') for kv in args.frwa_db_connection_params.split(',')))

        logger.debug('final args: %s', args)
        with db.db_connection(**args.frwa_db_connection_params):
            app.run(host=args.frwa_host, port=args.frwa_port,
                    debug=args.frwa_debug)


if __name__ == '__main__':
    FaceRecognitionWebApp().run()
