import inspect
import logging

from flask import Flask, jsonify, request

import face_recognition.database as db
from util import CLI

logger = logging.getLogger(__name__)
app = Flask(__name__.split('.')[0])


@app.route('/')
def index():
    return app.send_static_file('index.html')


def get_entity_class(entity_type):
    attr = getattr(db, entity_type.title(), None)
    if attr and inspect.isclass(attr):
        return attr


@app.route('/entity/<str:entity_type>', methods=['GET'])
def get_entities(entity_type):
    entity_class = get_entity_class(entity_type)
    if not entity_class:
        return 400
    with db.db_transaction:
        return jsonify([entity.to_dict(with_collections=True) for entity in entity_class.select()])


@app.route('/entity/<str:entity_type>/<uuid:entity_id>', methods=['GET'])
def get_entity(entity_type, entity_id):
    entity_class = get_entity_class(entity_type)
    if not entity_class:
        return 400
    entity = entity_class[entity_id]
    if not entity:
        return 404
    with db.db_transaction:
        return jsonify(entity.to_dict(with_collections=True))


@app.route('/entity/<str:entity_type>/<uuid:entity_id>', methods=['DELETE'])
def delete_entity(entity_type, entity_id):
    entity_class = get_entity_class(entity_type)
    if not entity_class:
        return 400
    entity = entity_class[entity_id]
    if not entity:
        return 404
    with db.db_transaction:
        entity.delete()


@app.route('/entity/<str:entity_type>', methods=['POST'])
def post_entity(entity_type):
    entity_class = get_entity_class(entity_type)
    if not entity_class:
        return 400
    with db.db_transaction:
        entity_class(**request.form.to_dict())


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
