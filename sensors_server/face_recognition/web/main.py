from flask import Flask, jsonify

from util import CLI
from ..database import db_connection, db_transaction, Person
import logging

logger = logging.getLogger(__name__)
app = Flask(__name__.split('.')[0])


@app.route('/')
def index():
    return app.send_static_file('index.html')


@app.route('/person', methods=['GET'])
@db_transaction
def persons():
    return jsonify(list(Person.select()))


@app.route('/person/<uuid:person_id>', methods=['GET'])
@db_transaction
def person(person_id):
    return jsonify(Person[person_id])


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
        with db_connection(**args.frwa_db_connection_params):
            app.run(host=args.frwa_host, port=args.frwa_port,
                    debug=args.frwa_debug)


if __name__ == '__main__':
    FaceRecognitionWebApp().run()
