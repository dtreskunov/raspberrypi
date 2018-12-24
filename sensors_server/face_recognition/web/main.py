from flask import Flask

from util import CLI

app = Flask(__name__.split('.')[0])


@app.route('/')
def hello_world():
    return 'Hello, World!'


class FaceRecognitionWebApp(CLI):
    def __init__(self, parser=None):
        super().__init__(parser)
        group = self.parser.add_argument_group(
            title='Face recognition web app options')
        group.add_argument(
            '--host', help='host embedded server listens on; restrict access by setting to "127.0.0.1"', default='0.0.0.0')
        group.add_argument(
            '--port', help='port embedded server listens on', default='5000')
        group.add_argument(
            '--debug', help='activate Flask debug mode', action='store_true', default=False)

    def main(self, args):
        app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == '__main__':
    FaceRecognitionWebApp().run()
