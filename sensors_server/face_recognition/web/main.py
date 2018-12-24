from flask import Flask

from util import CLI

app = Flask(__name__.split('.')[0])


@app.route('/')
def index():
    app.send_static_file('index.html')


class FaceRecognitionWebApp(CLI):
    def __init__(self, parser=None):
        super().__init__(parser)
        group = self.parser.add_argument_group(
            title='Face recognition web app options (prefixed with `frwa-` to avoid conflicts)')
        group.add_argument(
            '--frwa-host', help='host embedded server listens on; restrict access by setting to "127.0.0.1"', default='0.0.0.0')
        group.add_argument(
            '--frwa-port', help='port embedded server listens on', type=int, default='5000')
        group.add_argument(
            '--frwa-debug', help='activate Flask debug mode', action='store_true', default=False)

    def main(self, args):
        app.run(host=args.frwa_host, port=args.frwa_port, debug=args.frwa_debug)


if __name__ == '__main__':
    FaceRecognitionWebApp().run()
