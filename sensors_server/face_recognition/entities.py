import contextlib
import logging
import uuid as uuid_lib

from pony import orm

db = orm.Database()
db_transaction = orm.db_session

logger = logging.getLogger(__name__)


class Person(db.Entity):
    id = orm.PrimaryKey(uuid_lib.UUID, default=uuid_lib.uuid4)
    name = orm.Required(str)
    detected_faces = orm.Set(lambda: DetectedFace)


class Image(db.Entity):
    id = orm.PrimaryKey(uuid_lib.UUID, default=uuid_lib.uuid4)
    mime_type = orm.Required(str)
    data = orm.Required(bytes)
    width = orm.Required(int)
    height = orm.Required(int)
    detected_faces = orm.Set(lambda: DetectedFace)


class DetectedFace(db.Entity):
    id = orm.PrimaryKey(uuid_lib.UUID, default=uuid_lib.uuid4)
    image = orm.Optional(lambda: Image)
    image_region = orm.Required(orm.Json)  # [left, top, right, bottom]
    descriptor = orm.Required(orm.Json)  # float[128]
    labeled_landmarks = orm.Required(orm.Json)  # {"left_eye":[(123,456)]}
    face_score = orm.Optional(float)
    joy_score = orm.Optional(float)
    person = orm.Optional(lambda: Person)


@contextlib.contextmanager
def db_connection(*args, **kwds):
    with orm.sql_debugging(debug=(logger.getEffectiveLevel() == logging.DEBUG)):
        db.bind(*args, **kwds)
        db.generate_mapping(create_tables=True)
        yield
        db.disconnect()


if __name__ == '__main__':
    orm.set_sql_debug()
    db.bind(provider='sqlite', filename=':memory:')
    db.generate_mapping(create_tables=True)

    Image.before_insert = lambda image: print(
        'need to update spatial index with new image')
    with orm.db_session:
        i = Image(mime_type='foo', data=b'b', width=1, height=1)
