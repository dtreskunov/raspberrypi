import contextlib
import datetime
import logging
import os.path
import uuid as uuid_lib

from pony import orm

db = orm.Database()
db_transaction = orm.db_session
db_rollback = orm.rollback

logger = logging.getLogger(__name__)


class Person(db.Entity):
    id = orm.PrimaryKey(uuid_lib.UUID, default=uuid_lib.uuid4)
    created_at = orm.Required(
        datetime.datetime, sql_default='CURRENT_TIMESTAMP')
    name = orm.Required(str)
    detected_faces = orm.Set(lambda: DetectedFace)


class Image(db.Entity):
    id = orm.PrimaryKey(uuid_lib.UUID, default=uuid_lib.uuid4)
    created_at = orm.Required(
        datetime.datetime, sql_default='CURRENT_TIMESTAMP')
    mime_type = orm.Required(str)
    data = orm.Required(bytes)
    width = orm.Required(int)
    height = orm.Required(int)
    detected_faces = orm.Set(lambda: DetectedFace)


class DetectedFace(db.Entity):
    id = orm.PrimaryKey(uuid_lib.UUID, default=uuid_lib.uuid4)
    created_at = orm.Required(
        datetime.datetime, sql_default='CURRENT_TIMESTAMP')
    image = orm.Optional(lambda: Image)
    image_region = orm.Optional(orm.Json)  # [left, top, right, bottom]
    descriptor = orm.Optional(orm.Json)  # float[128]
    labeled_landmarks = orm.Optional(orm.Json)  # {"left_eye":[(123,456)]}
    face_score = orm.Optional(float)
    joy_score = orm.Optional(float)
    person = orm.Optional(lambda: Person)


def get_descriptor_person_id_pairs():
    with orm.db_session:
        return orm.select((df.descriptor, df.person.id) for df in DetectedFace if df.person)[:]


@contextlib.contextmanager
def db_connection(*args, **kwds):
    with orm.sql_debugging(debug=(logger.getEffectiveLevel() == logging.DEBUG)):
        kwds['create_db'] = True
        filename = kwds.get('filename', None)
        if filename:
            kwds['filename'] = os.path.expanduser(filename)
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
