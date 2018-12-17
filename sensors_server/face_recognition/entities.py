import contextlib
import uuid

from pony import orm

db = orm.Database()


class Person(db.Entity):
    id = orm.PrimaryKey(uuid.UUID, default=uuid.uuid4)
    name = orm.Required(str)
    detected_faces = orm.Set(lambda: DetectedFace)
    avg_face_descriptor = orm.Required(orm.Json)  # float[128]
    n_samples = orm.Required(int)


class Image(db.Entity):
    id = orm.PrimaryKey(uuid.UUID, default=uuid.uuid4)
    mime_type = orm.Required(str)
    data = orm.Required(bytes)
    width = orm.Required(int)
    height = orm.Required(int)
    detected_faces = orm.Set(lambda: DetectedFace)


class DetectedFace(db.Entity):
    id = orm.PrimaryKey(uuid.UUID, default=uuid.uuid4)
    person = orm.Required(lambda: Person)
    image = orm.Required(lambda: Image)
    face_descriptor = orm.Required(orm.Json)  # float[128]
    face_landmarks = orm.Required(orm.Json)  # {"left_eye":[(123,456)]}
    bbox_left = orm.Required(int)
    bbox_top = orm.Required(int)
    bbox_right = orm.Required(int)
    bbox_bottom = orm.Required(int)


@contextlib.contextmanager
def db_connection(*args, **kwds):
    db.bind(*args, **kwds)
    db.generate_mapping(create_tables=True)
    yield
    db.disconnect()


db_transaction = orm.db_session


if __name__ == '__main__':
    orm.set_sql_debug()
    db.bind(provider='sqlite', filename=':memory:')
    db.generate_mapping(create_tables=True)

    Image.before_insert = lambda image: print(
        'need to update spatial index with new image')
    with orm.db_session:
        i = Image(mime_type='foo', data=b'b', width=1, height=1)
