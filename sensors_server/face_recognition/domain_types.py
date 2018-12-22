from collections import namedtuple

Region = namedtuple('Region', ('left', 'top', 'right', 'bottom'))
Face = namedtuple('Face', ('image_region', 'raw_landmarks', 'labeled_landmarks',
                           'descriptor', 'face_score', 'joy_score',
                           'person'),
                  defaults=(None, None, None, None, None, None))
Person = namedtuple('Person', ('id', 'dist', 'name'),
                    defaults=(None, None))
InputOutput = namedtuple('Input', ('image', 'faces'),
                         defaults=([],))
