import abc
import logging

logger = logging.getLogger(__name__)


class ConfigItem(abc.ABC):
    def __init__(self, name, display_name, description, default_value):
        self._name = name
        self._display_name = display_name
        self._description = description
        self._default_value = default_value
        self._value = default_value
    
    @property
    def name(self):
        return self._name
    
    @property
    def display_name(self):
        return self._display_name

    @property
    def description(self):
        return self._description
    
    @property
    def value(self):
        return self._value
    
    @value.setter
    def value(self, value):
        self.set_value(value, validate=True)
    
    def set_value(self, value, validate=False):
        if validate:
            if self.validate(value):
                self._value = value
            else:
                raise ValueError()
        else:
            self._value = value
    
    def validate(self, value):
        pass

    def serialize_value(self, value):
        return value
    
    def deserialize_value(self, value):
        return value
    
    def metadata(self):
        return {
            'name': self._name,
            'display_name': self._display_name,
            'description': self._description,
            'default_value': self.serialize_value(self._default_value),
            'type': self.__class__.__name__
        }
    
    def extra_metadata(self):
        return None


class TypedConfigItem(ConfigItem):
    def __init__(self, item_type, **args):
        ConfigItem.__init__(self, **args)
        self._item_type = item_type

    def validate(self, value):
        return type(value) is self._item_type
    
    def extra_metadata(self):
        return {'type': self._item_type.__name__}


class MultipleChoiceConfigItem(ConfigItem):
    def __init__(self, choices, **args):
        ConfigItem.__init__(self, **args)
        self._choices = choices
    
    def validate(self, value):
        return value in self._choices

    def extra_metadata(self):
        return {'choices': [self.serialize_value(choice) for choice in self._choices]}

import weakref

class Configurable(abc.ABC):
    
    _name_to_wrapper = weakref.WeakValueDictionary()

    class ItemsWrapper:
        def __init__(self, items, name, description):
            self._items = items
            self._items_by_name = dict([
                (item.name, item)
                for item in items
            ])
            self._name = name
            self._description = description
            Configurable._name_to_wrapper[self._name] = self
    
        @property
        def name(self):
            return self._name
        
        @property
        def description(self):
            return self._description

        def __getattr__(self, name):
            try:
                return self._items_by_name[name]
            except KeyError:
                raise AttributeError('Config item "{}" not found'.format(name))

        def metadata(self):
            return {
                'name': self.name,
                'description': self.description,
                'items': [{
                    'metadata': item.metadata(),
                    'extra_metadata': item.extra_metadata()
                } for item in self._items]
            }
        
        def serialized_values_by_name(self):
            return dict([
                (item.name, item.serialize_value(item.value))
                for item in self._items
            ])
        
        def validated_values_by_item(self, serialized_values_by_name):
            def get_item_and_validated_value(name, serialized_value):
                item = self.__getattr__(name)
                value = item.deserialize_value(serialized_value)
                if not item.validate(value):
                    raise ValueError('Config item "{}" value "{}" is invalid'.format(name, serialized_value))
                return item, value

            return dict([
                get_item_and_validated_value(name, serialized_value)
                for name, serialized_value in serialized_values_by_name.items()
            ])
        
        def apply(self, serialized_values_by_name):
            updated = False
            validated_values_by_item = self.validated_values_by_item(serialized_values_by_name)
            for item, value in validated_values_by_item.items():
                if item.value != value:
                    item.set_value(value, validate=False)
                    updated = True
            return updated


    def __init__(self, *config_items, config_name=None, config_description=None):
        self._config_items_wrapper = Configurable.ItemsWrapper(
            config_items,
            config_name or self.__class__.__name__,
            config_description)
    
    @property
    def config(self):
        return self._config_items_wrapper
        
    @staticmethod
    def metadata():
        return dict([
            (name, wrapper.metadata()) for name, wrapper in Configurable._name_to_wrapper.items()
        ])
    
    @staticmethod
    def apply(serialized_values_by_name):
        applied = False
        for name, serialized_values in serialized_values_by_name.items():
            wrapper = Configurable._name_to_wrapper.get(name)
            if wrapper is None:
                logger.warning('Configurable `%s` not defined, unable to apply its config', name)
                continue
            applied = applied or wrapper.apply(serialized_values)
        return applied
        
    @staticmethod
    def serialized_values_by_name():
        return dict([
            (name, wrapper.serialized_values_by_name())
            for name, wrapper in Configurable._name_to_wrapper.items()
        ])
    
###

if __name__=='__main__':
    class TestClass(Configurable):
        def __init__(self):
            Configurable.__init__(self,
                TypedConfigItem(
                    name='enabled',
                    display_name='Enabled',
                    description=None,
                    default_value=False,
                    item_type=bool),
                MultipleChoiceConfigItem(
                    name='refinement_method',
                    display_name='Refinement method',
                    description='Method for refining face detection region',
                    default_value=None,
                    choices=[None, 'HOG', 'CNN']))

        def foo(self):
            print(self.config.enabled.value)
            self.config.enabled.value = True
            print(self.config.metadata())
            print(self.config.serialized_values_by_name())
            print(self.config.validated_values_by_item({'refinement_method': 'CNN'}))
        
    t = TestClass()
    t.foo()
    print(Configurable.metadata())
    old_refinement_method_value = t.config.refinement_method.value
    serialized = Configurable.serialized_values_by_name()
    print(serialized)
    t.config.refinement_method.value = 'HOG'
    assert(old_refinement_method_value != t.config.refinement_method.value)
    Configurable.apply(serialized)
    assert(old_refinement_method_value == t.config.refinement_method.value)

