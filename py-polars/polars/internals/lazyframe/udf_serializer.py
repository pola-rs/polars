from typing import Callable, Union
from abc import ABC

class UdfSerializer(ABC):
    """
    A custom UDF (User-Defined Function) serializer/deserializer, for use in
    ``LazyFrame`` serialization and deserialization functions.

    See Also
    --------
    polars.LazyFrame.from_json, polars.LazyFrame.read_json, polars.LazyFrame.write_json

    """

    def serialize_udf(self, udf: Callable) -> Union[str, bytes]:
        """
        Defines how an UDF is serialized.

        Parameters
        ----------
        udf
            The UDF to serialize.

        """
        pass

    def deserialize_udf(self, data: Union[str, bytes]) -> Callable:
        """
        Defines how an UDF is deserialized.

        Parameters
        ----------
        data
            The data representing the UDF.

        """
        pass

class PickleUdfSerializer(UdfSerializer):
    """
    An UDF (User-Defined Function) serializer/deserializer that uses pickle,
    followed by base64 encoding (so that it can be embedded in a json).

    See Also
    --------
    UdfSerializer

    """
    
    def serialize_udf(self, udf: Callable) -> Union[str, bytes]:
        import pickle
        import base64

        return str(base64.b64encode(pickle.dumps(udf)), 'ascii')

    def deserialize_udf(self, data: Union[str, bytes]) -> Callable:
        import pickle
        import base64

        return pickle.loads(base64.b64decode(data))