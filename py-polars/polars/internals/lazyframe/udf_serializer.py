from typing import Any, Union
from abc import ABC

class UdfSerializer(ABC):
    def serialize_udf(udf: Any) -> Union[str, bytes]:
        pass
    def deserialize_udf(udf: Union[str, bytes]) -> Any:
        pass

class PickleUdfSerializer(UdfSerializer):
    def serialize_udf(udf: Any) -> Union[str, bytes]:
        import pickle
        return pickle.dumps(udf)
    def deserialize_udf(udf: Union[str, bytes]) -> Any:
        import pickle
        return pickle.loads(udf)