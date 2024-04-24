from abc import abstractmethod, ABC
from json import load
from numbers import Real
from pathlib import Path
from typing import Dict, Iterable, Iterator, Tuple, Union, Any, List, Callable
from enum import Enum
from collections.abc import MutableSequence


class Type(Enum):
    Float = 0
    String = 1


def to_float(obj) -> float:
    """
    cast object to float with support of None objects (None is cast to None)
    """
    return float(obj) if obj is not None else None


def to_str(obj) -> str:
    """
    cast object to float with support of None objects (None is cast to None)
    """
    return str(obj) if obj is not None else None


def common(iterator): # from ChatGPT
    try:
        # Nejprve zkusíme získat první prvek iterátoru
        iterator = iter(iterator)
        first_value = next(iterator)
    except StopIteration:
        # Vyvolá výjimku, pokud je iterátor prázdný
        raise ValueError("Iterator is empty")

    # Kontrola, zda jsou všechny další prvky stejné jako první prvek
    for value in iterator:
        if value != first_value:
            raise ValueError("Not all values are the same")

    # Vrací hodnotu, pokud všechny prvky jsou stejné
    return first_value


class Column(MutableSequence):  # implement MutableSequence (some method are mixed from abc)
    def __init__(self, data: Iterable, dtype: Type):
        self.dtype = dtype
        self._cast = to_float if self.dtype == Type.Float else to_str # cast function
        self._data = [self._cast(value) for value in data]

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, item: Union[int, slice]) -> Union[float, str]:
        return self._data[item]

    def __setitem__(self, key: Union[int, slice], value: Any) -> None:
        self._data[key] = self._cast(value)

    def append(self, item: Any) -> None:
        self._data.append(self._cast(item))

    def insert(self, index: int, value: Any) -> None:
        self._data.insert(index, self._cast(value))

    def __delitem__(self, index: Union[int, slice]) -> None:
        del self._data[index]

    def permute(self, indices: List[int]):
        assert len(indices) == len(self)
        ...

    def copy(self) -> 'Column':
        # FIXME: value is casted to the same type (minor optimisation problem)
        return Column(self._data, self.dtype)

    def get_formatted_item(self, index:int, *, width: int):
        assert width > 0
        if self._data[index] is None:
            return "n/a".rjust(width)
        return format(self._data[index],
                      f"{width}s" if self.dtype == Type.String else f"-{width}.2g")


class DataFrame:
    def __init__(self, columns: Dict[str, Column]):
        """
        :param columns: columns of dataframe (key: name of dataframe),
                        lengths of all columns has to be the same
        """
        assert len(columns) > 0, "Dataframe without columns is not supported"
        self._size = common(len(column) for column in columns.values())
        # deep copy od dict `columns`
        self._columns = {name: column.copy() for name, column in columns.items()}

    def __getitem__(self, index: int) -> Tuple[Union[str,float]]:
        ...

    def __iter__(self) -> Iterator[Tuple[Union[str, float]]]:
        """
        :return: iterator over lines of dataframe
        """
        for i in range(len(self)):
            yield tuple(c[i] for c in self._columns.values())

    def __len__(self) -> int:
        return self._size

    @property
    def columns(self) -> Iterable[str]:
        return self._columns.keys()

    def __repr__(self) -> str:
        lines = []
        lines.append(" ".join(f"{name:12s}" for name in self.columns))
        for i in range(len(self)):
            lines.append(" ".join(self._columns[cname].get_formatted_item(i, width=12)
                                     for cname in self.columns))
        return "\n".join(lines)

    def append_column(self, column: Column) -> None:
        ...

    def append_row(self, row: Iterable) -> None:
        ...

    def filter(self, col_name:str, predicate: Callable[[Union[int, str]], bool]) -> 'DataFrame':
        ...

    def sort(self, col_name:str, ascending=True) -> 'DataFrame':
        ...

    def describe(self) -> str:
        """
        similar to pandas but only with min, max and avg statistics for floats and count"
        :return: string with decription
        """
        ...

    def inner_join(self, other: 'DataFrame', self_key_column: str,
                   other_key_column: str) -> 'DataFrame':
        """
            Inner join between self and other dataframe with join predicate
            `self.key_column == other.key_column`.

            Possible collision of column identifiers is resolved by prefixing `_other` to
            columns from `other` data table.
        """
        ...

    @staticmethod
    def read_csv(path: Union[str, Path]) -> 'DataFrame':
        return CSVReader(path).read()

    @staticmethod
    def read_json(path: Union[str, Path]) -> 'DataFrame':
        return JSONReader(path).read()


class Reader(ABC):
    def __init__(self, path: Union[Path, str]):
        self.path = Path(path)
    @abstractmethod
    def read(self) -> 'DataFrame':
        raise NotImplemented("Abstract method")


class JSONReader(Reader):
    def read(self) -> 'DataFrame':
        with open(self.path, "rt") as f:
            json = load(f)
        columns = {}
        for cname in json.keys():
            dtype = Type.Float if all(value is None or isinstance(value, Real)
                                      for value in json[cname]) else Type.String
            columns[cname] = Column(json[cname], dtype)
        return DataFrame(columns)


class CSVReader(Reader):
    def read(self) -> 'DataFrame':
        ...


if __name__ == "__main__":
    df = DataFrame(dict(
        a=Column([None, 3.1415], Type.Float),
        b=Column(["a", 2], Type.String),
        c=Column(range(2), Type.Float)
        ))
    print(df)

    df = DataFrame.read_json("data.json")
    print(df)

for line in df:
    print(line)
