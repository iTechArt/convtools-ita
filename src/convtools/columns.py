"""
Implements base conversions to reference and define columns in table
conversions
"""
import typing as t
from collections import defaultdict

from .base import BaseConversion, ConversionException, GetItem


class ColumnRef(BaseConversion):
    """Helper conversion, which allows to reference a column of a table
    conversion"""

    def __init__(self, name: str, id_=None):
        super().__init__()
        if not isinstance(name, str):
            raise ValueError("name should be str")
        self.name = name
        self.index: "t.Optional[t.Union[str, int]]" = None
        self.id_ = id_

    def set_index(self, index: t.Union[str, int]):
        if not isinstance(index, (str, int)):
            raise AssertionError("bad index")
        self.index = index
        return self

    def _gen_code_and_update_ctx(self, code_input, ctx):
        if self.index is None:
            raise ConversionException(
                f"{self.__class__.__name__} used outside of TableConversion"
            )
        return GetItem(self.index).gen_code_and_update_ctx(code_input, ctx)


class ColumnDef:
    """
    Defines a column within a table conversion. It holds the following:

     * name of the column in the output
     * index of the column in the input in simple cases, otherwise None
     * conversion to obtain the value from the input, None in simple cases

    """
    __slots__ = ["name", "index", "conversion"]

    def __init__(
        self,
        name: str,
        index: t.Optional[t.Any],
        conversion: t.Optional[BaseConversion],
    ):
        if not bool(index is not None) ^ bool(conversion is not None):
            raise ValueError("provide either index or conversion")
        self.name = name
        self.index = index
        self.conversion = conversion


class MetaColumns:
    """A helper container for naming & keeping column definitions"""

    def __init__(
        self,
        # t.Literal["raise", "keep", "drop", "mangle"]
        duplicate_columns="raise",
    ):
        self.columns: "t.List[ColumnDef]" = []
        self.column_to_number = defaultdict(int)
        if duplicate_columns not in ("raise", "keep", "drop", "mangle"):
            raise ValueError("invalid duplicate_columns value")
        self.duplicate_columns = duplicate_columns

    def add(self, name, index, conversion):
        column_number = self.column_to_number[name]
        self.column_to_number[name] += 1

        if name is None:
            name = f"COLUMN_{column_number}"
        elif column_number:
            if self.duplicate_columns == "raise":
                raise ValueError("such column already exists", name)
            elif self.duplicate_columns == "mangle":
                name = f"{name}_{column_number}"
            elif self.duplicate_columns == "drop":
                return None

        column = ColumnDef(name, index, conversion)
        self.columns.append(column)
        return column

    def take(self, *column_names: str):
        existing_column_names = {column.name for column in self.columns}
        unique_column_names = set(column_names)
        missing_columns = unique_column_names.difference(existing_column_names)
        if missing_columns:
            raise ValueError("missing columns", missing_columns)

        new_columns = []
        for column in self.columns:
            if column.name in unique_column_names:
                new_columns.append(column)
            else:
                self.column_to_number.pop(column.name, None)
        self.columns = new_columns
        return self

    def drop(self, *column_names: str):
        existing_column_names = {column.name for column in self.columns}
        unique_column_names = set(column_names)
        missing_columns = unique_column_names.difference(existing_column_names)
        if missing_columns:
            raise ValueError("missing columns", missing_columns)

        new_columns = []
        for column in self.columns:
            if column.name in unique_column_names:
                self.column_to_number.pop(column.name, None)
            else:
                new_columns.append(column)
        self.columns = new_columns
        return self

    def get_name_to_column(self) -> "t.Dict[str, ColumnDef]":
        return {column.name: column for column in reversed(self.columns)}
