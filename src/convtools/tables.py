import typing as t
from itertools import chain
import csv

from .base import BaseConversion, ensure_conversion, GetItem
from .joins import ColumnRef



class TableConversion:
    def __init__(
        self,
        rows,
        column_to_conversion: t.Dict[str, BaseConversion],
    ):
        self.column_to_conversion = column_to_conversion
        self.rows = rows
        self.file_to_close = None

    def __del__(self):
        if self.file_to_close is not None:
            file_to_close, self.file_to_close = self.file_to_close, None
            file_to_close.close()

    @classmethod
    def from_tuples(
        cls,
        rows: t.Iterable[tuple],
        header: t.Optional[t.List[str]],
    ) -> "TableConversion":
        if header is None:
            it = iter(rows)
            first_row = next(it)
            rows = chain((first_row,), it)
            header = list(range(len(first_row)))
        return cls(
            rows,
            {
                column_name: GetItem(index)
                for index, column_name in enumerate(header)
            },
        )

    @classmethod
    def from_tuples_with_header(
        cls,
        rows: t.Iterable[tuple],
    ) -> "TableConversion":
        it = iter(rows)
        return cls.from_tuples(it, next(it))

    @classmethod
    def from_dicts(
        cls, rows: t.Iterable[dict], header: t.List[str]
    ) -> "TableConversion":
        return cls.from_tuples(rows, header)

    @classmethod
    def from_dicts_n_infer_header(
        cls, rows: t.Iterable[dict]
    ) -> "TableConversion":
        it = iter(rows)
        first_row = next(it)
        header = list(first_row.keys())
        return cls.from_tuples(chain((first_row,), it), header)

    @classmethod
    def from_csv(
        cls,
        filename,
        header: t.Union[bool, t.List[str]] = None,
        open_file_kwargs=None,
        csv_reader_kwargs=None,
    ):
        file_to_close = open(filename, **(open_file_kwargs or {}))
        try:
            reader = csv.reader(file_to_close, **(csv_reader_kwargs or {}))
            if isinstance(header, list):
                table = cls.from_tuples(reader, header)
            elif header is True:
                table = cls.from_tuples_with_header(reader)
            else:
                table = cls.from_tuples(reader, None)
            table.file_to_close = file_to_close
        except Exception:
            file_to_close.close()
            raise

        return table

    @classmethod
    def resolve_col_refs(cls, column_to_conversion, conversion):
        conversion = ensure_conversion(conversion)
        for column_ in conversion.get_dependencies(types=ColumnRef):
            conversion_ = column_to_conversion[column_.column_name]
            conversion.depends_on(conversion_)
            column_.set_conversion(conversion_)
        return conversion

    def update(self, **column_to_conversion):
        for column, conversion in column_to_conversion.items():
            conversion = ensure_conversion(conversion)

            for column_ in conversion.get_dependencies(types=ColumnRef):
                self.column_to_index[column_.column_name]
                self.column_to_conversion[]

            self.column_to_conversion[column] = self.resolve_col_refs(
                self.column_to_conversion,
                conversion
            )
        return self

    def take(self, *column_names: str):
        self.column_to_conversion = {
            column_name: self.column_to_conversion[column_name]
            for column_name in column_names
        }
        return self

    def drop(self, *column_names: str):
        column_names = set(column_names)
        self.column_to_conversion = {
            column_name: conversion
            for column_name, conversion in self.column_to_conversion.items()
            if column_name not in column_names
        }
        return self

    def join(self, table: "TableConversion", on):
        from .conversion import conversion as c

        self.column_to_conversion
        table.column_to_conversion
        join_conditions = []
        if isinstance(on, BaseConversion):
            self.gt
        for item in on:
            if isinstance(item, tuple):
                left_column_name, right_column_name = item
            else:
                left_column_name = right_column_name = item
            join_conditions.append(
                c.LEFT.pipe(self.column_to_conversion[left_column_name])
                == c.RIGHT.pipe(self.column_to_conversion[right_column_name])
            )
        c.join(c.this(), c.input_arg("right"), c.LEFT.pipe())

    def into_iter_tuples(self, with_header=False, **kwargs):
        from .conversion import conversion as c

        if with_header:
            yield tuple(self.column_to_conversion)
        yield from c.iter(
            tuple(self.column_to_conversion.values())
        ).gen_converter(**kwargs)(self.rows)

    def into_iter_dicts(self, **kwargs):
        from .conversion import conversion as c

        yield from c.iter(self.column_to_conversion).gen_converter(**kwargs)(
            self.rows
        )
