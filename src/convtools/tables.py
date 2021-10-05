"""
Implements streaming operations on table-like data and csv files.  Conversions
are defined in realtime based on table headers and called methods:
 - update
 - take
 - drop
 - join
"""
import csv
import typing as t
from itertools import chain

from .base import (
    And,
    BaseConversion,
    ConverterOptionsCtx,
    GeneratorComp,
    GetItem,
    If,
    InputArg,
    ensure_conversion,
)
from .columns import ColumnRef, MetaColumns
from .joins import JoinConversion, LeftJoinCondition, RightJoinCondition


class CloseFileIterator:
    """
    Utility to close the corresponding file once the iterator is exhausted
    """

    def __init__(self, file_to_close):
        self.file_to_close = file_to_close

    def __iter__(self):
        return self

    def __next__(self):
        raise StopIteration

    def __del__(self):
        self.file_to_close.close()


def close_file_after_iterable(it: t.Iterable, file_to_close) -> t.Iterable:
    return chain(it, CloseFileIterator(file_to_close))


class CustomCsvDialect(csv.Dialect):
    """A helper to define custom csv dialects without defining classes"""

    def __init__(
        self,
        delimiter=csv.excel.delimiter,
        quotechar=csv.excel.quotechar,
        escapechar=csv.excel.escapechar,
        doublequote=csv.excel.doublequote,
        skipinitialspace=csv.excel.skipinitialspace,
        lineterminator=csv.excel.lineterminator,
        quoting=csv.excel.quoting,
    ):
        self.delimiter = delimiter
        self.quotechar = quotechar
        self.escapechar = escapechar
        self.doublequote = doublequote
        self.skipinitialspace = skipinitialspace
        self.lineterminator = lineterminator
        self.quoting = quoting
        super().__init__()


class TableConversion:
    """Table conversion exposes streaming operations on table-like data and csv
    files"""

    def __init__(
        self,
        rows: t.Iterable,
        meta_columns: MetaColumns,
        pipeline: t.Optional[BaseConversion] = None,
        renamed_only: bool = None,
    ):
        self.meta_columns = meta_columns
        self.rows = rows
        self.pipeline = pipeline
        self.renamed_only = renamed_only

    @classmethod
    def from_rows(
        cls,
        rows: t.Iterable[t.Union[dict, tuple, list]],
        header: t.Optional[
            t.Union[
                bool, t.List[str], t.Tuple[str], t.Dict[str, t.Union[str, int]]
            ]
        ] = None,
        # t.Literal["raise", "keep", "drop", "mangle"]
        duplicate_columns="raise",
        skip_rows=0,
    ) -> "TableConversion":
        columns = MetaColumns(duplicate_columns=duplicate_columns)

        if skip_rows:
            rows = iter(rows)
            for _ in range(skip_rows):
                next(rows)

        skipped_columns = 0

        index: "t.Union[str, int]"
        if isinstance(header, (tuple, list)):
            for index, column in enumerate(header):
                skipped_columns += bool(
                    columns.add(column, index, None) is None
                )

        elif isinstance(header, dict):
            for name, index in header.items():
                skipped_columns += bool(columns.add(name, index, None) is None)

        else:
            it = iter(rows)
            first_row = next(it)

            if isinstance(first_row, dict):
                if header is False:
                    for key in first_row:
                        skipped_columns += bool(
                            columns.add(None, key, None) is None
                        )
                else:
                    for key in first_row:
                        skipped_columns += bool(
                            columns.add(key, key, None) is None
                        )

                rows = chain((first_row,), it)

            elif isinstance(first_row, (tuple, list)):
                if header is True:
                    for index, column_name in enumerate(first_row):
                        skipped_columns += bool(
                            columns.add(column_name, index, None) is None
                        )
                    rows = it

                else:
                    for index in range(len(first_row)):
                        skipped_columns += bool(
                            columns.add(None, index, None) is None
                        )
                    rows = chain((first_row,), it)

            else:
                raise ValueError(
                    "failed to infer header: unsupported row type",
                    type(first_row),
                )

        return cls(
            rows,
            columns,
            renamed_only=True if skipped_columns else None,
        )

    csv_dialect = CustomCsvDialect

    @classmethod
    def from_csv(
        cls,
        filepath_or_buffer: t.Union[str, t.TextIO],
        header: t.Optional[
            t.Union[
                bool, t.List[str], t.Tuple[str], t.Dict[str, t.Union[str, int]]
            ]
        ] = None,
        duplicate_columns="mangle",
        skip_rows=0,  # TODO
        dialect: t.Union[str, CustomCsvDialect] = "excel",
        encoding: str = "utf-8",
    ):
        if isinstance(filepath_or_buffer, str):
            file_to_close = open(  # pylint: disable=consider-using-with
                filepath_or_buffer,
                encoding=encoding,
            )
            rows = close_file_after_iterable(
                csv.reader(file_to_close, dialect=dialect), file_to_close
            )
        else:
            rows = csv.reader(filepath_or_buffer, dialect=dialect)

        return cls.from_rows(
            rows, header, duplicate_columns, skip_rows=skip_rows
        )

    def embed_conversions(self) -> "TableConversion":
        if any(column.conversion for column in self.meta_columns.columns):
            column_conversions = []
            for index, column in enumerate(self.meta_columns.columns):
                if column.index is None:
                    column_conversions.append(column.conversion)
                    column.conversion = None
                else:
                    column_conversions.append(GetItem(column.index))
                column.index = index

            conversion = GeneratorComp(tuple(column_conversions))

            self.pipeline = (
                conversion
                if self.pipeline is None
                else self.pipeline.pipe(conversion)
            )

        return self

    def update_all(self, *conversions):
        conversion = GetItem()
        for conversion_ in conversions:
            conversion = conversion.pipe(conversion_)
        column_to_conversion = {
            column.name: (
                column.conversion.pipe(conversion)
                if column.index is None
                else GetItem(column.index).pipe(conversion)
            )
            for column in self.meta_columns.columns
        }
        return self.update(**column_to_conversion)

    def rename(
        self, columns: "t.Union[t.Tuple[str], t.List[str], t.Dict[str, str]]"
    ) -> "TableConversion":
        if isinstance(columns, dict):
            for column_ in self.meta_columns.columns:
                if column_.name in columns:
                    column_.name = columns[column_.name]
        elif isinstance(columns, (tuple, list)):
            if len(columns) != len(self.meta_columns.columns):
                raise ValueError("non-matching number of columns")
            for column_, new_column_name in zip(
                self.meta_columns.columns, columns
            ):
                column_.name = new_column_name
        else:
            raise TypeError("unsupported columns type")
        return self

    def get_columns(self) -> "t.List[str]":
        return [column.name for column in self.meta_columns.columns]

    columns = property(get_columns)

    def update(self, **column_to_conversion) -> "TableConversion":
        if column_to_conversion:
            self.renamed_only = False

        column_name_to_column = self.meta_columns.get_name_to_column()

        for column_name in list(column_to_conversion):
            conversion = ensure_conversion(column_to_conversion[column_name])
            column_refs = list(conversion.get_dependencies(types=ColumnRef))

            depends_on_complex_columns = any(
                column_name_to_column[ref.name].conversion is not None
                for ref in column_refs
            )
            if depends_on_complex_columns:
                return self.embed_conversions().update(**column_to_conversion)

            del column_to_conversion[column_name]

            for ref in column_refs:
                ref.set_index(column_name_to_column[ref.name].index)

            if column_name in column_name_to_column:
                column = column_name_to_column[column_name]
                column.conversion = conversion
                column.index = None
            else:
                column = self.meta_columns.add(column_name, None, conversion)
                column_name_to_column[column_name] = column

        return self

    def take(self, *column_names: str):
        if column_names and self.renamed_only is None:
            self.renamed_only = True
        self.meta_columns.take(*column_names)
        return self

    def drop(self, *column_names: str):
        if column_names and self.renamed_only is None:
            self.renamed_only = True
        self.meta_columns.drop(*column_names)
        return self

    def join(
        self,
        table: "TableConversion",
        on,
        how: str,
        suffixes=("_LEFT", "_RIGHT"),
    ):
        how = JoinConversion.validate_how(how)
        left = self.embed_conversions() if self.renamed_only is False else self
        right = (
            table.embed_conversions() if table.renamed_only is False else table
        )
        left_join_conversion = LeftJoinCondition()
        right_join_conversion = RightJoinCondition()
        left_column_name_to_column = left.meta_columns.get_name_to_column()
        right_column_name_to_column = right.meta_columns.get_name_to_column()

        after_join_conversions: "t.List[BaseConversion]" = []
        after_join_column_names: "t.List[str]" = []

        if isinstance(on, BaseConversion):
            # intentionally left blank to force suffixing
            join_columns = set()
            join_condition = on
            for ref in join_condition.get_dependencies(types=ColumnRef):
                if ref.id_ == left_join_conversion.NAME:
                    column = left_column_name_to_column[ref.name]
                elif ref.id_ == right_join_conversion.NAME:
                    column = right_column_name_to_column[ref.name]
                else:
                    raise ValueError("ambiguous column", ref.name)
                ref.set_index(column.index)
        else:
            on = [on] if isinstance(on, str) else list(on)
            join_columns = set(on)
            join_condition = (
                And(
                    *(
                        left_join_conversion.item(
                            left_column_name_to_column[column_name].index
                        )
                        == right_join_conversion.item(
                            right_column_name_to_column[column_name].index
                        )
                        for column_name in on
                    )
                )
                if len(on) > 1
                else left_join_conversion.item(
                    left_column_name_to_column[on[0]].index
                )
                == right_join_conversion.item(
                    right_column_name_to_column[on[0]].index
                )
            )
        del on

        only_left_values_matter = how in ("left", "inner")
        left_is_optional = how in ("right", "outer")
        right_is_optional = how in ("left", "outer")
        for column in left.meta_columns.columns:
            index = column.index
            column_name = column.name
            if column_name in right_column_name_to_column:
                if column_name in join_columns:
                    after_join_column_names.append(column_name)
                    if only_left_values_matter:
                        after_join_conversions.append(GetItem(0, index))
                    elif how == "right":
                        after_join_conversions.append(GetItem(1, index))
                    else:  # outer
                        after_join_conversions.append(
                            If(
                                GetItem(0).is_(None),
                                GetItem(1, index),
                                GetItem(0, index),
                            )
                        )
                else:
                    after_join_column_names.append(
                        f"{column_name}{suffixes[0]}"
                    )
                    after_join_conversions.append(
                        If(GetItem(0).is_(None), None, GetItem(0, index))
                        if left_is_optional
                        else GetItem(0, index)
                    )
            else:
                after_join_column_names.append(column_name)
                after_join_conversions.append(
                    If(GetItem(0).is_(None), None, GetItem(0, index))
                    if left_is_optional
                    else GetItem(0, index)
                )

        for column in right.meta_columns.columns:
            index = column.index
            column_name = column.name
            if column_name in left_column_name_to_column:
                if column_name in join_columns:
                    # handled above
                    pass
                else:
                    after_join_column_names.append(
                        f"{column_name}{suffixes[1]}"
                    )
                    after_join_conversions.append(
                        If(GetItem(1).is_(None), None, GetItem(1, index))
                        if right_is_optional
                        else GetItem(1, index)
                    )
            else:
                after_join_column_names.append(column_name)
                after_join_conversions.append(
                    If(GetItem(1).is_(None), None, GetItem(1, index))
                    if right_is_optional
                    else GetItem(1, index)
                )

        new_rows = JoinConversion(
            left.pipeline or GetItem(),
            InputArg("right").pipe(right.pipeline or GetItem()),
            join_condition,
            how,
        ).execute(
            left.rows,
            right=right.rows,
            debug=ConverterOptionsCtx.get_option_value("debug"),
        )
        new_columns = MetaColumns(self.meta_columns.duplicate_columns)
        for column_name, conversion in zip(
            after_join_column_names, after_join_conversions
        ):
            new_columns.add(column_name, None, conversion)

        return TableConversion(
            new_rows,
            new_columns,
            renamed_only=False,
        )

    def into_iter_rows(self, type_=None, include_header=None):
        if type_ not in (None, tuple, list, dict):
            raise TypeError("unsupported type_", type_)

        if type_ is None and self.renamed_only is None:
            tuples = self.rows
        else:
            if type_ is dict:
                row_conversion = {
                    column.name: (
                        column.conversion
                        if column.index is None
                        else GetItem(column.index)
                    )
                    for column in self.meta_columns.columns
                }
                include_header = False
            else:
                row_conversion = (type_ or tuple)(
                    (
                        column.conversion
                        if column.index is None
                        else GetItem(column.index)
                    )
                    for column in self.meta_columns.columns
                )
            tuples = (
                (self.pipeline or GetItem())
                .pipe(GeneratorComp(row_conversion))
                .execute(
                    self.rows,
                    debug=ConverterOptionsCtx.get_option_value("debug"),
                )
            )
        if include_header:
            tuples = chain(
                (
                    (type_ or tuple)(
                        column.name for column in self.meta_columns.columns
                    ),
                ),
                tuples,
            )
        return tuples

    def into_csv(
        self,
        filepath_or_buffer: t.Union[str, t.TextIO],
        include_header=True,
        dialect: t.Union[str, CustomCsvDialect] = "excel",
        encoding="utf-8",
    ):
        if isinstance(filepath_or_buffer, str):
            with open(filepath_or_buffer, "w", encoding=encoding) as f:
                writer = csv.writer(f, dialect=dialect)
                writer.writerows(
                    self.into_iter_rows(include_header=include_header)
                )
        else:
            writer = csv.writer(filepath_or_buffer, dialect=dialect)
            writer.writerows(
                self.into_iter_rows(include_header=include_header)
            )
