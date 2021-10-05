import pytest

from convtools import conversion as c
from convtools.columns import ColumnDef, MetaColumns


def test_table_base_init():
    result = list(
        c.table.from_rows(
            [(1, 2, 3), (2, 3, 4)], ["a", "b", "c"]
        ).into_iter_rows(include_header=True)
    )
    assert result == [
        ("a", "b", "c"),
        (1, 2, 3),
        (2, 3, 4),
    ]
    result = list(
        c.table.from_rows(
            [(1, 2, 3), (2, 3, 4)], {"a": 2, "b": 1, "c": 0}
        ).into_iter_rows(dict)
    )
    assert result == [
        {"a": 3, "b": 2, "c": 1},
        {"a": 4, "b": 3, "c": 2},
    ]

    input_data = [("a", "a", "b"), (1, 2, 3)]
    with pytest.raises(ValueError):
        c.table.from_rows(input_data, True)
    with pytest.raises(ValueError):
        c.table.from_rows(input_data, True, duplicate_columns="raise")

    result = list(
        c.table.from_rows(
            input_data, True, duplicate_columns="keep"
        ).into_iter_rows(include_header=True)
    )
    assert result == input_data

    result = list(
        c.table.from_rows(
            input_data, True, duplicate_columns="drop"
        ).into_iter_rows(include_header=True)
    )
    assert result == [("a", "b"), (1, 3)]

    result = list(
        c.table.from_rows(
            input_data, True, duplicate_columns="mangle"
        ).into_iter_rows(include_header=True)
    )
    assert result == [("a", "a_1", "b"), (1, 2, 3)]

    result = list(
        c.table.from_rows(input_data, None).into_iter_rows(include_header=True)
    )
    assert result == [
        ("COLUMN_0", "COLUMN_1", "COLUMN_2"),
        ("a", "a", "b"),
        (1, 2, 3),
    ]

    result = list(
        c.table.from_rows(input_data, {"a": 0, "b": 1, "c": 2}).into_iter_rows(
            include_header=True
        )
    )
    assert result == [("a", "b", "c"), ("a", "a", "b"), (1, 2, 3)]

    result = list(
        c.table.from_rows(
            input_data, {"a": 1, "b": 0, "c": 2}, skip_rows=1
        ).into_iter_rows(dict)
    )
    assert result == [{"a": 2, "b": 1, "c": 3}]

    result = list(
        c.table.from_rows([{"a": 1, "b": 2, "c": 3}]).into_iter_rows(dict)
    )
    assert result == [{"a": 1, "b": 2, "c": 3}]

    result = list(
        c.table.from_rows(
            [{"a": 1, "b": 2, "c": 3}], header=False
        ).into_iter_rows(dict)
    )
    assert result == [{"COLUMN_0": 1, "COLUMN_1": 2, "COLUMN_2": 3}]

    with pytest.raises(ValueError):
        c.table.from_rows([{1}])


def test_table_take():
    result = list(
        c.table.from_rows([(1, 2, 3), (2, 3, 4)], ["a", "b", "c"])
        .take("a", "c")
        .into_iter_rows(include_header=True)
    )
    assert result == [
        ("a", "c"),
        (1, 3),
        (2, 4),
    ]


def test_table_drop():
    result = list(
        c.table.from_rows([(1, 2, 3), (2, 3, 4)], ["a", "b", "c"])
        .drop("a", "c")
        .into_iter_rows(include_header=True)
    )
    assert result == [
        ("b",),
        (2,),
        (3,),
    ]


def test_table_update():
    result = list(
        c.table.from_rows([(1,), (2,)], ["a"])
        .update(b=c.col("a") + 1)
        .into_iter_rows(include_header=True)
    )
    assert result == [
        ("a", "b"),
        (1, 2),
        (2, 3),
    ]
    result = list(
        c.table.from_rows(result, True)
        .update(c=c.col("a") + c.col("b") + 1)
        .update(d=c.col("c") * -1)
        .take("a", "c", "d")
        .into_iter_rows(dict)
    )
    assert result == [
        {"a": 1, "c": 4, "d": -4},
        {"a": 2, "c": 6, "d": -6},
    ]


def test_table_rename():
    result = list(
        c.table.from_rows([(1, 2), (3, 4)], ["a", "b"])
        .rename(["A", "B"])
        .into_iter_rows(dict)
    )
    assert result == [{"A": 1, "B": 2}, {"A": 3, "B": 4}]

    result = list(
        c.table.from_rows([(1, 2), (3, 4)], ["a", "b"])
        .rename({"b": "B"})
        .into_iter_rows(tuple, include_header=True)
    )
    assert result == [("a", "B"), (1, 2), (3, 4)]


def test_table_columns():
    ref_columns = ["a", "b"]
    table = c.table.from_rows([(1, 2), (3, 4)], ref_columns)
    columns = table.get_columns()
    columns2 = table.columns
    assert columns == ref_columns and columns2 == ref_columns


def test_table_inner_join():
    result = list(
        c.table.from_rows([(1, 2), (2, 3)], ["a", "b"])
        .join(
            c.table.from_rows([(1, 3), (2, 4)], ["a", "c"]),
            how="inner",
            on=["a"],
        )
        .into_iter_rows(dict)
    )
    assert result == [
        {"a": 1, "b": 2, "c": 3},
        {"a": 2, "b": 3, "c": 4},
    ]

    result = list(
        c.table.from_rows([(1, 2), (2, 3)], ["a", "b"])
        .embed_conversions()
        .update()
        .drop()
        .join(
            c.table.from_rows([(1, 3), (2, 4)], ["a", "c"]),
            how="inner",
            on=["a"],
        )
        .into_iter_rows(include_header=True)
    )
    assert result == [("a", "b", "c"), (1, 2, 3), (2, 3, 4)]

    result = list(
        c.table.from_rows([(1, 2), (2, 3)], ["a", "b"])
        .join(
            c.table.from_rows([(0, -1), (3, 4)], ["a", "c"]),
            how="inner",
            on=c.LEFT.col("a") < c.RIGHT.col("a"),
        )
        .into_iter_rows(include_header=True)
    )
    assert result == [
        ("a_LEFT", "b", "a_RIGHT", "c"),
        (1, 2, 3, 4),
        (2, 3, 3, 4),
    ]


def test_table_left_simple():
    result = list(
        c.table.from_rows([(1, 2), (2, 3)], ["a", "b"])
        .join(
            c.table.from_rows([(1, 3), (2, 4)], ["a", "c"]),
            how="left",
            on=["a"],
        )
        .into_iter_rows(include_header=True)
    )
    assert result == [
        ("a", "b", "c"),
        (1, 2, 3),
        (2, 3, 4),
    ]


def test_table_left_join():
    with c.OptionsCtx() as options:
        options.debug = False
        result = list(
            c.table.from_rows([(1, 2), (2, 3)], ["a", "b"])
            .join(
                c.table.from_rows([(1, 3), (2, 4)], ["a", "c"]),
                how="left",
                on=["a"],
            )
            .join(
                c.table.from_rows(
                    [
                        (2, 4, 7),
                    ],
                    ["a", "c", "d"],
                ),
                how="left",
                on=["a", "c"],
            )
            .into_iter_rows(include_header=True)
        )
    assert result == [
        ("a", "b", "c", "d"),
        (1, 2, 3, None),
        (2, 3, 4, 7),
    ]


def test_table_right_join():
    result = list(
        c.table.from_rows([(1, 2), (2, 3)], ["a", "b"])
        .join(
            c.table.from_rows([(4, 3), (2, 4)], ["a", "c"]),
            how="right",
            on=["a"],
        )
        .into_iter_rows(include_header=True)
    )
    assert result == [
        ("a", "b", "c"),
        (4, None, 3),
        (2, 3, 4),
    ]


def test_table_outer_join():
    result = list(
        c.table.from_rows([(1, 2, 10), (2, 3, 11)], ["a", "b", "d"])
        .join(
            c.table.from_rows([(4, 3, 20), (2, 4, 21)], ["a", "c", "d"]),
            how="outer",
            on=["a"],
        )
        .into_iter_rows(include_header=True)
    )
    assert result == [
        ("a", "b", "d_LEFT", "c", "d_RIGHT"),
        (1, 2, 10, None, None),
        (2, 3, 11, 4, 21),
        (4, None, None, 3, 20),
    ]


def _test_table_speed():
    from time import time

    size = 10000
    left_data = [(i, i + 1) for i in range(size)]
    right_data = [(i, i + 2) for i in range(size)]

    with c.OptionsCtx() as options:
        options.debug = False

        for _ in range(2):
            t = time()
            result = list(
                c.table.from_rows(left_data, ["a", "b"])
                .join(
                    c.table.from_rows(right_data, ["a", "c"]),
                    how="inner",
                    on=["a"],
                )
                .into_iter_rows(dict)
            )
            tables_time = time() - t
            print(f"tables took {tables_time}; result length: {len(result)}")

        t = time()
        result2 = list(
            c.join(
                c.this(),
                c.input_arg("right"),
                c.LEFT.item(0) == c.RIGHT.item(0),
            )
            .pipe(
                c.iter(
                    {"a": c.item(0, 0), "b": c.item(0, 1), "c": c.item(1, 1)}
                )
            )
            .execute(
                left_data,
                right=right_data,
            )
        )
        raw_join_time = time() - t
        print(f"raw join took {time() - t}; result length: {len(result2)}")

        assert result == result2
        assert tables_time / raw_join_time < 1.2


def _test_table_speed2():
    import csv

    from convtools import conversion as c

    class CustomDialect2(csv.excel_tab):
        lineterminator = "\n"

    report_file = "Subscription_Event_88542188_20210913_V1_2.txt"
    report_file2 = "Subscription_Event_88542188_20210913_V1_2.txt2"
    from time import time

    for _ in range(2):
        t = time()
        with c.OptionsCtx() as options:
            options.debug = False
            # TODO: accept wither file descriptor or file path
            # + introduce dialect-specific args/kwargs
            # + CHECK pandas read_csv useful params
            table = c.table.from_csv(
                report_file,
                header=True,
                dialect=c.table.csv_dialect(
                    skipinitialspace=True, delimiter="\t"
                ),
            ).join(
                c.table.from_csv(
                    report_file2,
                    header=True,
                    dialect=c.table.csv_dialect(
                        skipinitialspace=True, delimiter="\t"
                    ),
                ),
                on=[
                    "Event Date",
                    "Event",
                    "Original Start Date",
                    "Subscription Apple ID",
                ],
                how="inner",
                suffixes=("", "_RIGHT"),
            )
            result = list(table.into_iter_rows())
            # with open("/tmp/events_after_join.csv", "w") as f:
            #     writer = csv.writer(f, dialect=CustomDialect2)
            #     writer.writerows(table.into_iter_rows(True))
            #     # writer.writerows(tqdm(table.into_iter_rows(True)))
            print(f"joining took {time() - t}")

    import pandas as pd

    t = time()
    result2 = list(
        pd.read_csv(
            report_file,
            skipinitialspace=True,
            sep="\t",
            dtype=str,
            keep_default_na=False,
        )
        .merge(
            pd.read_csv(
                report_file2,
                skipinitialspace=True,
                sep="\t",
                dtype=str,
                keep_default_na=False,
            ),
            on=[
                "Event Date",
                "Event",
                "Original Start Date",
                "Subscription Apple ID",
            ],
            how="inner",
            suffixes=("", "_RIGHT"),
        )
        .itertuples(index=False)
    )

    # ).to_csv("/tmp/events_after_join__pandas.csv", index=False, sep="\t")
    print(f"joining via PANDAS took {time() - t}")

    from collections import defaultdict

    import polars as pl

    t = time()
    result3 = (
        pl.read_csv(
            report_file,
            sep="\t",
            dtype=defaultdict(str),
            null_values=[""],
            infer_schema_length=0,
        )
        .join(
            pl.read_csv(
                report_file2,
                sep="\t",
                dtype=defaultdict(str),
                null_values=[""],
                infer_schema_length=0,
            ),
            on=[
                "Event Date",
                "Event",
                "Original Start Date",
                "Subscription Apple ID",
            ],
            how="inner",
            suffix="_RIGHT",
        )
        .rows()
    )

    # ).to_csv("/tmp/events_after_join__pandas.csv", index=False, sep="\t")
    print(f"joining via POLARS took {time() - t}")

    # for t1, t2 in zip(sorted(result), sorted(result3)):
    #     if t1 != t2:
    #         import pdb

    #         pdb.set_trace()


def test_table_csv():
    c.table.from_csv("tests/csvs/ab.csv", True).into_csv("tests/csvs/out.csv")
    result = list(
        c.table.from_csv("tests/csvs/out.csv", True).into_iter_rows()
    )
    assert result == [["1", "2"], ["2", "3"]]

    result = list(
        c.table.from_csv("tests/csvs/ab.csv", True)
        .update_all(int)
        .join(
            c.table.from_csv(
                "tests/csvs/ac.csv",
                True,
                dialect=c.table.csv_dialect(delimiter="\t"),
            ).update_all(int),
            on=["a"],
            how="inner",
        )
        .into_iter_rows(include_header=True)
    )
    assert result == [
        ("a", "b", "c"),
        (1, 2, 3),
        (2, 3, 4),
    ]

    with open("tests/csvs/ab.csv", "r") as f_in, open(
        "tests/csvs/out.csv", "w"
    ) as f_out:
        result = c.table.from_csv(f_in, header=["A", "B"], skip_rows=True).into_csv(f_out)
    result = list(c.table.from_csv("tests/csvs/out.csv").into_iter_rows())
    assert result == [["A", "B"], ["1", "2"], ["2", "3"]]


def test_table_exceptions():
    with pytest.raises(c.ConversionException):
        c.col("tst").gen_converter()
    with pytest.raises(ValueError):
        c.col(1)
    with pytest.raises(ValueError):
        ColumnDef("abc", None, None)
    with pytest.raises(ValueError):
        ColumnDef("abc", 0, c.this())
    with pytest.raises(ValueError):
        MetaColumns(duplicate_columns="abc")
    with pytest.raises(ValueError):
        c.table.from_rows([("a",), (1,)], True).take("b")
    with pytest.raises(ValueError):
        c.table.from_rows([("a",), (1,)], True).drop("b")

    with pytest.raises(ValueError):
        it = (
            c.table.from_rows([(1, 2), (2, 3)], ["a", "b"])
            .join(
                c.table.from_rows([(0, -1), (3, 4)], ["a", "c"]),
                how="inner",
                on=c.LEFT.col("a") < c.col("d"),
            )
            .into_iter_rows(include_header=True)
        )
    table = c.table.from_rows([(1, 2), (2, 3)], ["a", "b"])
    with pytest.raises(TypeError):
        table.into_iter_rows(object)
    with pytest.raises(ValueError):
        table.rename(["A"])
    with pytest.raises(TypeError):
        table.rename({"A"})
