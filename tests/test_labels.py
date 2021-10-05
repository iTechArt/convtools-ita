from convtools import conversion as c
import pytest


def test_labels():
    conv1 = c.if_(
        1,
        c.input_arg("y")
        .item("abc")
        .add_label("abc")
        .pipe(
            c.input_arg("x").pipe(
                c.inline_expr("{cde} + 10").pass_args(cde=c.this().item("cde"))
            )
        )
        .pipe(
            c.inline_expr("{this} + {abc}").pass_args(
                this=c.this(), abc=c.label("abc")
            )
        ),
        2,
    ).gen_converter(debug=False)
    assert conv1(data_=1, x={"cde": 2}, y={"abc": 3}) == 15

    list(c.generator_comp(c.this().add_label("a")).execute([1, 2]))
    c.list_comp(c.this().add_label("a")).execute([1, 2])

    with pytest.raises(c.ConversionException):
        c.this().add_label(123)
    with pytest.raises(ValueError):
        c.label(123)
