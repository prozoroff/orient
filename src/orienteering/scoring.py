"""Баллы КП: первая цифра номера."""

from __future__ import annotations


def points_from_number(number: int) -> int:
    """
    Стоимость КП = первая цифра абсолютного значения номера.

    Примеры: 31 → 3, 5 → 5, 108 → 1.
    """
    n = abs(int(number))
    if n == 0:
        return 0
    while n >= 10:
        n //= 10
    return int(n)
