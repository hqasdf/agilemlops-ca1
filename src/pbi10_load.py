"""Transaction adapter for the PBI 04 staged CSV loader.

Pass an authenticated connection and the PBI 04 load_table callable. Callers
must verify the live schema and keep the connection's TLS verification enabled.
"""
from pathlib import Path
from tempfile import TemporaryDirectory

from src.pbi10_db_contract import COLUMNS, validate_db_contract


def load_cleaned_batch(connection, cleaned, load_table=None):
    if load_table is None:
        from src.pbi04_load import load_table
    names = ('customers', 'orders', 'labels')
    if set(cleaned) != set(names):
        raise ValueError('A cleaned frame is required for each table')
    for name in names:
        validate_db_contract(name, cleaned[name])
    with TemporaryDirectory(prefix='pbi10-cleaned-') as temporary:
        directory = Path(temporary)
        for name in names:
            cleaned[name].loc[:, list(COLUMNS[name])].to_csv(directory / f'{name}_cleaned.csv', index=False)
        try:
            with connection.cursor() as cursor:
                cursor.execute("SET SESSION sql_mode = 'STRICT_ALL_TABLES,NO_ENGINE_SUBSTITUTION'")
                cursor.execute('SET TRANSACTION ISOLATION LEVEL SERIALIZABLE')
                connection.begin()
                counts = {}
                for name in names:
                    received, unchanged = load_table(cursor, directory, name, tuple(COLUMNS[name]))
                    counts[name] = {'received': received, 'unchanged': unchanged, 'inserted': received-unchanged}
            connection.commit()
            return counts
        except Exception:
            connection.rollback()
            raise

