"""Read current analytics tables using the fixed PBI 04 column contract."""
import pandas as pd
from src.pbi10_db_contract import COLUMNS


def read_tables(connection, chunk_size=50000):
    if chunk_size < 1:
        raise ValueError('chunk_size must be positive')
    frames={}
    with connection.cursor() as cursor:
        for table in ('customers','orders','labels'):
            columns=', '.join('`'+column+'`' for column in COLUMNS[table])
            cursor.execute('SELECT '+columns+' FROM `'+table+'`')
            result_columns=[item[0] for item in cursor.description]
            chunks=[]
            while True:
                rows=cursor.fetchmany(chunk_size)
                if not rows: break
                chunks.append(pd.DataFrame.from_records(rows,columns=result_columns))
            frames[table]=pd.concat(chunks,ignore_index=True) if chunks else pd.DataFrame(columns=result_columns)
    return frames

