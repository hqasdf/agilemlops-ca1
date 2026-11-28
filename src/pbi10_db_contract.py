"""Preflight the versioned PBI 04 NOT NULL and text-width contract.

Live schema equality and numeric/date compatibility require separate checks.
"""
from src.data_cleaning import CUSTOMER_COLUMNS, ORDER_COLUMNS, LABEL_COLUMNS

COLUMNS = {'customers': CUSTOMER_COLUMNS, 'orders': ORDER_COLUMNS, 'labels': LABEL_COLUMNS}
NULLABLE = {'customers': {'mobile_verified', 'first_order_datetime'},
            'orders': {'payment_method', 'num_items_ordered'}, 'labels': set()}
WIDTHS = {'country_code': 2, 'customer_id': 32, 'order_id': 32,
          'mobile_verified': 5, 'collect_type': 32, 'payment_method': 64}


def validate_db_contract(table, frame):
    if table not in COLUMNS:
        raise ValueError('Unknown table')
    missing = set(COLUMNS[table]) - set(frame.columns)
    if missing:
        raise ValueError('Missing database columns: ' + ', '.join(sorted(missing)))
    errors = []
    for column in COLUMNS[table]:
        values = frame[column]
        if column not in NULLABLE[table]:
            count = int(values.isna().sum())
            if count:
                errors.append(f'{column}: {count} null values')
        if column in WIDTHS:
            too_long = int(values.dropna().astype(str).str.len().gt(WIDTHS[column]).sum())
            if too_long:
                errors.append(f'{column}: {too_long} values exceed {WIDTHS[column]} characters')
    if errors:
        raise ValueError(f'{table} incompatible with documented schema: ' + '; '.join(errors))
    return len(frame)

