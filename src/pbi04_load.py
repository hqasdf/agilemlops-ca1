"""Load the three cleaned CA1 exports into the shared MySQL schema."""
import argparse
import csv
import getpass
import os
from pathlib import Path



TABLES = {
    "customers": (
        "country_code", "customer_id", "mobile_verified",
        "num_orders_last_50days", "num_cancelled_orders_last_50days",
        "num_refund_orders_last_50days", "total_payment_last_50days",
        "num_associated_customers", "first_order_datetime",
    ),
    "orders": (
        "country_code", "order_id", "collect_type", "payment_method",
        "order_value", "num_items_ordered", "refund_value", "order_date",
    ),
    "labels": ("country_code", "order_id", "customer_id", "is_fraud"),
}


def load_table(cursor, directory, table, columns):
    names = ", ".join(columns)
    placeholders = ", ".join(["%s"] * len(columns))
    staging = f"pbi04_stage_{table}"
    cursor.execute(f"CREATE TEMPORARY TABLE {staging} LIKE {table}")

    def flush(batch):
        values = ", ".join([f"({placeholders})"] * len(batch))
        params = tuple(value for row in batch for value in row)
        cursor.execute(f"INSERT INTO {staging} ({names}) VALUES {values}", params)

    count = 0
    seen = set()
    batch = []
    with (directory / f"{table}_cleaned.csv").open(encoding="utf-8-sig", newline="") as source:
        rows = csv.reader(source)
        if tuple(next(rows, ())) != columns:
            raise ValueError(f"Unexpected columns in {table}_cleaned.csv")
        for line, row in enumerate(rows, start=2):
            if len(row) != len(columns) or not all(row[:2]):
                raise ValueError(f"Invalid row in {table}_cleaned.csv at line {line}")
            key = tuple(row[:2])
            if key in seen:
                raise ValueError(f"Duplicate key in {table}_cleaned.csv at line {line}")
            seen.add(key)
            batch.append(tuple(value if value != "" else None for value in row))
            count += 1
            if len(batch) == 1000:
                flush(batch)
                batch.clear()
        if batch:
            flush(batch)
    key_match = " AND ".join(f"source.{key} = target.{key}" for key in columns[:2])
    same_payload = " AND ".join(
        f"(CAST(source.{column} AS BINARY) <=> CAST(target.{column} AS BINARY))"
        for column in columns[2:]
    )
    cursor.execute(
        f"SELECT COUNT(*), COALESCE(SUM(NOT ({same_payload})), 0) "
        f"FROM {staging} source JOIN {table} target ON {key_match}"
    )
    existing, conflicts = cursor.fetchone()
    if conflicts:
        raise ValueError(f"{table}: {conflicts:,} conflicting keys. Entire batch rolled back.")
    cursor.execute(
        f"INSERT INTO {table} ({names}) SELECT {names} FROM {staging} source "
        f"WHERE NOT EXISTS (SELECT 1 FROM {table} target WHERE {key_match})"
    )
    if cursor.rowcount != count - existing:
        raise RuntimeError(f"{table}: concurrent changes detected. Retry this batch.")
    return count, existing


def report_relationships(cursor):
    queries = {
        "labels without orders": (
            "SELECT COUNT(*) FROM labels l LEFT JOIN orders o "
            "ON l.country_code=o.country_code AND l.order_id=o.order_id "
            "WHERE o.order_id IS NULL"
        ),
        "orders without labels": (
            "SELECT COUNT(*) FROM orders o LEFT JOIN labels l "
            "ON o.country_code=l.country_code AND o.order_id=l.order_id "
            "WHERE l.order_id IS NULL"
        ),
        "labels without customers": (
            "SELECT COUNT(*) FROM labels l LEFT JOIN customers c "
            "ON l.country_code=c.country_code AND l.customer_id=c.customer_id "
            "WHERE c.customer_id IS NULL"
        ),
    }
    for name, query in queries.items():
        cursor.execute(query)
        print(f"{name}: {cursor.fetchone()[0]:,}")


def main():
    import pymysql
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="Directory containing the cleaned CSVs")
    env = os.environ.get
    parser.add_argument("--host", default=env("CA1_DB_HOST"), required=not env("CA1_DB_HOST"), help="default: CA1_DB_HOST")
    parser.add_argument("--port", type=int, default=int(env("CA1_DB_PORT", "3306")), help="default: CA1_DB_PORT or 3306")
    parser.add_argument("--user", default=env("CA1_DB_USER"), required=not env("CA1_DB_USER"), help="default: CA1_DB_USER")
    parser.add_argument("--database", default=env("CA1_DB_NAME"), required=not env("CA1_DB_NAME"), help="default: CA1_DB_NAME")
    parser.add_argument("--ssl-ca", default=env("CA1_DB_SSL_CA"), required=not env("CA1_DB_SSL_CA"), help="default: CA1_DB_SSL_CA")
    args = parser.parse_args()
    for table in TABLES:
        if not (args.directory / f"{table}_cleaned.csv").is_file():
            parser.error(f"Missing {table}_cleaned.csv in {args.directory}")

    password = os.environ.get("CA1_DB_PASSWORD") or getpass.getpass("MySQL password: ")
    connection = pymysql.connect(
        host=args.host, port=args.port, user=args.user, password=password, database=args.database,
        ssl_ca=args.ssl_ca, ssl_verify_cert=True, ssl_verify_identity=True,
        charset="utf8mb4", autocommit=False, connect_timeout=10,
    )
    try:
        with connection.cursor() as cursor:
            cursor.execute("SET SESSION sql_mode = 'STRICT_ALL_TABLES,NO_ENGINE_SUBSTITUTION'")
            cursor.execute("SET TRANSACTION ISOLATION LEVEL SERIALIZABLE")
            connection.begin()
            counts = {
                table: load_table(cursor, args.directory, table, columns)
                for table, columns in TABLES.items()
            }
            for table, (source_count, unchanged) in counts.items():
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                print(f"{table}: {source_count - unchanged:,} inserted, {unchanged:,} unchanged, {cursor.fetchone()[0]:,} database rows")
            report_relationships(cursor)
        connection.commit()
        print("Import committed.")
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


if __name__ == "__main__":
    main()


