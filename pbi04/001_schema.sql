-- Run against the CA1 MySQL database selected by the connection.
-- Apply this migration to a fresh schema. Existing incompatible tables require migration.
CREATE TABLE IF NOT EXISTS customers (
    country_code CHAR(2) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
    customer_id VARCHAR(32) COLLATE utf8mb4_0900_bin NOT NULL,
    mobile_verified VARCHAR(5) NULL,
    num_orders_last_50days INT NOT NULL,
    num_cancelled_orders_last_50days INT NOT NULL,
    num_refund_orders_last_50days INT NOT NULL,
    total_payment_last_50days DECIMAL(38,18) NOT NULL,
    num_associated_customers INT NOT NULL,
    first_order_datetime DATETIME NULL,
    PRIMARY KEY (country_code, customer_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS orders (
    country_code CHAR(2) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
    order_id VARCHAR(32) COLLATE utf8mb4_0900_bin NOT NULL,
    collect_type VARCHAR(32) NOT NULL,
    payment_method VARCHAR(64) NULL,
    order_value DECIMAL(38,18) NOT NULL,
    num_items_ordered INT NULL,
    refund_value DECIMAL(38,18) NOT NULL,
    order_date DATE NOT NULL,
    PRIMARY KEY (country_code, order_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- Labels link orders to customers through country-qualified IDs. The cleaned
-- data contains unmatched labels, so these relationships are checked by queries
-- instead of rejecting those records with foreign-key constraints.
CREATE TABLE IF NOT EXISTS labels (
    country_code CHAR(2) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
    order_id VARCHAR(32) COLLATE utf8mb4_0900_bin NOT NULL,
    customer_id VARCHAR(32) COLLATE utf8mb4_0900_bin NOT NULL,
    is_fraud TINYINT NOT NULL,
    PRIMARY KEY (country_code, order_id),
    KEY labels_customer (country_code, customer_id),
    CHECK (is_fraud IN (0, 1))
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

