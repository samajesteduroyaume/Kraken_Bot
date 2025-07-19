-- Connexion à la base de données en tant que postgres (exécuter séparément)
-- psql -U postgres -c "CREATE USER kraken_bot WITH PASSWORD 'kraken_bot_password' CREATEDB SUPERUSER;"
-- psql -U postgres -c "CREATE DATABASE kraken_bot WITH OWNER = kraken_bot;"

-- Connexion à la base de données (exécuter séparément)
-- psql -U kraken_bot -d kraken_bot -f ce_fichier.sql

-- Création des tables
CREATE TABLE IF NOT EXISTS market_data (
    id SERIAL PRIMARY KEY,
    pair VARCHAR(20) NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    open_price DECIMAL(20,10) NOT NULL,
    high_price DECIMAL(20,10) NOT NULL,
    low_price DECIMAL(20,10) NOT NULL,
    close_price DECIMAL(20,10) NOT NULL,
    volume DECIMAL(20,10) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS trades (
    id SERIAL PRIMARY KEY,
    pair VARCHAR(20) NOT NULL,
    strategy_id VARCHAR(50) NOT NULL,
    direction VARCHAR(10) NOT NULL,
    entry_price DECIMAL(20,10) NOT NULL,
    entry_time TIMESTAMP NOT NULL,
    exit_price DECIMAL(20,10),
    exit_time TIMESTAMP,
    quantity DECIMAL(20,10) NOT NULL,
    profit_loss DECIMAL(20,10),
    status VARCHAR(20) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS orders (
    id SERIAL PRIMARY KEY,
    trade_id INTEGER REFERENCES trades(id),
    order_id VARCHAR(50) NOT NULL,
    type VARCHAR(20) NOT NULL,
    side VARCHAR(10) NOT NULL,
    price DECIMAL(20,10),
    size DECIMAL(20,10),
    status VARCHAR(20) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS strategies (
    id VARCHAR(50) PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    parameters JSONB NOT NULL,
    active BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS performance_metrics (
    id SERIAL PRIMARY KEY,
    strategy_id VARCHAR(50) REFERENCES strategies(id),
    pair VARCHAR(20) NOT NULL,
    metric_name VARCHAR(50) NOT NULL,
    value DECIMAL(20,10) NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Création des index
CREATE INDEX idx_market_data_pair ON market_data(pair);
CREATE INDEX idx_market_data_timestamp ON market_data(timestamp);
CREATE INDEX idx_trades_strategy ON trades(strategy_id);
CREATE INDEX idx_trades_pair ON trades(pair);
CREATE INDEX idx_orders_trade ON orders(trade_id);
CREATE INDEX idx_performance_metrics ON performance_metrics(strategy_id, pair, timestamp);
