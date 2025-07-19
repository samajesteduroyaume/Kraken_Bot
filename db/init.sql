-- Script d'initialisation de la base de données PostgreSQL pour Kraken Trading Bot

-- Création de l'extension pour les UUID si elle n'existe pas déjà
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Table des paires de trading
CREATE TABLE IF NOT EXISTS pairs (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL UNIQUE,
    base_currency VARCHAR(10) NOT NULL,
    quote_currency VARCHAR(10) NOT NULL,
    is_active BOOLEAN DEFAULT TRUE,
    min_trade_size DECIMAL(20, 10),
    max_trade_size DECIMAL(20, 10),
    tick_size DECIMAL(20, 10),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Index pour la recherche par symbole
CREATE INDEX IF NOT EXISTS idx_pairs_symbol ON pairs(symbol);

-- Table des données de marché
CREATE TABLE IF NOT EXISTS market_data (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    pair_id INTEGER REFERENCES pairs(id) ON DELETE CASCADE,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    open DECIMAL(20, 10) NOT NULL,
    high DECIMAL(20, 10) NOT NULL,
    low DECIMAL(20, 10) NOT NULL,
    close DECIMAL(20, 10) NOT NULL,
    volume DECIMAL(30, 10) NOT NULL,
    vwap DECIMAL(20, 10),
    count INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(pair_id, timestamp)
);

-- Index pour les requêtes temporelles
CREATE INDEX IF NOT EXISTS idx_market_data_timestamp ON market_data(timestamp);
CREATE INDEX IF NOT EXISTS idx_market_data_pair_timestamp ON market_data(pair_id, timestamp);

-- Table des transactions
CREATE TABLE IF NOT EXISTS trades (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    trade_id VARCHAR(50) NOT NULL UNIQUE,
    pair_id INTEGER REFERENCES pairs(id) ON DELETE CASCADE,
    order_id VARCHAR(100),
    order_type VARCHAR(20) NOT NULL, -- 'buy' ou 'sell'
    order_side VARCHAR(10) NOT NULL, -- 'buy' ou 'sell'
    price DECIMAL(20, 10) NOT NULL,
    volume DECIMAL(30, 10) NOT NULL,
    cost DECIMAL(30, 10) NOT NULL,
    fee DECIMAL(20, 10) NOT NULL,
    fee_currency VARCHAR(10) NOT NULL,
    status VARCHAR(20) NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Index pour les requêtes sur les transactions
CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON trades(timestamp);
CREATE INDEX IF NOT EXISTS idx_trades_pair_id ON trades(pair_id);
CREATE INDEX IF NOT EXISTS idx_trades_status ON trades(status);

-- Table des ordres
CREATE TABLE IF NOT EXISTS orders (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    order_id VARCHAR(100) NOT NULL UNIQUE,
    client_order_id VARCHAR(100),
    pair_id INTEGER REFERENCES pairs(id) ON DELETE CASCADE,
    order_type VARCHAR(20) NOT NULL, -- 'limit', 'market', 'stop-loss', 'take-profit', etc.
    order_side VARCHAR(10) NOT NULL, -- 'buy' ou 'sell'
    price DECIMAL(20, 10),
    stop_price DECIMAL(20, 10),
    amount DECIMAL(30, 10) NOT NULL,
    filled DECIMAL(30, 10) DEFAULT 0,
    remaining DECIMAL(30, 10),
    cost DECIMAL(30, 10),
    status VARCHAR(20) NOT NULL, -- 'open', 'closed', 'canceled', 'expired', 'rejected'
    fee DECIMAL(20, 10),
    fee_currency VARCHAR(10),
    trades JSONB,
    params JSONB,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    closed_at TIMESTAMP WITH TIME ZONE
);

-- Index pour les requêtes sur les ordres
CREATE INDEX IF NOT EXISTS idx_orders_timestamp ON orders(timestamp);
CREATE INDEX IF NOT EXISTS idx_orders_status ON orders(status);
CREATE INDEX IF NOT EXISTS idx_orders_pair_id ON orders(pair_id);

-- Table des signaux de trading
CREATE TABLE IF NOT EXISTS signals (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    pair_id INTEGER REFERENCES pairs(id) ON DELETE CASCADE,
    signal_type VARCHAR(20) NOT NULL, -- 'buy', 'sell', 'hold'
    price DECIMAL(20, 10) NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    indicators JSONB,
    confidence DECIMAL(5, 4), -- Niveau de confiance du signal (0-1)
    source VARCHAR(50) NOT NULL, -- 'strategy', 'ml_model', 'manual'
    is_executed BOOLEAN DEFAULT FALSE,
    order_id UUID REFERENCES orders(id) ON DELETE SET NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Index pour les requêtes sur les signaux
CREATE INDEX IF NOT EXISTS idx_signals_timestamp ON signals(timestamp);
CREATE INDEX IF NOT EXISTS idx_signals_pair_id ON signals(pair_id);
CREATE INDEX IF NOT EXISTS idx_signals_signal_type ON signals(signal_type);

-- Table des performances du bot
CREATE TABLE IF NOT EXISTS performance_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    equity DECIMAL(30, 10) NOT NULL,
    balance DECIMAL(30, 10) NOT NULL,
    profit_loss DECIMAL(30, 10),
    profit_loss_percent DECIMAL(10, 4),
    drawdown DECIMAL(10, 4),
    sharpe_ratio DECIMAL(10, 4),
    sortino_ratio DECIMAL(10, 4),
    win_rate DECIMAL(5, 2),
    total_trades INTEGER DEFAULT 0,
    winning_trades INTEGER DEFAULT 0,
    losing_trades INTEGER DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Index pour les requêtes de performance
CREATE INDEX IF NOT EXISTS idx_performance_metrics_timestamp ON performance_metrics(timestamp);

-- Table des logs d'erreurs
CREATE TABLE IF NOT EXISTS error_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    level VARCHAR(20) NOT NULL, -- 'error', 'warning', 'info', 'debug'
    source VARCHAR(100) NOT NULL,
    message TEXT NOT NULL,
    traceback TEXT,
    context JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Index pour les requêtes de logs
CREATE INDEX IF NOT EXISTS idx_error_logs_timestamp ON error_logs(timestamp);
CREATE INDEX IF NOT EXISTS idx_error_logs_level ON error_logs(level);
CREATE INDEX IF NOT EXISTS idx_error_logs_source ON error_logs(source);

-- Fonction pour mettre à jour automatiquement les champs updated_at
CREATE OR REPLACE FUNCTION update_modified_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Déclencheurs pour la mise à jour automatique des champs updated_at
CREATE TRIGGER update_pairs_modtime
    BEFORE UPDATE ON pairs
    FOR EACH ROW EXECUTE FUNCTION update_modified_column();

CREATE TRIGGER update_trades_modtime
    BEFORE UPDATE ON trades
    FOR EACH ROW EXECUTE FUNCTION update_modified_column();

CREATE TRIGGER update_orders_modtime
    BEFORE UPDATE ON orders
    FOR EACH ROW EXECUTE FUNCTION update_modified_column();

-- Insertion des paires de trading par défaut
INSERT INTO pairs (symbol, base_currency, quote_currency, is_active, min_trade_size, tick_size)
VALUES 
    ('XBT/USD', 'XBT', 'USD', TRUE, 0.0002, 0.1),
    ('ETH/USD', 'ETH', 'USD', TRUE, 0.02, 0.01),
    ('XRP/USD', 'XRP', 'USD', TRUE, 1, 0.0001),
    ('LTC/USD', 'LTC', 'USD', TRUE, 0.1, 0.01),
    ('BCH/USD', 'BCH', 'USD', TRUE, 0.01, 0.01),
    ('DOT/USD', 'DOT', 'USD', TRUE, 0.1, 0.001),
    ('LINK/USD', 'LINK', 'USD', TRUE, 0.1, 0.001),
    ('ADA/USD', 'ADA', 'USD', TRUE, 1, 0.0001),
    ('XLM/USD', 'XLM', 'USD', TRUE, 10, 0.00001),
    ('EOS/USD', 'EOS', 'USD', TRUE, 0.3, 0.0001)
ON CONFLICT (symbol) DO NOTHING;

-- Création d'un rôle en lecture seule pour les requêtes d'analyse
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'readonly') THEN
        CREATE ROLE readonly WITH LOGIN PASSWORD 'readonly_password' NOSUPERUSER INHERIT NOCREATEDB NOCREATEROLE NOREPLICATION VALID UNTIL 'infinity';
    END IF;
END
$$;

-- Attribution des droits en lecture seule
GRANT CONNECT ON DATABASE kraken_db TO readonly;
GRANT USAGE ON SCHEMA public TO readonly;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO readonly;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO readonly;

-- Commentaires pour la documentation
COMMENT ON TABLE pairs IS 'Liste des paires de trading disponibles sur Kraken avec leurs métadonnées';
COMMENT ON TABLE market_data IS 'Données de marché historiques (OHLCV) pour chaque paire de trading';
COMMENT ON TABLE trades IS 'Historique des transactions exécutées par le bot';
COMMENT ON TABLE orders IS 'Ordres passés par le bot avec leur statut actuel';
COMMENT ON TABLE signals IS 'Signaux de trading générés par les stratégies et modèles';
COMMENT ON TABLE performance_metrics IS 'Métriques de performance du bot au fil du temps';
COMMENT ON TABLE error_logs IS 'Journal des erreurs et avertissements du système';

-- Vues utiles pour l'analyse
CREATE OR REPLACE VIEW vw_daily_performance AS
SELECT 
    DATE_TRUNC('day', timestamp) AS day,
    COUNT(DISTINCT pair_id) AS active_pairs,
    COUNT(*) AS total_signals,
    SUM(CASE WHEN signal_type = 'buy' THEN 1 ELSE 0 END) AS buy_signals,
    SUM(CASE WHEN signal_type = 'sell' THEN 1 ELSE 0 END) AS sell_signals,
    AVG(confidence) AS avg_confidence
FROM 
    signals
GROUP BY 
    DATE_TRUNC('day', timestamp)
ORDER BY 
    day DESC;

-- Fonction pour calculer le rendement quotidien
CREATE OR REPLACE FUNCTION calculate_daily_return()
RETURNS TRIGGER AS $$
BEGIN
    -- Cette fonction serait appelée par un déclencheur après l'insertion de nouvelles métriques de performance
    -- Implémentation simplifiée à des fins d'illustration
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;
