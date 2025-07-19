-- Script d'initialisation amélioré pour Kraken Trading Bot
-- Version: 1.0
-- Date: 2025-07-18

-- Suppression des extensions existantes si nécessaire
DROP EXTENSION IF EXISTS "uuid-ossp";

-- Création de l'extension pour les UUID
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

-- Table des signaux de trading
CREATE TABLE IF NOT EXISTS trading_signals (
    id SERIAL PRIMARY KEY,
    pair TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    signal_type TEXT NOT NULL,
    signal_value TEXT NOT NULL,
    confidence_score DECIMAL(5,2),
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Table des données de marché
CREATE TABLE IF NOT EXISTS market_data (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    pair_id INTEGER NOT NULL REFERENCES pairs(id) ON DELETE CASCADE,
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

-- Table des stratégies
CREATE TABLE IF NOT EXISTS strategies (
    id VARCHAR(50) PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    description TEXT,
    parameters JSONB,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Table des transactions
CREATE TABLE IF NOT EXISTS trades (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    trade_id VARCHAR(50) NOT NULL UNIQUE,
    pair_id INTEGER NOT NULL REFERENCES pairs(id) ON DELETE CASCADE,
    strategy_id VARCHAR(50) REFERENCES strategies(id) ON DELETE SET NULL,
    order_id VARCHAR(100),
    order_type VARCHAR(20) NOT NULL, -- 'market', 'limit', 'stop-loss', etc.
    order_side VARCHAR(10) NOT NULL, -- 'buy' ou 'sell'
    price DECIMAL(20, 10) NOT NULL,
    size DECIMAL(20, 10) NOT NULL,
    filled_size DECIMAL(20, 10) DEFAULT 0,
    fee DECIMAL(20, 10) DEFAULT 0,
    status VARCHAR(20) NOT NULL, -- 'open', 'closed', 'canceled', 'expired'
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    closed_at TIMESTAMP WITH TIME ZONE,
    profit_loss DECIMAL(20, 10),
    profit_loss_percentage DECIMAL(10, 4)
);

-- Table des positions
CREATE TABLE IF NOT EXISTS positions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    trade_id UUID NOT NULL REFERENCES trades(id) ON DELETE CASCADE,
    pair_id INTEGER NOT NULL REFERENCES pairs(id) ON DELETE CASCADE,
    strategy_id VARCHAR(50) REFERENCES strategies(id) ON DELETE SET NULL,
    type VARCHAR(10) NOT NULL, -- 'long' ou 'short'
    size DECIMAL(20, 10) NOT NULL,
    entry_price DECIMAL(20, 10) NOT NULL,
    exit_price DECIMAL(20, 10),
    entry_time TIMESTAMP WITH TIME ZONE NOT NULL,
    exit_time TIMESTAMP WITH TIME ZONE,
    status VARCHAR(20) NOT NULL, -- 'open' ou 'closed'
    pnl DECIMAL(20, 10),
    pnl_percentage DECIMAL(10, 4),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Table des indicateurs techniques
CREATE TABLE IF NOT EXISTS indicators (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    pair_id INTEGER NOT NULL REFERENCES pairs(id) ON DELETE CASCADE,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    rsi DECIMAL(5, 2),
    macd DECIMAL(20, 10),
    signal_line DECIMAL(20, 10),
    bollinger_upper DECIMAL(20, 10),
    bollinger_middle DECIMAL(20, 10),
    bollinger_lower DECIMAL(20, 10),
    atr DECIMAL(20, 10),
    vwap DECIMAL(20, 10),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(pair_id, timestamp)
);

-- Table des métriques de performance
CREATE TABLE IF NOT EXISTS performance_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    equity DECIMAL(30, 10) NOT NULL,
    balance DECIMAL(30, 10) NOT NULL,
    total_trades INTEGER DEFAULT 0,
    winning_trades INTEGER DEFAULT 0,
    losing_trades INTEGER DEFAULT 0,
    win_rate DECIMAL(5, 2),
    profit_factor DECIMAL(10, 4),
    max_drawdown DECIMAL(10, 4),
    sharpe_ratio DECIMAL(10, 4),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Table des logs d'erreurs
CREATE TABLE IF NOT EXISTS error_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    error_code VARCHAR(50),
    error_message TEXT NOT NULL,
    stack_trace TEXT,
    context JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Création des index pour optimiser les performances
CREATE INDEX IF NOT EXISTS idx_pairs_symbol ON pairs(symbol);
CREATE INDEX IF NOT EXISTS idx_market_data_pair_timestamp ON market_data(pair_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_trades_pair_status ON trades(pair_id, status);
CREATE INDEX IF NOT EXISTS idx_trades_created_at ON trades(created_at);
CREATE INDEX IF NOT EXISTS idx_positions_status ON positions(status);
CREATE INDEX IF NOT EXISTS idx_indicators_pair_timestamp ON indicators(pair_id, timestamp);

-- Fonction pour mettre à jour automatiquement les champs updated_at
CREATE OR REPLACE FUNCTION update_modified_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Déclencheurs pour mettre à jour automatiquement les champs updated_at
CREATE TRIGGER update_pairs_modtime
    BEFORE UPDATE ON pairs
    FOR EACH ROW EXECUTE FUNCTION update_modified_column();

CREATE TRIGGER update_trades_modtime
    BEFORE UPDATE ON trades
    FOR EACH ROW EXECUTE FUNCTION update_modified_column();

CREATE TRIGGER update_positions_modtime
    BEFORE UPDATE ON positions
    FOR EACH ROW EXECUTE FUNCTION update_modified_column();

-- Vues utiles
CREATE OR REPLACE VIEW vw_open_positions AS
SELECT p.*, s.symbol as pair_symbol
FROM positions p
JOIN pairs s ON p.pair_id = s.id
WHERE p.status = 'open';

CREATE OR REPLACE VIEW vw_trade_history AS
SELECT 
    t.*, 
    s.symbol as pair_symbol,
    st.name as strategy_name
FROM trades t
JOIN pairs s ON t.pair_id = s.id
LEFT JOIN strategies st ON t.strategy_id = st.id
ORDER BY t.created_at DESC;

-- Insertion des données initiales si nécessaire
INSERT INTO strategies (id, name, description, is_active)
VALUES 
    ('mean_reversion', 'Moyenne mobile', 'Stratégie de retour à la moyenne', true),
    ('breakout', 'Cassure de niveau', 'Stratégie de cassure de résistance/support', true)
ON CONFLICT (id) DO NOTHING;
