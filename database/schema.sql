-- ============================================================
-- Manchester United Performance Intelligence Platform
-- PostgreSQL Database Schema
-- Author: Yash Chabukswar | MSc Data Science, UoM
-- ============================================================

-- Matches
CREATE TABLE IF NOT EXISTS matches (
    match_id        BIGINT PRIMARY KEY,
    match_date      DATE,
    opponent        VARCHAR(100),
    venue           VARCHAR(10) CHECK (venue IN ('Home', 'Away')),
    mu_score        SMALLINT,
    opp_score       SMALLINT,
    result          CHAR(1) CHECK (result IN ('W', 'D', 'L')),
    created_at      TIMESTAMP DEFAULT NOW()
);

-- Players
CREATE TABLE IF NOT EXISTS players (
    player_id       SERIAL PRIMARY KEY,
    player_name     VARCHAR(150) UNIQUE NOT NULL,
    team            VARCHAR(100),
    position        VARCHAR(50)
);

-- Raw Events (partitioned by match for performance)
CREATE TABLE IF NOT EXISTS events (
    event_id        UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    match_id        BIGINT REFERENCES matches(match_id),
    event_index     INTEGER,
    period          SMALLINT,
    timestamp       TIME,
    event_type      VARCHAR(50),
    team            VARCHAR(100),
    player          VARCHAR(150),
    location_x      FLOAT,
    location_y      FLOAT,
    outcome         VARCHAR(100),
    created_at      TIMESTAMP DEFAULT NOW()
);
CREATE INDEX idx_events_match ON events(match_id);
CREATE INDEX idx_events_type ON events(event_type);
CREATE INDEX idx_events_team ON events(team);
CREATE INDEX idx_events_player ON events(player);

-- Expected Goals (xG)
CREATE TABLE IF NOT EXISTS xg_shots (
    shot_id         UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    match_id        BIGINT REFERENCES matches(match_id),
    player          VARCHAR(150),
    team            VARCHAR(100),
    location_x      FLOAT,
    location_y      FLOAT,
    distance        FLOAT,
    angle           FLOAT,
    is_header       BOOLEAN,
    is_open_play    BOOLEAN,
    shot_outcome    VARCHAR(50),
    is_goal         BOOLEAN,
    xg_value        FLOAT,
    created_at      TIMESTAMP DEFAULT NOW()
);
CREATE INDEX idx_xg_player ON xg_shots(player);
CREATE INDEX idx_xg_match ON xg_shots(match_id);

-- Player Season Stats
CREATE TABLE IF NOT EXISTS player_season_stats (
    stat_id             SERIAL PRIMARY KEY,
    player              VARCHAR(150),
    team                VARCHAR(100),
    season              VARCHAR(10) DEFAULT '2015/16',
    shots               INTEGER DEFAULT 0,
    goals               INTEGER DEFAULT 0,
    xg_total            FLOAT DEFAULT 0,
    xg_diff             FLOAT DEFAULT 0,
    passes              INTEGER DEFAULT 0,
    pass_accuracy       FLOAT DEFAULT 0,
    pressures           INTEGER DEFAULT 0,
    pressing_eff_rate   FLOAT DEFAULT 0,
    dribble_attempts    INTEGER DEFAULT 0,
    dribble_success     INTEGER DEFAULT 0,
    influence_score     FLOAT DEFAULT 0,
    betweenness         FLOAT DEFAULT 0,
    pre_shot_passes     INTEGER DEFAULT 0,
    created_at          TIMESTAMP DEFAULT NOW(),
    UNIQUE(player, team, season)
);
CREATE INDEX idx_pss_player ON player_season_stats(player);
CREATE INDEX idx_pss_team ON player_season_stats(team);

-- Pass Network Edges
CREATE TABLE IF NOT EXISTS pass_network (
    edge_id         SERIAL PRIMARY KEY,
    match_id        BIGINT REFERENCES matches(match_id),
    passer          VARCHAR(150),
    recipient       VARCHAR(150),
    weight          INTEGER,
    season          VARCHAR(10) DEFAULT '2015/16',
    result_context  CHAR(1),
    UNIQUE(match_id, passer, recipient)
);
CREATE INDEX idx_pn_passer ON pass_network(passer);
CREATE INDEX idx_pn_recipient ON pass_network(recipient);

-- Pressing Events
CREATE TABLE IF NOT EXISTS pressing_events (
    press_id        UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    match_id        BIGINT REFERENCES matches(match_id),
    player          VARCHAR(150),
    team            VARCHAR(100),
    location_x      FLOAT,
    location_y      FLOAT,
    pitch_zone      VARCHAR(30),
    won_ball        BOOLEAN,
    created_at      TIMESTAMP DEFAULT NOW()
);
CREATE INDEX idx_press_player ON pressing_events(player);
CREATE INDEX idx_press_zone ON pressing_events(pitch_zone);

-- Recruitment Profiles
CREATE TABLE IF NOT EXISTS recruitment_profiles (
    profile_id      SERIAL PRIMARY KEY,
    player          VARCHAR(150),
    team            VARCHAR(100),
    season          VARCHAR(10),
    target_player   VARCHAR(150),
    similarity      FLOAT,
    goals           INTEGER,
    passes          INTEGER,
    pass_accuracy   FLOAT,
    pressures       INTEGER,
    conversion_rate FLOAT,
    created_at      TIMESTAMP DEFAULT NOW()
);
CREATE INDEX idx_rp_target ON recruitment_profiles(target_player);
CREATE INDEX idx_rp_similarity ON recruitment_profiles(similarity DESC);

-- ============================================================
-- Analytical Views
-- ============================================================

CREATE OR REPLACE VIEW v_top_xg_performers AS
SELECT player, team,
       SUM(CASE WHEN is_goal THEN 1 ELSE 0 END) AS goals,
       ROUND(SUM(xg_value)::numeric, 2) AS total_xg,
       ROUND((SUM(CASE WHEN is_goal THEN 1 ELSE 0 END) - SUM(xg_value))::numeric, 2) AS xg_diff,
       COUNT(*) AS shots
FROM xg_shots
GROUP BY player, team
ORDER BY total_xg DESC;

CREATE OR REPLACE VIEW v_pressing_effectiveness AS
SELECT player, team,
       COUNT(*) AS total_pressures,
       SUM(CASE WHEN won_ball THEN 1 ELSE 0 END) AS balls_won,
       ROUND(100.0 * SUM(CASE WHEN won_ball THEN 1 ELSE 0 END) / COUNT(*), 1) AS effectiveness_pct,
       pitch_zone
FROM pressing_events
GROUP BY player, team, pitch_zone
HAVING COUNT(*) >= 10
ORDER BY effectiveness_pct DESC;

CREATE OR REPLACE VIEW v_season_summary AS
SELECT result,
       COUNT(*) AS matches,
       ROUND(AVG(mu_score), 2) AS avg_scored,
       ROUND(AVG(opp_score), 2) AS avg_conceded,
       SUM(mu_score) AS total_scored,
       SUM(opp_score) AS total_conceded
FROM matches
GROUP BY result
ORDER BY result;
