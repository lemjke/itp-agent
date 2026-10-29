-- Схема прототипа: индивидуальные тепловые пункты (ИТП), часовые измерения,
-- эталонная разметка внесённых аномалий и таблица нормативных требований.

DROP TABLE IF EXISTS anomalies_truth, measurements, requirements, substations CASCADE;

CREATE TABLE substations (
    id            int PRIMARY KEY,
    name          text NOT NULL,
    address       text NOT NULL,
    building_type text NOT NULL
);

-- Часовые измерения. NULL = пропуск данных.
CREATE TABLE measurements (
    ts            timestamp NOT NULL,
    substation_id int NOT NULL REFERENCES substations(id),
    t_outdoor     real,   -- температура наружного воздуха, °C
    t1            real,   -- температура в подающем трубопроводе, °C
    t2            real,   -- температура в обратном трубопроводе, °C
    t_dhw         real,   -- температура горячей воды (ГВС), °C
    p1            real,   -- давление в подающем, бар
    p2            real,   -- давление в обратном, бар
    flow          real,   -- расход теплоносителя, т/ч
    PRIMARY KEY (substation_id, ts)
);

-- Где и что было внесено генератором (эталон для оценки качества).
CREATE TABLE anomalies_truth (
    id            serial PRIMARY KEY,
    substation_id int NOT NULL REFERENCES substations(id),
    category      text NOT NULL,   -- sensor_fault | regime_violation
    kind          text NOT NULL,   -- shift | drift | spike | frozen | gap | underheat | high_return | dhw_low
    parameter     text NOT NULL,
    start_ts      timestamp NOT NULL,
    end_ts        timestamp NOT NULL,
    description   text NOT NULL
);

CREATE TABLE requirements (
    code          text PRIMARY KEY,
    parameter     text NOT NULL,
    title         text NOT NULL,
    rule          text NOT NULL,
    source        text NOT NULL,
    verified      boolean NOT NULL DEFAULT false   -- проверено экспертом-теплотехником
);
