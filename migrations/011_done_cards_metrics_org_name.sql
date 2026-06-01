-- Migration 011: expose organization name in done_cards_metrics.
--
-- Metrics are grouped by organization and visit date so callers can filter:
--     SELECT * FROM done_cards_metrics WHERE organization_name = 'Alenka';

DROP VIEW IF EXISTS done_cards_metrics;

CREATE VIEW done_cards_metrics AS
WITH cards AS (
    SELECT
        organizations.name AS organization_name,
        to_date(done_cards.card_data -> 'Прием' ->> 'DATE', 'DD.MM.YYYY') AS visit_date,
        done_cards.ignored,
        done_cards.token_count,
        done_cards.started_at,
        done_cards.finished_at
    FROM done_cards
    LEFT JOIN organizations ON organizations.id = done_cards.organization_id
),
ordered_intervals AS (
    SELECT
        organization_name,
        visit_date,
        started_at,
        finished_at,
        max(finished_at) OVER (
            PARTITION BY organization_name, visit_date
            ORDER BY started_at, finished_at
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
        ) AS previous_finished_at
    FROM cards
    WHERE ignored = FALSE
      AND started_at IS NOT NULL
      AND finished_at IS NOT NULL
),
marked_intervals AS (
    SELECT
        organization_name,
        visit_date,
        started_at,
        finished_at,
        CASE
            WHEN previous_finished_at IS NULL OR started_at > previous_finished_at THEN 1
            ELSE 0
        END AS starts_new_interval
    FROM ordered_intervals
),
grouped_intervals AS (
    SELECT
        organization_name,
        visit_date,
        started_at,
        finished_at,
        sum(starts_new_interval) OVER (
            PARTITION BY organization_name, visit_date
            ORDER BY started_at, finished_at
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) AS interval_group
    FROM marked_intervals
),
merged_intervals AS (
    SELECT
        organization_name,
        visit_date,
        min(started_at) AS started_at,
        max(finished_at) AS finished_at
    FROM grouped_intervals
    GROUP BY organization_name, visit_date, interval_group
),
active_time AS (
    SELECT
        organization_name,
        visit_date,
        sum(extract(epoch FROM finished_at - started_at))::numeric AS wall_clock_sec
    FROM merged_intervals
    GROUP BY organization_name, visit_date
),
per_date AS (
    SELECT
        organization_name,
        visit_date,
        count(*)                                               AS total_cards,
        count(*) FILTER (WHERE ignored = TRUE)                 AS ignored_cards,
        count(*) FILTER (WHERE ignored = FALSE)                AS audited_cards,
        sum(token_count) FILTER (WHERE ignored = FALSE)        AS total_tokens,
        round(avg(token_count) FILTER (WHERE ignored = FALSE)) AS avg_tokens
    FROM cards
    GROUP BY organization_name, visit_date
)
SELECT
    per_date.visit_date,
    per_date.total_cards,
    per_date.ignored_cards,
    per_date.total_tokens,
    per_date.avg_tokens,
    round(active_time.wall_clock_sec, 2) AS wall_clock_sec,
    round(active_time.wall_clock_sec / nullif(per_date.audited_cards, 0), 2) AS avg_time_sec,
    per_date.organization_name
FROM per_date
LEFT JOIN active_time
    ON active_time.organization_name IS NOT DISTINCT FROM per_date.organization_name
   AND active_time.visit_date = per_date.visit_date
ORDER BY per_date.organization_name, per_date.visit_date;
