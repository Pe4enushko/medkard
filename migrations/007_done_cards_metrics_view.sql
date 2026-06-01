-- Migration 007: view for per-date token and time metrics.
--
-- visit_date      — parsed from card_data->'Прием'->>'DATE' (format DD.MM.YYYY)
-- total_cards     — all cards for the date (including ignored)
-- ignored_cards   — cards skipped due to ICD ignore-list
-- token columns   — non-ignored cards only
-- wall_clock_sec  — summed merged audit intervals per date (non-ignored)
-- avg_time_sec    — wall_clock_sec / audited cards (non-ignored)
--
-- Do not derive per-date duration from max(finished_at) - min(started_at):
-- visit_date is the appointment date, not an audit-run date, so cards for the
-- same visit_date may be audited hours or days apart. Do not sum time_ms either:
-- parallel cards would be double-counted. Instead, merge overlapping
-- started_at/finished_at intervals and sum the merged intervals.

DROP VIEW IF EXISTS done_cards_metrics;

CREATE VIEW done_cards_metrics AS
WITH cards AS (
    SELECT
        to_date(card_data -> 'Прием' ->> 'DATE', 'DD.MM.YYYY') AS visit_date,
        ignored,
        token_count,
        started_at,
        finished_at
    FROM done_cards
),
ordered_intervals AS (
    SELECT
        visit_date,
        started_at,
        finished_at,
        max(finished_at) OVER (
            PARTITION BY visit_date
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
        visit_date,
        started_at,
        finished_at,
        sum(starts_new_interval) OVER (
            PARTITION BY visit_date
            ORDER BY started_at, finished_at
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) AS interval_group
    FROM marked_intervals
),
merged_intervals AS (
    SELECT
        visit_date,
        min(started_at) AS started_at,
        max(finished_at) AS finished_at
    FROM grouped_intervals
    GROUP BY visit_date, interval_group
),
active_time AS (
    SELECT
        visit_date,
        sum(extract(epoch FROM finished_at - started_at))::numeric AS wall_clock_sec
    FROM merged_intervals
    GROUP BY visit_date
),
per_date AS (
    SELECT
        visit_date,
        count(*)                                              AS total_cards,
        count(*) FILTER (WHERE ignored = TRUE)                AS ignored_cards,
        count(*) FILTER (WHERE ignored = FALSE)               AS audited_cards,
        sum(token_count) FILTER (WHERE ignored = FALSE)       AS total_tokens,
        round(avg(token_count) FILTER (WHERE ignored = FALSE)) AS avg_tokens
    FROM cards
    GROUP BY visit_date
)
SELECT
    per_date.visit_date,
    per_date.total_cards,
    per_date.ignored_cards,
    per_date.total_tokens,
    per_date.avg_tokens,
    round(active_time.wall_clock_sec, 2) AS wall_clock_sec,
    round(active_time.wall_clock_sec / nullif(per_date.audited_cards, 0), 2) AS avg_time_sec
FROM per_date
LEFT JOIN active_time USING (visit_date)
ORDER BY per_date.visit_date;
