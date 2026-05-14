-- Migration 007: view for per-date token and time metrics.
--
-- visit_date      — parsed from card_data->'Прием'->>'DATE' (format DD.MM.YYYY)
-- total_cards     — all cards for the date (including ignored)
-- ignored_cards   — cards skipped due to ICD ignore-list
-- token columns   — non-ignored cards only
-- wall_clock_sec  — max(finished_at) - min(started_at) per date (actual elapsed time)
-- avg_time_sec    — wall_clock_sec / audited cards (non-ignored)

CREATE OR REPLACE VIEW done_cards_metrics AS
SELECT
    to_date(card_data -> 'Прием' ->> 'DATE', 'DD.MM.YYYY')        AS visit_date,
    count(*)                                                        AS total_cards,
    count(*) FILTER (WHERE ignored = TRUE)                          AS ignored_cards,
    sum(token_count)   FILTER (WHERE ignored = FALSE)               AS total_tokens,
    round(avg(token_count) FILTER (WHERE ignored = FALSE))          AS avg_tokens,
    round(extract(epoch FROM (
        max(finished_at) FILTER (WHERE ignored = FALSE) -
        min(started_at)  FILTER (WHERE ignored = FALSE)
    ))::numeric, 2)                                                 AS wall_clock_sec,
    round(extract(epoch FROM (
        max(finished_at) FILTER (WHERE ignored = FALSE) -
        min(started_at)  FILTER (WHERE ignored = FALSE)
    ))::numeric / nullif(count(*) FILTER (WHERE ignored = FALSE), 0), 2) AS avg_time_sec
FROM done_cards
GROUP BY visit_date
ORDER BY visit_date;
