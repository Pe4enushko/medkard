-- Migration 005: view for per-date token and time metrics.
--
-- visit_date   — parsed from card_data->'Прием'->>'DATE' (format DD.MM.YYYY)
-- total_cards  — all cards for the date (including ignored)
-- ignored_cards — cards skipped due to ICD ignore-list
-- total_tokens / total_time_ms / avg_tokens / avg_time_ms — non-ignored cards only

CREATE OR REPLACE VIEW done_cards_metrics AS
SELECT
    to_date(card_data -> 'Прием' ->> 'DATE', 'DD.MM.YYYY')   AS visit_date,
    count(*)                                                   AS total_cards,
    count(*) FILTER (WHERE ignored = TRUE)                     AS ignored_cards,
    sum(token_count)   FILTER (WHERE ignored = FALSE)          AS total_tokens,
    round(avg(token_count) FILTER (WHERE ignored = FALSE))     AS avg_tokens,
    sum(time_ms)       FILTER (WHERE ignored = FALSE)          AS total_time_ms,
    round(avg(time_ms)     FILTER (WHERE ignored = FALSE))     AS avg_time_ms
FROM done_cards
GROUP BY visit_date
ORDER BY visit_date;
