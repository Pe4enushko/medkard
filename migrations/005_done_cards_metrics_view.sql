-- Migration 010: view for per-date token and time metrics.
--
-- visit_date is parsed from card_data->'Прием'->>'DATE' (format DD.MM.YYYY).
-- Only non-ignored rows with non-null metrics are included.

CREATE OR REPLACE VIEW done_cards_metrics AS
SELECT
    to_date(card_data -> 'Прием' ->> 'DATE', 'DD.MM.YYYY') AS visit_date,
    count(*)                                                 AS total_cards,
    sum(token_count)                                         AS total_tokens,
    round(avg(token_count))                                  AS avg_tokens,
    sum(time_ms)                                             AS total_time_ms,
    round(avg(time_ms))                                      AS avg_time_ms
FROM done_cards
WHERE ignored = FALSE
  AND token_count IS NOT NULL
  AND time_ms IS NOT NULL
GROUP BY visit_date
ORDER BY visit_date;
