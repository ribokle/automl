-- Detect missing weeks per (sku, store_id). Returns offending (sku, store) pairs.
{{ config(severity='warn') }}

with bounds as (
    select sku, store_id, min(week_start) as min_w, max(week_start) as max_w, count(*) as n_obs
    from {{ ref('panel') }}
    group by 1, 2
)
select sku, store_id, min_w, max_w, n_obs,
       date_diff('week', min_w, max_w) + 1 as expected_obs
from bounds
where n_obs <> date_diff('week', min_w, max_w) + 1
