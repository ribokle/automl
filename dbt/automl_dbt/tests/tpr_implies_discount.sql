-- Rows where TPR is on but price is not below base. Returns offending rows.
{{ config(severity='warn') }}

select sku, week_start, store_id, price, base_price, tpr_flag
from {{ ref('panel') }}
where tpr_flag = 1 and price >= base_price
