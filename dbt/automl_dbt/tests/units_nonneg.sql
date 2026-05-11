select sku, week_start, store_id, units
from {{ ref('panel') }}
where units < 0
