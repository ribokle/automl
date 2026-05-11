select sku, week_start, store_id, price
from {{ ref('panel') }}
where price <= 0
