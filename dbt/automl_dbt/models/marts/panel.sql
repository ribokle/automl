{{ config(materialized='table') }}

-- Canonical panel: deduplicated, ordered. Downstream agents read from this.
select
    sku,
    week_start,
    store_id,
    region,
    category,
    brand,
    pack_size,
    segment,
    ppg_id,
    units,
    price,
    base_price,
    case when base_price > 0 then 1.0 - (price / base_price) else 0.0 end as discount_depth,
    tpr_flag,
    display_flag,
    feature_flag,
    distribution_acv,
    competitor_price,
    holiday
from {{ ref('stg_panel') }}
