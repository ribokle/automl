{{ config(materialized='view') }}

-- Type-cast and lightly normalize the raw panel.
select
    cast(sku as varchar)              as sku,
    cast(week_start as date)          as week_start,
    cast(store_id as varchar)         as store_id,
    cast(region as varchar)           as region,
    cast(units as integer)            as units,
    cast(price as double)             as price,
    cast(base_price as double)        as base_price,
    cast(tpr_flag as integer)         as tpr_flag,
    cast(display_flag as integer)     as display_flag,
    cast(feature_flag as integer)     as feature_flag,
    cast(distribution_acv as double)  as distribution_acv,
    cast(competitor_price as double)  as competitor_price,
    cast(holiday as varchar)          as holiday,
    cast(category as varchar)         as category,
    cast(brand as varchar)            as brand,
    cast(pack_size as varchar)        as pack_size,
    cast(segment as varchar)          as segment,
    cast(ppg_id as varchar)           as ppg_id
from {{ source('raw', 'raw_panel') }}
