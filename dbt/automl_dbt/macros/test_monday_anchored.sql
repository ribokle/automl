{% test monday_anchored(model, column_name) %}
    select {{ column_name }}
    from {{ model }}
    where extract('isodow' from {{ column_name }}) <> 1
{% endtest %}
