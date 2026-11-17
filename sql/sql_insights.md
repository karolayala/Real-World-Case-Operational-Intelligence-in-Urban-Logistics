1. Average delivery time, by experience years of delivery people considering only those with at least 50 deliveries and who are still active

```sql
WITH repartidores_validos AS (
    SELECT dvp.delivery_person_id,
    CAST((julianday(MAX(dv.order_placed_at)) - julianday(dvp.hired_date)) / 365 AS INTEGER) AS experience_years
    FROM deliveries dv
    JOIN delivery_persons dvp
    ON dv.delivery_person_id = dvp.delivery_person_id
    WHERE dvp.is_active = 1
    GROUP BY dvp.delivery_person_id, dvp.hired_date
    HAVING COUNT(*) >= 50
)
SELECT rv.experience_years,
ROUND(AVG(dv.delivery_time_min), 2) AS avg_delivery_time,
COUNT(*)  AS no_deliveries,
COUNT(DISTINCT rv.delivery_person_id) AS no_delivery_persons
FROM repartidores_validos rv
JOIN deliveries dv
ON dv.delivery_person_id = rv.delivery_person_id
GROUP BY rv.experience_years
ORDER BY avg_delivery_time DESC;
```

2. Average delivery time, by region of delivery people considering only those with at least 50 deliveries and who are still active


```sql
WITH repartidores_validos AS (
    SELECT dvp.delivery_person_id,
    dvp.region
    FROM deliveries dv
    JOIN delivery_persons dvp
    ON dv.delivery_person_id = dvp.delivery_person_id
    WHERE dvp.is_active = 1
    GROUP BY dvp.delivery_person_id, dvp.region
    HAVING COUNT(*) >= 50
)
SELECT rv.region,
ROUND(AVG(dv.delivery_time_min), 2) AS avg_delivery_time,
COUNT(*)  AS no_deliveries,
COUNT(DISTINCT rv.delivery_person_id) AS no_delivery_persons
FROM repartidores_validos rv
JOIN deliveries dv
ON dv.delivery_person_id = rv.delivery_person_id
GROUP BY rv.region
ORDER BY avg_delivery_time DESC;
```

3. Top 10 customers with the highest percentage of slow deliveries (slowest 25%), considering only those with at least 4 orders.

```sql
WITH entregas_lentas AS (
    SELECT ord.customer_id,
    dv.delivery_time_min,
    NTILE(4) OVER (ORDER BY dv.delivery_time_min) = 4 AS es_lenta
    FROM orders ord
    JOIN deliveries dv ON dv.delivery_id = ord.delivery_id
)
SELECT customer_id,
COUNT(*) AS no_orders,
SUM(es_lenta)  AS no_lentas,
ROUND(AVG(es_lenta) * 100, 1) AS pct_lentas,
ROUND(AVG(delivery_time_min), 2)  AS avg_delivery_time
FROM entregas_lentas
GROUP BY customer_id
HAVING COUNT(*) >= 4
ORDER BY pct_lentas DESC, avg_delivery_time DESC
LIMIT 10;
```

4. Do higher-value orders take longer on average?

```sql
WITH valor_orden AS (
    SELECT ord.order_value,
    dv.delivery_time_min,
    NTILE(4) OVER (ORDER BY ord.order_value)  AS cuartil_valor,
    NTILE(4) OVER (ORDER BY dv.delivery_time_min) = 4 AS es_lenta
    FROM orders ord
    JOIN deliveries dv ON dv.delivery_id = ord.delivery_id
)
SELECT cuartil_valor,
ROUND(MIN(order_value), 2) AS valor_min,
ROUND(MAX(order_value), 2) AS valor_max,
ROUND(AVG(delivery_time_min), 2) AS avg_delivery_time
FROM valor_orden
GROUP BY cuartil_valor
ORDER BY AVG(delivery_time_min) DESC;
```