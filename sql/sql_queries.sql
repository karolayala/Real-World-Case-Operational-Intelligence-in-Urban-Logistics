-- 1. Top 5 customer areas with highest average delivery time in the last 30 days
SELECT customer_area, ROUND(AVG(delivery_time_min), 2) AS avg_delivery_time
FROM deliveries
WHERE order_placed_at >= (SELECT datetime(MAX(order_placed_at), '-30 days') FROM deliveries)
GROUP BY customer_area
ORDER BY avg_delivery_time DESC
LIMIT 5;

-- 2. Average delivery time per traffic condition, by restaurant area and cuisine type.
SELECT DV.RESTAURANT_AREA,
       RTR.CUISINE_TYPE,
       DV.TRAFFIC_CONDITION,
       ROUND(AVG(DV.DELIVERY_TIME_MIN), 2) AS AVG_DELIVERY_TIME
 FROM DELIVERIES DV
 LEFT JOIN ORDERS ORD ON ORD.DELIVERY_ID = DV.DELIVERY_ID
 LEFT JOIN RESTAURANTS RTR ON RTR.RESTAURANT_ID = ORD.RESTAURANT_ID
 WHERE DV.TRAFFIC_CONDITION IS NOT NULL
 GROUP BY DV.RESTAURANT_AREA, RTR.CUISINE_TYPE, DV.TRAFFIC_CONDITION
 ORDER BY AVG_DELIVERY_TIME DESC

-- 3. Top 10 delivery people with the fastest average delivery time, considering only those with at least 50 deliveries 
-- and who are still active.
SELECT dv.delivery_person_id, 
dvp.name, ROUND(AVG(dv.delivery_time_min), 2) as avg_delivery_time,
COUNT(*) as No_deliveries 
FROM deliveries dv
LEFT JOIN delivery_persons dvp 
ON dv.delivery_person_id = dvp.delivery_person_id
WHERE dvp.is_active = 1
GROUP BY dv.delivery_person_id, dvp.name
HAVING COUNT(*) >= 50
ORDER BY AVG(dv.delivery_time_min) ASC
LIMIT 10

-- 4. The most profitable restaurant area in the last 3 months, defined as the area with the highest total order value.
SELECT dv.restaurant_area, SUM(ord.order_value) AS Total_order_value
FROM deliveries dv
LEFT JOIN orders ord ON dv.delivery_id = ord.delivery_id
WHERE dv.order_placed_at >=  (SELECT DATETIME(MAX(order_placed_at), '-3 months') FROM deliveries )
GROUP BY dv.restaurant_area
ORDER BY SUM(ord.order_value) DESC
LIMIT 1

-- 5. Identify whether any delivery people show an increasing trend in average delivery time.
WITH entregas_person AS (
    SELECT
        dv.delivery_person_id,
        JULIANDAY(dv.order_placed_at)
          - JULIANDAY((SELECT MIN(order_placed_at) FROM deliveries)) AS x, -- días transcurridos desde la primera entrega
        dv.delivery_time_min AS y
    FROM deliveries dv
),
sumas_person AS (
    SELECT
        delivery_person_id,
        COUNT(*)     AS n,     -- número de entregas
        SUM(x)       AS sx,    -- Σx
        SUM(y)       AS sy,    -- Σy
        SUM(x * y)   AS sxy,   -- Σ(x*y)
        SUM(x * x)   AS sxx    -- Σ(x²)
    FROM entregas_person
    GROUP BY delivery_person_id
),
pendientes AS (
    SELECT
        delivery_person_id,
        n,
        30.0 * (n * sxy - sx * sy) / NULLIF(n * sxx - sx * sx, 0) AS pendiente_min_mes
    FROM sumas_person   
)
SELECT
    p.delivery_person_id,
    dvp.name,
    p.n AS no_deliveries,
    ROUND(p.pendiente_min_mes, 2) AS pendiente_min_per_month,
    CASE
        WHEN p.pendiente_min_mes >  1 THEN 'Tendencia a subir'
        WHEN p.pendiente_min_mes < -1 THEN 'Tendencia a bajar'
        ELSE 'Estable'
    END AS tendencia
FROM pendientes p
INNER JOIN delivery_persons dvp
    ON dvp.delivery_person_id = p.delivery_person_id
WHERE p.n >= 30 AND p.pendiente_min_mes >  1
ORDER BY p.pendiente_min_mes DESC
