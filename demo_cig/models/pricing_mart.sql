CREATE TABLE pricing_mart AS
SELECT 
    customer_id,
    base_price,
    discount_pct,
    base_price * (1 - discount_pct) AS final_price,
    quantity * base_price * (1 - discount_pct) AS revenue
FROM raw_orders;