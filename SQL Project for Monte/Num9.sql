SELECT Description, COUNT(Quantity)
FROM ecommerce
GROUP BY Description
ORDER BY COUNT(Quantity) DESC;