SELECT InvoiceNo, SUM(Quantity*UnitPrice) AS SPEND
FROM ecommerce
GROUP BY InvoiceNo;