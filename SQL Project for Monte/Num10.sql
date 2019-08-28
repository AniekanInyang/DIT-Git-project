SELECT Customer AS Customer_ID, SUM(Quantity) AS Total_Quantity,SUM(UnitPrice) AS Total_Unit_Price
FROM  ecommerce
GROUP BY Customer_ID
ORDER BY Total_Unit_Price DESC;