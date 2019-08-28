SELECT NewsType, COUNT(Date)
FROM news
GROUP BY NewsType;