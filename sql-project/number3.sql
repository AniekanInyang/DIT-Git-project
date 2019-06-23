-- 97 Sport articles were published in 2016.

SELECT COUNT(article) 
FROM newsarticles 
WHERE newstype = 'sports' 
AND extract(ISOYEAR from dates) = 2016;
--AND dates BETWEEN '2016-01-01' AND '2016-12-31';