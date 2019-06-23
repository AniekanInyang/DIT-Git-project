SELECT article, dates, heading
FROM newsarticles 
WHERE extract(ISOYEAR from dates) = 2016
AND heading ILIKE '%FIFA%';