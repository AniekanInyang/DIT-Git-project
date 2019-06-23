-- NONE, There is no store in Columbus that does not open on sunday

SELECT id FROM targetshedule
WHERE address_city = 'Columbus'
AND isopen_sun = 'false';