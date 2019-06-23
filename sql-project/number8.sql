SELECT name, fax_number FROM targetshedule
WHERE address_city = 'Atlanta'
AND isopen_sat = 'true'
AND '22:30:00' BETWEEN begintime_mf AND thrutime_mf;