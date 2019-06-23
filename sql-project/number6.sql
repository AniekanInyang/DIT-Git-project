CREATE TABLE targetShedule(
targetShedule_id serial PRIMARY KEY,
address_city VARCHAR (250),
ID integer,
name VARCHAR (100),
fax_number VARCHAR (50),
begintime_mf time,
thrutime_mf time,
isopen_sat boolean,
isopen_sun boolean
);
